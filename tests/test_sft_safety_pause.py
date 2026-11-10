"""CPU-only real Trainer proof: epoch schedule survives an operational pause."""
import dataclasses
from functools import partial
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import TrainerControl, TrainerState
from audiogen.training import sft
from test_sft_integration import make_fixture


class SafetyPauseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_optional_operational_config(self):
        self.assertIn('stop_after_step', sft.SFTConfig.__dataclass_fields__)
        self.assertIsNone(sft.SFTConfig().stop_after_step)
        self.assertNotIn('stop_after_step', sft._RESUME_KEYS)

    def test_callback_requests_all_health_actions_only_at_boundary(self):
        self.assertTrue(hasattr(sft, 'SafetyPauseCallback'))
        callback = sft.SafetyPauseCallback(100)
        args = SimpleNamespace(eval_strategy='steps')
        for step in (0, 99, 100, 101):
            control = callback.on_step_end(args, TrainerState(global_step=step), TrainerControl())
            for name in ('should_log', 'should_evaluate', 'should_save', 'should_training_stop'):
                self.assertEqual(getattr(control, name), step >= 100, (step, name))
        with self.assertRaisesRegex(ValueError, 'stop_after_step'):
            callback.on_train_begin(args, TrainerState(global_step=100), TrainerControl())

    def test_cli(self):
        import scripts.train_sft as cli
        with patch('sys.argv', ['train_sft', '--train_path', 'a', '--val_path', 'b', '--output_dir', 'c', '--stop_after_step', '100']), patch.object(cli, 'run_sft') as run:
            cli.main()
        self.assertEqual(run.call_args.args[0].stop_after_step, 100)
        self.assertEqual(run.call_args.args[0].max_steps, -1)

    def test_invalid_stop_fails_before_load(self):
        self.assertIn('stop_after_step', sft.SFTConfig.__dataclass_fields__)
        for value in (0, -1, 1.5, True):
            with patch.object(sft, 'load_base_model') as loader, self.assertRaisesRegex(ValueError, 'stop_after_step'):
                sft.run_sft(sft.SFTConfig(output_dir='unused', data_path='unused', stop_after_step=value))
            loader.assert_not_called()

    def test_epoch_pause_resume_exact_weights_lr_rng_and_token_normalization(self):
        self.assertIn('stop_after_step', sft.SFTConfig.__dataclass_fields__)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'WANDB_MODE': 'disabled'}), patch.object(sft, 'TrainingArguments', partial(sft.TrainingArguments, use_cpu=True, disable_tqdm=True)):
            root = Path(tmp)
            base = make_fixture(root)
            # Different supervised lengths within every accumulation group.
            records = [{'prompt': 'hello', 'tokens': [0, 1] * frames, 'num_codebooks': 2, 'num_frames': frames} for frames in (1, 3, 2, 4, 1, 2, 4, 3)]
            (root / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
            # Match scheduled eval at the pause boundary (as Phase1 does at
            # update100): creating an eval iterator itself consumes RNG in HF.
            # Comparing against a run with a different eval cadence is not a
            # same-policy RNG/weight comparison when the model has dropout.
            cfg = sft.SFTConfig(output_dir=str(root / 'full'), base_model=str(base), train_path=str(root / 'train.jsonl'), val_path=str(root / 'val.jsonl'), num_codebooks=2, codebook_size=2, dtype='float32', prompt_format='speech_v1', max_length=64, max_steps=-1, num_train_epochs=1, warmup_steps=1, lr_scheduler_type='cosine', learning_rate=0.001, gradient_accumulation_steps=2, per_device_train_batch_size=1, per_device_eval_batch_size=1, bf16=False, report_to='none', logging_steps=1, eval_steps=2, save_steps=99, seed=123, gradient_checkpointing=True)
            # Inspect the denominator that the real Trainer supplies for each
            # microbatch. It must be the sum across this accumulation group,
            # not each microbatch's varying number of supervised tokens.
            observed = []
            original = sft.RobustTrainer.compute_loss
            def inspect_loss(trainer, model, inputs, return_outputs=False, num_items_in_batch=None):
                if model.training:
                    observed.append((int((inputs['labels'][..., 1:] != -100).sum()), int(num_items_in_batch) if num_items_in_batch is not None else None))
                result = original(trainer, model, inputs, True, num_items_in_batch)
                if model.training:
                    loss, outputs = result
                    logits = outputs.logits[..., :-1, :].contiguous()
                    labels = inputs['labels'][..., 1:].contiguous()
                    expected = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100, reduction='sum') / num_items_in_batch
                    torch.testing.assert_close(loss, expected)
                return result if return_outputs else result[0]
            with patch.object(sft.RobustTrainer, 'compute_loss', inspect_loss):
                full = sft.run_sft(cfg)
            self.assertEqual(full.state.global_step, 4)
            self.assertEqual(len(observed), 8)
            self.assertGreater(len(set(n for n, _ in observed)), 1)
            for i in range(0, len(observed), 2):
                group = observed[i:i + 2]
                self.assertEqual([d for _, d in group], [sum(n for n, _ in group)] * 2)
            paused_cfg = dataclasses.replace(cfg, output_dir=str(root / 'paused'), stop_after_step=2)
            paused = sft.run_sft(paused_cfg)
            self.assertEqual(paused.state.global_step, 2)
            self.assertEqual(paused.state.max_steps, 4)
            self.assertEqual(paused.args.max_steps, -1)
            checkpoint = root / 'paused' / 'checkpoint-2'
            for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'sft_run_policy.json', 'audio_token_metadata.json'):
                self.assertTrue((checkpoint / name).is_file(), name)
            self.assertTrue(any('eval_loss' in row for row in paused.state.log_history))
            for threshold in (1, 2):
                with patch.object(sft, 'load_trained_model') as loader, self.assertRaisesRegex(ValueError, 'stop_after_step'):
                    sft.run_sft(dataclasses.replace(paused_cfg, stop_after_step=threshold, trainer_resume_from_checkpoint=str(checkpoint)))
                loader.assert_not_called()
            resumed = sft.run_sft(dataclasses.replace(paused_cfg, output_dir=str(root / 'resumed'), stop_after_step=None, trainer_resume_from_checkpoint=str(checkpoint)))
            self.assertEqual(resumed.state.global_step, 4)
            self.assertEqual(resumed.state.max_steps, 4)
            self.assertEqual(resumed.lr_scheduler.state_dict(), full.lr_scheduler.state_dict())
            self.assertEqual([(r['step'], r['learning_rate']) for r in resumed.state.log_history if 'learning_rate' in r], [(r['step'], r['learning_rate']) for r in full.state.log_history if 'learning_rate' in r])
            for a, b in zip(full.model.parameters(), resumed.model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            # A later explicit operational gate is also allowed without touching
            # any scheduler/data/optimizer policy key.
            later = sft.run_sft(dataclasses.replace(paused_cfg, output_dir=str(root / 'later'), stop_after_step=3, trainer_resume_from_checkpoint=str(checkpoint)))
            self.assertEqual(later.state.global_step, 3)
            self.assertEqual(later.state.max_steps, 4)


if __name__ == '__main__':
    unittest.main()
