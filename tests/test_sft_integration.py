import dataclasses
from functools import partial
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from audiogen.model.setup import load_trained_model
from audiogen.training import sft


def make_fixture(root):
    base = root / 'base'
    raw = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'hello': 3, 'user': 4, 'assistant': 5, ':': 6, 'Generate': 7, 'speech': 8, 'audio': 9}, unk_token='[UNK]'))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] }}{% if message['role'] == 'assistant' %}{{ eos_token }}{% else %}{{ ' ' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    tokenizer.save_pretrained(base)
    GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_embd=8, n_layer=1, n_head=2, n_positions=64, bos_token_id=2, eos_token_id=2, pad_token_id=1)).save_pretrained(base)
    record = {'prompt': 'hello', 'tokens': [0, 1, 1, 0], 'num_codebooks': 2, 'num_frames': 2}
    for name, count in [('train', 4), ('val', 1)]:
        (root / f'{name}.jsonl').write_text((json.dumps(record) + '\n') * count)
    return base


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        # CUDA visibility may already be cached by torch/Accelerate. Explicit
        # TrainingArguments isolation also covers resume and plateau runs.
        cpu_args = partial(sft.TrainingArguments, use_cpu=True)
        patcher = patch.object(sft, 'TrainingArguments', cpu_args)
        patcher.start()
        self.addCleanup(patcher.stop)

    def config(self, root, base):
        self.assertIn('train_path', sft.SFTConfig.__dataclass_fields__, 'explicit split config is missing')
        return sft.SFTConfig(output_dir=str(root / 'run'), base_model=str(base), train_path=str(root / 'train.jsonl'), val_path=str(root / 'val.jsonl'), num_codebooks=2, codebook_size=2, dtype='float32', prompt_format='speech_v1', max_length=64, max_steps=2, warmup_steps=0, gradient_accumulation_steps=1, per_device_train_batch_size=1, per_device_eval_batch_size=1, bf16=False, report_to='none', logging_steps=1, eval_steps=1, save_steps=1, seed=123, gradient_checkpointing=True)

    def test_two_real_steps_portable_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'WANDB_MODE': 'disabled'}):
            root = Path(tmp)
            cfg = self.config(root, make_fixture(root))
            original = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (root / 'train.jsonl', root / 'val.jsonl')}
            with patch.object(sft, 'split_records', side_effect=AssertionError('must not resplit')), patch.object(sft, 'save_records_jsonl', side_effect=AssertionError('must not copy')):
                trainer = sft.run_sft(cfg)
            self.assertTrue(trainer.args.use_cpu)
            self.assertEqual(trainer.model.device.type, 'cpu')
            self.assertEqual(trainer.state.global_step, 2)
            self.assertEqual(trainer.train_dataset.prompt_format, 'speech_v1')
            self.assertIs(trainer.processing_class, trainer.train_dataset.tokenizer)
            self.assertEqual(trainer.args.gradient_checkpointing_kwargs, {'use_reentrant': False})
            for p, before in original.items():
                self.assertEqual((p.read_bytes(), p.stat().st_mtime_ns), before)
            self.assertFalse((root / 'run' / 'train.jsonl').exists())
            for path in (root / 'run', root / 'run' / 'checkpoint-1', root / 'run' / 'checkpoint-2'):
                for filename in ('audio_token_metadata.json', 'tokenizer_config.json', 'config.json', 'sft_run_policy.json', 'trainer_state.json'):
                    self.assertTrue((path / filename).is_file(), str(path / filename))
                model, tokenizer, meta = load_trained_model(path)
                self.assertEqual(model.config.audiogen['prompt_format'], 'speech_v1')
                self.assertEqual(tokenizer.convert_tokens_to_ids('[A0_0]'), meta.audio_token_start_id)
            resumed_cfg = dataclasses.replace(cfg, output_dir=str(root / 'resumed'), trainer_resume_from_checkpoint=str(root / 'run' / 'checkpoint-1'))
            resumed = sft.run_sft(resumed_cfg)
            self.assertEqual(resumed.state.global_step, 2)
            for a, b in zip(trainer.model.parameters(), resumed.model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            # Historical --resume_from_checkpoint intentionally means weights-only init.
            fresh = sft.run_sft(dataclasses.replace(cfg, output_dir=str(root / 'fresh'), max_steps=1, resume_from_checkpoint=str(root / 'run' / 'checkpoint-2')))
            self.assertEqual(fresh.state.global_step, 1)
            self.assertEqual(len([x for x in fresh.state.log_history if 'loss' in x]), 1)
            with self.assertRaisesRegex(ValueError, 'policy|freeze'):
                sft.run_sft(dataclasses.replace(resumed_cfg, output_dir=str(root / 'bad'), unfreeze_all=True))
            with self.assertRaisesRegex(ValueError, 'budget|global_step'):
                sft.run_sft(dataclasses.replace(resumed_cfg, output_dir=str(root / 'exhausted'), trainer_resume_from_checkpoint=str(root / 'run' / 'checkpoint-2')))

    def test_plateau_stops_real_trainer_with_resumable_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = dataclasses.replace(self.config(root, make_fixture(root)), max_steps=8, learning_rate=0.0, plateau_patience=1, plateau_min_delta=0.1, save_steps=8)
            trainer = sft.run_sft(cfg)
            self.assertTrue(trainer.args.use_cpu)
            self.assertEqual(trainer.model.device.type, 'cpu')
            self.assertEqual(trainer.state.global_step, 2)
            checkpoint = root / 'run' / 'checkpoint-2'
            self.assertTrue((checkpoint / 'optimizer.pt').is_file())
            self.assertTrue((checkpoint / 'audio_token_metadata.json').is_file())
            state = json.loads((checkpoint / 'trainer_state.json').read_text())
            self.assertEqual(state['stateful_callbacks']['PlateauCallback']['attributes']['bad_evaluations'], 1)
            initialized = sft.run_sft(dataclasses.replace(cfg, output_dir=str(root / 'initialized'), initialize_from_checkpoint=str(checkpoint), max_steps=1, plateau_patience=0))
            self.assertEqual(initialized.state.global_step, 1)

    def test_invalid_inputs_rejected_before_model_load(self):
        self.assertIn('train_path', sft.SFTConfig.__dataclass_fields__)
        with tempfile.TemporaryDirectory() as tmp, patch.object(sft, 'load_base_model') as loader:
            with self.assertRaisesRegex(ValueError, 'train_path.*val_path'):
                sft.run_sft(sft.SFTConfig(output_dir=tmp, train_path='x'))
            with patch.dict(os.environ, {'WORLD_SIZE': '2'}):
                with self.assertRaisesRegex(ValueError, 'single|rank|distributed'):
                    sft.run_sft(sft.SFTConfig(output_dir=tmp, data_path='unused'))
            loader.assert_not_called()

    def test_cli_accepts_prepared_files_and_launch_controls(self):
        import scripts.train_sft as cli
        argv = ['train_sft', '--train_path', 'train.jsonl', '--val_path', 'val.jsonl', '--output_dir', 'out', '--dtype', 'bfloat16', '--revision', 'pinned', '--prompt_format', 'speech_v1', '--seed', '17', '--run_name', 'smoke', '--max_steps', '2', '--eval_batch_size', '1', '--save_total_limit', '2', '--report_to', 'none', '--initialize_from_checkpoint', 'weights', '--plateau_patience', '3', '--plateau_min_steps', '20', '--plateau_warmup_steps', '10', '--plateau_min_delta', '0.01']
        with patch('sys.argv', argv), patch.object(cli, 'run_sft') as run:
            cli.main()
        cfg = run.call_args.args[0]
        self.assertEqual(cfg.train_path, 'train.jsonl')
        self.assertEqual(cfg.dtype, 'bfloat16')
        self.assertEqual(cfg.revision, 'pinned')
        self.assertEqual(cfg.max_steps, 2)
        self.assertEqual(cfg.report_to, 'none')
        self.assertEqual(cfg.initialize_from_checkpoint, 'weights')
        self.assertEqual(cfg.plateau_patience, 3)


if __name__ == '__main__':
    unittest.main()
