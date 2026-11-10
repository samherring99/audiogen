"""Native original-format resume: offline tiny CPU fixtures, opt-in GPU0 smoke."""
import copy
import dataclasses
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import transformers
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

from audiogen.model import setup
from audiogen.model.setup import _load_model
from audiogen.training import sft
from test_nemotron_checkpointing import tiny_model


def digest_tree(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob('*') if p.is_file()}


def original_fixture(root, native=None):
    """Bootstrap source names, then actual from_pretrained -> original save roundtrip."""
    native = tiny_model() if native is None else native
    native.config.attention_dropout = 0.2
    native.config.hidden_dropout = 0.2
    native.config.save_pretrained(root / 'source')
    source = {}
    for key, value in native.state_dict().items():
        key = key.replace('model.', 'backbone.').replace('embeddings.weight', 'embedding.weight')
        if key.endswith(('experts.up_proj', 'experts.down_proj')):
            prefix, proj = key.rsplit('.', 1)
            for i, expert in enumerate(value):
                source[f'{prefix}.{i}.{proj}.weight'] = expert.contiguous()
        else:
            source[key] = value
    save_file(source, root / 'source/model.safetensors')
    loaded, info = AutoModelForCausalLM.from_pretrained(root / 'source', output_loading_info=True)
    assert not any(info.values()), info
    for key, value in native.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    checkpoint = root / 'original'
    loaded.save_pretrained(checkpoint, max_shard_size='2KB')
    keys = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    assert 'backbone.embedding.weight' in keys
    assert any('experts.0.up_proj.weight' in key for key in keys)
    assert len(set(keys.values())) > 1
    return checkpoint


@unittest.skipUnless(transformers.__version__ == '5.6.2', 'Targets native Transformers 5.6.2')
class NemotronResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def trainer(self, model, root, cls=sft.RobustTrainer):
        return cls(model=model, args=TrainingArguments(output_dir=str(root / 'run'),
                   use_cpu=True, report_to='none', disable_tqdm=True))

    def test_raw_hf_reload_warns_leaves_backbone_but_restores_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = original_fixture(root)
            model = _load_model(checkpoint, dtype=torch.float32)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            trainer = self.trainer(model, root, Trainer)
            # Good preloaded weights are retained: warnings alone do NOT prove corruption.
            with self.assertLogs('transformers.trainer', level='WARNING') as logs:
                trainer._load_from_checkpoint(str(checkpoint))
            self.assertIn('missing keys', str(logs.output))
            self.assertIn('unexpected keys', str(logs.output))
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]), key)
            with torch.no_grad():
                model.model.embeddings.weight.add_(1)
                model.lm_head.weight.add_(1)
            with self.assertLogs('transformers.trainer', level='WARNING'):
                trainer._load_from_checkpoint(str(checkpoint))
            self.assertFalse(torch.equal(model.model.embeddings.weight, before['model.embeddings.weight']))
            self.assertTrue(torch.equal(model.lm_head.weight, before['lm_head.weight']))

    def test_verified_preload_avoids_raw_reload_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = original_fixture(root)
            digest = digest_tree(checkpoint)
            for _ in range(2):
                model = _load_model(checkpoint, dtype=torch.float32)
                before = {k: v.clone() for k, v in model.state_dict().items()}
                trainer = self.trainer(model, root)
                with patch.object(model, 'load_state_dict', wraps=model.load_state_dict) as raw, \
                     self.assertNoLogs('transformers.trainer', level='WARNING'):
                    trainer._load_from_checkpoint(str(checkpoint))
                self.assertEqual(raw.call_count, 0, 'raw HF loader cannot convert original-format Nemotron')
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, before[key]), key)
            self.assertEqual(digest_tree(checkpoint), digest)

    def test_preload_integrity_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = original_fixture(root)
            for problem in ('path', 'missing_keys', 'unexpected_keys', 'mismatched_keys',
                            'error_msgs', 'absent_info', 'mutated', 'instance', 'consumed', 'files'):
                with self.subTest(problem=problem):
                    model = _load_model(checkpoint, dtype=torch.float32)
                    trainer = self.trainer(model, root)
                    path = str(checkpoint)
                    if problem == 'path':
                        path = str(root / 'source')
                    elif problem in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
                        model.audiogen_loading_info[problem] = ['bad']
                    elif problem == 'absent_info':
                        model.audiogen_loading_info = {}
                    elif problem == 'mutated':
                        with torch.no_grad():
                            model.model.embeddings.weight.add_(1)
                    elif problem == 'instance':
                        trainer.model = copy.deepcopy(model)
                    elif problem == 'consumed':
                        trainer._load_from_checkpoint(path)
                    elif problem == 'files':
                        cfg = checkpoint / 'config.json'
                        cfg.write_text(cfg.read_text() + '\n')
                    with self.assertRaisesRegex(ValueError, 'verified conversion preload'):
                        trainer._load_from_checkpoint(path)

    def test_buffer_mutation_invalidates_preload(self):
        self._assert_changed_preload_rejected('buffer')

    def test_data_mutation_invalidates_preload(self):
        self._assert_changed_preload_rejected('data')

    def test_dtype_conversion_invalidates_preload(self):
        self._assert_changed_preload_rejected('dtype')

    def _assert_changed_preload_rejected(self, change):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = original_fixture(root)
            model = _load_model(checkpoint, dtype=torch.float32)
            trainer = self.trainer(model, root)
            versions = [(id(p), p._version) for p in model.parameters()]
            if change == 'buffer':
                bias = next(b for n, b in model.named_buffers() if n.endswith('e_score_correction_bias'))
                bias.add_(1)
            elif change == 'data':
                model.model.embeddings.weight.data.add_(1)
            else:
                model.to(torch.bfloat16)
            # All three defects bypass the old identity/version-only receipt.
            self.assertEqual(versions, [(id(p), p._version) for p in model.parameters()])
            with self.assertRaisesRegex(ValueError, 'verified conversion preload'):
                trainer._load_from_checkpoint(str(checkpoint))

    def test_fingerprint_covers_full_bytes_shape_dtype_and_buffers(self):
        model = tiny_model()
        expected = setup._native_state_fingerprint(model)
        # Chunk boundaries must not affect the digest, even through BF16 scalars.
        with patch.object(setup, '_RESUME_HASH_CHUNK_BYTES', 7):
            self.assertEqual(expected, setup._native_state_fingerprint(model))
        parameter = model.model.embeddings.weight
        parameter.data = parameter.data.view(-1)
        self.assertNotEqual(expected, setup._native_state_fingerprint(model))
        parameter.data = parameter.data.view(32, 16)
        self.assertEqual(expected, setup._native_state_fingerprint(model))
        parameter.data.view(-1)[-1].add_(1)
        self.assertNotEqual(expected, setup._native_state_fingerprint(model))
        model.register_buffer('extra_state', torch.tensor(0., dtype=torch.bfloat16), persistent=False)
        before = setup._native_state_fingerprint(model)
        model.extra_state.fill_(-0.)  # numeric equality is not byte equality
        self.assertNotEqual(before, setup._native_state_fingerprint(model))
        parameter.data = parameter.data.t()
        with self.assertRaisesRegex(ValueError, 'dense contiguous'):
            setup._native_state_fingerprint(model)

    @unittest.skipUnless(os.environ.get('AUDIOGEN_TEST_GPU_RESUME') == '1', 'Opt-in tiny GPU0 fast-kernel test')
    def test_bf16_gpu_move_resume_guards_and_optimizer_update(self):
        from transformers import TrainerCallback
        from transformers.models.nemotron_h import modeling_nemotron_h as native
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), '0')
        self.assertTrue(torch.cuda.is_available())
        config = tiny_model().config
        config.use_mamba_kernels = True
        config.chunk_size = 32
        model = native.NemotronHForCausalLM(config)
        self.assertTrue(native.is_fast_path_available)
        ids = list(range(3, 19))
        data = [{'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': ids}] * 4
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = original_fixture(root, model)
            original_digest = digest_tree(checkpoint)
            def load():
                result = _load_model(checkpoint, dtype=torch.bfloat16)
                self.assertEqual(result.device.type, 'cpu')
                self.assertEqual(result.model.embeddings.weight.dtype, torch.bfloat16)
                return result
            def trainer_for(model, name):
                sft.configure_sft_model(model, sft.SFTConfig(unfreeze_all=True))
                return sft.RobustTrainer(model=model, train_dataset=data,
                    args=TrainingArguments(output_dir=str(root / name), bf16=True,
                        max_steps=2, learning_rate=0.001, lr_scheduler_type='constant',
                        per_device_train_batch_size=1, save_steps=1, logging_steps=1,
                        report_to='none', disable_tqdm=True, dataloader_pin_memory=False))
            for change in ('buffer', 'data', 'dtype'):
                with self.subTest(gpu_change=change):
                    moved = load()
                    trainer = trainer_for(moved, 'guards')
                    self.assertEqual(moved.device.type, 'cuda')
                    if change == 'buffer':
                        next(b for n, b in moved.named_buffers() if n.endswith('e_score_correction_bias')).add_(1)
                    elif change == 'data':
                        moved.model.embeddings.weight.data.add_(1)
                    else:
                        moved.to(torch.float32)
                    with self.assertRaisesRegex(ValueError, 'verified conversion preload'):
                        trainer._load_from_checkpoint(str(checkpoint))
            model = load()
            before_move = setup._native_state_fingerprint(model)
            trainer = trainer_for(model, 'pause')  # real Trainer CPU -> CUDA move
            self.assertEqual(model.device.type, 'cuda')
            # Force multiple tiny GPU -> CPU chunks to exercise the bounded-copy path.
            with patch.object(setup, '_RESUME_HASH_CHUNK_BYTES', 127):
                self.assertEqual(before_move, setup._native_state_fingerprint(model))
            with patch.object(Trainer, '_load_from_checkpoint', side_effect=AssertionError('raw loader forbidden')):
                trainer._load_from_checkpoint(str(checkpoint))
            with self.assertRaisesRegex(ValueError, 'verified conversion preload'):
                trainer._load_from_checkpoint(str(checkpoint))
            class Pause(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    control.should_save = control.should_training_stop = True
            trainer.add_callback(Pause())
            # Exercise real installed kernels; any torch fallback is a hard failure.
            with patch.object(native.NemotronHMamba2Mixer, 'torch_forward', side_effect=AssertionError('slow fallback forbidden')), \
                 patch.object(native, 'mamba_split_conv1d_scan_combined', wraps=native.mamba_split_conv1d_scan_combined) as fast:
                trainer.train()
                self.assertEqual(trainer.state.global_step, 1)
                paused = root / 'pause/checkpoint-1'
                digest = digest_tree(paused)
                resumed_model = _load_model(paused, dtype=torch.bfloat16)
                self.assertEqual(resumed_model.device.type, 'cpu')
                before = resumed_model.model.embeddings.weight.detach().clone()
                resumed = trainer_for(resumed_model, 'resume')
                with patch.object(Trainer, '_load_from_checkpoint', side_effect=AssertionError('raw loader forbidden')):
                    resumed.train(resume_from_checkpoint=str(paused))
                self.assertEqual(resumed.state.global_step, 2)
                self.assertFalse(hasattr(resumed_model, '_audiogen_resume_preload'))
                self.assertFalse(torch.equal(before, resumed_model.model.embeddings.weight.cpu()))
                self.assertTrue(all(torch.isfinite(p).all() for p in resumed_model.parameters()))
                steps = {int(state['step']) for state in resumed.optimizer.state.values() if 'step' in state}
                self.assertEqual(steps, {2})
                self.assertGreater(fast.call_count, 0)
                self.assertEqual(digest, digest_tree(paused))
                self.assertEqual(original_digest, digest_tree(checkpoint))
                print(f'NATIVE_GPU_RESUME_PASS device={torch.cuda.get_device_name(0)} dtype=BF16 '
                      f'pause=1 final=2 optimizer_steps={steps} fast_calls={fast.call_count} '
                      f'peak_bytes={torch.cuda.max_memory_allocated()} guards=buffer,data,dtype,one-use')

    def test_actual_missing_checkpoint_tensor_cannot_get_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = original_fixture(root)
            index_path = checkpoint / 'model.safetensors.index.json'
            index = json.loads(index_path.read_text())
            shard = checkpoint / index['weight_map'].pop('backbone.embedding.weight')
            state = load_file(shard)
            del state['backbone.embedding.weight']
            save_file(state, shard)
            index_path.write_text(json.dumps(index))
            model = _load_model(checkpoint, dtype=torch.float32)
            self.assertIn('model.embeddings.weight', model.audiogen_loading_info['missing_keys'])
            self.assertFalse(hasattr(model, '_audiogen_resume_preload'))
            with self.assertRaisesRegex(ValueError, 'verified conversion preload'):
                self.trainer(model, root)._load_from_checkpoint(str(checkpoint))

    def test_qwen_uses_normal_trainer_reload(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1))
            model.save_pretrained(root / 'qwen', max_shard_size='2KB')
            saved = {k: v.clone() for k, v in model.state_dict().items()}
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(1)
            with patch.object(model, 'load_state_dict', wraps=model.load_state_dict) as raw:
                self.trainer(model, root)._load_from_checkpoint(str(root / 'qwen'))
            self.assertGreater(raw.call_count, 0)
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(saved[k], v), k)

    def test_native_pause_resume_exact_continuity(self):
        self._assert_native_continuity(unfreeze_all=True)

    def test_native_embedding_only_pause_resume_exact_continuity(self):
        self._assert_native_continuity(unfreeze_all=False)

    def _assert_native_continuity(self, unfreeze_all):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        from audiogen.model.setup import expand_for_audio
        from audiogen.tokens.vocab import save_token_metadata
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.manual_seed(42)
            checkpoint = original_fixture(root)
            raw = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'hello': 3}, unk_token='[UNK]'))
            raw.pre_tokenizer = Whitespace()
            tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
            tok.chat_template = "{% for m in messages %}{{ m['content'] + ' ' }}{% endfor %}"
            model = _load_model(checkpoint, dtype=torch.float32)
            model, tok, meta = expand_for_audio(model, tok, 'moss', 2, 2)
            model.config.audiogen = {'prompt_format': 'speech_v1', 'chat_template_kwargs': {'enable_thinking': False}, 'base_model': 'local-fixture', 'revision': 'fixture'}
            initial = root / 'initial'
            model.save_pretrained(initial, max_shard_size='2KB')
            tok.save_pretrained(initial)
            save_token_metadata(meta, initial / 'audio_token_metadata.json')
            records = [{'prompt': 'hello', 'tokens': [0, 1] * f, 'num_codebooks': 2, 'num_frames': f}
                       for f in (1, 3, 2, 4, 1, 2, 4, 3)]
            for split in ('train', 'val'):
                (root / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
            cfg = sft.SFTConfig(output_dir=str(root / 'full'), initialize_from_checkpoint=str(initial),
                train_path=str(root / 'train.jsonl'), val_path=str(root / 'val.jsonl'),
                num_codebooks=2, codebook_size=2, dtype='float32', bf16=False,
                max_length=64, max_steps=-1, num_train_epochs=1, warmup_steps=1,
                lr_scheduler_type='cosine', learning_rate=0.001, gradient_accumulation_steps=2,
                per_device_train_batch_size=1, per_device_eval_batch_size=1, report_to='none',
                logging_steps=1, eval_steps=2, save_steps=2, seed=123,
                gradient_checkpointing=True, unfreeze_all=unfreeze_all)
            traces = []
            original_loss = sft.RobustTrainer.compute_loss
            def traced(trainer, model, inputs, return_outputs=False, num_items_in_batch=None):
                if model.training:
                    traces.append((trainer.state.global_step, inputs['input_ids'].tolist(), int(num_items_in_batch), int(inputs['attention_mask'].sum())))
                return original_loss(trainer, model, inputs, return_outputs, num_items_in_batch)
            original_save = transformers.PreTrainedModel.save_pretrained
            def sharded(model, *args, **kwargs):
                kwargs['max_shard_size'] = '2KB'
                return original_save(model, *args, **kwargs)
            def run(config):
                traces.clear()
                with patch.object(sft, 'TrainingArguments', partial(sft.TrainingArguments, use_cpu=True, disable_tqdm=True)), \
                     patch.object(sft.RobustTrainer, 'compute_loss', traced), \
                     patch.object(transformers.PreTrainedModel, 'save_pretrained', sharded):
                    result = sft.run_sft(config)
                return result, copy.deepcopy(traces), torch.get_rng_state().clone()
            full, full_trace, full_rng = run(cfg)
            paused, first_trace, _ = run(dataclasses.replace(cfg, output_dir=str(root / 'paused'), stop_after_step=2))
            resume_path = root / 'paused/checkpoint-2'
            before = digest_tree(resume_path)
            self.assertEqual(paused.state.global_step, 2)
            self.assertEqual(paused.state.max_steps, 4)
            keys = json.loads((resume_path / 'model.safetensors.index.json').read_text())['weight_map']
            self.assertIn('backbone.embedding.weight', keys)
            with patch.object(Trainer, '_load_from_checkpoint', side_effect=AssertionError('raw loader must not run')):
                resumed, second_trace, resumed_rng = run(dataclasses.replace(cfg,
                    output_dir=str(root / 'resumed'), initialize_from_checkpoint=None,
                    trainer_resume_from_checkpoint=str(resume_path)))
            self.assertEqual(resumed.state.global_step, 4)
            self.assertEqual(resumed.state.max_steps, 4)
            self.assertEqual(full_trace, first_trace + second_trace)
            self.assertGreater(len({t[3] for t in full_trace}), 1)
            self.assertTrue(torch.equal(full_rng, resumed_rng))
            self.assertEqual(full.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())
            for k, v in full.model.state_dict().items():
                torch.testing.assert_close(v, resumed.model.state_dict()[k], rtol=0, atol=0)
            def exact(a, b):
                if isinstance(a, torch.Tensor):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                elif isinstance(a, dict):
                    self.assertEqual(a.keys(), b.keys())
                    for k in a:
                        exact(a[k], b[k])
                elif isinstance(a, (tuple, list)):
                    self.assertEqual(len(a), len(b))
                    for x, y in zip(a, b):
                        exact(x, y)
                else:
                    self.assertEqual(a, b)
            exact(full.optimizer.state_dict(), resumed.optimizer.state_dict())
            def history(t):
                return [(x['step'], x['learning_rate'], x['loss']) for x in t.state.log_history if 'learning_rate' in x]
            self.assertEqual(history(full), history(resumed))
            self.assertEqual(before, digest_tree(resume_path))
            self.assertEqual(resumed.model.model.layers[2].mixer.attention_dropout, 0.2)
            self.assertTrue(resumed.model.is_gradient_checkpointing)
            print('NATIVE_RESUME_EXACT: weights/buffers optimizer scheduler RNG data LR/loss; pause=2 final=4 gradaccum=2 dropout=0.2')


if __name__ == '__main__':
    unittest.main()
