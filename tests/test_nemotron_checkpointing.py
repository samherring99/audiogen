"""Native, random-weight NemotronH regressions: CPU only, no downloads or W&B."""
from collections import Counter
from functools import partial
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import transformers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    GPT2Config, GPT2LMHeadModel, NemotronHConfig, NemotronHForCausalLM,
    PreTrainedTokenizerFast, TrainerCallback,
)
from transformers.models.nemotron_h import modeling_nemotron_h as native

from audiogen.model import compat
from audiogen.training import sft


def tiny_model():
    return NemotronHForCausalLM(NemotronHConfig(
        vocab_size=32, hidden_size=16,
        layers_block_type=['mamba', 'moe', 'attention', 'mlp'],
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        intermediate_size=32, mamba_num_heads=4, mamba_head_dim=8,
        ssm_state_size=4, n_groups=1, conv_kernel=4, chunk_size=4,
        n_routed_experts=2, n_shared_experts=1, num_experts_per_tok=1,
        n_group=1, topk_group=1, moe_intermediate_size=16,
        moe_shared_expert_intermediate_size=16, moe_latent_size=8,
        use_mamba_kernels=False, use_cache=True, attention_dropout=0.0,
        hidden_dropout=0.0, attn_implementation='sdpa', experts_implementation='eager',
        pad_token_id=1, bos_token_id=2, eos_token_id=2,
    ))


@unittest.skipUnless(transformers.__version__ == '5.6.2', 'Regression targets installed Transformers 5.6.2')
class NemotronCheckpointingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def configure(self, model, **kwargs):
        sft.configure_sft_model(model, sft.SFTConfig(gradient_checkpointing=True, **kwargs))

    def test_original_native_rejection(self):
        model = tiny_model()
        self.assertFalse(model.supports_gradient_checkpointing)
        with self.assertRaisesRegex(ValueError, 'does not support gradient checkpointing'):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    def test_setup_prepares_instance_only_and_records_policy(self):
        model, untouched = tiny_model(), tiny_model()
        model.config.audiogen = {'prompt_format': 'speech_v1', 'revision': 'keep-me'}
        self.configure(model)
        self.assertTrue(model.supports_gradient_checkpointing)
        self.assertFalse(model.model.gradient_checkpointing)
        self.assertFalse(model.is_gradient_checkpointing, 'Trainer, not setup, must enable')
        self.assertFalse(model.config.use_cache)
        self.assertFalse(untouched.supports_gradient_checkpointing)
        self.assertNotIn('gradient_checkpointing', untouched.model.__dict__)
        self.assertFalse(native.NemotronHPreTrainedModel.supports_gradient_checkpointing)
        self.assertNotIn('supports_gradient_checkpointing', native.NemotronHForCausalLM.__dict__)
        self.assertEqual(model.config.audiogen['revision'], 'keep-me')
        policy = model.config.audiogen['gradient_checkpointing_compat']
        self.assertEqual(policy['id'], 'nemotron_h_transformers_5_6_2_native_gc_v1')
        self.assertEqual(policy['transformers_version'], '5.6.2')
        self.assertEqual(policy['native_block_count'], 4)
        self.assertEqual(json.loads(model.config.to_json_string())['audiogen']['gradient_checkpointing_compat'], policy)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.configure(model)
        self.assertTrue(model.model.gradient_checkpointing, 'Repeated setup must not reset active flags')
        model.gradient_checkpointing_disable()
        model.disable_input_require_grads()

    def test_checkpointing_not_requested_does_not_apply_compatibility(self):
        model = tiny_model()
        sft.configure_sft_model(model, sft.SFTConfig(gradient_checkpointing=False))
        self.assertFalse(model.supports_gradient_checkpointing)
        self.assertNotIn('gradient_checkpointing', model.model.__dict__)
        self.assertFalse(model.config.use_cache)

    def test_other_versions_and_already_supported_models_are_not_patched(self):
        for version in ['5.6.1', '5.6.3', '5.7.0', '5.6.2.dev0']:
            # Transformers replaces its lazy module during imports; patch the
            # live module the implementation actually consults, not a stale alias.
            with self.subTest(version=version), patch.object(compat.transformers, '__version__', version):
                model = tiny_model()
                self.configure(model)
                self.assertFalse(model.supports_gradient_checkpointing)
                self.assertNotIn('gradient_checkpointing', model.model.__dict__)
                with self.assertRaisesRegex(ValueError, 'does not support'):
                    model.gradient_checkpointing_enable()
        model = tiny_model()
        model.supports_gradient_checkpointing = True
        self.configure(model)
        self.assertNotIn('gradient_checkpointing', model.model.__dict__)
        self.assertNotIn('gradient_checkpointing_compat', getattr(model.config, 'audiogen', {}))

    def test_model_type_string_is_not_sufficient(self):
        model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=8, n_layer=1, n_head=2))
        model.config.model_type = 'nemotron_h'
        model.supports_gradient_checkpointing = False
        self.configure(model)
        self.assertFalse(model.supports_gradient_checkpointing)
        with self.assertRaisesRegex(ValueError, 'does not support'):
            model.gradient_checkpointing_enable()
        model = tiny_model()
        model.config.model_type = 'other'
        self.configure(model)
        self.assertFalse(model.supports_gradient_checkpointing)

    def test_non_native_subclass_not_enabled(self):
        class CustomNemotron(NemotronHForCausalLM):
            pass
        model = CustomNemotron(tiny_model().config)
        self.configure(model)
        self.assertFalse(model.supports_gradient_checkpointing)

    def test_invalid_blocks_fail_closed_before_setting_support(self):
        for invalid in ['empty', 'replacement', 'count']:
            with self.subTest(invalid=invalid):
                model = tiny_model()
                if invalid == 'empty':
                    model.model.layers = torch.nn.ModuleList()
                elif invalid == 'count':
                    model.model.layers = model.model.layers[:-1]
                else:
                    model.model.layers[0] = torch.nn.Identity()
                with self.assertRaisesRegex(ValueError, 'native.*checkpoint|checkpoint.*native'):
                    self.configure(model)
                self.assertFalse(model.supports_gradient_checkpointing)
                self.assertNotIn('gradient_checkpointing', model.model.__dict__)

    def test_public_enable_disable_recomputation_and_cache_lifecycle(self):
        model = tiny_model().train()
        self.configure(model)
        self.assertTrue(model.supports_gradient_checkpointing)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.addCleanup(model.disable_input_require_grads)
        self.assertTrue(model.model.gradient_checkpointing)
        counts = Counter()
        for i, block in enumerate(model.model.layers):
            original = block.forward
            def counted(*args, _i=i, _original=original, **kwargs):
                counts[_i] += 1
                return _original(*args, **kwargs)
            block.forward = counted
            self.assertTrue(block.gradient_checkpointing)
            self.assertFalse(block._gradient_checkpointing_func.keywords['use_reentrant'])
        ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
        with patch.object(native, 'DynamicCache', wraps=native.DynamicCache) as cache:
            output = model(input_ids=ids, labels=ids, use_cache=True)
            self.assertIsNone(output.past_key_values)
            self.assertEqual(list(counts.values()), [1] * 4)
            output.loss.backward()
            self.assertEqual(list(counts.values()), [2] * 4)
            cache.assert_not_called()
        model.gradient_checkpointing_disable()
        self.assertFalse(model.model.gradient_checkpointing)
        self.assertFalse(model.is_gradient_checkpointing)
        counts.clear()
        model.zero_grad(set_to_none=True)
        model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
        self.assertEqual(list(counts.values()), [1] * 4)
        model.eval()
        with torch.no_grad():
            prefill = model(input_ids=ids[:, :-1], use_cache=True)
            self.assertIsNotNone(prefill.past_key_values)
            decode = model(input_ids=ids[:, -1:], past_key_values=prefill.past_key_values, use_cache=True)
            reference = model(input_ids=ids, use_cache=False)
        self.assertEqual(decode.past_key_values.get_seq_length(), ids.shape[1])
        self.assertTrue(torch.isfinite(decode.logits).all())
        torch.testing.assert_close(decode.logits[:, -1], reference.logits[:, -1], rtol=1e-4, atol=1e-5)

    def test_real_sft_trainer_one_step_both_freeze_modes(self):
        testcase = self
        for full in (False, True):
            with self.subTest(unfreeze_all=full), tempfile.TemporaryDirectory() as tmp:
                torch.manual_seed(42)
                root = Path(tmp)
                raw = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'hello': 3}, unk_token='[UNK]'))
                raw.pre_tokenizer = Whitespace()
                tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
                tokenizer.chat_template = "{% for m in messages %}{{ m['content'] + ' ' }}{% endfor %}"
                base = root / 'base'
                tokenizer.save_pretrained(base)
                tiny_model().save_pretrained(base)
                record = {'prompt': 'hello', 'tokens': [0, 1, 1, 0], 'num_codebooks': 2, 'num_frames': 2}
                for split in ('train', 'val'):
                    (root / f'{split}.jsonl').write_text(json.dumps(record) + '\n')
                observed = {}
                class CheckGradients(TrainerCallback):
                    def on_train_begin(self, args, state, control, model=None, **kwargs):
                        testcase.assertTrue(model.is_gradient_checkpointing)
                        testcase.assertTrue(model.model.gradient_checkpointing)
                        testcase.assertEqual(args.gradient_checkpointing_kwargs, {'use_reentrant': False})
                        observed['before'] = {n: p.detach().clone() for n, p in model.named_parameters()}
                    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
                        observed['gradient_names'] = []
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                testcase.assertIsNotNone(param.grad, name)
                                testcase.assertTrue(torch.isfinite(param.grad).all(), name)
                                observed['gradient_names'].append(name)
                            else:
                                testcase.assertIsNone(param.grad, name)
                class CheckedTrainer(sft.RobustTrainer):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.add_callback(CheckGradients())
                config = sft.SFTConfig(
                    output_dir=str(root / 'run'), base_model=str(base),
                    train_path=str(root / 'train.jsonl'), val_path=str(root / 'val.jsonl'),
                    num_codebooks=2, codebook_size=2, dtype='float32', bf16=False,
                    max_length=32, max_steps=1, warmup_steps=0, learning_rate=1e-3,
                    gradient_accumulation_steps=1, per_device_train_batch_size=1,
                    per_device_eval_batch_size=1, report_to='none', logging_steps=1,
                    eval_steps=1, save_steps=1, gradient_checkpointing=True, unfreeze_all=full,
                )
                with patch.object(sft, 'TrainingArguments', partial(sft.TrainingArguments, use_cpu=True, disable_tqdm=True)), patch.object(sft, 'RobustTrainer', CheckedTrainer):
                    trainer = sft.run_sft(config)
                self.assertEqual(trainer.state.global_step, 1)
                self.assertEqual(trainer.model.device.type, 'cpu')
                self.assertTrue(observed['gradient_names'])
                if not full:
                    self.assertEqual(set(observed['gradient_names']), {'model.embeddings.weight', 'lm_head.weight'})
                for name, param in trainer.model.named_parameters():
                    self.assertTrue(torch.isfinite(param).all(), name)
                    if not param.requires_grad:
                        self.assertTrue(torch.equal(observed['before'][name], param), name)
                for name in ('model.embeddings.weight', 'lm_head.weight'):
                    self.assertFalse(torch.equal(observed['before'][name], trainer.model.get_parameter(name)), name)
                losses = [entry[k] for entry in trainer.state.log_history for k in ('loss', 'eval_loss', 'train_loss') if k in entry]
                self.assertGreaterEqual(len(losses), 3)
                self.assertTrue(all(math.isfinite(loss) for loss in losses))
                for destination in (root / 'run', root / 'run' / 'checkpoint-1'):
                    policy = json.loads((destination / 'sft_run_policy.json').read_text())
                    self.assertEqual(policy['audiogen']['gradient_checkpointing_compat']['id'], 'nemotron_h_transformers_5_6_2_native_gc_v1')
                print(f'NATIVE_TRAINER_PASS full={full} step={trainer.state.global_step} gradients={len(observed["gradient_names"])} losses={losses}')


if __name__ == '__main__':
    unittest.main()
