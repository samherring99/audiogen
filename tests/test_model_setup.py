import types
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from audiogen.model import setup


def tiny_tokenizer():
    backend = Tokenizer(WordLevel({'[UNK]': 0, 'hello': 1, 'world': 2, '[EOS]': 3}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', eos_token='[EOS]', pad_token='[EOS]')
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    return tokenizer


class GetterModel(torch.nn.Module):
    """Small model whose embedding attribute names deliberately differ from Qwen."""
    def __init__(self, tied=False):
        super().__init__()
        self.words = torch.nn.Embedding(4, 32)
        self.projection = torch.nn.Linear(32, 4, bias=False)
        self.backbone = torch.nn.Linear(32, 32)
        self.config = types.SimpleNamespace(vocab_size=4)
        self.tied = tied
        if tied:
            self.projection.weight = self.words.weight

    def get_input_embeddings(self):
        return self.words

    def get_output_embeddings(self):
        return self.projection

    def resize_token_embeddings(self, size):
        old_input = self.words.weight.detach().clone()
        old_output = self.projection.weight.detach().clone()
        self.words.weight = torch.nn.Parameter(torch.zeros(size, 32, dtype=old_input.dtype))
        self.words.weight.data[:len(old_input)] = old_input
        self.projection.weight = torch.nn.Parameter(torch.zeros(size, 32, dtype=old_output.dtype))
        self.projection.weight.data[:len(old_output)] = old_output
        if self.tied:
            self.projection.weight = self.words.weight
        self.config.vocab_size = size


class ModelSetupTests(unittest.TestCase):
    def test_expansion_uses_getters_preserves_old_rows_and_text_scale_initialization(self):
        torch.manual_seed(19)
        model, tokenizer = GetterModel(), tiny_tokenizer()
        old_input = model.words.weight.detach().clone()
        old_output = model.projection.weight.detach().clone()
        template = tokenizer.chat_template
        model, tokenizer, meta = setup.expand_for_audio(model, tokenizer, 'moss', 2, 32)
        self.assertTrue(torch.equal(model.words.weight[:4], old_input))
        self.assertTrue(torch.equal(model.projection.weight[:4], old_output))
        self.assertEqual(tokenizer.chat_template, template)
        self.assertEqual(meta.audio_token_start_id, 4)
        self.assertEqual(model.config.audiogen['audio_init'], setup.AUDIO_INIT_TEXT_SCALE)
        for weight, old in ((model.words.weight, old_input), (model.projection.weight, old_output)):
            new = weight[4:]
            # Same RMS as text rows (within noise), centred on the text-row mean.
            ratio = new.pow(2).mean().sqrt().item() / old.pow(2).mean().sqrt().item()
            self.assertGreater(ratio, .5)
            self.assertLess(ratio, 2.)
            self.assertLess((new.mean(dim=0) - old.mean(dim=0)).abs().max().item(), old.std().item())
            # Rows are distinct (not a constant vector), so the backbone can tell them apart.
            self.assertGreater(new.std(dim=0).mean().item(), old.std().item() * .5)

    def test_tied_embeddings_new_rows_are_initialized_only_once(self):
        model, tokenizer = GetterModel(tied=True), tiny_tokenizer()
        old = model.words.weight.detach().clone()
        torch.manual_seed(29)
        expected = old.mean(dim=0) + torch.empty(10, 32).normal_(mean=0, std=old.std().item())
        torch.manual_seed(29)
        setup.expand_for_audio(model, tokenizer, 'moss', 2, 4)
        self.assertIs(model.words.weight, model.projection.weight)
        self.assertTrue(torch.allclose(model.words.weight[4:], expected))

    def test_legacy_tiny_init_is_explicit_and_reproducible(self):
        model, tokenizer = GetterModel(), tiny_tokenizer()
        std = model.words.weight.std().item() * .001
        torch.manual_seed(29)
        expected = torch.empty(66, 32).normal_(mean=0, std=std)
        torch.manual_seed(29)
        setup.expand_for_audio(model, tokenizer, 'moss', 2, 32, audio_init=setup.AUDIO_INIT_LEGACY_TINY)
        self.assertTrue(torch.equal(model.words.weight[4:], expected))
        self.assertEqual(model.config.audiogen['audio_init'], setup.AUDIO_INIT_LEGACY_TINY)
        with self.assertRaises(ValueError):
            setup.expand_for_audio(GetterModel(), tiny_tokenizer(), 'moss', 2, 32, audio_init='nope')

    def test_text_scale_rows_survive_pre_norm_rmsnorm_but_legacy_rows_collapse(self):
        """Regression for the Nemotron ~6.2 CE floor: RMSNorm(eps=1e-5) must not
        squash new audio rows to ~0 relative to text rows."""
        eps, hidden = 1e-5, 2688

        def rms_norm(x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

        class Wide(GetterModel):
            def __init__(self):
                super().__init__()
                torch.manual_seed(3)
                # Nemotron-like text rows: per-element std ~0.014 -> row norm ~0.73
                self.words = torch.nn.Embedding(4, hidden)
                self.words.weight.data.normal_(0, .014)
                self.projection = torch.nn.Linear(hidden, 4, bias=False)
                self.projection.weight.data.normal_(0, .02)

            def resize_token_embeddings(self, size):
                for name in ('words', 'projection'):
                    old = getattr(self, name).weight.detach().clone()
                    param = torch.nn.Parameter(torch.zeros(size, hidden))
                    param.data[:len(old)] = old
                    getattr(self, name).weight = param
                self.config.vocab_size = size

        for audio_init, expect_alive in ((setup.AUDIO_INIT_TEXT_SCALE, True), (setup.AUDIO_INIT_LEGACY_TINY, False)):
            model, tokenizer, _ = setup.expand_for_audio(Wide(), tiny_tokenizer(), 'moss', 2, 32, audio_init=audio_init)
            text_out = rms_norm(model.words.weight[:4].detach()).pow(2).mean().sqrt().item()
            audio_out = rms_norm(model.words.weight[4:].detach()).pow(2).mean().sqrt().item()
            self.assertAlmostEqual(text_out, 1., delta=.1)
            if expect_alive:
                self.assertGreater(audio_out, .9 * text_out, audio_init)
            else:
                self.assertLess(audio_out, .05 * text_out, audio_init)

    def test_freeze_helper_trains_full_embedding_and_head_via_getters(self):
        model = GetterModel()
        helper = getattr(setup, 'configure_trainable_parameters', None)
        self.assertTrue(callable(helper), 'getter-based freezing helper is required')
        helper(model)
        self.assertTrue(model.words.weight.requires_grad)
        self.assertTrue(model.projection.weight.requires_grad)
        self.assertFalse(model.backbone.weight.requires_grad)
        self.assertFalse(model.backbone.bias.requires_grad)
        helper(model, unfreeze_all=True)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_base_load_preserves_explicit_bf16_revision_and_conversion_diagnostics(self):
        model = GetterModel().to(torch.bfloat16)
        info = {'missing_keys': [], 'unexpected_keys': ['mtp.layers.0.weight'], 'mismatched_keys': [], 'error_msgs': []}
        with patch.object(setup.AutoTokenizer, 'from_pretrained', return_value=tiny_tokenizer()) as tok_load, patch.object(setup.AutoModelForCausalLM, 'from_pretrained', return_value=(model, info)) as model_load:
            with self.assertLogs(setup.__name__, level='WARNING') as logs:
                loaded, _ = setup.load_base_model('nvidia/tiny-nemotron', dtype=torch.bfloat16, revision='pinned-sha')
        self.assertEqual(loaded.words.weight.dtype, torch.bfloat16)
        self.assertEqual(model_load.call_args.kwargs['torch_dtype'], torch.bfloat16)
        self.assertEqual(model_load.call_args.kwargs['revision'], 'pinned-sha')
        self.assertTrue(model_load.call_args.kwargs['output_loading_info'])
        self.assertEqual(tok_load.call_args.kwargs['revision'], 'pinned-sha')
        self.assertEqual(loaded.audiogen_loading_info, info)
        self.assertIn('not loaded', ' '.join(logs.output))
        self.assertEqual(loaded.config.audiogen, {
            'base_model': 'nvidia/tiny-nemotron', 'revision': 'pinned-sha',
            'prompt_format': 'speech_v1', 'chat_template_kwargs': {'enable_thinking': False},
        })

    def test_invalid_prompt_policy_fails_before_model_load(self):
        with patch.object(setup.AutoModelForCausalLM, 'from_pretrained') as model_load:
            with self.assertRaisesRegex(ValueError, 'prompt_format'):
                setup.load_base_model('unused', prompt_format='unknown')
        model_load.assert_not_called()


if __name__ == '__main__':
    unittest.main()