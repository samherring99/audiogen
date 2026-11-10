import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel, Qwen2Config, Qwen2ForCausalLM

from audiogen.training import sft


class FreezingTests(unittest.TestCase):
    def test_getter_based_embedding_freeze(self):
        model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=8, n_layer=2, n_head=2))
        self.assertTrue(hasattr(sft, 'configure_sft_model'), 'SFT needs architecture-independent freeze helper')
        sft.configure_sft_model(model, sft.SFTConfig(data_path='unused', output_dir='unused'))
        expected = {id(p) for m in (model.get_input_embeddings(), model.get_output_embeddings()) for p in m.parameters()}
        self.assertEqual({id(p) for p in model.parameters() if p.requires_grad}, expected)
        self.assertFalse(model.config.use_cache)

    def test_last_n_only_supported_qwen_layers(self):
        self.assertTrue(hasattr(sft, 'configure_sft_model'))
        cfg = sft.SFTConfig(data_path='unused', output_dir='unused', unfreeze_last_n_layers=1)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2))
        sft.configure_sft_model(model, cfg)
        self.assertTrue(all(p.requires_grad for p in model.model.layers[-1].parameters()))
        self.assertFalse(any(p.requires_grad for p in model.model.layers[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.model.norm.parameters()))
        with self.assertRaisesRegex(ValueError, 'last.*layers|unsupported'):
            sft.configure_sft_model(GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=8, n_layer=1, n_head=2)), cfg)

    def test_all_parameters(self):
        self.assertTrue(hasattr(sft, 'configure_sft_model'))
        model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=8, n_layer=1, n_head=2))
        sft.configure_sft_model(model, sft.SFTConfig(data_path='unused', output_dir='unused', unfreeze_all=True))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
