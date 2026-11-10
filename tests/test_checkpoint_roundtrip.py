import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from audiogen.model import setup
from audiogen.tokens.vocab import expand_tokenizer, save_token_metadata
from test_model_setup import tiny_tokenizer


class CheckpointRoundtripTests(unittest.TestCase):
    def make_checkpoint(self, directory, codec='moss', dtype=torch.bfloat16):
        tokenizer = tiny_tokenizer()
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=len(tokenizer), n_embd=16, n_layer=1, n_head=2,
            n_positions=32, bos_token_id=3, eos_token_id=3, pad_token_id=3,
        )).to(dtype)
        model.config.audiogen = {
            'base_model': 'nvidia/tiny-nemotron', 'revision': 'source-sha',
            'prompt_format': 'speech_v1', 'chat_template_kwargs': {'enable_thinking': False},
        }
        model, tokenizer, meta = setup.expand_for_audio(model, tokenizer, codec, 2, 4, directory)
        model.eval().save_pretrained(directory)
        return model, tokenizer, meta

    def test_saved_tokenizer_and_bf16_weights_roundtrip_offline(self):
        for codec in ('moss', 'snac'):
            with self.subTest(codec=codec), tempfile.TemporaryDirectory() as tmp:
                model, tokenizer, meta = self.make_checkpoint(tmp, codec)
                # This deliberately invalid remote identity must never be consulted.
                loaded, restored, restored_meta = setup.load_trained_model(tmp)
                self.assertEqual(restored.get_vocab(), tokenizer.get_vocab())
                self.assertEqual(restored.chat_template, tokenizer.chat_template)
                self.assertEqual(restored_meta, meta)
                self.assertEqual(loaded.dtype, torch.bfloat16)
                self.assertEqual(loaded.config.audiogen, model.config.audiogen)
                self.assertFalse(loaded.training)
                self.assertFalse(loaded.audiogen_loading_info['unexpected_keys'])
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(loaded.state_dict()[name], value), name)
                with torch.no_grad():
                    self.assertTrue(torch.equal(loaded(torch.tensor([[1, 2]])).logits, model(torch.tensor([[1, 2]])).logits))

    def test_interior_audio_id_permutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_checkpoint(tmp)
            path = Path(tmp) / 'tokenizer.json'
            data = json.loads(path.read_text())
            tokens = data['added_tokens']
            a = next(t for t in tokens if t['content'] == '[A0_1]')
            b = next(t for t in tokens if t['content'] == '[A0_2]')
            a['content'], b['content'] = b['content'], a['content']
            path.write_text(json.dumps(data))
            # Remove duplicate declarations so the edited backend remains internally consistent.
            config_path = Path(tmp) / 'tokenizer_config.json'
            config = json.loads(config_path.read_text())
            config.pop('added_tokens_decoder', None)
            config_path.write_text(json.dumps(config))
            with patch.object(setup.AutoModelForCausalLM, 'from_pretrained') as model_load:
                with self.assertRaisesRegex(ValueError, 'token.*ID|ID.*token'):
                    setup.load_trained_model(tmp)
            model_load.assert_not_called()

    def test_scale_token_metadata_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_checkpoint(tmp, 'snac')
            path = Path(tmp) / 'audio_token_metadata.json'
            data = json.loads(path.read_text())
            data['scale_close_ids']['1'] = data['scale_close_ids']['0']
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'token.*ID|ID.*token'):
                setup.load_trained_model(tmp)

    def test_missing_tokenizer_reconstructs_only_pinned_declared_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, meta = self.make_checkpoint(tmp)
            for path in Path(tmp).glob('*token*'):
                if path.name != 'audio_token_metadata.json':
                    path.unlink()
            with patch.object(setup.AutoTokenizer, 'from_pretrained', return_value=tiny_tokenizer()) as loader:
                _, tokenizer, restored_meta = setup.load_trained_model(tmp)
            loader.assert_called_once_with('nvidia/tiny-nemotron', revision='source-sha')
            self.assertEqual(restored_meta, meta)
            self.assertEqual(tokenizer.convert_tokens_to_ids('[A1_2]'), meta.audio_token_start_id + 6)

    def test_unpinned_native_source_fallback_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_checkpoint(tmp)
            for path in Path(tmp).glob('*token*'):
                if path.name != 'audio_token_metadata.json':
                    path.unlink()
            path = Path(tmp) / 'config.json'
            data = json.loads(path.read_text())
            data['audiogen']['revision'] = None
            path.write_text(json.dumps(data))
            with patch.object(setup.AutoTokenizer, 'from_pretrained') as loader:
                with self.assertRaisesRegex(ValueError, 'revision'):
                    setup.load_trained_model(tmp)
            loader.assert_not_called()

    def test_unknown_legacy_checkpoint_never_defaults_to_qwen(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, meta = expand_tokenizer(tiny_tokenizer(), 'moss', 2, 4)
            save_token_metadata(meta, Path(tmp) / 'audio_token_metadata.json')
            (Path(tmp) / 'config.json').write_text(json.dumps({'model_type': 'nemotron_h', '_name_or_path': 'checkpoint-20'}))
            with patch.object(setup.AutoTokenizer, 'from_pretrained') as loader:
                with self.assertRaisesRegex(ValueError, 'source|base_model'):
                    setup.load_trained_model(tmp)
            loader.assert_not_called()

    def test_saved_tokenizer_without_policy_requires_identified_legacy_qwen(self):
        for model_type, source in (
            ('nemotron_h', 'nvidia/tiny-nemotron'),
            ('nemotron_h', 'Qwen/Qwen2.5-0.5B-Instruct'),
            ('gpt2', 'Qwen/Qwen2.5-0.5B-Instruct'),
            ('qwen2', 'checkpoint-20'),
        ):
            with self.subTest(model_type=model_type, source=source), tempfile.TemporaryDirectory() as tmp:
                model, tokenizer, _ = self.make_checkpoint(tmp)
                config_path = Path(tmp) / 'config.json'
                config = json.loads(config_path.read_text())
                config.pop('audiogen')
                config.update(model_type=model_type, _name_or_path=source)
                config_path.write_text(json.dumps(config))
                del model.config.audiogen
                with patch.object(setup.AutoTokenizer, 'from_pretrained', return_value=tokenizer) as tok_load, patch.object(
                    setup.AutoModelForCausalLM, 'from_pretrained', return_value=(model, {})
                ) as model_load:
                    with self.assertRaisesRegex(ValueError, 'policy|source'):
                        setup.load_trained_model(tmp)
                tok_load.assert_not_called()
                model_load.assert_not_called()

    def test_known_legacy_qwen_with_saved_tokenizer_retains_legacy_policy(self):
        for model_type, source in (('qwen2', 'Qwen/Qwen2.5-0.5B-Instruct'), ('qwen3', 'Qwen/Qwen3-0.6B')):
            with self.subTest(model_type=model_type), tempfile.TemporaryDirectory() as tmp:
                model, tokenizer, meta = self.make_checkpoint(tmp)
                config_path = Path(tmp) / 'config.json'
                config = json.loads(config_path.read_text())
                config.pop('audiogen')
                config.update(model_type=model_type, _name_or_path=source)
                config_path.write_text(json.dumps(config))
                del model.config.audiogen
                with patch.object(setup.AutoModelForCausalLM, 'from_pretrained', return_value=(model, {})):
                    loaded, restored, restored_meta = setup.load_trained_model(tmp)
                self.assertEqual(restored.get_vocab(), tokenizer.get_vocab())
                self.assertEqual(restored_meta, meta)
                self.assertEqual(loaded.config.audiogen['prompt_format'], 'legacy')
                self.assertEqual(loaded.config.audiogen['base_model'], source)

    def test_known_legacy_qwen_source_can_reconstruct(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, _, meta = self.make_checkpoint(tmp)
            config_path = Path(tmp) / 'config.json'
            config_path.write_text(json.dumps({'model_type': 'qwen2', '_name_or_path': 'Qwen/Qwen2.5-0.5B-Instruct'}))
            for path in Path(tmp).glob('*token*'):
                if path.name != 'audio_token_metadata.json':
                    path.unlink()
            del model.config.audiogen
            with patch.object(setup.AutoTokenizer, 'from_pretrained', return_value=tiny_tokenizer()) as tok_load, patch.object(setup.AutoModelForCausalLM, 'from_pretrained', return_value=(model, {})):
                loaded, _, loaded_meta = setup.load_trained_model(tmp)
            tok_load.assert_called_once_with('Qwen/Qwen2.5-0.5B-Instruct', revision=None)
            self.assertEqual(meta, loaded_meta)
            self.assertEqual(loaded.config.audiogen['prompt_format'], 'legacy')

    def test_declared_source_cannot_be_overridden_with_another_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_checkpoint(tmp)
            with self.assertRaisesRegex(ValueError, 'source|base_model'):
                setup.load_trained_model(tmp, base_model='Qwen/Qwen2.5-0.5B-Instruct')

    def test_checkpoint_embedding_size_must_cover_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, meta = self.make_checkpoint(tmp)
            undersized = GPT2LMHeadModel(GPT2Config(vocab_size=4, n_embd=16, n_layer=1, n_head=2))
            with patch.object(setup.AutoModelForCausalLM, 'from_pretrained', return_value=(undersized, {})):
                with self.assertRaisesRegex(ValueError, 'embedding|vocab'):
                    setup.load_trained_model(tmp)


    def test_legacy_nemotron_cannot_be_reconstructed_from_qwen_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, meta = expand_tokenizer(tiny_tokenizer(), 'moss', 2, 4)
            save_token_metadata(meta, Path(tmp) / 'audio_token_metadata.json')
            (Path(tmp) / 'config.json').write_text(json.dumps({'model_type': 'nemotron_h'}))
            with patch.object(setup.AutoTokenizer, 'from_pretrained') as loader:
                with self.assertRaisesRegex(ValueError, 'source|base_model'):
                    setup.load_trained_model(tmp, base_model='Qwen/Qwen2.5-0.5B', revision='some-sha')
            loader.assert_not_called()

    def test_native_prompt_policy_cannot_enable_thinking_on_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_checkpoint(tmp)
            path = Path(tmp) / 'config.json'
            config = json.loads(path.read_text())
            config['audiogen']['chat_template_kwargs']['enable_thinking'] = True
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'thinking|chat_template_kwargs'):
                setup.load_trained_model(tmp)


if __name__ == '__main__':
    unittest.main()
