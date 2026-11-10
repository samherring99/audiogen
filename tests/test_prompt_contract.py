import re
import unittest
from types import SimpleNamespace

import torch

from audiogen.tokens import formatting
from audiogen.tokens.vocab import TokenMetadata
from audiogen.inference.generate import generate_token_ids


def metadata():
    # The interval end includes both boundary specials, not just codec tokens.
    return TokenMetadata('moss', 2, 4, 100, 110, 108, 109)


class TinyTokenizer:
    chat_template = 'native fixture'
    pad_token_id = eos_token_id = 11

    def __init__(self):
        self.calls = []
        self.specials = {'<audio>': 108, '</audio>': 109, '<eos>': 11}
        self.specials.update({f'[A{c}_{v}]': 100 + c * 4 + v for c in range(2) for v in range(4)})

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        self.calls.append((messages, kwargs))
        text = ''.join(f'[{m["role"]}]{m["content"]}' for m in messages)
        return text + ('[assistant]' if add_generation_prompt else '<eos>')

    def encode(self, text, add_special_tokens=False):
        chunks = re.split('(' + '|'.join(re.escape(s) for s in self.specials) + ')', text)
        return [v for chunk in chunks for v in ([self.specials[chunk]] if chunk in self.specials else [500 + ord(c) for c in chunk])]

    def __call__(self, text, **kwargs):
        ids = self.encode(text)
        if kwargs.get('truncation'):
            ids = ids[:kwargs['max_length']]
        return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones((1, len(ids)), dtype=torch.long)}

    def decode(self, ids, **kwargs):
        inv = {v: k for k, v in self.specials.items()}
        return ''.join(inv.get(int(i), '?') for i in ids)


class TinyModel:
    def __init__(self, native=True, output=None):
        self.config = SimpleNamespace(audiogen={'prompt_format': 'speech_v1', 'chat_template_kwargs': {'enable_thinking': False}} if native else {})
        self.output = [100, 105, 109] if output is None else output

    def generate(self, input_ids, **kwargs):
        self.input_ids = input_ids
        self.kwargs = kwargs
        return torch.cat([input_ids, torch.tensor([self.output])], dim=1)


class PromptContractTests(unittest.TestCase):
    def test_speech_prompt_is_used_once_and_raw_text_normalized(self):
        formatter = formatting.get_formatter('moss', 2)
        for prompt, expected in [('Generate speech audio [female]: Hello.', 'Generate speech audio [female]: Hello.'), ('Hello.', 'Generate speech audio: Hello.')]:
            with self.subTest(prompt=prompt):
                messages = formatter.format_chat_messages(prompt, prompt_format='speech_v1')
                self.assertEqual(messages[0]['content'], expected)

    def test_legacy_double_wrapper_is_default(self):
        self.assertEqual(formatting.get_formatter('moss').format_chat_messages('Generate speech audio: Hello.')[0]['content'], 'Generate audio: Generate speech audio: Hello.')

    def test_native_shared_prefix_and_inference_use_saved_policy(self):
        tokenizer = TinyTokenizer()
        model = TinyModel()
        ids = generate_token_ids(model, tokenizer, 'Hello.', metadata())
        prefix = formatting.build_audio_prefix(tokenizer, 'Hello.', metadata(), prompt_format='speech_v1')
        self.assertEqual(model.input_ids[0].tolist(), prefix)
        self.assertEqual(prefix[-1], 108)
        self.assertEqual(ids.tolist(), [100, 105, 109])
        self.assertTrue(all(kwargs == {'enable_thinking': False} for _, kwargs in tokenizer.calls))
        self.assertEqual(model.kwargs['eos_token_id'], 109)

    def test_missing_native_template_fails_without_fallback(self):
        tokenizer = TinyTokenizer()
        tokenizer.chat_template = None
        from unittest.mock import patch
        with patch('transformers.AutoTokenizer.from_pretrained', side_effect=AssertionError('No network fallback allowed')):
            with self.assertRaisesRegex(ValueError, 'chat template'):
                generate_token_ids(TinyModel(), tokenizer, 'Hello.', metadata())

    def test_missing_template_fallback_requires_identified_legacy_qwen(self):
        from unittest.mock import patch
        for model_type, source in (
            ('nemotron_h', 'nvidia/Nemotron-H'),
            ('nemotron_h', 'Qwen/Qwen2.5-0.5B-Instruct'),
            ('gpt2', 'Qwen/Qwen2.5-0.5B-Instruct'),
            ('qwen2', 'checkpoint-20'),
            (None, None),
        ):
            for policy in (None, {'prompt_format': 'legacy', 'base_model': source}):
                with self.subTest(model_type=model_type, source=source, policy=policy):
                    tokenizer = TinyTokenizer()
                    tokenizer.chat_template = None
                    model = TinyModel(native=False)
                    model.config = SimpleNamespace(model_type=model_type, _name_or_path=source, audiogen=policy)
                    with patch('transformers.AutoTokenizer.from_pretrained', return_value=TinyTokenizer()) as loader:
                        with self.assertRaisesRegex(ValueError, 'policy|chat template'):
                            generate_token_ids(model, tokenizer, 'Hello.', metadata())
                    loader.assert_not_called()
                    self.assertIsNone(tokenizer.chat_template)
                    self.assertFalse(hasattr(model, 'input_ids'))

    def test_nemotron_without_policy_fails_even_with_saved_template(self):
        from unittest.mock import patch
        model = TinyModel(native=False)
        model.config = SimpleNamespace(model_type='nemotron_h', _name_or_path='nvidia/Nemotron-H')
        with patch('transformers.AutoTokenizer.from_pretrained') as loader:
            with self.assertRaisesRegex(ValueError, 'policy'):
                generate_token_ids(model, TinyTokenizer(), 'Hello.', metadata())
        loader.assert_not_called()
        self.assertFalse(hasattr(model, 'input_ids'))

    def test_identified_legacy_qwen_can_restore_missing_template(self):
        from unittest.mock import patch
        for model_type, source in (('qwen2', 'Qwen/Qwen2.5-0.5B-Instruct'), ('qwen3', 'Qwen/Qwen3-0.6B')):
            for policy in (None, {'prompt_format': 'legacy', 'base_model': source}):
                with self.subTest(model_type=model_type, policy=policy):
                    tokenizer = TinyTokenizer()
                    tokenizer.chat_template = None
                    model = TinyModel(native=False)
                    model.config = SimpleNamespace(model_type=model_type, _name_or_path=source, audiogen=policy)
                    with patch('transformers.AutoTokenizer.from_pretrained', return_value=TinyTokenizer()) as loader:
                        ids = generate_token_ids(model, tokenizer, 'Hello.', metadata())
                    loader.assert_called_once_with('Qwen/Qwen2.5-0.5B-Instruct')
                    self.assertEqual(ids.tolist(), [100, 105, 109])
                    self.assertEqual(tokenizer.chat_template, TinyTokenizer.chat_template)
                    self.assertEqual(tokenizer.calls[0][0][0]['content'], 'Generate audio: Hello.')

    def test_native_qwen_cannot_use_legacy_template_fallback(self):
        from unittest.mock import patch
        tokenizer = TinyTokenizer()
        tokenizer.chat_template = None
        model = TinyModel()
        model.config.model_type = 'qwen2'
        model.config.audiogen['base_model'] = 'Qwen/Qwen2.5-0.5B-Instruct'
        with patch('transformers.AutoTokenizer.from_pretrained') as loader:
            with self.assertRaisesRegex(ValueError, 'chat template'):
                generate_token_ids(model, tokenizer, 'Hello.', metadata())
        loader.assert_not_called()

    def test_native_rejects_thinking_and_template_control_overrides(self):
        for kwargs in ({'enable_thinking': True}, {'tokenize': True}, {'add_generation_prompt': False}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                formatting.build_audio_prefix(TinyTokenizer(), 'Hello', metadata(), 'speech_v1', kwargs)

    def test_legacy_inference_uses_default_wrapper(self):
        tokenizer = TinyTokenizer()
        generate_token_ids(TinyModel(native=False), tokenizer, 'Generate speech audio: Hello.', metadata())
        messages, kwargs = tokenizer.calls[0]
        self.assertEqual(messages[0]['content'], 'Generate audio: Generate speech audio: Hello.')
        self.assertEqual(kwargs, {})

    def test_unknown_policy_fails(self):
        with self.assertRaises(ValueError):
            formatting.get_formatter('moss').format_chat_messages('Hello', prompt_format='typo')


if __name__ == '__main__':
    unittest.main()
