import json
from pathlib import Path
import tempfile
import unittest

import torch

from audiogen.data.dataset import AudioDataCollator, AudioGenerationDataset
from audiogen.inference.generate import generate_token_ids
from test_prompt_contract import TinyTokenizer, TinyModel, metadata


class AudioDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'fixture.jsonl'
        self.record = {'prompt': 'Generate speech audio [female]: Hello.', 'tokens': [0, 1, 2, 3], 'num_codebooks': 2, 'num_frames': 2}
        self.tokenizer = TinyTokenizer()

    def dataset(self, **kwargs):
        self.path.write_text(json.dumps(self.record) + '\n')
        return AudioGenerationDataset(self.path, self.tokenizer, metadata(), prompt_format='speech_v1', **kwargs)

    def test_prefix_matches_inference_and_only_targets_supervised(self):
        sample = self.dataset()[0]
        model = TinyModel()
        generate_token_ids(model, self.tokenizer, self.record['prompt'], metadata())
        prefix = model.input_ids[0]
        self.assertEqual(sample['input_ids'][:len(prefix)].tolist(), prefix.tolist())
        self.assertEqual(sample['labels'][:len(prefix)].tolist(), [-100] * len(prefix))
        # Real EOS shares the padding id but must remain supervised.
        self.assertEqual(sample['labels'][len(prefix):].tolist(), [100, 105, 102, 107, 109, 11])
        self.assertTrue(all(kwargs == {'enable_thinking': False} for _, kwargs in self.tokenizer.calls))

    def test_collator_masks_only_padding_positions(self):
        sample = self.dataset()[0]
        batch = AudioDataCollator(11, pad_to_multiple_of=8)([sample])
        n = len(sample['input_ids'])
        self.assertEqual(batch['labels'][0, n - 1].item(), 11)
        self.assertTrue(torch.all(batch['labels'][0, n:] == -100))
        self.assertTrue(torch.all(batch['attention_mask'][0, n:] == 0))

    def test_dataloaders_forward_native_prompt_policy(self):
        from audiogen.data.dataset import create_dataloaders
        self.path.write_text(json.dumps(self.record) + '\n')
        train, val = create_dataloaders(
            self.path, self.path, self.tokenizer, metadata(), batch_size=1,
            prompt_format='speech_v1', chat_template_kwargs={'enable_thinking': False},
        )
        for loader in (train, val):
            self.assertEqual(loader.dataset.prompt_format, 'speech_v1')
            self.assertTrue((next(iter(loader))['labels'] != -100).any())

    def test_rejects_missing_or_multiple_closing_boundary(self):
        original = self.tokenizer.encode
        for replacement in ([], [109, 109]):
            self.tokenizer.encode = lambda text, add_special_tokens=False: [v for token in original(text) for v in (replacement if token == 109 else [token])]
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, 'boundary'):
                self.dataset()[0]

    def test_rejects_overlength_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, 'length|max_length'):
            self.dataset(max_length=8)[0]

    def test_rejects_invalid_codec_targets(self):
        for tokens in ([0, 1, 2], [0, 1, 2, 4], [0, -1, 2, 3], [0, True, 2, 3], [0, 1.5, 2, 3], []):
            with self.subTest(tokens=tokens):
                self.record['tokens'] = tokens
                with self.assertRaises(ValueError):
                    self.dataset()[0]

    def test_rejects_zero_frames_and_wrong_codebooks(self):
        for field, value in [('num_frames', 0), ('num_codebooks', 1)]:
            original = self.record[field]
            self.record[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.dataset()[0]
            self.record[field] = original

    def test_rejects_injected_boundary(self):
        self.record['prompt'] += '<audio>'
        with self.assertRaisesRegex(ValueError, 'boundary|<audio>'):
            self.dataset()[0]

    def test_rejects_missing_or_multiple_encoded_boundaries(self):
        original = self.tokenizer.encode
        for replacement in ([], [108, 108]):
            self.tokenizer.encode = lambda text, add_special_tokens=False: [v for token in original(text) for v in (replacement if token == 108 else [token])]
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, 'boundary|<audio>'):
                self.dataset()[0]

    def test_cached_nemotron_native_template_prefix_offline(self):
        from transformers import AutoTokenizer
        from audiogen.tokens.vocab import expand_tokenizer
        from audiogen.tokens.formatting import build_audio_prefix
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16',
                revision='a9904d24bcc1d289a1950fa9d2b978c47cf903b9',
                local_files_only=True,
            )
        except OSError:
            self.skipTest('Pinned native tokenizer is not cached; never download in tests')
        tokenizer, meta = expand_tokenizer(tokenizer, 'moss', 8, 1024)
        self.record.update(tokens=[0, 1, 1023, 2, 3, 4, 5, 6], num_frames=1, num_codebooks=8)
        self.path.write_text(json.dumps(self.record) + '\n')
        sample = AudioGenerationDataset(self.path, tokenizer, meta, prompt_format='speech_v1')[0]
        prefix = build_audio_prefix(tokenizer, self.record['prompt'], meta, 'speech_v1')
        self.assertEqual(sample['input_ids'][:len(prefix)].tolist(), prefix)
        self.assertTrue(torch.all(sample['labels'][:len(prefix)] == -100))
        self.assertEqual(sample['labels'][len(prefix)].item(), meta.audio_token_start_id)
        self.assertIn(meta.audio_end_token_id, sample['labels'].tolist())
        self.assertEqual(tokenizer.pad_token_id, tokenizer.eos_token_id)

    def test_legacy_default_still_double_wraps(self):
        self.path.write_text(json.dumps(self.record) + '\n')
        sample = AudioGenerationDataset(self.path, self.tokenizer, metadata())[0]
        self.assertIn('Generate audio: Generate speech audio', self.tokenizer.calls[0][0][0]['content'])
        self.assertGreater(len(sample['input_ids']), 0)


if __name__ == '__main__':
    unittest.main()
