import json
from pathlib import Path
import tempfile
import unittest

import torch

from audiogen.inference import generate
from audiogen.tokens.formatting import get_formatter
from test_prompt_contract import TinyTokenizer, TinyModel, metadata


class AudioStreamValidationTests(unittest.TestCase):
    def diagnostic(self, ids, cap=None):
        return generate.validate_audio_stream(ids, metadata(), max_new_tokens=cap)

    def test_valid_complete_frames_and_eos(self):
        result = self.diagnostic([100, 105, 102, 107, 109], 10)
        self.assertTrue(result['valid'])
        self.assertEqual(result['num_frames'], 2)
        self.assertTrue(result['eos_seen'])
        self.assertFalse(result['hit_token_cap'])
        self.assertEqual(result['termination'], 'audio_eos')

    def test_wrong_order_range_incomplete_and_boundary_tokens_fail(self):
        for ids in ([105, 100, 109], [100, 500, 109], [100, 109], [108, 100, 105, 109], [109], [100, 105, 109, 100], [100, 105, 109, 109]):
            with self.subTest(ids=ids):
                result = self.diagnostic(ids)
                self.assertFalse(result['valid'])
                self.assertTrue(result['errors'])

    def test_snac_scale_order_range_and_ratios_are_strict(self):
        from audiogen.tokens.vocab import TokenMetadata
        meta = TokenMetadata('snac', 3, 4, 100, 120, 112, 113,
                             scale_open_ids={0: 114, 1: 116, 2: 118},
                             scale_close_ids={0: 115, 1: 117, 2: 119})
        valid = [114, 100, 115, 116, 104, 105, 117, 118, 108, 109, 110, 111, 119, 113]
        result = generate.validate_audio_stream(valid, meta)
        self.assertTrue(result['valid'])
        self.assertEqual(result['flat_tokens'], [0, 0, 1, 0, 1, 2, 3])
        for invalid in (valid[1:], valid[:9] + valid[10:], [114, 104] + valid[2:]):
            with self.subTest(invalid=invalid):
                self.assertFalse(generate.validate_audio_stream(invalid, meta)['valid'])

    def test_legacy_generation_remains_parse_compatible_but_reports_invalid_stream(self):
        class Codec:
            def decode(self, output):
                return torch.zeros(1, 12)
        diagnostic = {}
        waveform = generate.generate_audio(TinyModel(native=False, output=[105, 100, 102, 109]), TinyTokenizer(), Codec(), metadata(), 'Hello.', diagnostics=diagnostic)
        self.assertIsNotNone(waveform)
        self.assertEqual(diagnostic['decode_policy'], 'legacy')
        self.assertFalse(diagnostic['stream']['valid'])

    def test_cap_without_eos_is_not_successful_parse(self):
        result = self.diagnostic([100, 105], 2)
        self.assertEqual(result['num_frames'], 1)
        self.assertFalse(result['valid'])
        self.assertTrue(result['hit_token_cap'])
        self.assertEqual(result['termination'], 'token_cap')
        self.assertFalse(self.diagnostic([100, 105], 4)['hit_token_cap'])

    def test_eos_at_exact_cap_is_success(self):
        result = self.diagnostic([100, 105, 109], 3)
        self.assertTrue(result['valid'])
        self.assertTrue(result['hit_token_cap'])
        self.assertEqual(result['termination'], 'audio_eos')

    def test_strict_decode_refuses_legacy_reordering_and_truncation(self):
        ids = torch.tensor([105, 100, 102, 109])
        formatter = get_formatter('moss', 2)
        self.assertIsNotNone(generate.decode_token_ids(ids, TinyTokenizer(), formatter, metadata()))
        with self.assertRaisesRegex(ValueError, 'audio stream'):
            generate.decode_token_ids(ids, TinyTokenizer(), formatter, metadata(), strict=True)

    def test_native_failed_generation_retains_raw_ids_and_diagnostic(self):
        class Codec:
            def decode(self, output):
                raise AssertionError('invalid native stream must never reach codec')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tokens.json'
            diagnostics = {}
            waveform = generate.generate_audio(TinyModel(output=[105, 100]), TinyTokenizer(), Codec(), metadata(), 'Hello.', max_new_tokens=2, raw_token_path=path, diagnostics=diagnostics)
            self.assertIsNone(waveform)
            saved = json.loads(path.read_text())
            self.assertEqual(saved['raw_token_ids'], [105, 100])
            self.assertFalse(saved['stream']['valid'])
            self.assertEqual(saved, diagnostics)
            self.assertEqual(saved['prompt_format'], 'speech_v1')

    def test_batch_cli_records_seed_top_p_and_raw_outputs(self):
        from unittest.mock import patch
        from scripts import batch_generate
        with tempfile.TemporaryDirectory() as tmp:
            args = ['batch_generate', '--model_path', 'unused', '--prompts', 'unused', '--output_dir', tmp, '--seed', '37', '--top_p', '0.8', '--device', 'cpu']
            calls = []
            def fake_generate(*args, **kwargs):
                calls.append(kwargs)
                kwargs['diagnostics'].update({'stream': {'valid': False}})
                return None
            with patch('sys.argv', args), patch.object(batch_generate, 'load_trained_model', return_value=(TinyModel(), TinyTokenizer(), metadata())), patch.object(batch_generate, 'load_codec'), patch.object(batch_generate, 'load_prompts', return_value=['Hello.', 'Again.']), patch.object(batch_generate, 'generate_audio', side_effect=fake_generate), patch('transformers.set_seed') as set_seed:
                batch_generate.main()
            rows = json.loads((Path(tmp) / 'results.json').read_text())
            self.assertEqual([r['seed'] for r in rows], [37, 38])
            self.assertEqual([r['top_p'] for r in rows], [0.8, 0.8])
            self.assertEqual([c['top_p'] for c in calls], [0.8, 0.8])
            self.assertTrue(all(c['raw_token_path'] for c in calls))
            self.assertEqual([c.args[0] for c in set_seed.call_args_list], [37, 38])

    def test_single_cli_retains_raw_output_by_default(self):
        from unittest.mock import patch
        from scripts import generate as cli
        with tempfile.TemporaryDirectory() as tmp:
            output = str(Path(tmp) / 'output.wav')
            with patch('sys.argv', ['generate', '--model_path', 'unused', '--prompt', 'Hello.', '--output', output, '--seed', '17', '--device', 'cpu']), patch.object(cli, 'load_trained_model', return_value=(TinyModel(), TinyTokenizer(), metadata())), patch.object(cli, 'load_codec'), patch.object(cli, 'generate_audio', return_value=None) as gen, patch('transformers.set_seed') as seed:
                cli.main()
            self.assertEqual(str(gen.call_args.kwargs['raw_token_path']), output + '.tokens.json')
            seed.assert_called_once_with(17)

    def test_valid_native_generation_preserves_return_api(self):
        class Codec:
            def decode(self, output):
                return torch.zeros(1, 12)
        waveform = generate.generate_audio(TinyModel(), TinyTokenizer(), Codec(), metadata(), 'Hello.')
        self.assertEqual(tuple(waveform.shape), (1, 12))


if __name__ == '__main__':
    unittest.main()
