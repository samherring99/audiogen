"""Phase 0 audit contract. Synthetic corpora and a real tiny fast tokenizer; no weights."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pytest

from audiogen.data.io import load_records
from audiogen.tokens.vocab import expand_tokenizer

MODULE = Path(__file__).resolve().parents[1] / 'experiments/nemotron35_tts_2026-09-18/audit_data.py'

pytestmark = pytest.mark.skipif(
    not MODULE.is_file(),
    reason='experiment harness not present (experiments/ is not part of this repo)',
)


def tokenizer_fixture():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]')
    tok.chat_template = "{% for m in messages %}{{ m['role'] + ': ' }}{% if m['role'] == 'assistant' and not enable_thinking %}{{ '<think></think>' }}{% endif %}{{ m['content'] + '\\n' }}{% endfor %}"
    return expand_tokenizer(tok, 'moss', 8, 1024)[0]


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('experiment_audit_cli_tests', MODULE)
        cls.api = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.api
        spec.loader.exec_module(cls.api)
        cls.tok = tokenizer_fixture()

    def setUp(self):
        self.assertTrue(hasattr(self.api, 'run_audit'), 'streaming audit CLI is not implemented')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = [dict(prompt=f'Generate speech audio: Sentence {i}.', tokens=list(range(8)),
                         num_codebooks=8, num_frames=1, source_path=f'clips/{i}') for i in range(100)]
        self.tagged = [dict(r, prompt=r['prompt'].replace('audio:', 'audio [gender=female]:')) for r in self.raw]
        self.meta = [dict(source_path=r['source_path'], client_id=f'speaker-{i}') for i, r in enumerate(self.raw)]

    def run_audit(self, output='out', **kwargs):
        for name, records in [('raw', self.raw), ('tagged', self.tagged), ('metadata', self.meta)]:
            (self.root / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        return self.api.run_audit(self.root/'raw.jsonl', self.root/'tagged.jsonl',
                    self.root/'metadata.jsonl', self.root/output, tokenizer=self.tok,
                    pilot_count=20, fixed_dev_count=3, smoke_count=2, **kwargs)

    def test_lengths_equal_full_encoding_for_edge_targets_and_exact_prompt(self):
        auditor = self.api.TokenLengthAuditor(self.tok)
        for prompt in ['', 'Generate speech audio: Hello!', '你好 [age=twenties]: café', '<audio>punctuation</audio>']:
            for frames in [1, 2, 256, 300]:
                tokens = [0, 1023, 1, 999, 10, 101, 1000, 17] * frames
                messages = self.api.chat_messages(prompt, tokens)
                self.assertEqual(messages[0]['content'], prompt)
                rendered = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
                self.assertEqual(auditor.length(prompt, len(tokens)), len(self.tok.encode(rendered, add_special_tokens=False)))
        self.assertEqual(auditor.proof['codec_tokens_verified'], 8192)

    def test_outputs_are_loader_compatible_deterministic_and_disjoint(self):
        report = self.run_audit()
        self.assertEqual(report['counts']['source_rows'], 100)
        self.assertEqual(report['counts']['splits'], {'train':90, 'dev':5, 'test':5})
        self.assertTrue(report['verification']['second_split_identical'])
        self.assertTrue(all(v == 0 for v in report['disjointness'].values()))
        self.assertEqual(report['counts']['selections'], {'pilot':20, 'fixed_dev':3, 'smoke_train':2, 'smoke_dev':2})
        pools = {}
        for name in ['train','dev','test','pilot','fixed_dev','smoke_train','smoke_dev']:
            pools[name] = set()
            for variant in ['untagged','tagged']:
                records = load_records(self.root/'out'/f'{name}.{variant}.jsonl')
                ids = {r.source_path for r in records}
                if variant == 'untagged': pools[name] = ids
                else: self.assertEqual(ids, pools[name])
                self.assertTrue(all(r.tokens == list(range(8)) for r in records))
        self.assertTrue(pools['pilot'] <= pools['train'])
        self.assertTrue(pools['fixed_dev'] <= pools['dev'])
        second = self.run_audit('out2')
        self.assertEqual(report, second)
        self.assertEqual((self.root/'out'/'source_manifest.jsonl').read_bytes(), (self.root/'out2'/'source_manifest.jsonl').read_bytes())
        with self.assertRaises(FileExistsError): self.run_audit()
        # Restore permissions for TemporaryDirectory cleanup.
        for d in [self.root/'out', self.root/'out2']:
            d.chmod(0o755)

    def test_duplicate_sidecar_fails_without_publishing(self):
        self.meta.append(self.meta[0])
        with self.assertRaisesRegex(ValueError, 'Duplicate sidecar'): self.run_audit()
        self.assertFalse((self.root/'out').exists())
        self.assertFalse(list(self.root.glob('.out.*')))

    def test_duplicate_source_fails_even_when_invalid(self):
        self.raw.append(dict(self.raw[0], tokens=[]))
        self.tagged.append(dict(self.tagged[0], tokens=[]))
        with self.assertRaisesRegex(ValueError, 'Duplicate source'): self.run_audit()
        self.assertFalse((self.root/'out').exists())

    def test_identity_and_row_order_mismatch_fail_closed(self):
        for field, value in [('source_path','clips/different'),('num_frames',2),('num_codebooks',7),('tokens',[2]*8),('prompt','Generate speech audio: different sentence')]:
            old = self.tagged[0][field]
            self.tagged[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'mismatch'):
                self.run_audit()
            self.assertFalse((self.root/'out').exists())
            self.tagged[0][field] = old
        self.tagged[0], self.tagged[1] = self.tagged[1], self.tagged[0]
        with self.assertRaisesRegex(ValueError, 'mismatch'): self.run_audit()

    def test_missing_speakers_joins_invalid_targets_and_overlength_are_reported(self):
        self.meta[0]['client_id'] = ''
        self.meta.pop(1)
        invalid = [{'tokens':[]}, {'tokens':[True]*8}, {'tokens':[1024]*8},
                   {'tokens':[-1]*8}, {'tokens':[1.5]*8}, {'tokens':[1]*7},
                   {'num_codebooks':7}, {'num_frames':0}, {'num_frames':1.5}]
        for i, change in enumerate(invalid, 2):
            self.raw[i].update(change); self.tagged[i].update(change)
        self.raw[11].update(tokens=[0]*2400, num_frames=300)
        self.tagged[11].update(tokens=[0]*2400, num_frames=300)
        # Tagged length alone excludes this row; never silently use raw length.
        self.tagged[12]['prompt'] = self.raw[12]['prompt'].replace('audio:', 'audio [accent='+' '.join(['long']*2100)+']:')
        report = self.run_audit()
        self.assertEqual(report['counts']['eligible_rows'],87)
        self.assertEqual(report['exclusions']['invalid_target'],9)
        self.assertEqual(report['exclusions']['missing_sidecar'],1)
        self.assertEqual(report['exclusions']['missing_speaker_id'],1)
        self.assertEqual(report['exclusions']['overlength'],2)
        self.assertEqual(report['join']['matched'],99)
        self.assertEqual(report['lengths']['longer']['count'],91)
        self.assertEqual(report['lengths']['eligible_longer']['count'],87)
        (self.root/'out').chmod(0o755)

    def test_row_count_mismatch_and_degenerate_split_fail_closed(self):
        self.tagged.pop()
        with self.assertRaisesRegex(ValueError, 'row count mismatch'): self.run_audit()
        self.tagged.append(dict(self.raw[-1]))
        for m in self.meta: m['client_id'] = 'one-speaker'
        with self.assertRaisesRegex(ValueError, 'degenerate split'): self.run_audit()
        self.assertFalse((self.root/'out').exists())


if __name__ == '__main__': unittest.main()
