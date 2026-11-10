"""Small synthetic fixtures only; never read the speech corpus or load a model."""
import copy
import importlib.util
import itertools
from pathlib import Path
import random
import sys
import unittest

import pytest

from audiogen.data.schema import AudioRecord

MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments/nemotron35_tts_2026-09-18/audit_data.py"

pytestmark = pytest.mark.skipif(
    not MODULE_PATH.is_file(),
    reason="experiment harness not present (experiments/ is not part of this repo)",
)


class ExperimentSplitTests(unittest.TestCase):
    def setUp(self):
        # Make the first RED a clear missing-feature assertion, not an import typo.
        self.assertTrue(MODULE_PATH.is_file(), "Phase 0 split/selection utility has not been implemented")
        spec = importlib.util.spec_from_file_location("experiment_audit_data", MODULE_PATH)
        assert spec is not None and spec.loader is not None
        self.api = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.api
        spec.loader.exec_module(self.api)

    def row(self, i, speaker=None, text=None, shard="shard-a", frames=50):
        source = f"clips/{i}"
        rec = dict(prompt=text or f"Generate speech audio: Sentence {i}.",
                   tokens=[0] * 16, num_codebooks=8, num_frames=frames,
                   source_path=source)
        meta = {source: {"client_id": speaker or f"speaker-{i}"}}
        return self.api.make_manifest_row(rec, meta, shard=shard)

    def rows(self, n=200):
        return [self.row(i, speaker=f"speaker-{i // 2}") for i in range(n)]

    def assert_disjoint(self, result):
        pools = (result.train, result.dev, result.test)
        for a, b in itertools.combinations(pools, 2):
            for key in ("source_path", "client_id", "transcript_key"):
                self.assertFalse({getattr(r, key) for r in a} & {getattr(r, key) for r in b})

    def test_normalization_ignores_wrapper_tags_case_unicode_and_punctuation(self):
        normalize = self.api.normalize_transcript
        expected = "hello café world"
        for text in ["Generate speech audio: HELLO, Café world!",
                     "Generate speech audio [gender=female] [age=twenties]: hello cafe\u0301 WORLD.",
                     "  Ｈｅｌｌｏ café\tworld "]:
            self.assertEqual(normalize(text), expected)
        self.assertEqual(normalize(normalize(text)), expected)

    def test_manifest_is_lightweight_and_preserves_original_targets(self):
        rec = dict(prompt="Generate speech audio: Hello.", tokens=list(range(16)),
                   num_codebooks=8, num_frames=2, source_path="clips/a")
        metadata = {"clips/a": {"client_id": "speaker-a", "gender": "female"}}
        before = copy.deepcopy((rec, metadata))
        token_object = rec["tokens"]
        row = self.api.make_manifest_row(rec, metadata, shard="input.jsonl")
        self.assertEqual(row.duration_seconds, 2 / 12.5)
        self.assertEqual(row.shard, "input.jsonl")
        self.assertEqual(row.transcript_key, "hello")
        self.assertFalse(hasattr(row, "tokens"))
        self.assertFalse(hasattr(row, "record"))
        self.api.split_manifest([row])
        self.assertEqual((rec, metadata), before)
        self.assertIs(rec["tokens"], token_object)

    def test_manifest_accepts_audio_record_without_reading_tokens(self):
        rec = AudioRecord(prompt="Hello", tokens=[], codec="moss", num_codebooks=8,
                          num_frames=25, source_path="clips/a")
        row = self.api.make_manifest_row(rec, {"clips/a": {"client_id": "s"}}, shard="a")
        self.assertEqual(row.duration_seconds, 2.0)
        self.assertEqual(row.client_id, "s")

    def test_source_identity_and_shard_are_required(self):
        for source in (None, "", "  "):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "source_path"):
                self.api.make_manifest_row(dict(source_path=source), {}, shard="a")
        with self.assertRaisesRegex(ValueError, "shard"):
            self.api.make_manifest_row(dict(source_path="a"), {}, shard="")

    def test_invalid_frame_metadata_is_rejected_without_token_audit(self):
        base = dict(source_path="a", prompt="hello", num_frames=25, num_codebooks=8)
        for update in ({"num_frames": -1}, {"num_frames": 1.5}, {"num_codebooks": 0}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.api.make_manifest_row(dict(base, **update), {}, shard="a")
        for rate in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.api.make_manifest_row(base, {}, shard="a", frame_rate=rate)

    def test_split_is_reorder_invariant_seeded_and_approximately_90_5_5(self):
        rows = self.rows()
        baseline = self.api.split_manifest(rows, seed=42)
        random.Random(812).shuffle(rows)
        self.assertEqual(baseline, self.api.split_manifest(iter(rows), seed=42))
        self.assertNotEqual(baseline, self.api.split_manifest(rows, seed=43))
        self.assertEqual([len(baseline.train), len(baseline.dev), len(baseline.test)], [180, 10, 10])
        self.assertEqual(sum(map(len, (baseline.train, baseline.dev, baseline.test))), len(rows))
        self.assert_disjoint(baseline)

    def test_transcript_links_connect_speaker_groups_transitively(self):
        rows = self.rows()
        links = [self.row("a1", "A", "Hello!"), self.row("b1", "B", "hello"),
                 self.row("b2", "B", "Other sentence"), self.row("c1", "C", "OTHER sentence.")]
        result = self.api.split_manifest(rows + links)
        self.assert_disjoint(result)
        owners = {r.source_path: partition for partition in ("train", "dev", "test")
                  for r in getattr(result, partition)}
        self.assertEqual(len({owners[r.source_path] for r in links}), 1)
        self.assertEqual(result, self.api.split_manifest(list(reversed(rows + links))))

    def test_duplicate_sources_raise_with_sorted_ids_even_if_excluded(self):
        a, b = self.row("a"), self.row("b")
        with self.assertRaises(self.api.DuplicateSourceError) as caught:
            self.api.split_manifest([b, a, b, a])
        self.assertEqual(caught.exception.source_ids, ("clips/a", "clips/b"))
        missing = self.api.make_manifest_row(dict(source_path="unknown", prompt="hello",
                    num_frames=1, num_codebooks=8), {}, shard="a")
        with self.assertRaises(self.api.DuplicateSourceError):
            self.api.split_manifest([missing, missing])

    def test_missing_speaker_ids_are_explicitly_excluded_not_assumed_unique(self):
        unknown = []
        for i, value in enumerate((None, "", "  ")):
            source = f"unknown/{i}"
            unknown.append(self.api.make_manifest_row(dict(source_path=source,
                prompt="Sentence 0", num_frames=1, num_codebooks=8),
                {source: {"client_id": value}}, shard="a"))
        unknown.append(self.api.make_manifest_row(dict(source_path="no-sidecar", prompt="hello",
                         num_frames=1, num_codebooks=8), {}, shard="a"))
        result = self.api.split_manifest(self.rows() + unknown)
        self.assertEqual(len(result.excluded), 4)
        self.assertEqual({x.reason for x in result.excluded}, {"missing_speaker_id"})
        self.assertEqual({x.row.source_path for x in result.excluded}, {r.source_path for r in unknown})
        self.assertTrue(all(r.client_id for name in ("train", "dev", "test") for r in getattr(result, name)))
        self.assert_disjoint(result)

    def test_empty_normalized_transcript_is_explicitly_excluded(self):
        result = self.api.split_manifest([self.row("empty", text="Generate speech audio: !!!")])
        self.assertEqual(result.excluded[0].reason, "empty_transcript")
        self.assertFalse(result.train or result.dev or result.test)

    def test_small_sets_and_one_large_connected_group_do_not_force_leakage(self):
        for n in (0, 1, 2, 3):
            result = self.api.split_manifest(self.rows(n))
            self.assert_disjoint(result)
            self.assertEqual(sum(len(getattr(result, p)) for p in ("train", "dev", "test")), n)
        result = self.api.split_manifest([self.row(i, speaker="one") for i in range(30)])
        self.assertEqual(len(result.train), 30)
        self.assertFalse(result.dev or result.test)

    def test_invalid_ratios_and_seed_fail_explicitly(self):
        for ratios in ((1, 1, 1), (1, 0), (-0.1, 0.1, 1), (float("nan"), 0, 1)):
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                self.api.split_manifest([], fractions=ratios)
        with self.assertRaises(ValueError):
            self.api.split_manifest([], seed="42")
        result = self.api.split_manifest(self.rows(), fractions=(1, 0, 0))
        self.assertEqual(len(result.train), 200)
        self.assertFalse(result.dev or result.test)

    def test_pilot_is_train_only_deterministic_and_prefix_nested(self):
        result = self.api.split_manifest(self.rows())
        pilot = self.api.select_pilot(result, count=23, seed=7)
        self.assertEqual(len(pilot), 23)
        self.assertTrue(set(pilot) <= set(result.train))
        self.assertTrue(set(pilot).isdisjoint(result.dev + result.test))
        self.assertEqual(pilot, self.api.select_pilot(self.api.split_manifest(list(reversed(self.rows()))), count=23, seed=7))
        self.assertEqual(pilot[:7], self.api.select_pilot(result, count=7, seed=7))
        self.assertNotEqual(pilot, self.api.select_pilot(result, count=23, seed=8))

    def test_pilot_round_robin_covers_shard_speaker_duration_strata(self):
        rows = [self.row(f"{shard}-{speaker}-{frames}-{j}", speaker=speaker,
                         shard=shard, frames=frames)
                for shard, speaker, frames, j in itertools.product(("a", "b"), ("s1", "s2"), (25, 75, 150), range(4))]
        split = self.api.split_manifest(rows, fractions=(1, 0, 0))
        pilot = self.api.select_pilot(split, count=12, duration_bins=(3, 8))
        strata = {(r.shard, r.client_id, r.duration_seconds) for r in pilot}
        self.assertEqual(len(strata), 12)

    def test_fixed_dev_is_dev_only_independent_of_pilot_budget_and_seed(self):
        result = self.api.split_manifest(self.rows())
        dev = self.api.select_fixed_dev(result, count=5)
        self.api.select_pilot(result, count=180, seed=99)
        self.assertEqual(dev, self.api.select_fixed_dev(result, count=5))
        self.assertEqual(len(dev), 5)
        self.assertTrue(set(dev) <= set(result.dev))
        self.assertTrue(set(dev).isdisjoint(result.train + result.test))

    def test_sampling_count_bounds_and_empty_pools(self):
        result = self.api.split_manifest(self.rows())
        for select, pool in ((self.api.select_pilot, result.train), (self.api.select_fixed_dev, result.dev)):
            self.assertEqual(select(result, count=0), ())
            self.assertEqual(set(select(result, count=10000)), set(pool))
            self.assertEqual(select(self.api.split_manifest([])), ())
            for count in (-1, 1.5, True):
                with self.assertRaises(ValueError):
                    select(result, count=count)

    def test_invalid_duration_bins_are_rejected(self):
        result = self.api.split_manifest(self.rows())
        for bins in ((3, 2), (2, 2), (-1,), (float("inf"),), (float("nan"),)):
            with self.subTest(bins=bins), self.assertRaises(ValueError):
                self.api.select_pilot(result, count=0, duration_bins=bins)


if __name__ == "__main__":
    unittest.main()
