import json
import tempfile
import unittest
from pathlib import Path

from audiogen.data.number_variants import (
    build_number_variant,
    classify_safe_variant,
    make_focused_dataset,
)


class NumberVariantTests(unittest.TestCase):
    def test_converts_single_cardinal(self):
        self.assertEqual(build_number_variant("Grayson has three sisters."), "Grayson has 3 sisters.")

    def test_converts_simple_compound_cardinal(self):
        self.assertEqual(build_number_variant("There were twenty one entries."), "There were 21 entries.")

    def test_converts_contextual_clock_time(self):
        self.assertEqual(
            build_number_variant("The meeting begins at nine forty five."),
            "The meeting begins at 9:45.",
        )

    def test_rejects_invalid_clock_minute(self):
        self.assertIsNone(build_number_variant("The new location is at ten eighty seven Oak Street."))

    def test_converts_simple_ordinal(self):
        self.assertEqual(build_number_variant("He finished third in the race."), "He finished 3rd in the race.")

    def test_rejects_ambiguous_large_number(self):
        self.assertIsNone(build_number_variant("There were twelve hundred people."))

    def test_rejects_sentence_without_number(self):
        self.assertIsNone(build_number_variant("The bridge remained closed."))

    def test_classifies_time(self):
        self.assertEqual(classify_safe_variant("The meeting begins at nine forty five."), "time")

    def test_dataset_builder_keeps_pair_targets_identical_and_source_disjoint(self):
        records = [
            {
                "prompt": "Generate speech audio: Grayson has three sisters.",
                "tokens": list(range(16)),
                "num_codebooks": 8,
                "num_frames": 2,
                "source_path": "shard/a",
            },
            {
                "prompt": "Generate speech audio: The bridge remained closed.",
                "tokens": list(range(16, 32)),
                "num_codebooks": 8,
                "num_frames": 2,
                "source_path": "shard/b",
            },
            {
                "prompt": "Generate speech audio: She has four cats.",
                "tokens": list(range(32, 48)),
                "num_codebooks": 8,
                "num_frames": 2,
                "source_path": "shard/c",
            },
        ]
        result = make_focused_dataset(records, pair_limit=1, replay_count=1, eval_count=1, seed=7)
        pair = result["pairs"][0]
        self.assertEqual(pair["word_record"]["tokens"], pair["digit_record"]["tokens"])
        train_sources = {r["source_path"] for r in result["train"]}
        eval_sources = {r["source_path"] for r in result["eval"]}
        self.assertTrue(train_sources.isdisjoint(eval_sources))
        self.assertEqual(len(result["train"]), 3)  # word + digit pair + one replay


if __name__ == "__main__":
    unittest.main()
