import unittest

from audiogen.data.metadata_dataset import assign_prompt_variant, make_metadata_dataset


def _rec(i, client, gender="male", age="twenties", accent="us"):
    return (
        {
            "prompt": f"Generate speech audio: Sentence number {i}.",
            "tokens": list(range(16)),
            "num_codebooks": 8,
            "num_frames": 2,
            "source_path": f"shard/{i:08d}",
        },
        {
            "source_path": f"shard/{i:08d}",
            "client_id": client,
            "gender": gender,
            "age": age,
            "accent": accent,
        },
    )


class VariantTests(unittest.TestCase):
    def test_full_variant_has_all_available_tags(self):
        rec, meta = _rec(1, "spkA")
        out = assign_prompt_variant(rec["prompt"], meta, mode="full")
        self.assertEqual(
            out,
            "Generate speech audio [gender=male] [age=twenties] [accent=us]: Sentence number 1.",
        )

    def test_none_variant_returns_original(self):
        rec, meta = _rec(2, "spkA")
        self.assertEqual(assign_prompt_variant(rec["prompt"], meta, mode="none"), rec["prompt"])

    def test_full_variant_with_missing_fields_omits_them(self):
        rec, meta = _rec(3, "spkA", age=None, accent=None)
        self.assertEqual(
            assign_prompt_variant(rec["prompt"], meta, mode="full"),
            "Generate speech audio [gender=male]: Sentence number 3.",
        )


class DatasetTests(unittest.TestCase):
    def _build(self):
        pairs = [
            _rec(i, f"spk{i % 10}", gender="male" if i % 2 else "female")
            for i in range(200)
        ]
        records = [r for r, _ in pairs]
        metadata = {m["source_path"]: m for _, m in pairs}
        return make_metadata_dataset(
            records, metadata,
            target_size=100,
            eval_speaker_fraction=0.2,
            per_speaker_cap=8,
            seed=7,
        )

    def test_speaker_disjoint(self):
        result = self._build()
        train_speakers = {r["client_id"] for r in result["train_manifest"]}
        eval_speakers = {r["client_id"] for r in result["eval_manifest"]}
        self.assertTrue(train_speakers)
        self.assertTrue(eval_speakers)
        self.assertFalse(train_speakers & eval_speakers)

    def test_dropout_mix_present(self):
        result = self._build()
        variants = [r["variant"] for r in result["train_manifest"]]
        self.assertIn("full", variants)
        self.assertIn("partial", variants)
        self.assertIn("none", variants)
        none_frac = variants.count("none") / len(variants)
        self.assertGreater(none_frac, 0.05)
        self.assertLess(none_frac, 0.45)

    def test_per_speaker_cap_respected(self):
        result = self._build()
        import collections
        counts = collections.Counter(r["client_id"] for r in result["train_manifest"])
        self.assertLessEqual(max(counts.values()), 8)

    def test_tokens_unchanged(self):
        result = self._build()
        for r in result["train_records"]:
            self.assertEqual(len(r["tokens"]), r["num_codebooks"] * r["num_frames"])


if __name__ == "__main__":
    unittest.main()
