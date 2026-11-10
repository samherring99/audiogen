import unittest

from audiogen.data.metadata_tags import (
    normalize_accent,
    normalize_age,
    normalize_gender,
    build_tagged_prompt,
)


class GenderTests(unittest.TestCase):
    def test_male(self):
        self.assertEqual(normalize_gender("male_masculine"), "male")

    def test_female(self):
        self.assertEqual(normalize_gender("female_feminine"), "female")

    def test_missing_or_junk(self):
        self.assertIsNone(normalize_gender(""))
        self.assertIsNone(normalize_gender(None))
        self.assertIsNone(normalize_gender("non_binary"))  # too sparse in CV17 to train


class AgeTests(unittest.TestCase):
    def test_passthrough_buckets(self):
        self.assertEqual(normalize_age("twenties"), "twenties")
        self.assertEqual(normalize_age("teens"), "teens")

    def test_fixes_cv_typo(self):
        self.assertEqual(normalize_age("fourties"), "forties")

    def test_merges_sparse_old_buckets(self):
        self.assertEqual(normalize_age("seventies"), "seventies_plus")
        self.assertEqual(normalize_age("eighties"), "seventies_plus")
        self.assertEqual(normalize_age("nineties"), "seventies_plus")

    def test_missing(self):
        self.assertIsNone(normalize_age(""))
        self.assertIsNone(normalize_age(None))


class AccentTests(unittest.TestCase):
    def test_us(self):
        self.assertEqual(normalize_accent("United States English"), "us")

    def test_junk_tail_ignored(self):
        self.assertEqual(normalize_accent("United States English,Midwestern,Low,Demure"), "us")

    def test_india_truncated_paren(self):
        self.assertEqual(
            normalize_accent("India and South Asia (India, Pakistan, Sri Lanka)"),
            "india_south_asia",
        )

    def test_england(self):
        self.assertEqual(normalize_accent("England English"), "england")

    def test_irish_merge(self):
        self.assertEqual(normalize_accent("Irish English"), "ireland")
        self.assertEqual(normalize_accent("Northern Irish"), "ireland")

    def test_unmapped_returns_none(self):
        self.assertIsNone(normalize_accent("Klingon"))
        self.assertIsNone(normalize_accent(""))
        self.assertIsNone(normalize_accent(None))


class PromptTests(unittest.TestCase):
    def test_full_tags_fixed_order(self):
        self.assertEqual(
            build_tagged_prompt("Generate speech audio: Hello.", gender="female", age="twenties", accent="us"),
            "Generate speech audio [gender=female] [age=twenties] [accent=us]: Hello.",
        )

    def test_partial_tags_omit_missing(self):
        self.assertEqual(
            build_tagged_prompt("Generate speech audio: Hello.", gender="male", age=None, accent=None),
            "Generate speech audio [gender=male]: Hello.",
        )

    def test_no_tags_returns_original(self):
        p = "Generate speech audio: Hello."
        self.assertEqual(build_tagged_prompt(p), p)


if __name__ == "__main__":
    unittest.main()
