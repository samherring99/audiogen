import unittest

from audiogen.training.grpo import _normalize_text, _wer


class NormalizeTextTests(unittest.TestCase):
    def test_tagged_reference_scores_perfect_speech_zero(self):
        ref = _normalize_text("Generate speech audio [gender=female] [accent=us]: Hello there.")
        self.assertEqual(_wer(ref, _normalize_text("Hello there.")), 0.0)

    def test_recursive_wrappers_and_numeric_reference(self):
        prompt = " Generate audio: Generate speech audio [gender=male]: Starts at 9:45."
        self.assertEqual(_normalize_text(prompt), "starts at nine forty five")
        self.assertEqual(_normalize_text(_normalize_text(prompt)), _normalize_text(prompt))

    def test_instruction_words_inside_sentence_are_preserved(self):
        self.assertEqual(_normalize_text("Please generate audio: hello."), "please generate audio hello")

    def test_time_colon_becomes_spoken_words(self):
        # "9:45" must match Whisper transcribing "nine forty five"
        self.assertEqual(
            _normalize_text("The meeting begins at 9:45."),
            _normalize_text("The meeting begins at nine forty five."),
        )

    def test_oclock_hour(self):
        self.assertEqual(
            _normalize_text("It starts at 9:00."),
            _normalize_text("It starts at nine o'clock."),
        )

    def test_plain_digit(self):
        self.assertEqual(
            _normalize_text("I have 9 red apples."),
            _normalize_text("I have nine red apples."),
        )

    def test_two_digit_compound(self):
        self.assertEqual(
            _normalize_text("We ordered 45 new chairs."),
            _normalize_text("We ordered forty five new chairs."),
        )

    def test_ordinal_digit(self):
        self.assertEqual(
            _normalize_text("She finished 3rd in the race."),
            _normalize_text("She finished third in the race."),
        )

    def test_year_stays_stable_under_double_normalization(self):
        once = _normalize_text("Born in 1999.")
        twice = _normalize_text(once)
        self.assertEqual(once, twice)

    def test_prompt_prefix_stripped(self):
        self.assertEqual(
            _normalize_text("Generate speech audio: Hello there."),
            "hello there",
        )

    def test_wer_zero_for_digit_vs_word_forms(self):
        ref = _normalize_text("Generate speech audio: The meeting begins at 9:45.")
        hyp = _normalize_text("The meeting begins at nine forty-five.")
        self.assertEqual(_wer(ref, hyp), 0.0)


if __name__ == "__main__":
    unittest.main()
