import unittest
from types import SimpleNamespace

import torch

from audiogen.training import grpo


class Processor:
    def __init__(self, text="Hello there."):
        self.text = text

    def __call__(self, audio, **kwargs):
        return SimpleNamespace(input_features=torch.zeros(1, 80, 10))

    def batch_decode(self, ids, **kwargs):
        return [self.text]


class ASR:
    def generate(self, *args, **kwargs):
        return torch.tensor([[1]])


class RewardTests(unittest.TestCase):
    def score(self, wav=None, processor=None, model=None):
        if wav is None:
            wav = torch.full((1, 16000), 0.1)
        return grpo.score_whisper_components(
            wav, "Generate speech audio [gender=female]: Hello there.", 16000,
            (processor or Processor(), model or ASR(), "cpu"),
        )

    def test_components_and_exact_total(self):
        r = self.score()
        self.assertEqual(r["wer"], 0.0)
        self.assertEqual(r["reference"], "hello there")
        self.assertEqual(r["transcription"], "Hello there.")
        self.assertAlmostEqual(r["reward"], 0.7 * r["wer_reward"] + 0.15 * r["tail_penalty"] + 0.15 * r["silence_penalty"])
        self.assertEqual(r["silence_penalty"], 0.0)

    def test_asr_error_is_not_a_policy_penalty(self):
        class Broken(ASR):
            def generate(self, *args, **kwargs):
                raise RuntimeError("ASR unavailable")
        with self.assertRaisesRegex(RuntimeError, "ASR unavailable"):
            self.score(model=Broken())

    def test_invalid_audio_fails_closed(self):
        for wav in (torch.tensor([[float("nan")]]), torch.empty(1, 0)):
            with self.assertRaises(ValueError):
                self.score(wav=wav)

    def test_wrong_text_scores_lower(self):
        self.assertGreater(self.score()["reward"], self.score(processor=Processor("Completely unrelated words."))["reward"])

    def test_silence_remains_bounded(self):
        r = self.score(wav=torch.zeros(1, 16000), processor=Processor(""))
        self.assertGreaterEqual(r["reward"], -1.0)
        self.assertLess(r["reward"], 0.0)
        self.assertEqual(r["silence_penalty"], -1.0)


if __name__ == "__main__":
    unittest.main()
