#!/usr/bin/env python3
"""Validate the GRPO whisper reward ranks known-quality audio correctly before training.

Uses real generated wavs with known human judgments + synthetic silence.
PASS requires: good samples > bad sample > silence, and digit-prompt scoring
works (9:45 canonicalization).
"""
import argparse
import sys

import torch
import soundfile as sf

from audiogen.training.grpo import (
    _load_whisper,
    compute_whisper_reward,
    silence_penalty,
    tail_noise_penalty,
)

# Reference wav paths are supplied on the command line; PASS requires the reward to
# rank the known-good samples above the known-bad one, silence, noise, and a
# wrong-text control, with nontrivial spread.
GOOD_PROMPTS = [
    ("She finished 3rd in the race.", "good"),
    ("The train leaves from platform 7.", "good"),
    ("The meeting begins at 9:45.", "good (digit-canon test)"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--good_wavs", nargs=3, required=True,
                        help="Three known-good wavs, in the order of GOOD_PROMPTS")
    parser.add_argument("--bad_wav", required=True, help="One known-bad wav")
    args = parser.parse_args()

    cases = [
        (src, f"Generate speech audio: {prompt}", label)
        for src, (prompt, label) in zip(args.good_wavs, GOOD_PROMPTS)
    ]
    cases.append((args.bad_wav, "Generate speech audio: I have 9 red apples.", "BAD"))
    cases.append(("SILENCE", "Generate speech audio: Hello there, how are you today?", "synthetic silence"))
    cases.append(("NOISE", "Generate speech audio: Hello there, how are you today?", "synthetic noise"))
    cases.append(("MISMATCH:" + args.good_wavs[1],
                  "Generate speech audio: The weather in Paris is lovely this evening.", "wrong-text control"))

    whisper = _load_whisper("openai/whisper-base")
    waveforms, prompts, labels = [], [], []
    for src, prompt, label in cases:
        if src == "SILENCE":
            wav = torch.zeros(1, 24000 * 4)
        elif src == "NOISE":
            wav = torch.randn(1, 24000 * 4) * 0.1
        else:
            path = src.split(":", 1)[1] if src.startswith("MISMATCH:") else src
            data, sr = sf.read(path)
            wav = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        waveforms.append(wav)
        prompts.append(prompt)
        labels.append(label)

    rewards = compute_whisper_reward(waveforms, prompts, 24000, whisper)

    print(f"{'reward':>8}  {'sil_pen':>8}  {'tail_pen':>8}  label")
    for wav, r, label in zip(waveforms, rewards, labels):
        print(f"{r:8.3f}  {silence_penalty(wav):8.3f}  {tail_noise_penalty(wav):8.3f}  {label}")

    good = [r for r, l in zip(rewards, labels) if l.startswith("good")]
    bad = rewards[3]
    silence = rewards[4]
    noise = rewards[5]
    mismatch = rewards[6]

    checks = {
        "all good samples > bad sample": min(good) > bad,
        "bad sample > pure silence": bad > silence,
        "all good > silence": min(good) > silence,
        "all good > noise": min(good) > noise,
        "all good > wrong-text control": min(good) > mismatch,
        "digit 9:45 sample scores positive": rewards[2] > 0,
        "reward variance nontrivial": (max(rewards) - min(rewards)) > 0.5,
    }
    print()
    failed = False
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
        failed |= not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
