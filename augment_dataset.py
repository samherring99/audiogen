import argparse
import json
import torch
import torchaudio
import random

from pathlib import Path
from tqdm import tqdm
from snac import SNAC
from typing import List

import torchaudio.transforms as T


def pitch_shift(audio: torch.Tensor, sr: int, n_steps: int) -> torch.Tensor:
    shift = T.PitchShift(sr, n_steps=n_steps)
    return shift(audio)


def time_stretch(audio: torch.Tensor, rate: float) -> torch.Tensor:
    stretch = T.TimeStretch(n_freq=1025, hop_length=512)
    spec = torch.stft(audio, n_fft=2048, hop_length=512, return_complex=True)
    stretched = stretch(spec, rate)
    audio_stretched = torch.istft(stretched, n_fft=2048, hop_length=512)
    return audio_stretched


def speed_change(audio: torch.Tensor, sr: int, speed: float) -> tuple[torch.Tensor, int]:
    target_sr = int(sr * speed)
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
    augmented = resampler(audio)
    return augmented, sr


def add_noise(audio: torch.Tensor, noise_level: float = 0.005) -> torch.Tensor:
    noise = torch.randn_like(audio) * noise_level
    return audio + noise


def volume_adjust(audio: torch.Tensor, gain_db: float) -> torch.Tensor:
    gain_linear = 10 ** (gain_db / 20)
    return audio * gain_linear


def augment_audio(
    audio: torch.Tensor,
    sr: int,
    aug_type: str,
    aug_params: dict
) -> tuple[torch.Tensor, int, str]:
    try:
        if aug_type == "pitch_shift":
            n_steps = aug_params.get("n_steps", 2)
            augmented = pitch_shift(audio, sr, n_steps)
            desc = f"pitch {'+' if n_steps > 0 else ''}{n_steps} semitones"
            return augmented, sr, desc

        elif aug_type == "speed":
            speed = aug_params.get("speed", 1.1)
            augmented, new_sr = speed_change(audio, sr, speed)
            desc = f"{speed:.1f}x speed"
            return augmented, new_sr, desc

        elif aug_type == "noise":
            noise_level = aug_params.get("noise_level", 0.005)
            augmented = add_noise(audio, noise_level)
            desc = "with background noise"
            return augmented, sr, desc

        elif aug_type == "volume":
            gain_db = aug_params.get("gain_db", 3)
            augmented = volume_adjust(audio, gain_db)
            desc = f"{'+' if gain_db > 0 else ''}{gain_db}dB"
            return augmented, sr, desc

        else:
            return audio, sr, ""

    except Exception as e:
        print(f"Warning: Augmentation {aug_type} failed: {e}")
        return audio, sr, ""


def encode_audio_with_snac(
    audio: torch.Tensor,
    snac_model,
    num_scales: int,
    device: str
) -> tuple[List[int], List[int]]:
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    if audio.dim() == 2:
        audio = audio.unsqueeze(0)

    audio = audio.to(device)

    with torch.no_grad():
        codes = snac_model.encode(audio)

    codes_for_training = codes[:num_scales]
    codes_cpu = [c.squeeze(0).cpu() for c in codes_for_training]

    scale_lengths = [c.shape[-1] for c in codes_cpu]
    flattened = []
    for code in codes_cpu:
        flattened.extend(code.flatten().tolist())

    return flattened, scale_lengths


def augment_dataset(
    input_dataset: str,
    output_dataset: str,
    data_dir: str,
    augmentations_per_sample: int = 4,
    num_scales: int = 3,
    target_sr: int = 44100,
    device: str = "cuda"
):
    print("="*50)
    print("Dataset Augmentation")
    print("="*50)

    print(f"Loading dataset from {input_dataset}")
    with open(input_dataset, 'r') as f:
        original_dataset = json.load(f)

    print(f"Original dataset: {len(original_dataset)} samples")
    print(f"Augmentations per sample: {augmentations_per_sample}")
    print(f"Expected output: {len(original_dataset) * (augmentations_per_sample + 1)} samples")

    print("\nLoading SNAC model...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval().to(device)

    aug_strategies = [
        {"type": "pitch_shift", "params": {"n_steps": 2}},
        {"type": "pitch_shift", "params": {"n_steps": -2}},
        {"type": "speed", "params": {"speed": 1.1}},
        {"type": "speed", "params": {"speed": 0.9}},
        {"type": "noise", "params": {"noise_level": 0.005}},
        {"type": "volume", "params": {"gain_db": 3}},
        {"type": "volume", "params": {"gain_db": -3}},
    ]

    output_path = Path(output_dataset).parent
    output_audio_path = output_path / "audio"
    output_audio_path.mkdir(parents=True, exist_ok=True)

    augmented_dataset = []
    base_idx = 0

    print(f"\nAugmenting {len(original_dataset)} samples...")

    for orig_idx, entry in enumerate(tqdm(original_dataset)):
        try:
            augmented_dataset.append(entry)
            base_idx += 1


            audio_path = Path(data_dir) / entry["audio_path"]
            if not audio_path.exists():
                print(f"\nWarning: Audio not found: {audio_path}")
                continue

            audio, sr = torchaudio.load(str(audio_path))

            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                audio = resampler(audio)
                sr = target_sr

            selected_augs = random.sample(aug_strategies, min(augmentations_per_sample, len(aug_strategies)))

            for aug_strategy in selected_augs:
                aug_type = aug_strategy["type"]
                aug_params = aug_strategy["params"]

                aug_audio, aug_sr, aug_desc = augment_audio(audio, sr, aug_type, aug_params)

                # resample if needed
                if aug_sr != target_sr:
                    resampler = torchaudio.transforms.Resample(aug_sr, target_sr)
                    aug_audio = resampler(aug_audio)

                aug_filename = f"sample_{base_idx:06d}.wav"
                aug_filepath = output_audio_path / aug_filename
                torchaudio.save(str(aug_filepath), aug_audio, target_sr)

                audio_tokens, scale_lengths = encode_audio_with_snac(
                    aug_audio, snac_model, num_scales, device
                )

                aug_prompt = f"{entry['prompt']} ({aug_desc})" if aug_desc else entry["prompt"]

                aug_entry = {
                    "prompt": aug_prompt,
                    "audio_tokens": audio_tokens,
                    "scale_lengths": scale_lengths,
                    "audio_path": f"audio/{aug_filename}",
                    "duration": entry["duration"],
                    "n_tokens": len(audio_tokens),
                    "num_scales": num_scales,
                    "augmented": True,
                    "original_idx": orig_idx,
                    "augmentation": aug_type
                }

                augmented_dataset.append(aug_entry)
                base_idx += 1

        except Exception as e:
            print(f"\nError processing sample {orig_idx}: {e}")
            continue

    print(f"\nSaving augmented dataset to {output_dataset}")
    with open(output_dataset, 'w') as f:
        json.dump(augmented_dataset, f, indent=2)

    print("\n" + "="*50)
    print("Augmentation Complete!")
    print("="*50)
    print(f"Original samples: {len(original_dataset)}")
    print(f"Augmented samples: {len(augmented_dataset) - len(original_dataset)}")
    print(f"Total samples: {len(augmented_dataset)}")
    print(f"Multiplier: {len(augmented_dataset) / len(original_dataset):.1f}x")
    print(f"Saved to: {output_dataset}")

    return augmented_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment audio dataset")
    parser.add_argument("--input", type=str, default="./data/dataset.json",
                       help="Input dataset.json")
    parser.add_argument("--output", type=str, default="./data_augmented/dataset.json",
                       help="Output dataset.json")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Base directory with audio files")
    parser.add_argument("--augmentations", type=int, default=4,
                       help="Augmentations per sample")
    parser.add_argument("--num_scales", type=int, default=3,
                       help="Number of SNAC scales")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device")

    args = parser.parse_args()

    augment_dataset(
        input_dataset=args.input,
        output_dataset=args.output,
        data_dir=args.data_dir,
        augmentations_per_sample=args.augmentations,
        num_scales=args.num_scales,
        device=args.device
    )
