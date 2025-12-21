import json
import os
from pathlib import Path
from typing import List, Dict
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from snac import SNAC
from tqdm import tqdm


def generate_audio_from_prompt(
    model,
    prompt: str,
    duration: float,
    sample_rate: int,
    sample_size: int,
    device: str
) -> torch.Tensor:
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": duration
    }]

    output = generate_diffusion_cond(
        model,
        steps=100,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device
    )

    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).cpu()
    output = output / torch.max(torch.abs(output))
    output = torch.clamp(output, -1, 1)

    expected_samples = int(duration * sample_rate)
    if output.shape[-1] > expected_samples:
        output = output[..., :expected_samples]
    elif output.shape[-1] < expected_samples:
        padding = expected_samples - output.shape[-1]
        output = torch.nn.functional.pad(output, (0, padding))

    return output


def encode_audio_with_snac(
    audio: torch.Tensor,
    snac_model,
    num_scales: int,
    device: str
) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # [batch, channels, samples]
    if audio.dim() == 2:
        audio = audio.unsqueeze(0)

    audio = audio.to(device)

    with torch.no_grad():
        codes = snac_model.encode(audio)

    # Return codes for training and all codes for validation
    return codes[:num_scales], codes


def flatten_codes(codes: List[torch.Tensor]) -> tuple[List[int], List[int]]:
    codes_cpu = [c.squeeze(0).cpu() for c in codes]  # Remove batch dimension

    scale_lengths = [c.shape[-1] for c in codes_cpu]

    # flatten all codes in order: [scale0_tokens, scale1_tokens, scale2_tokens, ...]
    flattened = []
    for code in codes_cpu:
        flattened.extend(code.flatten().tolist())

    return flattened, scale_lengths


def validate_reconstruction(
    original_audio: torch.Tensor,
    codes: List[torch.Tensor],
    snac_model,
    device: str
) -> bool:
    try:
        with torch.no_grad():
            reconstructed = snac_model.decode(codes)
        return True
    except Exception as e:
        print(f"Reconstruction validation failed: {e}")
        return False


def generate_dataset(
    prompts: List[str],
    output_dir: str,
    num_scales: int = 3,
    duration: float = 30.0,
    device: str = "cuda",
    resume: bool = True
):
    output_path = Path(output_dir)
    audio_path = output_path / "audio"
    audio_path.mkdir(parents=True, exist_ok=True)

    dataset_file = output_path / "dataset.json"

    dataset = []
    failed_samples = []
    processed_prompts = set()

    if resume and dataset_file.exists():
        print(f"Resuming from existing dataset: {dataset_file}")
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
        processed_prompts = {entry["prompt"] for entry in dataset}
        print(f"Found {len(dataset)} existing samples")

    remaining_prompts = [p for p in prompts if p not in processed_prompts]
    print(f"Processing {len(remaining_prompts)} new samples (skipping {len(prompts) - len(remaining_prompts)} already processed)")

    if len(remaining_prompts) == 0:
        print("All prompts already processed!")
        return dataset, failed_samples

    print("Loading Stable Audio model...")
    stable_audio_model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    stable_audio_model = stable_audio_model.to(device)

    print("Loading SNAC model...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval()
    snac_model = snac_model.to(device)

    target_sr = 44100

    print(f"\nGenerating {len(remaining_prompts)} samples...")
    start_idx = len(dataset)

    for i, prompt in enumerate(tqdm(remaining_prompts)):
        idx = start_idx + i
        try:
            audio = generate_audio_from_prompt(
                stable_audio_model,
                prompt,
                duration,
                sample_rate,
                sample_size,
                device
            )

            audio_filename = f"sample_{idx:06d}.wav"
            audio_filepath = audio_path / audio_filename
            torchaudio.save(str(audio_filepath), audio, sample_rate)

            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                audio_for_snac = resampler(audio)
            else:
                audio_for_snac = audio

            # get both training codes and all codes for validation
            codes_for_training, codes_all = encode_audio_with_snac(audio_for_snac, snac_model, num_scales, device)

            if not validate_reconstruction(audio_for_snac, codes_all, snac_model, device):
                print(f"\nWarning: Reconstruction failed for sample {idx}")
                failed_samples.append(idx)
                continue

            audio_tokens, scale_lengths = flatten_codes(codes_for_training)

            entry = {
                "prompt": prompt,
                "audio_tokens": audio_tokens,
                "scale_lengths": scale_lengths,
                "audio_path": f"audio/{audio_filename}",
                "duration": duration,
                "n_tokens": len(audio_tokens),
                "num_scales": num_scales
            }

            dataset.append(entry)

            with open(dataset_file, 'w') as f:
                json.dump(dataset, f, indent=2)

        except Exception as e:
            print(f"\nError processing prompt {idx}: {prompt}")
            print(f"Error: {e}")
            failed_samples.append(idx)
            continue

    print(f"\nDataset generation complete!")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Failed samples: {len(failed_samples)}")
    print(f"  Dataset saved to: {dataset_file}")

    if dataset:
        token_counts = [entry["n_tokens"] for entry in dataset]
        print(f"\nToken statistics:")
        print(f"  Mean tokens: {sum(token_counts) / len(token_counts):.0f}")
        print(f"  Min tokens: {min(token_counts)}")
        print(f"  Max tokens: {max(token_counts)}")

    return dataset, failed_samples


if __name__ == "__main__":
    import sys

    prompts_file = Path("prompts.json")
    if prompts_file.exists():
        print(f"Loading prompts from {prompts_file}")
        with open(prompts_file, 'r') as f:
            prompts = json.load(f)
        print(f"Loaded {len(prompts)} prompts")
    else:
        print("prompts.json not found, using example prompts")
        prompts = [
            "upbeat electronic dance music",
            "a dog barking loudly in a park",
            "rain falling on a tin roof",
            "jazz piano in a small cafe",
            "ocean waves crashing on the beach",
        ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset, failed = generate_dataset(
        prompts=prompts,
        output_dir="./data",
        num_scales=3,
        duration=10.0,
        device=device,
        resume=True
    )

    print(f"\n{'='*50}")
    print(f"Final Statistics:")
    print(f"  Total successful samples: {len(dataset)}")
    print(f"  Failed samples: {len(failed)}")
    if failed:
        print(f"  Failed indices: {failed}")
