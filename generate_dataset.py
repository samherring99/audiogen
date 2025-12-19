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

    return output


def encode_audio_with_snac(
    audio: torch.Tensor,
    snac_model,
    num_scales: int,
    device: str
) -> List[torch.Tensor]:
    if audio.dim() == 2:
        audio = audio.unsqueeze(0)

    audio = audio.to(device)

    with torch.no_grad():
        codes = snac_model.encode(audio)

    return codes[:num_scales]


def flatten_codes(codes: List[torch.Tensor]) -> List[int]:
    codes_cpu = [c.squeeze(0).cpu() for c in codes]  # Remove batch dimension

    lengths = [c.shape[-1] for c in codes_cpu]

    # TODO: Could implement interleaving if scales have different temporal resolutions
    flattened = []
    for code in codes_cpu:
        flattened.extend(code.flatten().tolist())

    return flattened


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
    device: str = "cuda"
):
    output_path = Path(output_dir)
    audio_path = output_path / "audio"
    audio_path.mkdir(parents=True, exist_ok=True)

    print("Loading Stable Audio model...")
    stable_audio_model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    stable_audio_model = stable_audio_model.to(device)

    print("Loading SNAC model...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval()
    snac_model = snac_model.to(device)

    target_sr = 44100

    dataset = []
    failed_samples = []

    print(f"\nGenerating {len(prompts)} samples...")
    for idx, prompt in enumerate(tqdm(prompts)):
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

            codes = encode_audio_with_snac(audio_for_snac, snac_model, num_scales, device)

            if not validate_reconstruction(audio_for_snac, codes, snac_model, device):
                print(f"\nWarning: Reconstruction failed for sample {idx}")
                failed_samples.append(idx)
                continue

            audio_tokens = flatten_codes(codes)

            entry = {
                "prompt": prompt,
                "audio_tokens": audio_tokens,
                "audio_path": f"audio/{audio_filename}",
                "duration": duration,
                "n_tokens": len(audio_tokens),
                "num_scales": num_scales
            }

            dataset.append(entry)

        except Exception as e:
            print(f"\nError processing prompt {idx}: {prompt}")
            print(f"Error: {e}")
            failed_samples.append(idx)
            continue

    dataset_file = output_path / "dataset.json"
    with open(dataset_file, 'w') as f:
        json.dump(dataset, f, indent=2)

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
    example_prompts = [
        "upbeat electronic dance music",
        "a dog barking loudly in a park",
        "rain falling on a tin roof",
        "jazz piano in a small cafe",
        "ocean waves crashing on the beach",
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset, failed = generate_dataset(
        prompts=example_prompts,
        output_dir="./data",
        num_scales=3,
        duration=10.0,
        device=device
    )
