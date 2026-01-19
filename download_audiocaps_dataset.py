import json
import time
import argparse
import random
from pathlib import Path
from typing import List, Dict
import torch
from datasets import load_dataset
from snac import SNAC
import torchaudio
from tqdm import tqdm

from download_youtube_audio import download_youtube_segment


def encode_audio_with_snac(audio_path: str, device: str = "cuda") -> Dict:
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval().to(device)
    audio, sr = torchaudio.load(audio_path)

    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    if sr != 44100:
        resampler = torchaudio.transforms.Resample(sr, 44100)
        audio = resampler(audio)

    audio = audio.unsqueeze(0).to(device)

    with torch.no_grad():
        codes = snac_model.encode(audio)

    audio_tokens = []
    scale_lengths = []

    for scale_idx in range(3):
        scale_codes = codes[scale_idx].squeeze(0).cpu().tolist()
        audio_tokens.extend(scale_codes)
        scale_lengths.append(len(scale_codes))

    return {
        "audio_tokens": audio_tokens,
        "scale_lengths": scale_lengths
    }


def download_and_process_audiocaps(
    output_dir: str = "./data/audiocaps",
    split: str = "train",
    max_samples: int = None,
    delay_seconds: float = 2.0,
    delay_jitter: float = 1.0,
    long_pause_every: int = 50,
    long_pause_duration: tuple = (30, 60),
    checkpoint_every: int = 100,
    device: str = "cuda",
    start_index: int = 0,
    max_retries: int = 3,
    shuffle: bool = False,
    random_seed: int = 42,
    cookies_from_browser: str = None,
    cookies_file: str = None
):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_dir = output_path / "audio"
    audio_dir.mkdir(exist_ok=True)

    dataset_file = output_path / f"{split}_dataset.json"
    checkpoint_file = output_path / f"{split}_checkpoint.json"
    failed_file = output_path / f"{split}_failed.json"
    shuffle_file = output_path / f"{split}_shuffle_order.json"

    print(f"{'='*60}")
    print(f"AudioCaps Dataset Download & Processing")
    print(f"{'='*60}")
    print(f"Split: {split}")
    print(f"Output: {output_path}")
    print(f"Shuffle: {shuffle} (seed: {random_seed if shuffle else 'N/A'})")
    print(f"Delay: {delay_seconds}s ± {delay_jitter}s (randomized)")
    print(f"Long pause every: {long_pause_every} downloads ({long_pause_duration[0]}-{long_pause_duration[1]}s)")
    print(f"Max retries: {max_retries}")
    print(f"Checkpoints every: {checkpoint_every} samples")
    print(f"{'='*60}\n")

    print("Loading AudioCaps dataset from HuggingFace...")
    dataset = load_dataset("d0rj/audiocaps", split=split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Loaded {len(dataset)} samples")

    # Load existing progress first
    processed_data = []
    failed_downloads = []
    processed_indices = set()

    if checkpoint_file.exists():
        print(f"Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            processed_data = checkpoint.get('data', [])
            processed_indices = {entry.get('original_index', i) for i, entry in enumerate(processed_data)}
        print(f"Found {len(processed_data)} already processed samples\n")

    if shuffle:
        if shuffle_file.exists() and not checkpoint_file.exists():
            print(f"Loading existing shuffle order from {shuffle_file}")
            with open(shuffle_file, 'r') as f:
                indices = json.load(f)
        else:
            remaining_indices = [i for i in range(len(dataset)) if i not in processed_indices]

            if remaining_indices:
                print(f"Shuffling {len(remaining_indices)} remaining samples (seed: {random_seed})")
                random.seed(random_seed)
                random.shuffle(remaining_indices)

                already_processed_in_order = [entry.get('original_index', i) for i, entry in enumerate(processed_data)]
                indices = already_processed_in_order + remaining_indices

                with open(shuffle_file, 'w') as f:
                    json.dump(indices, f)
            else:
                indices = list(range(len(dataset)))

        print(f"Processing in shuffled order\n")
    else:
        indices = list(range(len(dataset)))
        print(f"Processing in original order\n")

    start_index = len(processed_data)

    if failed_file.exists():
        with open(failed_file, 'r') as f:
            failed_downloads = json.load(f)

    print("Loading SNAC encoder...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval().to(device)
    print("SNAC loaded!\n")

    successful_downloads = 0

    for progress_idx in tqdm(range(start_index, len(indices)), desc="Processing"):
        idx = indices[progress_idx]
        sample = dataset[idx]

        youtube_id = sample['youtube_id']
        start_time = sample['start_time']
        caption = sample['caption']

        audio_file = audio_dir / f"{idx:06d}_{youtube_id}_{int(start_time)}.wav"

        try:
            if not audio_file.exists():
                result = None
                for retry in range(max_retries):
                    result = download_youtube_segment(
                        video_id=youtube_id,
                        start_time=start_time,
                        duration=10.0,
                        output_path=str(audio_file),
                        sample_rate=44100,
                        cookies_from_browser=cookies_from_browser,
                        cookies_file=cookies_file
                    )

                    if result is not None:
                        break

                    if retry < max_retries - 1:
                        retry_delay = delay_seconds * (2 ** retry)
                        print(f"\n  Retry {retry + 1}/{max_retries} in {retry_delay:.1f}s...")
                        time.sleep(retry_delay)

                if result is None:
                    print(f"\nFailed to download {youtube_id} after {max_retries} attempts")
                    failed_downloads.append({
                        'index': idx,
                        'youtube_id': youtube_id,
                        'start_time': start_time,
                        'caption': caption,
                        'error': 'download_failed'
                    })

                    with open(failed_file, 'w') as f:
                        json.dump(failed_downloads, f, indent=2)

                    actual_delay = delay_seconds + random.uniform(-delay_jitter, delay_jitter)
                    time.sleep(max(0.5, actual_delay))
                    continue

                successful_downloads += 1

                actual_delay = delay_seconds + random.uniform(-delay_jitter, delay_jitter)
                time.sleep(max(0.5, actual_delay))

                if successful_downloads % long_pause_every == 0:
                    pause = random.uniform(long_pause_duration[0], long_pause_duration[1])
                    print(f"\nTaking a {pause:.0f}s break after {successful_downloads} downloads...")
                    time.sleep(pause)

            audio = torchaudio.load(str(audio_file))[0]

            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

            audio = audio.unsqueeze(0).to(device)

            with torch.no_grad():
                codes = snac_model.encode(audio)

            audio_tokens = []
            scale_lengths = []

            for scale_idx in range(3):
                scale_codes = codes[scale_idx].squeeze(0).cpu().tolist()
                audio_tokens.extend(scale_codes)
                scale_lengths.append(len(scale_codes))

            entry = {
                "prompt": caption,
                "audio_tokens": audio_tokens,
                "scale_lengths": scale_lengths,
                "youtube_id": youtube_id,
                "start_time": start_time,
                "audio_file": str(audio_file.relative_to(output_path)),
                "original_index": idx
            }

            processed_data.append(entry)

            if (progress_idx + 1) % checkpoint_every == 0:
                print(f"\nSaving checkpoint at {progress_idx + 1} samples...")
                with open(checkpoint_file, 'w') as f:
                    json.dump({'data': processed_data}, f, indent=2)

                print(f"   Processed: {len(processed_data)}")
                print(f"   Failed: {len(failed_downloads)}")

        except Exception as e:
            print(f"\nError processing {youtube_id}: {e}")
            failed_downloads.append({
                'index': idx,
                'youtube_id': youtube_id,
                'start_time': start_time,
                'caption': caption,
                'error': str(e)
            })

            with open(failed_file, 'w') as f:
                json.dump(failed_downloads, f, indent=2)

            continue

    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(processed_data)}")
    print(f"Failed: {len(failed_downloads)}")

    with open(dataset_file, 'w') as f:
        json.dump(processed_data, f, indent=2)

    print(f"\nDataset saved to: {dataset_file}")
    print(f"   Total samples: {len(processed_data)}")

    if failed_downloads:
        print(f"\nFailed downloads saved to: {failed_file}")
        print(f"   Total failed: {len(failed_downloads)}")

    if checkpoint_file.exists():
        checkpoint_file.unlink()

    return processed_data, failed_downloads


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process AudioCaps dataset")
    parser.add_argument("--output_dir", type=str, default="./data/audiocaps",
                       help="Output directory")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "val", "test"],
                       help="Dataset split to process")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples to process (None = all)")
    parser.add_argument("--delay", type=float, default=3.0,
                       help="Base delay in seconds between downloads (will be randomized)")
    parser.add_argument("--delay_jitter", type=float, default=2.0,
                       help="Random jitter to add/subtract from delay (±seconds)")
    parser.add_argument("--long_pause_every", type=int, default=50,
                       help="Take a longer pause every N successful downloads")
    parser.add_argument("--long_pause_min", type=float, default=30.0,
                       help="Minimum long pause duration in seconds")
    parser.add_argument("--long_pause_max", type=float, default=90.0,
                       help="Maximum long pause duration in seconds")
    parser.add_argument("--max_retries", type=int, default=3,
                       help="Maximum retry attempts for failed downloads")
    parser.add_argument("--checkpoint_every", type=int, default=100,
                       help="Save checkpoint every N samples")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for SNAC encoding")
    parser.add_argument("--start_index", type=int, default=0,
                       help="Start from this index (for resuming)")
    parser.add_argument("--shuffle", action="store_true",
                       help="Shuffle processing order to avoid clusters of unavailable videos")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for shuffling (only used if --shuffle is set)")
    parser.add_argument("--cookies-from-browser", type=str, default=None,
                       help="Browser to extract cookies from (e.g., 'chrome', 'firefox', 'safari')")
    parser.add_argument("--cookies", type=str, default=None,
                       help="Path to cookies.txt file exported from browser")

    args = parser.parse_args()

    download_and_process_audiocaps(
        output_dir=args.output_dir,
        split=args.split,
        max_samples=args.max_samples,
        delay_seconds=args.delay,
        delay_jitter=args.delay_jitter,
        long_pause_every=args.long_pause_every,
        long_pause_duration=(args.long_pause_min, args.long_pause_max),
        max_retries=args.max_retries,
        checkpoint_every=args.checkpoint_every,
        device=args.device,
        start_index=args.start_index,
        shuffle=args.shuffle,
        random_seed=args.random_seed,
        cookies_from_browser=getattr(args, 'cookies_from_browser', None),
        cookies_file=getattr(args, 'cookies', None)
    )
