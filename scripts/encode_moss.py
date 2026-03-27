import argparse
from pathlib import Path

import torch

from tqdm import tqdm

from audiogen.codecs import load_codec
from audiogen.data.io import save_records_jsonl
from audiogen.data.sources.tar_source import process_tar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar_dir", required=True)
    parser.add_argument("--output", default="encoded_dataset.jsonl")
    parser.add_argument("--num_codebooks", type=int, default=8)
    parser.add_argument("--model_path", default=None, help="Local path or HF repo for MOSS")
    parser.add_argument("--min_duration", type=float, default=1.0)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    codec = load_codec("moss", model_path=args.model_path, device=args.device, num_codebooks=args.num_codebooks, dtype=dtype)

    tar_files = sorted(Path(args.tar_dir).glob("**/*.tar"))
    if not tar_files:
        tar_files = sorted(Path(args.tar_dir).glob("**/*.tar.gz"))
    print(f"Found {len(tar_files)} tar files")

    total = 0
    all_records = []
    for tar_path in tqdm(tar_files, desc="Tars"):
        records = process_tar(tar_path, codec, args.min_duration, args.max_duration)
        all_records.extend(records)
        total += len(records)
        if args.max_samples and total >= args.max_samples:
            all_records = all_records[: args.max_samples]
            break

    save_records_jsonl(all_records, args.output)
    print(f"Done. Wrote {len(all_records)} samples to {args.output}")


if __name__ == "__main__":
    main()
