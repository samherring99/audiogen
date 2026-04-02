import argparse
from pathlib import Path

import torch

from tqdm import tqdm

from audiogen.codecs import load_codec
from audiogen.data.io import save_records_jsonl
from audiogen.data.sources.tar_source import process_tar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar_dir", default=None)
    parser.add_argument("--tar_file", default=None, help="Single tar file to process (overrides --tar_dir)")
    parser.add_argument("--output", default="encoded_dataset.jsonl")
    parser.add_argument("--num_codebooks", type=int, default=8)
    parser.add_argument("--model_path", default=None, help="Local path or HF repo for MOSS")
    parser.add_argument("--min_duration", type=float, default=1.0)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--language", default=None, help="Filter by language code, e.g. 'en'")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None, choices=["float32", "float16", "bfloat16"],
                        help="Model dtype. Defaults to float32 on CPU, bfloat16 on CUDA.")
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype] if args.dtype else None
    codec = load_codec("moss", model_path=args.model_path, device=args.device, num_codebooks=args.num_codebooks, dtype=dtype)

    if args.tar_file:
        tar_files = [Path(args.tar_file)]
    elif args.tar_dir:
        tar_files = sorted(Path(args.tar_dir).glob("**/*.tar"))
        if not tar_files:
            tar_files = sorted(Path(args.tar_dir).glob("**/*.tar.gz"))
    else:
        raise ValueError("Must provide either --tar_dir or --tar_file")
    print(f"Found {len(tar_files)} tar files")

    total = 0
    all_records = []
    for tar_path in tqdm(tar_files, desc="Tars"):
        remaining = (args.max_samples - total) if args.max_samples else None
        records = process_tar(tar_path, codec, args.min_duration, args.max_duration, max_samples=remaining, language=args.language)
        all_records.extend(records)
        total += len(records)
        if args.max_samples and total >= args.max_samples:
            all_records = all_records[: args.max_samples]
            break

    save_records_jsonl(all_records, args.output)
    print(f"Done. Wrote {len(all_records)} samples to {args.output}")


if __name__ == "__main__":
    main()
