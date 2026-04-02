"""
Parallel MOSS encoding — one process per tar file.
Usage:
  python scripts/encode_moss_parallel.py \
    --tar_dir /home/shared/omnimodel/common_voice_1 \
    --output /home/nightwing/Projects/audiogen/data/cv17_en_moss_1m.jsonl \
    --language en \
    --max_samples 1000000 \
    --workers 4 \
    --device cuda
"""
import argparse
import subprocess
import sys
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def encode_tar(args):
    tar_path, output_path, language, max_samples, device, dtype, min_duration, max_duration, gpu_id = args
    env_device = f"cuda:{gpu_id}" if device == "cuda" else device
    cmd = [
        sys.executable, "scripts/encode_moss.py",
        "--tar_file", str(tar_path),
        "--output", str(output_path),
        "--device", env_device,
        "--min_duration", str(min_duration),
        "--max_duration", str(max_duration),
    ]
    if language:
        cmd += ["--language", language]
    if max_samples:
        cmd += ["--max_samples", str(max_samples)]
    if dtype:
        cmd += ["--dtype", dtype]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return tar_path, result.returncode, result.stderr.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel encoding processes")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--min_duration", type=float, default=1.0)
    parser.add_argument("--max_duration", type=float, default=30.0)
    args = parser.parse_args()

    tar_files = sorted(Path(args.tar_dir).glob("**/*.tar"))
    if not tar_files:
        tar_files = sorted(Path(args.tar_dir).glob("**/*.tar.gz"))
    print(f"Found {len(tar_files)} tar files, using {args.workers} workers")

    output_dir = Path(args.output).parent / "tar_shards"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Estimate per-tar sample budget if max_samples set
    per_tar_max = None
    if args.max_samples:
        per_tar_max = (args.max_samples // len(tar_files)) + 1000  # buffer

    tasks = [
        (
            tar, output_dir / f"{tar.stem}.jsonl",
            args.language, per_tar_max,
            args.device, args.dtype,
            args.min_duration, args.max_duration,
            i % args.workers,  # round-robin GPU assignment
        )
        for i, tar in enumerate(tar_files)
    ]

    completed = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(encode_tar, t): t[0] for t in tasks}
        for future in as_completed(futures):
            tar_path, returncode, stderr = future.result()
            completed += 1
            if returncode != 0:
                failed += 1
                print(f"[{completed}/{len(tar_files)}] FAILED {tar_path.name}: {stderr[-200:]}")
            else:
                # Print last line of stderr (the stats line)
                last_line = stderr.strip().split("\n")[-1] if stderr else ""
                print(f"[{completed}/{len(tar_files)}] OK {tar_path.name} — {last_line}")

    # Merge all shards into final output, respecting max_samples
    print(f"\nMerging shards into {args.output} ...")
    shard_files = sorted(output_dir.glob("*.jsonl"))
    total = 0
    with open(args.output, "w") as out:
        for shard in shard_files:
            with open(shard) as f:
                for line in f:
                    if args.max_samples and total >= args.max_samples:
                        break
                    out.write(line)
                    total += 1
            if args.max_samples and total >= args.max_samples:
                break

    print(f"Done. Wrote {total} samples to {args.output} ({failed} tars failed)")


if __name__ == "__main__":
    main()
