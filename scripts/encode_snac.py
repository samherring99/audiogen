import argparse
import json
from pathlib import Path

import torchaudio
from tqdm import tqdm

from audiogen.codecs import load_codec
from audiogen.data.io import save_records_jsonl
from audiogen.data.schema import AudioRecord


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True, help="Directory of audio files")
    parser.add_argument("--prompts", required=True, help="JSON file mapping filename -> prompt text")
    parser.add_argument("--output", default="encoded_snac.jsonl")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.prompts) as f:
        prompt_map = json.load(f)  # {"filename.wav": "prompt text", ...}

    codec = load_codec("snac", device=args.device)

    audio_files = sorted(Path(args.audio_dir).glob("**/*.wav")) + \
                  sorted(Path(args.audio_dir).glob("**/*.mp3"))

    records = []
    for audio_path in tqdm(audio_files, desc="Encoding"):
        prompt = prompt_map.get(audio_path.name)
        if prompt is None:
            continue
        try:
            waveform, orig_sr = torchaudio.load(str(audio_path))
            if orig_sr != codec.sample_rate:
                waveform = torchaudio.functional.resample(waveform, orig_sr, codec.sample_rate)
            output = codec.encode(waveform)
        except Exception as e:
            print(f"Warning: failed {audio_path.name}: {e}")
            continue

        records.append(AudioRecord(
            prompt=prompt,
            tokens=output.to_flat_tokens(),
            codec="snac",
            num_codebooks=output.num_codebooks,
            num_frames=output.num_frames,
            scale_lengths=output.scale_lengths,
            audio_path=str(audio_path),
            source_path=audio_path.name,
        ))

    save_records_jsonl(records, args.output)
    print(f"Done. Wrote {len(records)} samples to {args.output}")


if __name__ == "__main__":
    main()
