import argparse
import json
from pathlib import Path

from tqdm import tqdm

from audiogen.codecs import load_codec
from audiogen.data.io import load_prompts
from audiogen.inference.generate import generate_audio
from audiogen.model.setup import load_lora_model, load_trained_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompts", required=True, help="JSON file of prompts")
    parser.add_argument("--output_dir", default="generated")
    parser.add_argument("--lora_base", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.lora_base:
        model, tokenizer, meta = load_lora_model(args.lora_base, args.model_path, device=args.device)
    else:
        model, tokenizer, meta = load_trained_model(args.model_path, device=args.device)

    codec = load_codec(meta.codec, num_codebooks=meta.num_codebooks, device=args.device)

    prompts = load_prompts(args.prompts)
    if args.max_samples:
        prompts = prompts[: args.max_samples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        out_path = output_dir / f"sample_{i:04d}.wav"
        waveform = generate_audio(
            model, tokenizer, codec, meta,
            prompt=prompt,
            output_path=out_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device,
        )
        results.append({"prompt": prompt, "output": str(out_path), "success": waveform is not None})

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    n_ok = sum(r["success"] for r in results)
    print(f"Done. {n_ok}/{len(results)} generated successfully.")


if __name__ == "__main__":
    main()
