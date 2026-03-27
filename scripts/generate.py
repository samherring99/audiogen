import argparse

from audiogen.codecs import load_codec
from audiogen.inference.generate import generate_audio
from audiogen.model.setup import load_lora_model, load_trained_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to trained checkpoint")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--lora_base", default=None, help="Base model path if model_path is a LoRA checkpoint")
    parser.add_argument("--max_new_tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.lora_base:
        model, tokenizer, meta = load_lora_model(args.lora_base, args.model_path, device=args.device)
    else:
        model, tokenizer, meta = load_trained_model(args.model_path, device=args.device)

    codec = load_codec(meta.codec, num_codebooks=meta.num_codebooks, device=args.device)

    waveform = generate_audio(
        model, tokenizer, codec, meta,
        prompt=args.prompt,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )

    if waveform is not None:
        print(f"Saved to {args.output}")
    else:
        print("Generation failed — could not parse audio tokens")


if __name__ == "__main__":
    main()
