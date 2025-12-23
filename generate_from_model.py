import torch
import torchaudio
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import json

import argparse
import re


def load_model_and_tokenizer(model_path: str):
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    metadata_path = "./models/qwen-audio/audio_token_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return model, tokenizer, metadata


def generate_audio_tokens(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 2000,
    min_new_tokens: int = 600,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: str = "cuda"
):
    messages = [
        {"role": "user", "content": f"Generate audio: [{prompt}]"},
        {"role": "assistant", "content": "<audio>"}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt"
    ).to(device)

    print(f"Input prompt: '{prompt}'")
    print(f"Input tokens shape: {input_ids.shape}")
    print(f"Generating {min_new_tokens}-{max_new_tokens} tokens...")

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,  # Enforce minimum length
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("</audio>"),
        )

    generated_ids = output[0][input_ids.shape[1]:]

    print(f"Generated {len(generated_ids)} tokens")

    return generated_ids


def parse_audio_tokens(generated_ids, tokenizer, metadata, num_scales=3):
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    print(f"\nGenerated text preview:")
    print(generated_text[:500] + "..." if len(generated_text) > 500 else generated_text)

    all_audio_pattern = r'\[AUDIO_S(\d+)_(\d+)\]'
    all_matches = re.findall(all_audio_pattern, generated_text)

    if not all_matches:
        print("Warning: No audio tokens found in generation!")
        return None

    tokens_by_scale = {scale: [] for scale in range(num_scales)}
    for scale_str, token_id_str in all_matches:
        scale = int(scale_str)
        token_id = int(token_id_str)
        if scale < num_scales:
            tokens_by_scale[scale].append(token_id)

    total_tokens = sum(len(tokens) for tokens in tokens_by_scale.values())

    print(f"\nFound {total_tokens} audio tokens")
    print(f"Raw tokens per scale (as generated):")
    for scale, tokens in sorted(tokens_by_scale.items()):
        print(f"  Scale {scale}: {len(tokens)} tokens")

    return tokens_by_scale


def decode_with_snac(
    tokens_by_scale,
    snac_model,
    num_total_scales: int = 4,
    device: str = "cuda"
):
    snac_model = snac_model.to(device)

    if 0 not in tokens_by_scale or len(tokens_by_scale[0]) == 0:
        print("Error: No tokens found for scale 0 (coarse)")
        return None

    # TODO: check if this is right in SNAC
    raw_base_length = len(tokens_by_scale[0])
    base_length = ((raw_base_length + 17) // 35) * 35  # round to nearest multiple of 35 

    if base_length != raw_base_length:
        print(f"\nAdjusting base length from {raw_base_length} to {base_length}")
        token_list_s0 = tokens_by_scale[0]
        if raw_base_length < base_length:
            # pad by repeating
            repeats = (base_length + raw_base_length - 1) // raw_base_length
            tokens_by_scale[0] = (token_list_s0 * repeats)[:base_length]
        else:
            # truncate
            tokens_by_scale[0] = token_list_s0[:base_length]

    print(f"Using length: {base_length} (structure: {base_length}:{base_length*2}:{base_length*4}:{base_length*8})")

    codes = []

    for scale in range(num_total_scales):
        expected_length = base_length * (2 ** scale)

        if scale in tokens_by_scale:
            token_list = tokens_by_scale[scale]
            actual_length = len(token_list)

            if actual_length < expected_length:
                repeats = (expected_length + actual_length - 1) // actual_length
                token_list = (token_list * repeats)[:expected_length]
                print(f"  Scale {scale}: Repeated {actual_length} -> {expected_length} tokens")
            elif actual_length > expected_length:
                token_list = token_list[:expected_length]
                print(f"  Scale {scale}: Truncated {actual_length} -> {expected_length} tokens")
            else:
                print(f"  Scale {scale}: {actual_length} tokens")

            code_tensor = torch.tensor(token_list, dtype=torch.long).unsqueeze(0)
        else:
            code_tensor = torch.zeros((1, expected_length), dtype=torch.long)
            print(f"  Scale {scale}: missing - filled with {expected_length} zeros")

        codes.append(code_tensor.to(device))

    print("\nDecoding with SNAC...")
    print(f"code shapes: {[c.shape for c in codes]}")

    with torch.no_grad():
        audio = snac_model.decode(codes)

    audio = audio.squeeze(0).cpu()

    print(f"Decoded audio shape: {audio.shape}")

    return audio


def generate_audio(
    model_path: str,
    prompt: str,
    output_path: str,
    max_new_tokens: int = 2000,
    min_new_tokens: int = 600,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: str = "cuda"
):
    print("="*50)
    print("Audio Generation Pipeline")
    print("="*50)

    model, tokenizer, metadata = load_model_and_tokenizer(model_path)
    print("\nLoading SNAC decoder...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval()

    generated_ids = generate_audio_tokens(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device
    )

    tokens_by_scale = parse_audio_tokens(generated_ids, tokenizer, metadata)

    if tokens_by_scale is None:
        print("Failed to parse audio tokens. Generation may have failed.")
        return None

    audio = decode_with_snac(tokens_by_scale, snac_model, num_total_scales=4, device=device)

    sample_rate = 44100
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torchaudio.save(str(output_path), audio, sample_rate)
    print(f"Saved audio to: {output_path}")
    print(f"Duration: {audio.shape[-1] / sample_rate:.2f} seconds")

    return audio


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate audio from trained model")
    parser.add_argument("--model_path", type=str, default="./checkpoints/final_model",
                       help="Path to trained model")
    parser.add_argument("--prompt", type=str, default="upbeat electronic dance music",
                       help="Text prompt for audio generation")
    parser.add_argument("--output", type=str, default="./generated_audio.wav",
                       help="Output audio file path")
    parser.add_argument("--max_tokens", type=int, default=2000,
                       help="Maximum tokens to generate")
    parser.add_argument("--min_tokens", type=int, default=600,
                       help="Minimum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature (0.1-2.0)")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Nucleus sampling threshold")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")

    args = parser.parse_args()

    generate_audio(
        model_path=args.model_path,
        prompt=args.prompt,
        output_path=args.output,
        max_new_tokens=args.max_tokens,
        min_new_tokens=args.min_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device
    )
