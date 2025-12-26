import re
import torch
import torchaudio
from snac import SNAC
from pathlib import Path
import argparse


def parse_constrained_output(text: str, num_scales: int = 3):
    print("Parsing constrained generation output...")

    tokens_by_scale = {}

    for scale in range(num_scales):
        pattern = rf'<s{scale}>(.*?)</s{scale}>'
        match = re.search(pattern, text)

        if match:
            scale_content = match.group(1)
            token_pattern = rf'\[AUDIO_S{scale}_(\d+)\]'
            token_matches = re.findall(token_pattern, scale_content)

            if token_matches:
                tokens_by_scale[scale] = [int(tid) for tid in token_matches]
                print(f"  Scale {scale}: {len(tokens_by_scale[scale])} tokens")

    if not tokens_by_scale:
        print("Error: No audio tokens found!")
        return None

    # verify ratios
    if 0 in tokens_by_scale:
        s0 = len(tokens_by_scale[0])
        print(f"\nRatio check:")
        for s in range(1, num_scales):
            if s in tokens_by_scale:
                actual = len(tokens_by_scale[s])
                expected = s0 * (2 ** s)
                ratio = actual / s0 if s0 > 0 else 0
                status = "✓" if actual == expected else "✗"
                print(f"  {status} Scale {s}: {actual} tokens (expected {expected}, ratio {ratio:.2f}x)")

    return tokens_by_scale


def decode_with_snac(tokens_by_scale, device="cuda"):
    print("\nLoading SNAC decoder...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval().to(device)

    if 0 not in tokens_by_scale:
        print("Error: No scale 0 tokens!")
        return None

    codes = []
    base_length = len(tokens_by_scale[0])

    for scale in range(4):
        if scale < 3 and scale in tokens_by_scale:
            token_list = tokens_by_scale[scale]
            code_tensor = torch.tensor(token_list, dtype=torch.long).unsqueeze(0)
            print(f"  Scale {scale}: {len(token_list)} tokens (generated)")
        else:
            expected_length = base_length * (2 ** scale)
            code_tensor = torch.zeros((1, expected_length), dtype=torch.long)
            print(f"  Scale {scale}: {expected_length} zeros (filled)")

        codes.append(code_tensor.to(device))

    print(f"\nDecoding with SNAC...")
    print(f"Code shapes: {[c.shape for c in codes]}")

    try:
        with torch.no_grad():
            audio = snac_model.decode(codes)

        audio = audio.squeeze(0).cpu()
        print(f"Decoded audio shape: {audio.shape}")
        return audio

    except Exception as e:
        print(f"SNAC decode failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Decode constrained audio generation")
    parser.add_argument("input_file", type=str, help="Input text file with generation")
    parser.add_argument("output_file", type=str, help="Output WAV file")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    print("="*60)
    print("Decoding Constrained Audio Generation")
    print("="*60)

    with open(args.input_file, 'r') as f:
        text = f.read()

    tokens_by_scale = parse_constrained_output(text, num_scales=3)

    if tokens_by_scale is None:
        print("\nFailed to parse tokens")
        return

    audio = decode_with_snac(tokens_by_scale, device=args.device)

    if audio is None:
        print("\nFailed to decode audio")
        return

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_rate = 44100
    torchaudio.save(str(output_path), audio, sample_rate)

    duration = audio.shape[-1] / sample_rate
    print(f"\n{'='*60}")
    print(f"Success!")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")


if __name__ == "__main__":
    main()
