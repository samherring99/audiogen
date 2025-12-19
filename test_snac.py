import numpy as np
import torch
import torchaudio
import traceback

from snac import SNAC

model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

audio_path = "path/to/audio.wav"
audio, sr = torchaudio.load(audio_path)

print(f"Original audio shape: {audio.shape}")
print(f"Sample rate: {sr}")
print(f"Duration: {audio.shape[-1] / sr:.2f} seconds")

# Resample to 44kHz
if sr != 44100:
    resampler = torchaudio.transforms.Resample(sr, 44100)
    audio = resampler(audio)
    sr = 44100
    print(f"Resampled to 44kHz")

if audio.shape[0] > 1:
    audio = audio.mean(dim=0, keepdim=True)
    print("Converted to mono")

if audio.dim() == 2:
    audio = audio.unsqueeze(0)  # batch dimension
print(f"Audio shape before encoding: {audio.shape}")

audio = audio.to(device)

print("\n--- Encoding ---")
with torch.no_grad():
    codes = model.encode(audio)

coarse_tokens = codes[0]  # [batch, time]

print(f"Coarse tokens shape: {coarse_tokens.shape}")
print(f"Number of coarse tokens: {coarse_tokens.shape[-1]}")
print(f"Tokens per second: {coarse_tokens.shape[-1] / (audio.shape[-1] / sr):.2f}")
print(f"Token value range: [{coarse_tokens.min()}, {coarse_tokens.max()}]")
print(f"Unique tokens: {len(torch.unique(coarse_tokens))}")

print(f"Number of code scales: {len(codes)}")
for i, code in enumerate(codes):
    print(f"Scale {i} shape: {code.shape}")

print("\n--- Decoding (all scales) ---")
with torch.no_grad():
    reconstructed_full = model.decode(codes)

# [batch, channels, samples] -> [channels, samples]
reconstructed_full = reconstructed_full.squeeze(0)
output_full_path = "reconstructed_full.wav"
torchaudio.save(output_full_path, reconstructed_full.cpu(), sr)
print(f"Saved full reconstruction to: {output_full_path}")

print("\n--- Decoding (coarse + medium) ---")
try:
    with torch.no_grad():
        medium_codes = [codes[0], codes[1]] + [torch.zeros_like(codes[i]) for i in range(2, len(codes))]
        reconstructed_medium = model.decode(medium_codes)

    reconstructed_medium = reconstructed_medium.squeeze(0)
    output_path = "reconstructed_medium.wav"
    torchaudio.save(output_path, reconstructed_medium.cpu(), sr)
    print(f"Saved coarse+medium reconstruction to: {output_path}")
except Exception as e:
    print(f"Medium decoding failed: {e}")
    traceback.print_exc()

print("\n--- Decoding (coarse + medium + fine) ---")
try:
    with torch.no_grad():
        three_scale_codes = codes[:3] + [torch.zeros_like(codes[3])]
        reconstructed_three = model.decode(three_scale_codes)

    reconstructed_three = reconstructed_three.squeeze(0)
    output_path = "reconstructed_three_scales.wav"
    torchaudio.save(output_path, reconstructed_three.cpu(), sr)
    print(f"Saved coarse+medium+fine reconstruction to: {output_path}")
except Exception as e:
    print(f"Three-scale decoding failed: {e}")
    traceback.print_exc()

print("\n--- Token Statistics ---")
print(f"Coarse tokens: {list(coarse_tokens[0][:20].cpu().numpy())}...")
print(f"\nCoarse - For 30-second audio, expect ~{30 * (coarse_tokens.shape[-1] / (audio.shape[-1] / sr)):.0f} tokens")

total_tokens = coarse_tokens.shape[-1]

if len(codes) >= 2:
    medium_tokens = codes[1]
    print(f"Medium tokens: {list(medium_tokens[0][:20].cpu().numpy())}...")
    print(f"Medium - For 30-second audio, expect ~{30 * (medium_tokens.shape[-1] / (audio.shape[-1] / sr)):.0f} tokens")
    total_tokens += medium_tokens.shape[-1]
    print(f"Coarse+Medium combined: ~{30 * (total_tokens / (audio.shape[-1] / sr)):.0f} tokens")

if len(codes) >= 3:
    fine_tokens = codes[2]
    print(f"Fine tokens: {list(fine_tokens[0][:20].cpu().numpy())}...")
    print(f"Fine - For 30-second audio, expect ~{30 * (fine_tokens.shape[-1] / (audio.shape[-1] / sr)):.0f} tokens")
    total_tokens += fine_tokens.shape[-1]
    print(f"Coarse+Medium+Fine combined: ~{30 * (total_tokens / (audio.shape[-1] / sr)):.0f} tokens")
