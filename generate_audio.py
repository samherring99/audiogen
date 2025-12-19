import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

device = "cuda" if torch.cuda.is_available() else "cpu"

model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)

duration = 10

conditioning = [{
    "prompt": "a bird chirping",
    "seconds_start": 0,
    "seconds_total": duration
}]

output = generate_diffusion_cond(
    model,
    steps=100,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=sample_size,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=device
)

# Resizing
output = rearrange(output, "b d n -> d (b n)")
output = output.to(torch.float32).cpu()
output = output / torch.max(torch.abs(output))
output = torch.clamp(output, -1, 1)

# Trimming
expected_samples = int(duration * sample_rate)
if output.shape[-1] > expected_samples:
    output = output[..., :expected_samples]
elif output.shape[-1] < expected_samples:
    padding = expected_samples - output.shape[-1]
    output = torch.nn.functional.pad(output, (0, padding))

torchaudio.save("output.wav", output, sample_rate)