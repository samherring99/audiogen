from llamppl import Model, CachedCausalLM, LMContext, smc_steer
from typing import List, Optional
import asyncio
import torch

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
END = "\033[0m"


class ConstrainedAudioModel(Model):

    def __init__(
        self,
        lm: CachedCausalLM,
        prompt: str,
        base_length: int = 140,  # scale 0 length
        temperature: float = 1.0,
        num_scales: int = 3
    ):
        super().__init__()
        self.lm = lm
        self.context = LMContext(lm, prompt, temperature)
        self.base_length = base_length
        self.num_scales = num_scales
        self.generated_tokens = []

        # track current state
        self.current_scale = -1  # -1 = before audio, 0-2 = in scale, 3 = after audio
        self.tokens_in_current_scale = 0

        # get delimiter token IDs
        self.audio_start = lm.tokenizer.encode("<audio>", add_special_tokens=False)
        self.audio_end = lm.tokenizer.encode("</audio>", add_special_tokens=False)

        self.scale_delimiters = {}
        for scale in range(num_scales):
            self.scale_delimiters[f"start_{scale}"] = lm.tokenizer.encode(f"<s{scale}>", add_special_tokens=False)
            self.scale_delimiters[f"end_{scale}"] = lm.tokenizer.encode(f"</s{scale}>", add_special_tokens=False)

        self.audio_token_ids = {scale: set() for scale in range(num_scales)}
        for scale in range(num_scales):
            for i in range(4096):  # SNAC codebook size
                token_str = f"[AUDIO_S{scale}_{i}]"
                token_id = lm.tokenizer.encode(token_str, add_special_tokens=False)
                if len(token_id) == 1:
                    self.audio_token_ids[scale].add(token_id[0])

        print(f"\n{GREEN}Constrained Audio Generation{END}")
        print(f"Base length: {base_length} (ratios: {base_length}:{base_length*2}:{base_length*4})")
        print(f"Total tokens: {base_length * 7} audio tokens + delimiters")

    def get_expected_length_for_scale(self, scale: int) -> int:
        return self.base_length * (2 ** scale)

    async def step(self):
        # state 0: Generate <audio> token
        if self.current_scale == -1:
            for token_id in self.audio_start:
                await self.observe(self.context.next_token(), token_id)
                token = await self.sample(self.context.next_token())
                self.generated_tokens.append(token)

            # Move to scale 0
            self.current_scale = 0
            self.tokens_in_current_scale = 0

            # Force <s0>
            for token_id in self.scale_delimiters["start_0"]:
                await self.observe(self.context.next_token(), token_id)
                token = await self.sample(self.context.next_token())
                self.generated_tokens.append(token)

            print(f"{GREEN}Starting Scale 0{END} (target: {self.get_expected_length_for_scale(0)} tokens)")
            return

        # states 1-3: Generate audio tokens for current scale
        if 0 <= self.current_scale < self.num_scales:
            expected_length = self.get_expected_length_for_scale(self.current_scale)

            # still generating tokens for this scale
            if self.tokens_in_current_scale < expected_length:
                # mask out scale end delimiter to prevent early stopping
                end_delimiter_ids = set(self.scale_delimiters[f"end_{self.current_scale}"])
                audio_end_ids = set(self.audio_end)
                forbidden = end_delimiter_ids | audio_end_ids

                # only allow audio tokens for current scale
                allowed = self.audio_token_ids[self.current_scale]

                # If not at end, also forbid next scale delimiters
                if self.current_scale < self.num_scales - 1:
                    next_start = set(self.scale_delimiters[f"start_{self.current_scale + 1}"])
                    forbidden |= next_start

                await self.observe(self.context.mask_dist(forbidden), False)

                # sample next token
                next_dist = self.context.next_token()
                token = await self.sample(next_dist)
                self.generated_tokens.append(token)
                self.tokens_in_current_scale += 1

                if self.tokens_in_current_scale % 50 == 0:
                    print(f"  Scale {self.current_scale}: {self.tokens_in_current_scale}/{expected_length} tokens")

            # force delimiter transition
            else:
                print(f"{GREEN}Completed Scale {self.current_scale}{END} ({self.tokens_in_current_scale} tokens)")

                # force </sN>
                for token_id in self.scale_delimiters[f"end_{self.current_scale}"]:
                    await self.observe(self.context.next_token(), token_id)
                    token = await self.sample(self.context.next_token())
                    self.generated_tokens.append(token)

                # move to next scale or finish
                self.current_scale += 1
                self.tokens_in_current_scale = 0

                if self.current_scale < self.num_scales:
                    # force <sN>
                    for token_id in self.scale_delimiters[f"start_{self.current_scale}"]:
                        await self.observe(self.context.next_token(), token_id)
                        token = await self.sample(self.context.next_token())
                        self.generated_tokens.append(token)

                    print(f"{GREEN}Starting Scale {self.current_scale}{END} (target: {self.get_expected_length_for_scale(self.current_scale)} tokens)")
                else:
                    # force </audio> and finish
                    for token_id in self.audio_end:
                        await self.observe(self.context.next_token(), token_id)
                        token = await self.sample(self.context.next_token())
                        self.generated_tokens.append(token)

                    print(f"{GREEN}✓ Generation Complete!{END}")
                    self.finish()

            return

        # should never reach here
        self.finish()

    def get_result(self) -> str:
        return self.lm.tokenizer.decode([t.token_id for t in self.generated_tokens])


async def generate_constrained_audio(
    model_path: str,
    prompt: str,
    base_length: int = 140,
    num_particles: int = 3,
    beam_factor: int = 1,
    temperature: float = 1.0
):
    print(f"\n{'='*60}")
    print(f"Constrained Audio Generation with SMC")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Prompt: {prompt}")
    print(f"Particles: {num_particles}, Beam: {beam_factor}")
    print(f"{'='*60}\n")

    # Load model
    lm = CachedCausalLM.from_pretrained(model_path)
    lm.batch_size = 8

    # Format prompt as chat
    full_prompt = f"Generate audio: [{prompt}]<|im_start|>assistant\n<audio>"

    # Create constrained model
    model = ConstrainedAudioModel(
        lm=lm,
        prompt=full_prompt,
        base_length=base_length,
        temperature=temperature,
        num_scales=3
    )

    # Run SMC
    print(f"Running SMC with {num_particles} particles...")
    particles = await smc_steer(model, num_particles, beam_factor)

    print(f"\n{'='*60}")
    print(f"Generated {len(particles)} particles:")
    print(f"{'='*60}\n")

    for i, particle in enumerate(particles):
        result = str(particle.context)
        print(f"{GREEN}Particle {i+1}{END} (weight: {RED}{particle.weight:.3f}{END})")
        print(f"Length: {len(result)} chars")
        print(f"Preview: {result[:200]}...\n")

    return particles


async def main():
    """Example usage."""
    # Example with a trained checkpoint
    model_path = "./checkpoints/llama3_100k_delimiters/checkpoint-5000" 
    prompt = "a cat meowing"

    particles = await generate_constrained_audio(
        model_path=model_path,
        prompt=prompt,
        base_length=144,  # Will generate perfect 1:2:4
        num_particles=3,
        beam_factor=1,
        temperature=1.0
    )

    # best particle (highest weight)
    best = max(particles, key=lambda p: p.weight)
    generated_text = str(best.context)

    print(f"\n{GREEN}Best generation:{END}")
    print(generated_text[:500])

    with open("constrained_generation.txt", "w") as f:
        f.write(generated_text)

    print(f"\n{GREEN}Saved to constrained_generation.txt{END}")
    print(f"Now decode with: python decode_audio_from_text.py constrained_generation.txt output.wav")


if __name__ == "__main__":
    asyncio.run(main())
