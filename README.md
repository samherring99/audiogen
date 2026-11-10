# audiogen

Autoregressive audio generation by post-training a small LLM on discrete audio tokens.

Encode audio with a neural codec (MOSS or SNAC) into token sequences, format them as chat
turns, and train a Qwen2.5 model to generate audio from text prompts. Two stages: SFT,
then GRPO with a Whisper-WER reward.

## Install

```bash
pip install -e .
```

Requires Python 3.10+. Install the codec you need:

```bash
pip install -e ".[snac]"   # SNAC codec
pip install -e ".[lora]"   # LoRA adapters
```

## Pipeline

1. Encode a speech corpus into token sequences.

   ```bash
   python scripts/encode_moss.py --tar_dir /path/to/tars --output encoded.jsonl --language en
   ```

2. Phase 1 SFT — frozen base, train embeddings only.

   ```bash
   python scripts/train_sft.py \
     --data_path encoded.jsonl \
     --output_dir checkpoints/phase1 \
     --base_model Qwen/Qwen2.5-1.5B-Instruct \
     --epochs 1 --batch_size 8 --grad_accum 8 --lr 5e-7
   ```

3. Phase 2 SFT — unfreeze the full model.

   ```bash
   python scripts/train_sft.py \
     --data_path encoded.jsonl \
     --output_dir checkpoints/phase2 \
     --resume_from_checkpoint checkpoints/phase1/checkpoint-N \
     --unfreeze_all --epochs 2 --batch_size 8 --grad_accum 8 --lr 1e-6
   ```

4. GRPO refinement — Whisper WER reward.

   ```bash
   python scripts/train_grpo.py \
     --resume_from_checkpoint checkpoints/phase2/checkpoint-N \
     --prompts_path prompts.json \
     --output_dir checkpoints/grpo \
     --reward_type whisper
   ```

5. Generate.

   ```bash
   python scripts/generate.py \
     --model_path checkpoints/phase2/checkpoint-N \
     --prompt "Generate speech audio: hello world" \
     --output out.wav
   ```

## Layout

```
audiogen/
  codecs/     MOSS and SNAC codec wrappers
  data/       dataset, schema, io, metadata conditioning, tar sources
  tokens/     vocabulary expansion, chat formatting
  model/      model loading, tokenizer expansion, embedding init
  training/   SFT and GRPO trainers, LoRA
  inference/  generation and decoding
  eval/       metrics and plotting
scripts/      CLI entry points
tests/        test suite
```

## Tests

```bash
pip install pytest
python -m pytest tests/
```

Some tests load helper scripts from a local `experiments/` directory; they skip
automatically when it is absent.

## Notes

- Cross-entropy loss is a poor proxy for audio quality. Select checkpoints by listening.
- Phase 1 should be stopped at the loss plateau; overtraining the embeddings destabilizes Phase 2.
- Whisper WER is the reward that worked. CLAP showed no reward variance and SQUIM rewarded silence.
