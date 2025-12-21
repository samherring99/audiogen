import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


def expand_tokenizer_and_model(
    base_model: str = "Qwen/Qwen2.5-0.5B",
    output_dir: str = "./models/qwen-audio",
    num_audio_tokens: int = 4096,  # SNAC uses 12-bit codebook = 4096 tokens per scale
    num_scales: int = 3  # default for dataset gen -  coarse, medium, fine
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model and tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(base_model)

    print(f"Original vocab size: {len(tokenizer)}")
    print(f"Original embedding size: {model.get_input_embeddings().weight.shape}")

    # special tokens for audio
    special_tokens = ["<audio>", "</audio>"]

    # format: [AUDIO_S0_0], [AUDIO_S0_1], ..., [AUDIO_S0_4095] for scale 0 (coarse)
    #         [AUDIO_S1_0], [AUDIO_S1_1], ..., [AUDIO_S1_4095] for scale 1 (medium)
    audio_tokens = []
    for scale in range(num_scales):
        for token_id in range(num_audio_tokens):
            audio_tokens.append(f"[AUDIO_S{scale}_{token_id}]")

    print(f"\nAdding {len(special_tokens)} special tokens: {special_tokens}")
    print(f"Adding {len(audio_tokens)} audio tokens ({num_scales} scales x {num_audio_tokens} tokens)")

    num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    num_added += tokenizer.add_tokens(audio_tokens)

    print(f"Added {num_added} tokens total")
    print(f"New vocab size: {len(tokenizer)}")

    model.resize_token_embeddings(len(tokenizer))

    print(f"New embedding size: {model.get_input_embeddings().weight.shape}")

    print(f"\nSaving to {output_path}")
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)

    metadata = {
        "base_model": base_model,
        "num_audio_tokens_per_scale": num_audio_tokens,
        "num_scales": num_scales,
        "total_audio_tokens": len(audio_tokens),
        "special_tokens": special_tokens,
        "audio_token_start_id": tokenizer.convert_tokens_to_ids(audio_tokens[0]),
        "audio_token_end_id": tokenizer.convert_tokens_to_ids(audio_tokens[-1]),
        "audio_start_token_id": tokenizer.convert_tokens_to_ids("<audio>"),
        "audio_end_token_id": tokenizer.convert_tokens_to_ids("</audio>"),
    }

    with open(output_path / "audio_token_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nModel and tokenizer prepared successfully!")
    print(f"Metadata saved to: {output_path / 'audio_token_metadata.json'}")

    print("\nVerifying special tokens:")
    for token in special_tokens[:5]:
        print(f"  {token}: {tokenizer.convert_tokens_to_ids(token)}")
    print(f"  First audio token: {audio_tokens[0]} -> {tokenizer.convert_tokens_to_ids(audio_tokens[0])}")
    print(f"  Last audio token: {audio_tokens[-1]} -> {tokenizer.convert_tokens_to_ids(audio_tokens[-1])}")

    return tokenizer, model, metadata


def test_tokenization(tokenizer, metadata):
    print("\n" + "="*50)
    print("Testing tokenization:")

    test_audio_tokens = [0, 1, 2, 3, 4]

    audio_token_strs = [f"[AUDIO_S0_{tid}]" for tid in test_audio_tokens]
    audio_sequence = "".join(audio_token_strs)

    messages = [
        {"role": "user", "content": f"Generate audio: [a dog barking]"},
        {"role": "assistant", "content": f"<audio>{audio_sequence}</audio>"}
    ]

    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    print("\nFormatted chat template:")
    print(formatted)

    tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
    print(f"\nTokenized shape: {tokens.shape}")
    print(f"Number of tokens: {tokens.shape[1]}")

    decoded = tokenizer.decode(tokens[0])
    print("\nDecoded:")
    print(decoded)

    return True


if __name__ == "__main__":
    tokenizer, model, metadata = expand_tokenizer_and_model(
        base_model="Qwen/Qwen2.5-0.5B",
        output_dir="./models/qwen-audio",
        num_audio_tokens=4096,
        num_scales=3
    )

    test_tokenization(tokenizer, metadata)
