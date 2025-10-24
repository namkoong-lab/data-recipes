#!/usr/bin/env python3
"""
Simple example of loading an OLMo model checkpoint for inference or fine-tuning.

Usage examples:
    # Load specific checkpoint
    python load_model_example.py /path/to/checkpoint

    # Load latest checkpoint from directory
    python load_model_example.py /path/to/checkpoints --latest

    # Load on GPU and test inference
    python load_model_example.py /path/to/checkpoint --device cuda --test-generation
"""

import sys
from pathlib import Path

import torch

# Add the olmo-data-recipe directory to path so we can import modules
sys.path.append(str(Path(__file__).parent))

from load_model import load_latest_checkpoint, load_model_from_checkpoint


def simple_load_example(checkpoint_path: str, use_latest: bool = False):
    """Simple example of loading a model."""

    print(f"Loading model from: {checkpoint_path}")

    try:
        if use_latest:
            model, _, trainer_state = load_latest_checkpoint(
                checkpoint_path, device="auto"  # Use GPU if available, otherwise CPU
            )
        else:
            model, _, trainer_state = load_model_from_checkpoint(checkpoint_path, device="auto")

        print("✅ Model loaded successfully!")
        print(f"📊 Total parameters: {model.num_params():,d}")

        # Set model to evaluation mode
        model.eval()

        return model, trainer_state

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def test_text_generation(model, tokenizer=None, prompt: str = "The future of artificial intelligence"):
    """Test text generation with the loaded model."""

    print(f"\n🧪 Testing text generation with prompt: '{prompt}'")

    # If you have a tokenizer, use it. Otherwise, use simple token IDs
    if tokenizer is None:
        print("⚠️  No tokenizer provided, using random token IDs for demo")
        # Create dummy input tokens (this won't produce meaningful text)
        input_ids = torch.randint(10, model.config.vocab_size - 10, (1, 20))
    else:
        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    print(f"📥 Input shape: {input_ids.shape}")

    # Generate text
    with torch.no_grad():
        # Simple greedy generation
        generated_ids = input_ids.clone()
        max_new_tokens = 50

        for _ in range(max_new_tokens):
            # Get next token logits
            outputs = model(generated_ids)
            next_token_logits = outputs.logits[0, -1, :]  # Last token logits

            # Get most likely next token
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)

            # Add to sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

            # Simple stopping condition (you might want to use proper EOS token)
            if next_token_id.item() == 0:  # Assuming 0 is pad/eos token
                break

        print(f"📤 Generated sequence length: {generated_ids.shape[1]}")

        if tokenizer is not None:
            # Decode if tokenizer is available
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"📝 Generated text: {generated_text}")
        else:
            print(f"🔢 Generated token IDs: {generated_ids[0].tolist()}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simple OLMo model loading example")
    parser.add_argument("checkpoint_path", help="Path to checkpoint directory")
    parser.add_argument("--latest", action="store_true", help="Load latest checkpoint from directory")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--test-generation", action="store_true", help="Test text generation")
    parser.add_argument(
        "--prompt", default="The future of artificial intelligence", help="Prompt for text generation test"
    )

    args = parser.parse_args()

    # Load the model
    model, trainer_state = simple_load_example(args.checkpoint_path, args.latest)

    # Print some info about the loaded model
    if trainer_state and "global_step" in trainer_state:
        print(f"🔄 Training step: {trainer_state['global_step']}")

    # Test generation if requested
    if args.test_generation:
        test_text_generation(model, prompt=args.prompt)

    print("\n✨ Example complete!")

    return model


if __name__ == "__main__":
    main()
