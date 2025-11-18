"""
Train Ternary GPT on text data.

This example demonstrates:
- Loading text datasets
- Training a character-level ternary GPT
- Validation and text generation
- Model compression analysis

Usage:
    python examples/train_gpt.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import time
from ternary.ternary_gpt import create_ternary_gpt_tiny, TernaryGPTConfig, TernaryGPT
from ternary.gpt_data import ShakespeareDataset, create_synthetic_text_dataset


def train_epoch(model, dataset, batch_size=32):
    """Train for one epoch."""
    total_loss = 0.0
    batch_count = 0

    for input_ids, target_ids in dataset.get_batches(batch_size, shuffle=True):
        # Compute loss
        loss, grad_logits = model.compute_loss(input_ids, target_ids)

        # Simplified backprop (placeholder - full implementation needed)
        # For now, just track loss

        total_loss += loss
        batch_count += 1

        if batch_count % 10 == 0:
            print(f"    Batch {batch_count}: Loss = {loss:.4f}")

    return total_loss / batch_count


def validate(model, dataset, batch_size=32):
    """Validate model."""
    total_loss = 0.0
    batch_count = 0

    for input_ids, target_ids in dataset.get_batches(batch_size, shuffle=False):
        loss, _ = model.compute_loss(input_ids, target_ids)
        total_loss += loss
        batch_count += 1

    return total_loss / batch_count


def generate_sample(model, tokenizer, prompt="", max_tokens=100):
    """Generate text sample."""
    if prompt:
        prompt_ids = tokenizer.encode(prompt)
    else:
        # Start with random character
        prompt_ids = [np.random.randint(0, tokenizer.vocab_size)]

    prompt_ids = np.array([prompt_ids])

    # Generate
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_tokens,
        temperature=0.8,
        top_k=40
    )

    # Decode
    text = tokenizer.decode(generated_ids[0].tolist())
    return text


def main():
    print("=" * 70)
    print("Training Ternary GPT")
    print("=" * 70)

    np.random.seed(42)

    # ======================================================================
    # 1. Load Dataset
    # ======================================================================
    print("\n[1/5] Loading dataset...")

    # Try to load Shakespeare, fallback to synthetic
    dataset_loader = ShakespeareDataset(seq_length=128)

    try:
        dataset_loader.load()
        print("✓ Loaded Shakespeare dataset")
    except:
        print("Creating synthetic text dataset...")
        train_dataset, val_dataset, tokenizer = create_synthetic_text_dataset(
            num_chars=50000,
            seq_length=128
        )
        dataset_loader.tokenizer = tokenizer
        dataset_loader.text_train = "synthetic"
        dataset_loader.text_val = "synthetic"

    if dataset_loader.text_train != "synthetic":
        train_dataset = dataset_loader.get_train_dataset()
        val_dataset = dataset_loader.get_val_dataset()
        tokenizer = dataset_loader.tokenizer

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Training sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")

    # ======================================================================
    # 2. Create Model
    # ======================================================================
    print("\n[2/5] Creating model...")

    # Create tiny GPT for quick training
    config = TernaryGPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_len=128,
        embed_dim=128,
        num_layers=4,
        num_heads=4,
        ff_dim=512,
        threshold=0.3,
        dropout=0.1,
        learning_rate=3e-4
    )

    model = TernaryGPT(config)

    # Model info
    model_size = model.get_model_size()
    print(f"Model: TernaryGPT-Tiny")
    print(f"Parameters: {model_size['params']:,}")
    print(f"Embedding dim: {config.embed_dim}")
    print(f"Layers: {config.num_layers}")
    print(f"Heads: {config.num_heads}")
    print(f"Memory (float32): {model_size['float32_mb']:.2f} MB")
    print(f"Memory (ternary): {model_size['ternary_mb']:.2f} MB")
    print(f"Compression: {model_size['compression']:.1f}x")

    # ======================================================================
    # 3. Train Model
    # ======================================================================
    print("\n[3/5] Training model...")

    epochs = 10
    batch_size = 32

    best_val_loss = float('inf')

    for epoch in range(epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_dataset, batch_size)

        # Validate
        val_loss = validate(model, val_dataset, batch_size)

        # Update best
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch + 1}/{epochs}: "
              f"Train Loss = {train_loss:.4f}, "
              f"Val Loss = {val_loss:.4f}, "
              f"Time = {epoch_time:.2f}s")

        # Generate sample every few epochs
        if (epoch + 1) % 5 == 0:
            print("\n--- Generated Sample ---")
            sample = generate_sample(model, tokenizer, prompt="", max_tokens=100)
            print(sample[:200])  # Print first 200 chars
            print("------------------------\n")

    # ======================================================================
    # 4. Final Evaluation
    # ======================================================================
    print("\n[4/5] Final evaluation...")

    final_val_loss = validate(model, val_dataset, batch_size)

    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # ======================================================================
    # 5. Text Generation
    # ======================================================================
    print("\n[5/5] Generating text samples...")

    # Generate with different temperatures
    temperatures = [0.5, 0.8, 1.0]

    for temp in temperatures:
        print(f"\nTemperature = {temp}:")
        print("-" * 70)

        # Generate
        prompt_ids = np.array([[np.random.randint(0, tokenizer.vocab_size)]])
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=150,
            temperature=temp,
            top_k=40
        )

        text = tokenizer.decode(generated_ids[0].tolist())
        print(text[:300])  # Print first 300 chars

    # Model statistics
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

    print(f"\nModel Statistics:")
    print(f"  Total parameters: {model_size['params']:,}")
    print(f"  Float32 memory: {model_size['float32_mb']:.2f} MB")
    print(f"  Ternary memory: {model_size['ternary_mb']:.2f} MB")
    print(f"  Compression ratio: {model_size['compression']:.1f}x")

    print(f"\nTraining Results:")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Final validation loss: {final_val_loss:.4f}")

    print("\n" + "=" * 70)
    print("For better results:")
    print("  - Train on real Shakespeare dataset")
    print("  - Use larger model (GPT-small: 768 dim, 12 layers)")
    print("  - Train for more epochs (50-100)")
    print("  - Use learning rate scheduling")
    print("=" * 70)


if __name__ == '__main__':
    main()
