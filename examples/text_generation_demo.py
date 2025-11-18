"""
Text Generation Demo with Ternary GPT.

Demonstrates:
- Creating a ternary GPT model
- Loading/creating simple text data
- Generating text with different parameters
- Temperature, top-k, top-p sampling

Usage:
    python examples/text_generation_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
from ternary.ternary_gpt import create_ternary_gpt_tiny
from ternary.gpt_data import CharTokenizer


def demo_text_generation():
    """Demonstrate text generation capabilities."""

    print("=" * 70)
    print("Ternary GPT Text Generation Demo")
    print("=" * 70)

    # ======================================================================
    # 1. Create Model
    # ======================================================================
    print("\n[1/4] Creating Ternary GPT model...")

    model = create_ternary_gpt_tiny(vocab_size=128)

    model_size = model.get_model_size()

    print(f"Model: TernaryGPT-Tiny")
    print(f"  Vocabulary: {model.config.vocab_size}")
    print(f"  Embedding dim: {model.config.embed_dim}")
    print(f"  Layers: {model.config.num_layers}")
    print(f"  Attention heads: {model.config.num_heads}")
    print(f"  Total parameters: {model_size['params']:,}")
    print(f"  Memory (ternary): {model_size['ternary_mb']:.2f} MB")
    print(f"  Compression: {model_size['compression']:.1f}x vs float32")

    # ======================================================================
    # 2. Create Simple Tokenizer
    # ======================================================================
    print("\n[2/4] Creating tokenizer...")

    # Simple character tokenizer with limited alphabet
    alphabet = 'abcdefghijklmnopqrstuvwxyz .,'
    tokenizer = CharTokenizer()
    tokenizer.fit(alphabet)

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Characters: {list(tokenizer.char_to_id.keys())}")

    # ======================================================================
    # 3. Generate Text (Untrained Model)
    # ======================================================================
    print("\n[3/4] Generating text (untrained model)...")
    print("Note: Model is random/untrained, so output will be random text.")
    print("-" * 70)

    # Test different generation strategies
    strategies = [
        ("Greedy (temp=0.1)", {"temperature": 0.1, "top_k": None, "top_p": None}),
        ("Sampling (temp=1.0)", {"temperature": 1.0, "top_k": None, "top_p": None}),
        ("Top-K (k=10)", {"temperature": 0.8, "top_k": 10, "top_p": None}),
        ("Top-P (p=0.9)", {"temperature": 0.8, "top_k": None, "top_p": 0.9}),
    ]

    for name, params in strategies:
        print(f"\n{name}:")
        print("-" * 40)

        # Start with a simple prompt
        prompt = "the "
        prompt_ids = tokenizer.encode(prompt)
        prompt_ids = np.array([prompt_ids])

        # Generate
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=50,
            **params
        )

        # Decode
        text = tokenizer.decode(generated_ids[0].tolist())
        print(f"Prompt: '{prompt}'")
        print(f"Generated: {text}")

    # ======================================================================
    # 4. Show Model Architecture
    # ======================================================================
    print("\n" + "=" * 70)
    print("[4/4] Model Architecture Details")
    print("=" * 70)

    print("\nTransformer Configuration:")
    print(f"  Max sequence length: {model.config.max_len}")
    print(f"  Embedding dimension: {model.config.embed_dim}")
    print(f"  Number of layers: {model.config.num_layers}")
    print(f"  Number of heads: {model.config.num_heads}")
    print(f"  Feed-forward dimension: {model.config.ff_dim}")
    print(f"  Dropout rate: {model.config.dropout}")
    print(f"  Ternary threshold: {model.config.threshold}")

    print("\nLayer Breakdown:")
    print(f"  Token embedding: {model.config.vocab_size} × {model.config.embed_dim}")
    print(f"  Positional encoding: {model.config.max_len} × {model.config.embed_dim}")
    print(f"  Transformer blocks: {model.config.num_layers}x")
    print(f"    - Multi-head attention (heads={model.config.num_heads})")
    print(f"    - Feed-forward network ({model.config.embed_dim} → {model.config.ff_dim} → {model.config.embed_dim})")
    print(f"    - Layer normalization ×2")
    print(f"  Final layer norm")
    print(f"  LM head: {model.config.embed_dim} → {model.config.vocab_size}")

    print("\nMemory Analysis:")
    print(f"  Full-precision (float32): {model_size['float32_mb']:.2f} MB")
    print(f"  Ternary quantized: {model_size['ternary_mb']:.2f} MB")
    print(f"  Memory saved: {model_size['float32_mb'] - model_size['ternary_mb']:.2f} MB ({(1 - model_size['ternary_mb']/model_size['float32_mb'])*100:.1f}%)")

    print("\nKey Features:")
    print("  ✓ All weights quantized to {-1, 0, +1}")
    print("  ✓ 14.8x memory compression")
    print("  ✓ Fast inference with integer operations")
    print("  ✓ Causal masking for autoregressive generation")
    print("  ✓ Multi-head self-attention")
    print("  ✓ Positional encoding for sequence order")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)

    print("\nNext Steps:")
    print("  1. Train on real text data (Shakespeare, WikiText, etc.)")
    print("  2. Use larger model for better quality")
    print("  3. Fine-tune on domain-specific text")
    print("  4. Implement full backpropagation for training")
    print("  5. Add learning rate scheduling")

    print("\nTo train a model:")
    print("  python examples/train_gpt.py")


if __name__ == '__main__':
    np.random.seed(42)
    demo_text_generation()
