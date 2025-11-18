"""
Ternary GPT - Decoder-only Transformer with Ternary Weights.

Implements a GPT-style language model with ternary quantized weights
for efficient text generation and language modeling.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from .ternary_transformer import (
    TernaryTransformerBlock,
    TernaryEmbedding,
    PositionalEncoding,
    TernaryLayerNorm,
    create_causal_mask
)
from .neural import TernaryLinear


class TernaryGPTConfig:
    """Configuration for TernaryGPT model."""

    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2 vocab size
        max_len: int = 1024,      # Maximum sequence length
        embed_dim: int = 768,     # Embedding dimension
        num_layers: int = 12,     # Number of transformer layers
        num_heads: int = 12,      # Number of attention heads
        ff_dim: int = 3072,       # Feed-forward dimension (4 * embed_dim)
        threshold: float = 0.3,   # Ternary quantization threshold
        dropout: float = 0.1,     # Dropout rate
        learning_rate: float = 3e-4
    ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.threshold = threshold
        self.dropout = dropout
        self.learning_rate = learning_rate


class TernaryGPT:
    """
    Ternary GPT - Generative Pre-trained Transformer with ternary weights.

    Architecture:
    - Token embedding + positional encoding
    - N transformer blocks
    - Layer normalization
    - Language modeling head

    All linear layers use ternary weights {-1, 0, +1}.
    """

    def __init__(self, config: TernaryGPTConfig):
        self.config = config

        # Token embedding
        self.token_embedding = TernaryEmbedding(
            config.vocab_size,
            config.embed_dim,
            config.threshold
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.max_len, config.embed_dim)

        # Transformer blocks
        self.blocks = []
        for _ in range(config.num_layers):
            block = TernaryTransformerBlock(
                config.embed_dim,
                config.num_heads,
                config.ff_dim,
                config.threshold,
                config.dropout
            )
            self.blocks.append(block)

        # Final layer norm
        self.ln_f = TernaryLayerNorm(config.embed_dim)

        # Language modeling head (vocabulary projection)
        self.lm_head = TernaryLinear(
            config.embed_dim,
            config.vocab_size,
            config.threshold
        )

        # Training state
        self.learning_rate = config.learning_rate

    def forward(
        self,
        input_ids: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Forward pass through GPT.

        Args:
            input_ids: (batch_size, seq_len) token indices
            training: Whether in training mode

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding.forward(input_ids, training)

        # Add positional encodings
        x = self.pos_encoding.forward(x)

        # Create causal mask (prevent attending to future tokens)
        mask = create_causal_mask(seq_len)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask=mask, training=training)

        # Final layer norm
        x = self.ln_f.forward(x, training)

        # Project to vocabulary
        logits = self.lm_head.forward(x, training)

        return logits

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate text autoregressively.

        Args:
            prompt_ids: (1, prompt_len) initial token indices
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k logits (nucleus sampling)
            top_p: Keep top tokens with cumulative probability p

        Returns:
            generated_ids: (1, prompt_len + max_new_tokens)
        """
        generated = prompt_ids.copy()

        for _ in range(max_new_tokens):
            # Get predictions for current sequence
            # Truncate to max_len if needed
            context = generated[:, -self.config.max_len:]

            # Forward pass (no training)
            logits = self.forward(context, training=False)

            # Get logits for last position
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < np.sort(next_token_logits, axis=-1)[:, -top_k]
                next_token_logits[indices_to_remove] = -float('inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_indices = np.argsort(next_token_logits, axis=-1)[:, ::-1]
                sorted_logits = np.take_along_axis(next_token_logits, sorted_indices, axis=-1)

                # Softmax
                sorted_probs = self._softmax(sorted_logits)

                # Cumulative probabilities
                cumulative_probs = np.cumsum(sorted_probs, axis=-1)

                # Remove tokens with cumulative probability > p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].copy()
                sorted_indices_to_remove[:, 0] = False

                # Map back to original indices
                indices_to_remove = np.zeros_like(next_token_logits, dtype=bool)
                np.put_along_axis(indices_to_remove, sorted_indices, sorted_indices_to_remove, axis=-1)
                next_token_logits[indices_to_remove] = -float('inf')

            # Sample from distribution
            probs = self._softmax(next_token_logits)
            next_token = self._sample(probs)

            # Append to sequence
            generated = np.concatenate([generated, next_token.reshape(1, 1)], axis=1)

        return generated

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        max_val = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_val)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _sample(self, probs: np.ndarray) -> np.ndarray:
        """Sample from probability distribution."""
        # Flatten to 1D, sample, return as array
        return np.array([np.random.choice(len(probs[0]), p=probs[0])])

    def compute_loss(
        self,
        input_ids: np.ndarray,
        target_ids: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compute cross-entropy loss for language modeling.

        Args:
            input_ids: (batch_size, seq_len) input tokens
            target_ids: (batch_size, seq_len) target tokens (shifted by 1)

        Returns:
            loss: Scalar loss value
            grad_logits: Gradient w.r.t. logits
        """
        # Forward pass
        logits = self.forward(input_ids, training=True)

        batch_size, seq_len, vocab_size = logits.shape

        # Flatten for easier computation
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = target_ids.reshape(-1)

        # Softmax
        max_logits = np.max(logits_flat, axis=1, keepdims=True)
        exp_logits = np.exp(logits_flat - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Cross-entropy loss
        loss = 0.0
        for i in range(len(targets_flat)):
            if targets_flat[i] >= 0:  # Ignore padding tokens (if any)
                loss -= np.log(probs[i, targets_flat[i]] + 1e-8)
        loss /= len(targets_flat)

        # Gradient
        grad_logits_flat = probs.copy()
        for i in range(len(targets_flat)):
            if targets_flat[i] >= 0:
                grad_logits_flat[i, targets_flat[i]] -= 1
        grad_logits_flat /= len(targets_flat)

        # Reshape back
        grad_logits = grad_logits_flat.reshape(batch_size, seq_len, vocab_size)

        return loss, grad_logits

    def get_num_params(self) -> Dict[str, int]:
        """Get parameter counts."""
        total = 0

        # Token embedding
        total += self.token_embedding.vocab_size * self.token_embedding.embed_dim

        # Transformer blocks
        for block in self.blocks:
            # Attention (Q, K, V, O projections)
            total += 4 * self.config.embed_dim * self.config.embed_dim
            # FFN (two layers)
            total += self.config.embed_dim * self.config.ff_dim
            total += self.config.ff_dim * self.config.embed_dim
            # Layer norms (gamma, beta)
            total += 2 * 2 * self.config.embed_dim

        # Final layer norm
        total += 2 * self.config.embed_dim

        # LM head
        total += self.config.embed_dim * self.config.vocab_size

        return {
            'total': total,
            'ternary': total,  # All weights are ternary
            'non_ternary': 0
        }

    def get_model_size(self) -> Dict[str, float]:
        """Get model size in MB."""
        params = self.get_num_params()

        # Float32 size
        float32_size = params['total'] * 4 / (1024 * 1024)

        # Ternary size (~0.27 bits per weight)
        ternary_size = params['total'] * 0.27 / 8 / (1024 * 1024)

        return {
            'params': params['total'],
            'float32_mb': float32_size,
            'ternary_mb': ternary_size,
            'compression': float32_size / ternary_size if ternary_size > 0 else 0
        }


def create_ternary_gpt_small(vocab_size: int = 50257) -> TernaryGPT:
    """Create small GPT (GPT-2 small config)."""
    config = TernaryGPTConfig(
        vocab_size=vocab_size,
        max_len=1024,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        ff_dim=3072
    )
    return TernaryGPT(config)


def create_ternary_gpt_medium(vocab_size: int = 50257) -> TernaryGPT:
    """Create medium GPT (GPT-2 medium config)."""
    config = TernaryGPTConfig(
        vocab_size=vocab_size,
        max_len=1024,
        embed_dim=1024,
        num_layers=24,
        num_heads=16,
        ff_dim=4096
    )
    return TernaryGPT(config)


def create_ternary_gpt_tiny(vocab_size: int = 256) -> TernaryGPT:
    """Create tiny GPT for testing/demo."""
    config = TernaryGPTConfig(
        vocab_size=vocab_size,
        max_len=256,
        embed_dim=128,
        num_layers=4,
        num_heads=4,
        ff_dim=512
    )
    return TernaryGPT(config)
