"""
Ternary Transformer Components.

Implements transformer building blocks with ternary weights:
- Multi-head self-attention
- Feed-forward networks
- Layer normalization
- Positional embeddings
- Causal masking for autoregressive generation
"""

import numpy as np
from typing import Optional, Tuple, List
from .neural import TernaryLinear, ternary_quantize


class TernaryMultiHeadAttention:
    """
    Multi-head self-attention with ternary weights.

    Computes attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    With ternary weight matrices for Q, K, V projections.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        threshold: float = 0.3,
        dropout: float = 0.1
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.threshold = threshold
        self.dropout = dropout

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        # Projections (ternary)
        self.q_proj = TernaryLinear(embed_dim, embed_dim, threshold)
        self.k_proj = TernaryLinear(embed_dim, embed_dim, threshold)
        self.v_proj = TernaryLinear(embed_dim, embed_dim, threshold)
        self.out_proj = TernaryLinear(embed_dim, embed_dim, threshold)

        # Cache for backprop
        self.cache = {}

    def forward(
        self,
        query: np.ndarray,
        key: Optional[np.ndarray] = None,
        value: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        training: bool = True
    ) -> np.ndarray:
        """
        Forward pass through multi-head attention.

        Args:
            query: (batch_size, seq_len, embed_dim)
            key: (batch_size, seq_len, embed_dim) or None (self-attention)
            value: (batch_size, seq_len, embed_dim) or None (self-attention)
            mask: (batch_size, seq_len, seq_len) or None
            training: Whether in training mode

        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = query.shape

        # Self-attention if key/value not provided
        if key is None:
            key = query
        if value is None:
            value = query

        # Project to Q, K, V
        Q = self.q_proj.forward(query, training)  # (batch, seq, embed_dim)
        K = self.k_proj.forward(key, training)
        V = self.v_proj.forward(value, training)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        # scores = QK^T / sqrt(d_k)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)

        # Apply mask (for causal attention)
        if mask is not None:
            scores = scores + mask  # mask is -inf for masked positions

        # Softmax
        attn_weights = self.softmax(scores)

        # Dropout (during training)
        if training and self.dropout > 0:
            dropout_mask = np.random.rand(*attn_weights.shape) > self.dropout
            attn_weights = attn_weights * dropout_mask / (1 - self.dropout)

        # Apply attention to values
        attn_output = np.matmul(attn_weights, V)  # (batch, heads, seq, head_dim)

        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

        # Output projection
        output = self.out_proj.forward(attn_output, training)

        # Cache for backprop
        if training:
            self.cache = {
                'query': query,
                'key': key,
                'value': value,
                'Q': Q,
                'K': K,
                'V': V,
                'scores': scores,
                'attn_weights': attn_weights,
                'attn_output': attn_output,
                'mask': mask
            }

        return output

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        max_val = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_val)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def backward(self, grad_output: np.ndarray):
        """Backward pass (simplified - full implementation needed for training)."""
        # This is a placeholder - full attention backprop is complex
        # For now, just pass gradient through output projection
        return self.out_proj.backward(grad_output)


class TernaryFeedForward:
    """
    Feed-forward network with ternary weights.

    FFN(x) = ReLU(xW1 + b1)W2 + b2
    Typically expands dimension by 4x then projects back.
    """

    def __init__(
        self,
        embed_dim: int,
        ff_dim: int,
        threshold: float = 0.3,
        dropout: float = 0.1
    ):
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout = dropout

        # Two linear layers
        self.fc1 = TernaryLinear(embed_dim, ff_dim, threshold)
        self.fc2 = TernaryLinear(ff_dim, embed_dim, threshold)

        self.cache = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: (batch_size, seq_len, embed_dim)
            training: Whether in training mode

        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        # First layer + ReLU
        hidden = self.fc1.forward(x, training)
        hidden = np.maximum(0, hidden)  # ReLU

        # Dropout
        if training and self.dropout > 0:
            dropout_mask = np.random.rand(*hidden.shape) > self.dropout
            hidden = hidden * dropout_mask / (1 - self.dropout)

        # Second layer
        output = self.fc2.forward(hidden, training)

        if training:
            self.cache = {'x': x, 'hidden': hidden}

        return output

    def backward(self, grad_output: np.ndarray):
        """Backward pass (simplified)."""
        # Placeholder
        return grad_output


class TernaryLayerNorm:
    """
    Layer normalization.

    LayerNorm(x) = γ * (x - μ) / σ + β
    where μ, σ are computed per layer.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Learnable parameters
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)

        self.cache = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: (..., normalized_shape)
            training: Whether in training mode

        Returns:
            output: Same shape as x
        """
        # Compute mean and variance along last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        # Normalize
        x_normalized = (x - mean) / np.sqrt(var + self.eps)

        # Scale and shift
        output = self.gamma * x_normalized + self.beta

        if training:
            self.cache = {
                'x': x,
                'mean': mean,
                'var': var,
                'x_normalized': x_normalized
            }

        return output

    def backward(self, grad_output: np.ndarray):
        """Backward pass (simplified)."""
        # Placeholder
        return grad_output


class TernaryEmbedding:
    """
    Token embedding with ternary weights.

    Maps token indices to dense vectors.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        threshold: float = 0.3
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.threshold = threshold

        # Embedding matrix (full precision for training)
        self.weights_fp = np.random.randn(vocab_size, embed_dim) * 0.02
        self.weights_ternary = None

        self.cache_indices = None

    def quantize(self):
        """Quantize embeddings to ternary."""
        self.weights_ternary = ternary_quantize(self.weights_fp, self.threshold)

    def forward(self, indices: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.

        Args:
            indices: (batch_size, seq_len) token indices
            training: Whether in training mode

        Returns:
            embeddings: (batch_size, seq_len, embed_dim)
        """
        # Quantize
        self.quantize()

        # Lookup embeddings
        embeddings = self.weights_ternary[indices]

        if training:
            self.cache_indices = indices

        return embeddings

    def backward(self, grad_output: np.ndarray):
        """Backward pass (simplified)."""
        # Placeholder
        return None


class PositionalEncoding:
    """
    Sinusoidal positional encoding.

    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """

    def __init__(self, max_len: int, embed_dim: int):
        self.max_len = max_len
        self.embed_dim = embed_dim

        # Precompute positional encodings
        pe = np.zeros((max_len, embed_dim))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = pe

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input.

        Args:
            x: (batch_size, seq_len, embed_dim)

        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        return x + self.pe[:seq_len, :embed_dim]


class TernaryTransformerBlock:
    """
    Complete transformer block with ternary weights.

    Block(x) = LayerNorm(x + Attention(x))
               LayerNorm(x + FFN(x))
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        threshold: float = 0.3,
        dropout: float = 0.1
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        # Sub-layers
        self.attn = TernaryMultiHeadAttention(embed_dim, num_heads, threshold, dropout)
        self.ffn = TernaryFeedForward(embed_dim, ff_dim, threshold, dropout)

        # Layer normalization
        self.ln1 = TernaryLayerNorm(embed_dim)
        self.ln2 = TernaryLayerNorm(embed_dim)

        self.dropout = dropout

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = True
    ) -> np.ndarray:
        """
        Forward pass through transformer block.

        Args:
            x: (batch_size, seq_len, embed_dim)
            mask: Attention mask
            training: Whether in training mode

        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        # Multi-head attention with residual
        attn_out = self.attn.forward(x, mask=mask, training=training)

        # Dropout
        if training and self.dropout > 0:
            dropout_mask = np.random.rand(*attn_out.shape) > self.dropout
            attn_out = attn_out * dropout_mask / (1 - self.dropout)

        x = self.ln1.forward(x + attn_out, training)

        # Feed-forward with residual
        ffn_out = self.ffn.forward(x, training)

        # Dropout
        if training and self.dropout > 0:
            dropout_mask = np.random.rand(*ffn_out.shape) > self.dropout
            ffn_out = ffn_out * dropout_mask / (1 - self.dropout)

        x = self.ln2.forward(x + ffn_out, training)

        return x


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal attention mask for autoregressive generation.

    Mask is -inf for future positions, 0 for current/past positions.

    Args:
        seq_len: Sequence length

    Returns:
        mask: (1, 1, seq_len, seq_len)
    """
    mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
    return mask.reshape(1, 1, seq_len, seq_len)
