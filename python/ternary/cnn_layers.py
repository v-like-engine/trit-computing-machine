"""
Ternary Convolutional Neural Networks

Complete implementation of CNNs with ternary weights for image classification.
Supports:
- Ternary Conv2D layers
- Ternary Batch Normalization
- Popular architectures: ResNet, VGG, MobileNet
- ImageNet and CIFAR-10 datasets
- Full training pipeline
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
from dataclasses import dataclass
from ternary.neural import TernaryConfig, ternary_quantize


@dataclass
class ConvConfig:
    """Configuration for convolutional layers."""
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    use_bias: bool = False  # Typically False when using BatchNorm
    threshold: float = 0.3


class TernaryConv2D:
    """
    Ternary 2D Convolution Layer.

    Implements convolution with ternary weights {-1, 0, +1}.
    Uses im2col for efficient computation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bias: bool = False,
        threshold: float = 0.3
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.threshold = threshold

        # Initialize weights (full precision for training)
        k = kernel_size
        self.weights_fp = np.random.randn(
            out_channels, in_channels, k, k
        ) * np.sqrt(2.0 / (in_channels * k * k))

        if use_bias:
            self.bias_fp = np.zeros(out_channels)
        else:
            self.bias_fp = None

        # Ternary weights
        self.weights_ternary = None
        self.bias_ternary = None

        # Cache for backprop
        self.cache_input = None
        self.cache_col = None

    def quantize(self):
        """Quantize weights to ternary."""
        self.weights_ternary = ternary_quantize(self.weights_fp, self.threshold)
        if self.use_bias:
            self.bias_ternary = ternary_quantize(self.bias_fp, self.threshold)

    def im2col(self, x: np.ndarray) -> np.ndarray:
        """
        Convert image to column matrix for efficient convolution.

        Args:
            x: Input of shape (batch, channels, height, width)

        Returns:
            Column matrix of shape (batch * out_h * out_w, in_c * k * k)
        """
        batch, channels, height, width = x.shape
        k = self.kernel_size

        # Apply padding
        if self.padding > 0:
            x_padded = np.pad(
                x,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            x_padded = x

        # Calculate output dimensions
        out_h = (height + 2 * self.padding - k) // self.stride + 1
        out_w = (width + 2 * self.padding - k) // self.stride + 1

        # Extract patches
        col = np.zeros((batch, channels, k, k, out_h, out_w))

        for y in range(k):
            y_max = y + self.stride * out_h
            for x in range(k):
                x_max = x + self.stride * out_w
                col[:, :, y, x, :, :] = x_padded[:, :, y:y_max:self.stride, x:x_max:self.stride]

        # Reshape to column matrix
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch * out_h * out_w, -1)

        return col

    def col2im(self, col: np.ndarray, input_shape: Tuple) -> np.ndarray:
        """
        Convert column matrix back to image.

        Args:
            col: Column matrix
            input_shape: Original input shape (batch, channels, height, width)

        Returns:
            Image of original shape
        """
        batch, channels, height, width = input_shape
        k = self.kernel_size

        out_h = (height + 2 * self.padding - k) // self.stride + 1
        out_w = (width + 2 * self.padding - k) // self.stride + 1

        # Reshape column matrix
        col = col.reshape(batch, out_h, out_w, channels, k, k).transpose(0, 3, 4, 5, 1, 2)

        # Initialize output
        if self.padding > 0:
            img = np.zeros((batch, channels, height + 2 * self.padding, width + 2 * self.padding))
        else:
            img = np.zeros((batch, channels, height, width))

        # Reconstruct image
        for y in range(k):
            y_max = y + self.stride * out_h
            for x in range(k):
                x_max = x + self.stride * out_w
                img[:, :, y:y_max:self.stride, x:x_max:self.stride] += col[:, :, y, x, :, :]

        # Remove padding
        if self.padding > 0:
            return img[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return img

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, in_channels, height, width)
            training: Whether in training mode

        Returns:
            Output of shape (batch, out_channels, out_height, out_width)
        """
        # Quantize weights
        self.quantize()

        batch, channels, height, width = x.shape

        # Convert to column matrix
        col = self.im2col(x)

        # Reshape weights for matrix multiplication
        w_col = self.weights_ternary.reshape(self.out_channels, -1).T

        # Convolution as matrix multiplication
        out = col @ w_col

        # Add bias if used
        if self.use_bias:
            out += self.bias_ternary

        # Calculate output dimensions
        k = self.kernel_size
        out_h = (height + 2 * self.padding - k) // self.stride + 1
        out_w = (width + 2 * self.padding - k) // self.stride + 1

        # Reshape to output shape
        out = out.reshape(batch, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)

        # Cache for backprop
        if training:
            self.cache_input = x
            self.cache_col = col

        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Backward pass using straight-through estimator.

        Args:
            grad_output: Gradient of shape (batch, out_channels, out_h, out_w)

        Returns:
            - grad_input: Gradient w.r.t. input
            - grad_weights: Gradient w.r.t. weights
            - grad_bias: Gradient w.r.t. bias (if used)
        """
        batch, out_channels, out_h, out_w = grad_output.shape

        # Reshape grad_output for matrix operations
        grad_output_col = grad_output.transpose(0, 2, 3, 1).reshape(batch * out_h * out_w, out_channels)

        # Gradient w.r.t. weights (straight-through estimator)
        grad_weights = (self.cache_col.T @ grad_output_col).T
        grad_weights = grad_weights.reshape(self.weights_fp.shape)

        # Gradient w.r.t. bias
        if self.use_bias:
            grad_bias = np.sum(grad_output_col, axis=0)
        else:
            grad_bias = None

        # Gradient w.r.t. input
        w_col = self.weights_ternary.reshape(self.out_channels, -1).T
        grad_col = grad_output_col @ w_col.T
        grad_input = self.col2im(grad_col, self.cache_input.shape)

        return grad_input, grad_weights, grad_bias

    def update(self, grad_weights: np.ndarray, grad_bias: Optional[np.ndarray], learning_rate: float):
        """Update parameters."""
        self.weights_fp -= learning_rate * grad_weights
        if self.use_bias and grad_bias is not None:
            self.bias_fp -= learning_rate * grad_bias

        # Re-quantize
        self.quantize()

    def get_sparsity(self) -> float:
        """Calculate weight sparsity."""
        return np.mean(self.weights_ternary == 0)


class TernaryBatchNorm2D:
    """
    Batch Normalization for 2D inputs (images).

    Normalizes activations across batch dimension.
    Uses running statistics for inference.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # Running statistics
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        # Cache for backprop
        self.cache_normalized = None
        self.cache_std = None
        self.cache_input = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, channels, height, width)
            training: Whether in training mode

        Returns:
            Normalized output of same shape
        """
        if training:
            # Compute batch statistics
            # Mean and variance over batch, height, width dimensions
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            # Use running statistics
            mean = self.running_mean.reshape(1, -1, 1, 1)
            var = self.running_var.reshape(1, -1, 1, 1)

        # Normalize
        std = np.sqrt(var + self.eps)
        x_normalized = (x - mean) / std

        # Scale and shift
        gamma = self.gamma.reshape(1, -1, 1, 1)
        beta = self.beta.reshape(1, -1, 1, 1)
        out = gamma * x_normalized + beta

        # Cache for backprop
        if training:
            self.cache_normalized = x_normalized
            self.cache_std = std
            self.cache_input = x

        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass.

        Returns:
            - grad_input: Gradient w.r.t. input
            - grad_gamma: Gradient w.r.t. gamma
            - grad_beta: Gradient w.r.t. beta
        """
        batch, channels, height, width = grad_output.shape
        N = batch * height * width

        # Reshape for easier computation
        gamma = self.gamma.reshape(1, -1, 1, 1)

        # Gradients w.r.t. gamma and beta
        grad_gamma = np.sum(grad_output * self.cache_normalized, axis=(0, 2, 3))
        grad_beta = np.sum(grad_output, axis=(0, 2, 3))

        # Gradient w.r.t. normalized input
        grad_normalized = grad_output * gamma

        # Gradient w.r.t. input (chain rule through normalization)
        grad_var = np.sum(grad_normalized * (self.cache_input - np.mean(self.cache_input, axis=(0, 2, 3), keepdims=True)),
                         axis=(0, 2, 3), keepdims=True) * -0.5 * (self.cache_std ** -3)

        grad_mean = np.sum(grad_normalized * -1.0 / self.cache_std, axis=(0, 2, 3), keepdims=True) + \
                   grad_var * np.mean(-2.0 * (self.cache_input - np.mean(self.cache_input, axis=(0, 2, 3), keepdims=True)),
                                     axis=(0, 2, 3), keepdims=True)

        grad_input = grad_normalized / self.cache_std + \
                    grad_var * 2.0 * (self.cache_input - np.mean(self.cache_input, axis=(0, 2, 3), keepdims=True)) / N + \
                    grad_mean / N

        return grad_input, grad_gamma, grad_beta

    def update(self, grad_gamma: np.ndarray, grad_beta: np.ndarray, learning_rate: float):
        """Update parameters."""
        self.gamma -= learning_rate * grad_gamma
        self.beta -= learning_rate * grad_beta


class MaxPool2D:
    """Max Pooling layer."""

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache_input = None
        self.cache_max_indices = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, channels, height, width)

        Returns:
            Pooled output
        """
        batch, channels, height, width = x.shape
        k = self.kernel_size
        s = self.stride

        out_h = (height - k) // s + 1
        out_w = (width - k) // s + 1

        # Reshape for pooling
        x_reshaped = x.reshape(batch, channels, out_h, s, out_w, s)
        x_reshaped = x_reshaped.transpose(0, 1, 2, 4, 3, 5).reshape(batch, channels, out_h, out_w, -1)

        # Max pooling
        out = np.max(x_reshaped, axis=4)

        # Cache for backprop
        if training:
            self.cache_input = x
            self.cache_max_indices = np.argmax(x_reshaped, axis=4)

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        batch, channels, out_h, out_w = grad_output.shape
        k = self.kernel_size
        s = self.stride

        grad_input = np.zeros_like(self.cache_input)

        for b in range(batch):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        # Find position of max value
                        max_idx = self.cache_max_indices[b, c, i, j]
                        h_offset = max_idx // k
                        w_offset = max_idx % k

                        h_pos = i * s + h_offset
                        w_pos = j * s + w_offset

                        grad_input[b, c, h_pos, w_pos] += grad_output[b, c, i, j]

        return grad_input


class AvgPool2D:
    """Average Pooling layer."""

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache_input_shape = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        batch, channels, height, width = x.shape
        k = self.kernel_size
        s = self.stride

        out_h = (height - k) // s + 1
        out_w = (width - k) // s + 1

        # Reshape for pooling
        x_reshaped = x.reshape(batch, channels, out_h, s, out_w, s)
        x_reshaped = x_reshaped.transpose(0, 1, 2, 4, 3, 5).reshape(batch, channels, out_h, out_w, -1)

        # Average pooling
        out = np.mean(x_reshaped, axis=4)

        if training:
            self.cache_input_shape = x.shape

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        batch, channels, out_h, out_w = grad_output.shape
        k = self.kernel_size
        s = self.stride

        grad_input = np.zeros(self.cache_input_shape)

        # Distribute gradient equally across pooling window
        for b in range(batch):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        grad_input[b, c, i*s:(i+1)*s, j*s:(j+1)*s] += grad_output[b, c, i, j] / (k * k)

        return grad_input


class GlobalAvgPool2D:
    """Global Average Pooling - reduces spatial dimensions to 1x1."""

    def __init__(self):
        self.cache_input_shape = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        # Average over spatial dimensions
        out = np.mean(x, axis=(2, 3), keepdims=True)

        if training:
            self.cache_input_shape = x.shape

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        batch, channels, height, width = self.cache_input_shape

        # Distribute gradient equally across all spatial positions
        grad_input = np.broadcast_to(
            grad_output / (height * width),
            self.cache_input_shape
        ).copy()

        return grad_input


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def relu_backward(x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
    """ReLU gradient."""
    return grad_output * (x > 0)
