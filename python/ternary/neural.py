"""
Ternary Neural Networks with proper gradient flow.

Implements neural networks with ternary weights {-1, 0, 1}
using straight-through estimators for backpropagation.
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from ternary import Tryte, Memory
from ternary.encoding import TritEncoder, TryteEncoder


@dataclass
class TernaryConfig:
    """Configuration for ternary neural network."""
    use_ternary_activations: bool = False  # If True, quantize activations too
    threshold: float = 0.3  # Threshold for ternary quantization
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 10


def ternary_quantize(x: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Quantize values to {-1, 0, +1}.

    Args:
        x: Input array
        threshold: Values with |x| < threshold become 0

    Returns:
        Quantized array with values in {-1, 0, 1}
    """
    quantized = np.zeros_like(x)
    quantized[x > threshold] = 1.0
    quantized[x < -threshold] = -1.0
    # Values in [-threshold, threshold] stay 0
    return quantized


def ternary_quantize_stochastic(x: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Stochastic ternary quantization for better gradient flow.

    Uses probability-based quantization to maintain expectation.
    """
    # Clip to [-1, 1]
    x_clipped = np.clip(x, -1, 1)

    # Calculate probabilities
    abs_x = np.abs(x_clipped)

    # For values > threshold, use deterministic quantization
    # For values < threshold, use stochastic
    quantized = np.zeros_like(x)

    # Stochastic quantization
    prob_positive = (x_clipped + 1) / 2  # Maps [-1, 1] to [0, 1]
    random_vals = np.random.rand(*x.shape)

    quantized = np.where(
        abs_x > threshold,
        np.sign(x_clipped),  # Deterministic for large values
        np.where(
            random_vals < prob_positive,
            1.0,
            np.where(random_vals < prob_positive + (1 - abs_x) / 2, 0.0, -1.0)
        )
    )

    return quantized


class TernaryLinear:
    """
    Ternary linear layer with {-1, 0, 1} weights.

    Uses straight-through estimator for backpropagation:
    - Forward: y = quantize(W) @ x
    - Backward: gradient flows through as if quantize is identity
    """

    def __init__(self, in_features: int, out_features: int, config: TernaryConfig):
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Initialize weights with small random values
        # We keep full-precision weights for training
        self.weights_fp = np.random.randn(out_features, in_features) * 0.1
        self.bias_fp = np.zeros(out_features)

        # Quantized weights (for forward pass)
        self.weights_ternary = ternary_quantize(self.weights_fp, config.threshold)
        self.bias_ternary = ternary_quantize(self.bias_fp, config.threshold)

        # Cache for backprop
        self.cache_input = None
        self.cache_weights_fp = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass with ternary weights.

        Args:
            x: Input of shape (batch_size, in_features)
            training: If True, cache for backprop

        Returns:
            Output of shape (batch_size, out_features)
        """
        # Quantize weights
        self.weights_ternary = ternary_quantize(self.weights_fp, self.config.threshold)

        # Forward pass: y = W_ternary @ x + b
        output = x @ self.weights_ternary.T + self.bias_ternary

        if training:
            self.cache_input = x
            self.cache_weights_fp = self.weights_fp.copy()

        return output

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass using straight-through estimator.

        Args:
            grad_output: Gradient from next layer, shape (batch_size, out_features)

        Returns:
            - grad_input: Gradient w.r.t. input, shape (batch_size, in_features)
            - grad_weights: Gradient w.r.t. weights, shape (out_features, in_features)
            - grad_bias: Gradient w.r.t. bias, shape (out_features,)
        """
        batch_size = grad_output.shape[0]

        # Gradient w.r.t. input: dL/dx = dL/dy @ W^T
        # Use ternary weights for backward pass too (matches forward)
        grad_input = grad_output @ self.weights_ternary

        # Gradient w.r.t. weights: dL/dW = dL/dy^T @ x
        # This is where straight-through estimator comes in:
        # We compute gradient as if quantization didn't exist
        grad_weights_fp = (grad_output.T @ self.cache_input) / batch_size

        # Gradient w.r.t. bias
        grad_bias_fp = np.mean(grad_output, axis=0)

        return grad_input, grad_weights_fp, grad_bias_fp

    def update(self, grad_weights: np.ndarray, grad_bias: np.ndarray):
        """Update full-precision weights using gradients."""
        lr = self.config.learning_rate

        # Update full-precision weights
        self.weights_fp -= lr * grad_weights
        self.bias_fp -= lr * grad_bias

        # Re-quantize for next forward pass
        self.weights_ternary = ternary_quantize(self.weights_fp, self.config.threshold)
        self.bias_ternary = ternary_quantize(self.bias_fp, self.config.threshold)

    def get_ternary_weights(self) -> np.ndarray:
        """Get quantized ternary weights."""
        return self.weights_ternary

    def count_nonzero_weights(self) -> int:
        """Count non-zero weights (sparsity metric)."""
        return np.count_nonzero(self.weights_ternary)


class TernaryActivation:
    """Ternary activation function."""

    def __init__(self, config: TernaryConfig):
        self.config = config
        self.cache_input = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.

        For ternary networks, we can use:
        1. Standard ReLU (keeps full precision)
        2. Ternary quantization (more aggressive)
        3. Scaled sign function
        """
        if training:
            self.cache_input = x

        if self.config.use_ternary_activations:
            # Ternary activation: quantize to {-1, 0, 1}
            return ternary_quantize(x, self.config.threshold)
        else:
            # Standard ReLU
            return np.maximum(0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        if self.config.use_ternary_activations:
            # Straight-through estimator: gradient = 1 for |x| > threshold
            mask = np.abs(self.cache_input) > self.config.threshold
            return grad_output * mask
        else:
            # ReLU gradient
            return grad_output * (self.cache_input > 0)


class TernaryNeuralNetwork:
    """
    Multi-layer ternary neural network.

    Architecture: Input -> TernaryLinear -> Activation -> ... -> Output
    """

    def __init__(self, layer_sizes: List[int], config: TernaryConfig):
        """
        Initialize network.

        Args:
            layer_sizes: List of layer sizes, e.g., [784, 256, 128, 10]
            config: Network configuration
        """
        self.layer_sizes = layer_sizes
        self.config = config
        self.layers = []

        # Create layers
        for i in range(len(layer_sizes) - 1):
            layer = TernaryLinear(layer_sizes[i], layer_sizes[i + 1], config)
            self.layers.append(layer)

        # Activations (one less than layers, no activation after output)
        self.activations = []
        for i in range(len(layer_sizes) - 2):
            self.activations.append(TernaryActivation(config))

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through network.

        Args:
            x: Input of shape (batch_size, input_size)

        Returns:
            Output of shape (batch_size, output_size)
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer.forward(x, training)
            x = self.activations[i].forward(x, training)

        # Last layer (no activation)
        x = self.layers[-1].forward(x, training)

        return x

    def backward(self, grad_output: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Backward pass through network.

        Args:
            grad_output: Gradient from loss

        Returns:
            List of (grad_weights, grad_bias) for each layer
        """
        gradients = []

        # Backward through last layer
        grad = grad_output
        grad_input, grad_w, grad_b = self.layers[-1].backward(grad)
        gradients.insert(0, (grad_w, grad_b))

        # Backward through remaining layers
        for i in range(len(self.layers) - 2, -1, -1):
            # Backward through activation
            grad = self.activations[i].backward(grad_input)

            # Backward through linear layer
            grad_input, grad_w, grad_b = self.layers[i].backward(grad)
            gradients.insert(0, (grad_w, grad_b))

        return gradients

    def update(self, gradients: List[Tuple[np.ndarray, np.ndarray]]):
        """Update all layers with computed gradients."""
        for layer, (grad_w, grad_b) in zip(self.layers, gradients):
            layer.update(grad_w, grad_b)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        logits = self.forward(x, training=False)
        return np.argmax(logits, axis=1)

    def get_sparsity(self) -> dict:
        """Get sparsity statistics for all layers."""
        stats = {}
        total_weights = 0
        total_nonzero = 0

        for i, layer in enumerate(self.layers):
            n_weights = layer.weights_ternary.size
            n_nonzero = layer.count_nonzero_weights()
            sparsity = 1.0 - (n_nonzero / n_weights)

            stats[f'layer_{i}'] = {
                'total': n_weights,
                'nonzero': n_nonzero,
                'sparsity': sparsity
            }

            total_weights += n_weights
            total_nonzero += n_nonzero

        stats['overall'] = {
            'total': total_weights,
            'nonzero': total_nonzero,
            'sparsity': 1.0 - (total_nonzero / total_weights)
        }

        return stats


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Cross-entropy loss with gradient.

    Args:
        logits: Raw scores, shape (batch_size, num_classes)
        labels: True labels, shape (batch_size,)

    Returns:
        - loss: Scalar loss value
        - grad: Gradient w.r.t. logits
    """
    batch_size = logits.shape[0]

    # Softmax probabilities
    probs = softmax(logits)

    # Cross-entropy loss
    log_probs = np.log(probs + 1e-8)
    loss = -np.mean(log_probs[np.arange(batch_size), labels])

    # Gradient
    grad = probs.copy()
    grad[np.arange(batch_size), labels] -= 1
    grad /= batch_size

    return loss, grad


def train_step(
    model: TernaryNeuralNetwork,
    x_batch: np.ndarray,
    y_batch: np.ndarray
) -> Tuple[float, float]:
    """
    Single training step.

    Args:
        model: Ternary neural network
        x_batch: Input batch
        y_batch: Label batch

    Returns:
        - loss: Training loss
        - accuracy: Training accuracy
    """
    # Forward pass
    logits = model.forward(x_batch, training=True)

    # Compute loss and gradient
    loss, grad_output = cross_entropy_loss(logits, y_batch)

    # Backward pass
    gradients = model.backward(grad_output)

    # Update weights
    model.update(gradients)

    # Compute accuracy
    predictions = np.argmax(logits, axis=1)
    accuracy = np.mean(predictions == y_batch)

    return loss, accuracy


def evaluate(model: TernaryNeuralNetwork, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate model.

    Returns:
        - loss: Evaluation loss
        - accuracy: Evaluation accuracy
    """
    logits = model.forward(x, training=False)
    loss, _ = cross_entropy_loss(logits, y)
    predictions = np.argmax(logits, axis=1)
    accuracy = np.mean(predictions == y)

    return loss, accuracy


if __name__ == "__main__":
    print("=" * 70)
    print("Ternary Neural Network")
    print("=" * 70)

    # Test on synthetic data
    np.random.seed(42)

    # Create synthetic dataset
    n_samples = 1000
    n_features = 20
    n_classes = 3

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)

    # Split train/test
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Create model
    config = TernaryConfig(
        use_ternary_activations=False,
        threshold=0.3,
        learning_rate=0.1,
        batch_size=32,
        epochs=10
    )

    model = TernaryNeuralNetwork([n_features, 50, n_classes], config)

    print(f"\nModel architecture: {model.layer_sizes}")
    print(f"Configuration: {config}")

    # Training
    print("\nTraining...")
    for epoch in range(config.epochs):
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        # Mini-batch training
        epoch_loss = 0
        epoch_acc = 0
        n_batches = 0

        for i in range(0, len(X_train), config.batch_size):
            x_batch = X_train_shuffled[i:i + config.batch_size]
            y_batch = y_train_shuffled[i:i + config.batch_size]

            loss, acc = train_step(model, x_batch, y_batch)
            epoch_loss += loss
            epoch_acc += acc
            n_batches += 1

        epoch_loss /= n_batches
        epoch_acc /= n_batches

        # Evaluate
        test_loss, test_acc = evaluate(model, X_test, y_test)

        print(f"Epoch {epoch + 1}/{config.epochs}: "
              f"Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, "
              f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

    # Print sparsity statistics
    print("\nSparsity Statistics:")
    sparsity = model.get_sparsity()
    for layer_name, stats in sparsity.items():
        if layer_name != 'overall':
            print(f"  {layer_name}: {stats['nonzero']}/{stats['total']} "
                  f"({stats['sparsity']*100:.1f}% sparse)")
    print(f"  Overall: {sparsity['overall']['sparsity']*100:.1f}% sparse")

    # Show some ternary weights
    print("\nSample ternary weights from first layer:")
    weights = model.layers[0].get_ternary_weights()
    print(weights[:5, :10])
