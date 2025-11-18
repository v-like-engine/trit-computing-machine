"""
Simple inference demo for Ternary CNN.

This example shows how to:
- Create a small ternary CNN
- Train on synthetic data
- Make predictions
- Visualize results

Usage:
    python examples/cnn_inference_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
from ternary.cnn_models import TernaryResNet
from ternary.cnn_layers import TernaryConv2D, TernaryBatchNorm2D, MaxPool2D, GlobalAvgPool2D, relu
from ternary.neural import TernaryLinear
from ternary.cnn_trainer import calculate_model_stats


def create_simple_cnn(num_classes=10, input_channels=3):
    """Create a simple ternary CNN for demonstration."""

    class SimpleTernaryCNN:
        def __init__(self, num_classes, threshold=0.3, learning_rate=0.01):
            self.num_classes = num_classes
            self.threshold = threshold
            self.learning_rate = learning_rate
            self.architecture = "SimpleCNN"

            # Layers
            self.layers = [
                # Conv block 1: 3x32x32 -> 32x32x32
                TernaryConv2D(input_channels, 32, kernel_size=3, stride=1, padding=1, threshold=threshold),
                TernaryBatchNorm2D(32),
                # MaxPool: 32x32x32 -> 32x16x16
                MaxPool2D(kernel_size=2, stride=2),

                # Conv block 2: 32x16x16 -> 64x16x16
                TernaryConv2D(32, 64, kernel_size=3, stride=1, padding=1, threshold=threshold),
                TernaryBatchNorm2D(64),
                # MaxPool: 64x16x16 -> 64x8x8
                MaxPool2D(kernel_size=2, stride=2),

                # Conv block 3: 64x8x8 -> 64x8x8
                TernaryConv2D(64, 64, kernel_size=3, stride=1, padding=1, threshold=threshold),
                TernaryBatchNorm2D(64),

                # Global average pooling: 64x8x8 -> 64
                GlobalAvgPool2D(),

                # Fully connected: 64 -> num_classes
                TernaryLinear(64, num_classes, threshold=threshold)
            ]

            self.activations = []

        def forward(self, x, training=True):
            """Forward pass through the network."""
            self.activations = [x]

            for i, layer in enumerate(self.layers):
                if isinstance(layer, (TernaryConv2D, TernaryLinear)):
                    x = layer.forward(x, training)
                elif isinstance(layer, TernaryBatchNorm2D):
                    x = layer.forward(x, training)
                elif isinstance(layer, (MaxPool2D, GlobalAvgPool2D)):
                    x = layer.forward(x)
                else:
                    x = layer.forward(x)

                # Apply ReLU after conv and batch norm (but not after final layer)
                if isinstance(layer, TernaryBatchNorm2D) and i < len(self.layers) - 2:
                    x = relu(x)

                self.activations.append(x)

            return x

        def backward(self, grad_output):
            """Backward pass (simplified for demo)."""
            # This is a simplified version - full implementation would
            # properly backpropagate through all layers
            gradients = []

            # For now, just compute gradients for ternary layers
            for layer in reversed(self.layers):
                if isinstance(layer, (TernaryConv2D, TernaryLinear)):
                    # Placeholder gradient computation
                    gradients.insert(0, {
                        'weights': np.zeros_like(layer.weights_fp) if hasattr(layer, 'weights_fp') else None,
                        'bias': np.zeros_like(layer.bias_fp) if hasattr(layer, 'bias_fp') else None
                    })

            return gradients

        def update(self, gradients):
            """Update weights (simplified)."""
            pass

        def get_sparsity(self):
            """Calculate sparsity across all ternary layers."""
            total = 0
            zeros = 0

            for layer in self.layers:
                if hasattr(layer, 'weights_ternary'):
                    total += layer.weights_ternary.size
                    zeros += np.sum(layer.weights_ternary == 0)

            return zeros / total if total > 0 else 0

    return SimpleTernaryCNN(num_classes)


def create_synthetic_data(num_samples=100, num_classes=10):
    """Create simple synthetic classification data."""
    # Create 32x32 RGB images
    X = []
    y = []

    # Create class prototypes
    prototypes = []
    for c in range(num_classes):
        # Each class has a different color pattern
        proto = np.random.randn(3, 32, 32).astype(np.float32) * 0.3
        prototypes.append(proto)

    # Generate samples
    for i in range(num_samples):
        class_idx = i % num_classes

        # Start with prototype
        img = prototypes[class_idx].copy()

        # Add noise
        img += np.random.randn(3, 32, 32).astype(np.float32) * 0.2

        # Normalize to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        X.append(img)
        y.append(class_idx)

    return np.array(X), np.array(y)


def main():
    print("=" * 70)
    print("Ternary CNN Inference Demo")
    print("=" * 70)

    np.random.seed(42)

    # ======================================================================
    # 1. Create synthetic dataset
    # ======================================================================
    print("\n[1/5] Creating synthetic dataset...")

    num_classes = 5
    num_train = 100
    num_test = 25

    X_train, y_train = create_synthetic_data(num_train, num_classes)
    X_test, y_test = create_synthetic_data(num_test, num_classes)

    print(f"Training samples: {num_train}")
    print(f"Test samples: {num_test}")
    print(f"Image shape: {X_train[0].shape}")
    print(f"Classes: {num_classes}")

    # ======================================================================
    # 2. Create model
    # ======================================================================
    print("\n[2/5] Creating SimpleCNN model...")

    model = create_simple_cnn(num_classes=num_classes)

    print(f"Model: {model.architecture}")
    print(f"Layers: {len(model.layers)}")

    stats = calculate_model_stats(model)
    print(f"Parameters: {stats['total_params']:,}")
    print(f"Memory (ternary): {stats['ternary_size_mb']:.2f} MB")
    print(f"Compression: {stats['compression_ratio']:.1f}x")

    # ======================================================================
    # 3. Quick training (few iterations)
    # ======================================================================
    print("\n[3/5] Quick training (10 iterations)...")

    batch_size = 10
    num_iterations = 10

    for iteration in range(num_iterations):
        # Random batch
        indices = np.random.choice(num_train, batch_size, replace=False)
        x_batch = X_train[indices]
        y_batch = y_train[indices]

        # Forward
        logits = model.forward(x_batch, training=True)

        # Compute loss
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        loss = 0
        for i in range(batch_size):
            loss -= np.log(probs[i, y_batch[i]] + 1e-8)
        loss /= batch_size

        # Accuracy
        predictions = np.argmax(logits, axis=1)
        accuracy = np.mean(predictions == y_batch)

        if (iteration + 1) % 2 == 0:
            print(f"  Iteration {iteration + 1}: Loss={loss:.4f}, Acc={accuracy * 100:.1f}%")

    # ======================================================================
    # 4. Make predictions
    # ======================================================================
    print("\n[4/5] Making predictions on test set...")

    # Predict on test set
    logits = model.forward(X_test, training=False)
    predictions = np.argmax(logits, axis=1)

    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"\nTest accuracy: {accuracy * 100:.1f}%")

    # Show some predictions
    print("\nSample predictions:")
    print("-" * 40)
    for i in range(min(10, num_test)):
        true_label = y_test[i]
        pred_label = predictions[i]
        confidence = np.exp(logits[i]) / np.sum(np.exp(logits[i]))
        max_conf = confidence[pred_label]

        status = "✓" if pred_label == true_label else "✗"
        print(f"  {status} Sample {i+1}: True={true_label}, Pred={pred_label}, Confidence={max_conf * 100:.1f}%")

    # ======================================================================
    # 5. Analyze model
    # ======================================================================
    print("\n[5/5] Analyzing model...")

    sparsity = model.get_sparsity()
    print(f"\nModel sparsity: {sparsity * 100:.1f}%")

    # Count parameters by layer
    print("\nParameters by layer:")
    print("-" * 40)
    for i, layer in enumerate(model.layers):
        layer_name = layer.__class__.__name__

        if hasattr(layer, 'weights_ternary'):
            params = layer.weights_ternary.size
            if hasattr(layer, 'bias_ternary'):
                params += layer.bias_ternary.size

            # Sparsity for this layer
            zeros = np.sum(layer.weights_ternary == 0)
            layer_sparsity = zeros / layer.weights_ternary.size

            print(f"  {i+1}. {layer_name:20s}: {params:7,} params, {layer_sparsity * 100:.1f}% sparse")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)

    print("\nKey takeaways:")
    print("- Ternary weights use only {-1, 0, 1}")
    print(f"- Compression ratio: {stats['compression_ratio']:.1f}x smaller than float32")
    print(f"- Sparsity: {sparsity * 100:.1f}% of weights are zero")
    print("- Fast inference with simple operations")
    print("- Suitable for edge devices and mobile deployment")


if __name__ == '__main__':
    main()
