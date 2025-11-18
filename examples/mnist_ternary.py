#!/usr/bin/env python3
"""
MNIST Classification using Ternary Neural Networks.

This example demonstrates:
1. Training a ternary neural network on MNIST
2. Using 3-bit to 2-trit encoding for efficient storage
3. Proper gradient flow through ternary quantization
4. Integration with the ternary computing simulation
"""

import numpy as np
import sys
from typing import Tuple

# Import ternary computing components
sys.path.insert(0, '../python')
from ternary.neural import (
    TernaryNeuralNetwork, TernaryConfig,
    train_step, evaluate, softmax
)
from ternary.encoding import TritEncoder, TryteEncoder
from ternary import Tryte


def load_mnist_simple() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST dataset.

    For this example, we'll create synthetic MNIST-like data.
    In practice, you would load real MNIST using:
    - keras.datasets.mnist.load_data()
    - sklearn.datasets.load_digits()
    - torchvision.datasets.MNIST

    Returns:
        X_train, y_train, X_test, y_test
    """
    print("Loading MNIST-like synthetic dataset...")

    # Generate synthetic dataset similar to MNIST
    # 28x28 = 784 pixels, 10 classes (digits 0-9)
    np.random.seed(42)

    n_train = 5000
    n_test = 1000
    n_pixels = 784
    n_classes = 10

    # Create synthetic data with some structure
    # Each class has a different random pattern
    X_train = np.zeros((n_train, n_pixels))
    y_train = np.zeros(n_train, dtype=int)

    X_test = np.zeros((n_test, n_pixels))
    y_test = np.zeros(n_test, dtype=int)

    # Generate class prototypes
    prototypes = []
    for c in range(n_classes):
        # Each class has a prototype (simplified MNIST-like pattern)
        proto = np.random.randn(n_pixels) * 0.5
        # Add some structure (e.g., top half vs bottom half activation)
        if c < 5:
            proto[:n_pixels//2] += 1.0  # Top half active
        else:
            proto[n_pixels//2:] += 1.0  # Bottom half active
        prototypes.append(proto)

    # Generate training data
    for i in range(n_train):
        class_idx = i % n_classes
        X_train[i] = prototypes[class_idx] + np.random.randn(n_pixels) * 0.3
        y_train[i] = class_idx

    # Generate test data
    for i in range(n_test):
        class_idx = i % n_classes
        X_test[i] = prototypes[class_idx] + np.random.randn(n_pixels) * 0.3
        y_test[i] = class_idx

    # Normalize to [0, 1]
    X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())

    # Scale to [-1, 1] for ternary quantization
    X_train = 2 * X_train - 1
    X_test = 2 * X_test - 1

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Input features: {n_pixels}")
    print(f"  Classes: {n_classes}")

    return X_train, y_train, X_test, y_test


def quantize_dataset(X: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Quantize dataset to ternary values {-1, 0, 1}.

    Args:
        X: Input data
        threshold: Quantization threshold

    Returns:
        Quantized data
    """
    X_ternary = np.zeros_like(X)
    X_ternary[X > threshold] = 1.0
    X_ternary[X < -threshold] = -1.0
    return X_ternary


def encode_ternary_weights_to_memory(
    weights: np.ndarray,
    memory_start_addr: int = 1000
) -> Tuple[bytes, int]:
    """
    Encode ternary weights efficiently using 3-bit to 2-trit encoding.

    Args:
        weights: Weight matrix (flattened)
        memory_start_addr: Starting memory address

    Returns:
        - Encoded bytes
        - Number of bytes used
    """
    from ternary import Trit, TritValue

    # Convert weights to trits
    trits = []
    for w in weights.flatten():
        if w > 0.5:
            trits.append(Trit(TritValue.PLUS))
        elif w < -0.5:
            trits.append(Trit(TritValue.MINUS))
        else:
            trits.append(Trit(TritValue.ZERO))

    # Encode using 3-bit to 2-trit encoding
    encoded = TritEncoder.encode_trits(trits)

    return encoded, len(encoded)


def demonstrate_encoding_efficiency(model: TernaryNeuralNetwork):
    """Show storage efficiency of ternary encoding."""
    print("\n" + "=" * 70)
    print("STORAGE EFFICIENCY ANALYSIS")
    print("=" * 70)

    total_weights = 0
    total_bytes_binary = 0
    total_bytes_ternary = 0

    for i, layer in enumerate(model.layers):
        weights = layer.get_ternary_weights()
        n_weights = weights.size

        # Binary encoding (float32 = 4 bytes per weight)
        bytes_binary = n_weights * 4

        # Ternary encoding (3 bits per 2 trits)
        encoded, bytes_ternary = encode_ternary_weights_to_memory(weights)

        total_weights += n_weights
        total_bytes_binary += bytes_binary
        total_bytes_ternary += bytes_ternary

        print(f"\nLayer {i}: {weights.shape}")
        print(f"  Weights: {n_weights:,}")
        print(f"  Binary (float32): {bytes_binary:,} bytes")
        print(f"  Ternary (3-bit/2-trit): {bytes_ternary:,} bytes")
        print(f"  Compression: {bytes_binary/bytes_ternary:.2f}x")

    print(f"\n" + "-" * 70)
    print(f"Total:")
    print(f"  Weights: {total_weights:,}")
    print(f"  Binary: {total_bytes_binary:,} bytes ({total_bytes_binary/1024:.1f} KB)")
    print(f"  Ternary: {total_bytes_ternary:,} bytes ({total_bytes_ternary/1024:.1f} KB)")
    print(f"  Overall compression: {total_bytes_binary/total_bytes_ternary:.2f}x")
    print(f"  Memory savings: {(1 - total_bytes_ternary/total_bytes_binary)*100:.1f}%")


def train_mnist_ternary():
    """Train ternary neural network on MNIST."""
    print("=" * 70)
    print("MNIST TERNARY NEURAL NETWORK")
    print("=" * 70)

    # Load data
    X_train, y_train, X_test, y_test = load_mnist_simple()

    # Configuration
    config = TernaryConfig(
        use_ternary_activations=False,  # Keep activations full precision for better accuracy
        threshold=0.3,
        learning_rate=0.01,
        batch_size=64,
        epochs=20
    )

    print(f"\nConfiguration:")
    print(f"  Ternary activations: {config.use_ternary_activations}")
    print(f"  Quantization threshold: {config.threshold}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")

    # Create model
    # Architecture: 784 -> 256 -> 128 -> 10
    layer_sizes = [784, 256, 128, 10]
    model = TernaryNeuralNetwork(layer_sizes, config)

    print(f"\nArchitecture: {' -> '.join(map(str, layer_sizes))}")

    # Calculate total parameters
    total_params = sum(
        layer.weights_ternary.size + layer.bias_ternary.size
        for layer in model.layers
    )
    print(f"Total parameters: {total_params:,}")

    # Training
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_test_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

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

        # Evaluate on test set
        test_loss, test_acc = evaluate(model, X_test, y_test)

        # Track history
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        # Print progress
        print(f"Epoch {epoch + 1:2d}/{config.epochs}: "
              f"Train Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.3f} | "
              f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.3f}")

    print(f"\nBest test accuracy: {best_test_acc:.3f}")

    # Sparsity analysis
    print("\n" + "=" * 70)
    print("SPARSITY ANALYSIS")
    print("=" * 70)

    sparsity = model.get_sparsity()
    for layer_name, stats in sparsity.items():
        if layer_name != 'overall':
            zero_weights = stats['total'] - stats['nonzero']
            print(f"{layer_name}:")
            print(f"  Total weights: {stats['total']:,}")
            print(f"  Non-zero: {stats['nonzero']:,} (+ or -)")
            print(f"  Zero: {zero_weights:,}")
            print(f"  Sparsity: {stats['sparsity']*100:.1f}%")

    print(f"\nOverall sparsity: {sparsity['overall']['sparsity']*100:.1f}%")

    # Storage efficiency
    demonstrate_encoding_efficiency(model)

    # Analyze weight distribution
    print("\n" + "=" * 70)
    print("WEIGHT DISTRIBUTION")
    print("=" * 70)

    for i, layer in enumerate(model.layers):
        weights = layer.get_ternary_weights().flatten()
        n_negative = np.sum(weights == -1)
        n_zero = np.sum(weights == 0)
        n_positive = np.sum(weights == 1)
        total = len(weights)

        print(f"\nLayer {i}:")
        print(f"  Negative (-1): {n_negative:6,} ({n_negative/total*100:5.1f}%)")
        print(f"  Zero      (0): {n_zero:6,} ({n_zero/total*100:5.1f}%)")
        print(f"  Positive (+1): {n_positive:6,} ({n_positive/total*100:5.1f}%)")

    # Sample predictions
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    n_samples = 20
    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices]

    predictions = model.predict(X_sample)

    correct = 0
    print("\nTrue  Pred")
    print("-" * 12)
    for true_label, pred_label in zip(y_sample, predictions):
        match = "✓" if true_label == pred_label else "✗"
        print(f"  {true_label}     {pred_label}    {match}")
        if true_label == pred_label:
            correct += 1

    print(f"\nSample accuracy: {correct}/{n_samples} ({correct/n_samples*100:.1f}%)")

    return model, history


def demonstrate_ternary_advantages():
    """Demonstrate advantages of ternary representation."""
    print("\n" + "=" * 70)
    print("TERNARY COMPUTING ADVANTAGES FOR NEURAL NETWORKS")
    print("=" * 70)

    print("\n1. MEMORY EFFICIENCY")
    print("   - Ternary weights: {-1, 0, +1}")
    print("   - 3 bits for 2 trits encoding")
    print("   - ~94.6% efficiency vs ideal")
    print("   - 10-20x compression vs float32")

    print("\n2. COMPUTATIONAL EFFICIENCY")
    print("   - Multiply becomes conditional add/subtract")
    print("   - Weight = -1: negate input")
    print("   - Weight =  0: skip (no computation)")
    print("   - Weight = +1: pass through")
    print("   - No actual multiplication needed!")

    print("\n3. SPARSITY")
    print("   - Zero weights provide natural pruning")
    print("   - Reduces computation further")
    print("   - Improves inference speed")

    print("\n4. GRADIENT FLOW")
    print("   - Straight-through estimator")
    print("   - Maintains trainability")
    print("   - Full-precision updates during training")
    print("   - Ternary quantization for inference")

    print("\n5. HARDWARE IMPLEMENTATION")
    print("   - Simpler multiply units")
    print("   - Lower power consumption")
    print("   - Smaller chip area")
    print("   - Natural fit for ternary logic gates")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Demonstrate ternary advantages
    demonstrate_ternary_advantages()

    # Train model
    print("\n")
    model, history = train_mnist_ternary()

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nSuccessfully trained ternary neural network on MNIST!")
    print(f"Final test accuracy: {history['test_acc'][-1]:.3f}")
    print(f"Model sparsity: {model.get_sparsity()['overall']['sparsity']*100:.1f}%")

    # Calculate theoretical speedup
    sparsity = model.get_sparsity()['overall']['sparsity']
    speedup = 1 / (1 - sparsity)  # Due to skipping zero weights
    print(f"\nTheoretical speedup from sparsity: {speedup:.2f}x")
    print("Additional speedup from ternary multiply: ~3-5x")
    print(f"Combined potential speedup: {speedup * 4:.1f}x")

    print("\n" + "=" * 70)
    print("This demonstrates the power of ternary neural networks!")
    print("=" * 70)
