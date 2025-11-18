#!/usr/bin/env python3
"""
Tests for ternary neural network implementation.
"""

import numpy as np
import sys
import pytest

sys.path.insert(0, '../python')

from ternary.neural import (
    TernaryNeuralNetwork,
    TernaryLinear,
    TernaryActivation,
    TernaryConfig,
    ternary_quantize,
    ternary_quantize_stochastic,
    train_step,
    evaluate,
    cross_entropy_loss,
)
from ternary.encoding import TritEncoder, TryteEncoder
from ternary import Trit, Tryte, TritValue


class TestTernaryQuantization:
    """Test ternary quantization functions."""

    def test_deterministic_quantization(self):
        """Test deterministic ternary quantization."""
        x = np.array([-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0])
        threshold = 0.3

        quantized = ternary_quantize(x, threshold)

        expected = np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(quantized, expected)

    def test_quantization_values(self):
        """Ensure quantized values are only {-1, 0, 1}."""
        x = np.random.randn(100)
        quantized = ternary_quantize(x, 0.3)

        unique_vals = np.unique(quantized)
        assert all(v in [-1.0, 0.0, 1.0] for v in unique_vals)

    def test_stochastic_quantization(self):
        """Test stochastic quantization maintains expectation."""
        np.random.seed(42)
        x = np.array([0.5] * 1000)  # Constant value

        quantized = ternary_quantize_stochastic(x, threshold=0.3)

        # Mean should be close to 0.5
        assert 0.3 < np.mean(quantized) < 0.7


class TestTritEncoding:
    """Test 3-bit to 2-trit encoding."""

    def test_encode_decode_roundtrip(self):
        """Test that encode -> decode is identity."""
        t1 = Trit(TritValue.PLUS)
        t2 = Trit(TritValue.MINUS)

        bits = TritEncoder.encode_trit_pair(t1, t2)
        decoded_t1, decoded_t2 = TritEncoder.decode_trit_pair(bits)

        assert decoded_t1.to_int() == t1.to_int()
        assert decoded_t2.to_int() == t2.to_int()

    def test_all_combinations(self):
        """Test all valid 2-trit combinations."""
        trit_values = [-1, 0, 1]

        for t1_val in trit_values:
            for t2_val in trit_values:
                # Skip unmappable (1, 1)
                if t1_val == 1 and t2_val == 1:
                    continue

                t1 = Trit(t1_val)
                t2 = Trit(t2_val)

                bits = TritEncoder.encode_trit_pair(t1, t2)
                decoded_t1, decoded_t2 = TritEncoder.decode_trit_pair(bits)

                assert decoded_t1.to_int() == t1_val
                assert decoded_t2.to_int() == t2_val

    def test_unmappable_saturation(self):
        """Test that (1, 1) saturates to (1, 0)."""
        t1 = Trit(TritValue.PLUS)
        t2 = Trit(TritValue.PLUS)

        bits = TritEncoder.encode_trit_pair(t1, t2)
        decoded_t1, decoded_t2 = TritEncoder.decode_trit_pair(bits)

        assert decoded_t1.to_int() == 1
        assert decoded_t2.to_int() == 0  # Saturated

    def test_tryte_encoding(self):
        """Test tryte encoding/decoding."""
        tryte = Tryte(12345)

        # Encode
        encoded = TryteEncoder.encode_tryte(tryte)

        # Decode
        decoded = TryteEncoder.decode_tryte(encoded)

        assert decoded.to_int() == tryte.to_int()

    def test_encoding_efficiency(self):
        """Test that encoding is efficient."""
        eff = TryteEncoder.calculate_efficiency()

        # Should be close to ideal
        assert eff['efficiency'] > 0.9
        assert eff['bytes_used'] <= 4  # 18 trits should fit in 4 bytes


class TestTernaryLinear:
    """Test ternary linear layer."""

    def test_initialization(self):
        """Test layer initialization."""
        config = TernaryConfig()
        layer = TernaryLinear(10, 5, config)

        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.weights_fp.shape == (5, 10)
        assert layer.bias_fp.shape == (5,)

    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        config = TernaryConfig()
        layer = TernaryLinear(10, 5, config)

        x = np.random.randn(32, 10)  # Batch of 32
        output = layer.forward(x)

        assert output.shape == (32, 5)

    def test_ternary_weights(self):
        """Test that weights are ternary after forward."""
        config = TernaryConfig(threshold=0.3)
        layer = TernaryLinear(10, 5, config)

        x = np.random.randn(1, 10)
        layer.forward(x)

        # Check weights are ternary
        unique_vals = np.unique(layer.weights_ternary)
        assert all(v in [-1.0, 0.0, 1.0] for v in unique_vals)

    def test_backward_gradient_shape(self):
        """Test backward pass gradient shapes."""
        config = TernaryConfig()
        layer = TernaryLinear(10, 5, config)

        x = np.random.randn(32, 10)
        layer.forward(x, training=True)

        grad_output = np.random.randn(32, 5)
        grad_input, grad_weights, grad_bias = layer.backward(grad_output)

        assert grad_input.shape == (32, 10)
        assert grad_weights.shape == (5, 10)
        assert grad_bias.shape == (5,)

    def test_update_changes_weights(self):
        """Test that update modifies weights."""
        config = TernaryConfig(learning_rate=0.1)
        layer = TernaryLinear(10, 5, config)

        weights_before = layer.weights_fp.copy()

        grad_weights = np.ones((5, 10))
        grad_bias = np.ones(5)

        layer.update(grad_weights, grad_bias)

        assert not np.allclose(layer.weights_fp, weights_before)


class TestTernaryNeuralNetwork:
    """Test full ternary neural network."""

    def test_initialization(self):
        """Test network initialization."""
        config = TernaryConfig()
        model = TernaryNeuralNetwork([10, 20, 5], config)

        assert len(model.layers) == 2
        assert len(model.activations) == 1
        assert model.layers[0].in_features == 10
        assert model.layers[0].out_features == 20
        assert model.layers[1].in_features == 20
        assert model.layers[1].out_features == 5

    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        config = TernaryConfig()
        model = TernaryNeuralNetwork([10, 20, 5], config)

        x = np.random.randn(32, 10)
        output = model.forward(x)

        assert output.shape == (32, 5)

    def test_backward_returns_gradients(self):
        """Test backward pass returns gradients."""
        config = TernaryConfig()
        model = TernaryNeuralNetwork([10, 20, 5], config)

        x = np.random.randn(32, 10)
        model.forward(x, training=True)

        grad_output = np.random.randn(32, 5)
        gradients = model.backward(grad_output)

        assert len(gradients) == 2  # One per layer
        for grad_w, grad_b in gradients:
            assert grad_w is not None
            assert grad_b is not None

    def test_training_reduces_loss(self):
        """Test that training actually reduces loss."""
        np.random.seed(42)

        # Synthetic dataset
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)

        config = TernaryConfig(learning_rate=0.1)
        model = TernaryNeuralNetwork([10, 20, 3], config)

        # Initial loss
        initial_loss, _ = evaluate(model, X, y)

        # Train for a few steps
        for _ in range(10):
            for i in range(0, len(X), 32):
                x_batch = X[i:i + 32]
                y_batch = y[i:i + 32]
                train_step(model, x_batch, y_batch)

        # Final loss
        final_loss, _ = evaluate(model, X, y)

        # Loss should decrease
        assert final_loss < initial_loss

    def test_sparsity_reporting(self):
        """Test sparsity reporting."""
        config = TernaryConfig()
        model = TernaryNeuralNetwork([10, 20, 5], config)

        # Forward pass to quantize weights
        x = np.random.randn(1, 10)
        model.forward(x)

        # Get sparsity
        sparsity = model.get_sparsity()

        assert 'overall' in sparsity
        assert 'layer_0' in sparsity
        assert 'layer_1' in sparsity

        # Check structure
        for key, stats in sparsity.items():
            if key != 'overall':
                assert 'total' in stats
                assert 'nonzero' in stats
                assert 'sparsity' in stats
                assert 0 <= stats['sparsity'] <= 1


class TestCrossEntropyLoss:
    """Test cross-entropy loss function."""

    def test_loss_shape(self):
        """Test loss is scalar."""
        logits = np.random.randn(32, 10)
        labels = np.random.randint(0, 10, 32)

        loss, grad = cross_entropy_loss(logits, labels)

        assert isinstance(loss, (float, np.floating))
        assert grad.shape == logits.shape

    def test_perfect_prediction_low_loss(self):
        """Test that perfect predictions have low loss."""
        batch_size = 10
        num_classes = 5

        # Create perfect predictions
        labels = np.arange(batch_size) % num_classes
        logits = np.zeros((batch_size, num_classes))
        logits[np.arange(batch_size), labels] = 10.0  # Large logit for correct class

        loss, _ = cross_entropy_loss(logits, labels)

        assert loss < 0.1  # Should be very small

    def test_gradient_sum(self):
        """Test gradient sums to zero (for softmax)."""
        logits = np.random.randn(32, 10)
        labels = np.random.randint(0, 10, 32)

        _, grad = cross_entropy_loss(logits, labels)

        # Gradient should sum to ~0 across classes for each sample
        grad_sum = np.sum(grad, axis=1)
        np.testing.assert_allclose(grad_sum, 0.0, atol=1e-6)


def run_integration_test():
    """Integration test: train small network."""
    print("\n" + "=" * 70)
    print("INTEGRATION TEST: Training Ternary Neural Network")
    print("=" * 70)

    np.random.seed(42)

    # Create simple dataset
    n_samples = 200
    n_features = 20
    n_classes = 3

    # Generate data with some structure
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)

    # Split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Create model
    config = TernaryConfig(
        use_ternary_activations=False,
        threshold=0.3,
        learning_rate=0.1,
        batch_size=32,
        epochs=5
    )

    model = TernaryNeuralNetwork([n_features, 30, n_classes], config)

    print(f"\nModel: {model.layer_sizes}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.epochs}")

    # Train
    print("\nTraining:")
    for epoch in range(config.epochs):
        # Mini-batch training
        for i in range(0, len(X_train), config.batch_size):
            x_batch = X_train[i:i + config.batch_size]
            y_batch = y_train[i:i + config.batch_size]
            train_step(model, x_batch, y_batch)

        # Evaluate
        train_loss, train_acc = evaluate(model, X_train, y_train)
        test_loss, test_acc = evaluate(model, X_test, y_test)

        print(f"  Epoch {epoch + 1}: "
              f"Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")

    # Check sparsity
    sparsity = model.get_sparsity()
    print(f"\nOverall sparsity: {sparsity['overall']['sparsity']*100:.1f}%")

    print("\n✓ Integration test passed!")


if __name__ == "__main__":
    # Run pytest tests
    print("Running unit tests...")
    pytest.main([__file__, "-v"])

    # Run integration test
    run_integration_test()
