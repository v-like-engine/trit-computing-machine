"""
Train Ternary ResNet on CIFAR-10.

This example demonstrates:
- Loading CIFAR-10 dataset
- Creating a TernaryResNet-18 model
- Training with data augmentation
- Validation and checkpointing
- Model compression analysis

Usage:
    python examples/train_cifar10_cnn.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
from ternary.cnn_models import create_ternary_resnet18
from ternary.cnn_data import CIFAR10Loader, DataAugmentation
from ternary.cnn_trainer import (
    TernaryCNNTrainer,
    TrainingConfig,
    print_model_summary,
    calculate_model_stats
)


def main():
    print("=" * 70)
    print("Training Ternary ResNet-18 on CIFAR-10")
    print("=" * 70)

    # Set random seed for reproducibility
    np.random.seed(42)

    # ======================================================================
    # 1. Load CIFAR-10 Dataset
    # ======================================================================
    print("\n[1/5] Loading CIFAR-10 dataset...")

    # Configure data augmentation
    augmentation = DataAugmentation(
        random_crop=True,
        random_flip=True,
        random_rotation=0.0,
        color_jitter=False,
        normalize=True
    )

    # Create data loader
    data_loader = CIFAR10Loader(
        data_dir='./data/cifar10',
        augmentation=augmentation
    )

    # Try to load real CIFAR-10, fallback to synthetic data
    try:
        data_loader.load()
        print("✓ Loaded real CIFAR-10 dataset")
    except FileNotFoundError:
        print("⚠ CIFAR-10 not found, creating synthetic data...")
        print("  (Download from: https://www.cs.toronto.edu/~kriz/cifar.html)")

        # Create synthetic CIFAR-10 for demonstration
        num_train = 5000
        num_test = 1000

        data_loader.X_train = np.random.randn(num_train, 3, 32, 32).astype(np.float32) * 0.2 + 0.5
        data_loader.X_train = data_loader.X_train.clip(0, 1)
        data_loader.y_train = np.random.randint(0, 10, num_train)

        data_loader.X_test = np.random.randn(num_test, 3, 32, 32).astype(np.float32) * 0.2 + 0.5
        data_loader.X_test = data_loader.X_test.clip(0, 1)
        data_loader.y_test = np.random.randint(0, 10, num_test)

        print(f"✓ Created synthetic dataset: {num_train} train, {num_test} test")

    print(f"Training samples: {len(data_loader.X_train)}")
    print(f"Test samples: {len(data_loader.X_test)}")
    print(f"Image shape: {data_loader.input_shape}")
    print(f"Number of classes: {data_loader.num_classes}")

    # ======================================================================
    # 2. Create TernaryResNet-18 Model
    # ======================================================================
    print("\n[2/5] Creating TernaryResNet-18 model...")

    model = create_ternary_resnet18(
        num_classes=10,
        threshold=0.3,
        learning_rate=0.1
    )

    print_model_summary(model)

    # ======================================================================
    # 3. Configure Training
    # ======================================================================
    print("\n[3/5] Configuring training...")

    config = TrainingConfig(
        epochs=100,
        batch_size=128,
        learning_rate=0.1,
        weight_decay=1e-4,
        momentum=0.9,

        # Learning rate schedule (reduce at epochs 30, 60, 90)
        lr_schedule='step',
        lr_decay_epochs=[30, 60, 90],
        lr_decay_factor=0.1,

        # Early stopping
        early_stopping=True,
        patience=15,
        min_delta=1e-4,

        # Checkpointing
        save_dir='./checkpoints/cifar10_resnet18',
        save_best=True,
        save_every=10,

        # Validation
        val_every=1,

        # Display
        print_every=10
    )

    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Initial learning rate: {config.learning_rate}")
    print(f"LR schedule: {config.lr_schedule} at epochs {config.lr_decay_epochs}")
    print(f"Early stopping patience: {config.patience}")

    # ======================================================================
    # 4. Train Model
    # ======================================================================
    print("\n[4/5] Training model...")

    trainer = TernaryCNNTrainer(
        model=model,
        train_loader=data_loader,
        val_loader=data_loader,  # Use test set for validation
        config=config
    )

    metrics = trainer.train()

    # ======================================================================
    # 5. Analyze Results
    # ======================================================================
    print("\n[5/5] Analyzing results...")

    # Model statistics
    stats = calculate_model_stats(model)

    print("\nFinal Model Statistics:")
    print("-" * 70)
    print(f"Total parameters: {stats['total_params']:,}")
    print(f"Ternary parameters: {stats['ternary_params']:,}")
    print(f"Float32 memory: {stats['float32_size_mb']:.2f} MB")
    print(f"Ternary memory: {stats['ternary_size_mb']:.2f} MB")
    print(f"Compression ratio: {stats['compression_ratio']:.1f}x")
    print(f"Sparsity: {stats.get('sparsity', 0) * 100:.1f}%")

    # Training metrics
    print("\nTraining Metrics:")
    print("-" * 70)
    print(f"Best validation accuracy: {metrics.get_best_val_acc() * 100:.2f}% (epoch {metrics.get_best_val_epoch() + 1})")
    print(f"Final training loss: {metrics.train_loss[-1]:.4f}")
    print(f"Final training accuracy: {metrics.train_acc[-1] * 100:.2f}%")

    if metrics.epoch_times:
        total_time = sum(metrics.epoch_times)
        avg_time = np.mean(metrics.epoch_times)
        print(f"\nTotal training time: {total_time / 60:.2f} minutes")
        print(f"Average epoch time: {avg_time:.2f} seconds")

    # Compare to full-precision baseline
    print("\nComparison to Full-Precision ResNet-18:")
    print("-" * 70)
    full_precision_size = stats['total_params'] * 4 / (1024 * 1024)
    print(f"Full-precision size: {full_precision_size:.2f} MB")
    print(f"Ternary size: {stats['ternary_size_mb']:.2f} MB")
    print(f"Memory savings: {(1 - stats['ternary_size_mb'] / full_precision_size) * 100:.1f}%")

    # Expected accuracy (approximate)
    print("\nExpected CIFAR-10 Accuracy:")
    print("-" * 70)
    print("Full-precision ResNet-18: ~95%")
    print("Ternary ResNet-18: ~90-93% (typical)")
    print(f"Your model: {metrics.get_best_val_acc() * 100:.2f}%")

    print("\n" + "=" * 70)
    print("Training complete! Model saved to:", config.save_dir)
    print("=" * 70)


if __name__ == '__main__':
    main()
