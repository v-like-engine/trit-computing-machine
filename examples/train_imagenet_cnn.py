"""
Train Ternary ResNet on ImageNet.

This example demonstrates:
- Loading ImageNet dataset (or synthetic subset)
- Creating a TernaryResNet-34 model for ImageNet
- Training with data augmentation
- Top-1 and Top-5 accuracy evaluation
- Model compression analysis

Usage:
    python examples/train_imagenet_cnn.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
from ternary.cnn_models import create_ternary_resnet34
from ternary.cnn_data import ImageNetLoader, DataAugmentation
from ternary.cnn_trainer import (
    TernaryCNNTrainer,
    TrainingConfig,
    print_model_summary,
    calculate_model_stats
)


def main():
    print("=" * 70)
    print("Training Ternary ResNet-34 on ImageNet")
    print("=" * 70)

    # Set random seed for reproducibility
    np.random.seed(42)

    # ======================================================================
    # 1. Load ImageNet Dataset
    # ======================================================================
    print("\n[1/5] Loading ImageNet dataset...")

    # Configure data augmentation
    augmentation = DataAugmentation(
        random_crop=True,
        random_flip=True,
        random_rotation=0.0,
        color_jitter=False,
        normalize=True
    )

    # Create data loader
    data_loader = ImageNetLoader(
        data_dir='./data/imagenet',
        input_size=224,
        augmentation=augmentation
    )

    # For this example, use a synthetic subset
    # (Real ImageNet is ~140GB and requires special download)
    print("Creating synthetic ImageNet subset for demonstration...")
    print("(For real ImageNet, download from: https://image-net.org/)")

    num_samples = 10000  # 10K samples for quick training
    data_loader.load_subset(num_samples=num_samples)

    print(f"Training samples: {len(data_loader.X_train)}")
    print(f"Validation samples: {len(data_loader.X_val)}")
    print(f"Image shape: {data_loader.input_shape}")
    print(f"Number of classes: {data_loader.num_classes}")

    # ======================================================================
    # 2. Create TernaryResNet-34 Model
    # ======================================================================
    print("\n[2/5] Creating TernaryResNet-34 model...")

    model = create_ternary_resnet34(
        num_classes=1000,
        threshold=0.3,
        learning_rate=0.1
    )

    print_model_summary(model)

    # ======================================================================
    # 3. Configure Training
    # ======================================================================
    print("\n[3/5] Configuring training...")

    config = TrainingConfig(
        epochs=90,
        batch_size=256,
        learning_rate=0.1,
        weight_decay=1e-4,
        momentum=0.9,

        # Learning rate schedule (reduce at epochs 30, 60)
        lr_schedule='step',
        lr_decay_epochs=[30, 60],
        lr_decay_factor=0.1,

        # Early stopping
        early_stopping=True,
        patience=10,
        min_delta=1e-4,

        # Checkpointing
        save_dir='./checkpoints/imagenet_resnet34',
        save_best=True,
        save_every=5,

        # Validation
        val_every=1,

        # Display
        print_every=5
    )

    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Initial learning rate: {config.learning_rate}")
    print(f"LR schedule: {config.lr_schedule} at epochs {config.lr_decay_epochs}")

    # ======================================================================
    # 4. Train Model
    # ======================================================================
    print("\n[4/5] Training model...")

    trainer = TernaryCNNTrainer(
        model=model,
        train_loader=data_loader,
        val_loader=data_loader,
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
    print(f"Best validation accuracy (Top-1): {metrics.get_best_val_acc() * 100:.2f}% (epoch {metrics.get_best_val_epoch() + 1})")

    if metrics.val_top5_acc:
        best_epoch = metrics.get_best_val_epoch()
        best_top5 = metrics.val_top5_acc[best_epoch]
        print(f"Best validation accuracy (Top-5): {best_top5 * 100:.2f}%")

    print(f"Final training loss: {metrics.train_loss[-1]:.4f}")
    print(f"Final training accuracy: {metrics.train_acc[-1] * 100:.2f}%")

    if metrics.epoch_times:
        total_time = sum(metrics.epoch_times)
        avg_time = np.mean(metrics.epoch_times)
        print(f"\nTotal training time: {total_time / 3600:.2f} hours")
        print(f"Average epoch time: {avg_time / 60:.2f} minutes")

    # Compare to full-precision baseline
    print("\nComparison to Full-Precision ResNet-34:")
    print("-" * 70)
    full_precision_size = stats['total_params'] * 4 / (1024 * 1024)
    print(f"Full-precision size: {full_precision_size:.2f} MB")
    print(f"Ternary size: {stats['ternary_size_mb']:.2f} MB")
    print(f"Memory savings: {(1 - stats['ternary_size_mb'] / full_precision_size) * 100:.1f}%")

    # Expected accuracy on real ImageNet
    print("\nExpected ImageNet Accuracy (Real Dataset):")
    print("-" * 70)
    print("Full-precision ResNet-34:")
    print("  Top-1: ~73.3%")
    print("  Top-5: ~91.4%")
    print("\nTernary ResNet-34 (typical):")
    print("  Top-1: ~68-71%")
    print("  Top-5: ~88-90%")
    print(f"\nYour model (synthetic data):")
    print(f"  Top-1: {metrics.get_best_val_acc() * 100:.2f}%")
    if metrics.val_top5_acc:
        print(f"  Top-5: {best_top5 * 100:.2f}%")

    print("\n" + "=" * 70)
    print("Training complete! Model saved to:", config.save_dir)
    print("=" * 70)

    # Tips for real ImageNet training
    print("\nTips for Training on Real ImageNet:")
    print("-" * 70)
    print("1. Download ImageNet from: https://image-net.org/")
    print("2. Preprocess images to 224x224 (or use torchvision)")
    print("3. Use distributed training for faster training (multi-GPU)")
    print("4. Typical training time: 3-5 days on 8 GPUs")
    print("5. Batch size: 256-512 (depending on GPU memory)")
    print("6. Learning rate: Start at 0.1, decay by 10x at epochs 30, 60")
    print("7. Data augmentation is critical for good accuracy")
    print("=" * 70)


if __name__ == '__main__':
    main()
