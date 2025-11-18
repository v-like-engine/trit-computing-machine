"""
Evaluate a trained Ternary CNN model.

This example demonstrates:
- Loading a trained model from checkpoint
- Evaluating on test set
- Computing detailed metrics
- Analyzing model compression and sparsity

Usage:
    python examples/evaluate_cnn.py --checkpoint checkpoints/cifar10_resnet18/best_model.pkl
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import pickle
from ternary.cnn_models import create_ternary_resnet18, create_ternary_vgg16
from ternary.cnn_data import CIFAR10Loader, ImageNetLoader, DataAugmentation
from ternary.cnn_trainer import calculate_model_stats, print_model_summary


def evaluate_model(model, data_loader, batch_size=128, split='test'):
    """
    Evaluate model on dataset.

    Args:
        model: Trained model
        data_loader: Data loader
        batch_size: Batch size for evaluation
        split: 'test' or 'val'

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating on {split} set...")

    total_correct = 0
    total_top5_correct = 0
    total_samples = 0
    total_loss = 0.0
    batch_count = 0

    # Per-class accuracy
    num_classes = model.num_classes
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    # Get batches
    batches = data_loader.get_batch(
        batch_size,
        split=split,
        shuffle=False,
        augment=False
    )

    for x_batch, y_batch in batches:
        batch_count += 1

        # Forward pass
        logits = model.forward(x_batch, training=False)

        # Softmax
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Loss
        for i in range(len(y_batch)):
            total_loss -= np.log(probs[i, y_batch[i]] + 1e-8)

        # Top-1 accuracy
        predictions = np.argmax(logits, axis=1)
        correct = predictions == y_batch
        total_correct += np.sum(correct)

        # Top-5 accuracy
        if num_classes > 5:
            top5_preds = np.argsort(logits, axis=1)[:, -5:]
            for i, true_label in enumerate(y_batch):
                if true_label in top5_preds[i]:
                    total_top5_correct += 1

        # Per-class accuracy
        for i, label in enumerate(y_batch):
            class_total[label] += 1
            if correct[i]:
                class_correct[label] += 1

        total_samples += len(y_batch)

        if batch_count % 10 == 0:
            print(f"  Processed {total_samples} samples...")

    # Calculate metrics
    metrics = {
        'accuracy': total_correct / total_samples,
        'top5_accuracy': total_top5_correct / total_samples if num_classes > 5 else None,
        'loss': total_loss / total_samples,
        'total_samples': total_samples,
        'per_class_accuracy': class_correct / (class_total + 1e-8)
    }

    return metrics


def print_evaluation_results(metrics, class_names=None):
    """Print evaluation results."""
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    print(f"Total samples: {metrics['total_samples']}")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Top-1 Accuracy: {metrics['accuracy'] * 100:.2f}%")

    if metrics['top5_accuracy'] is not None:
        print(f"Top-5 Accuracy: {metrics['top5_accuracy'] * 100:.2f}%")

    # Per-class accuracy
    if class_names is not None and len(class_names) == len(metrics['per_class_accuracy']):
        print("\nPer-Class Accuracy:")
        print("-" * 70)

        sorted_indices = np.argsort(metrics['per_class_accuracy'])

        print("Top 5 classes:")
        for idx in sorted_indices[-5:][::-1]:
            print(f"  {class_names[idx]:20s}: {metrics['per_class_accuracy'][idx] * 100:.2f}%")

        print("\nBottom 5 classes:")
        for idx in sorted_indices[:5]:
            print(f"  {class_names[idx]:20s}: {metrics['per_class_accuracy'][idx] * 100:.2f}%")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Ternary CNN')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/cifar10_resnet18/best_model.pkl',
                        help='Path to checkpoint file')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'],
                        help='Dataset to evaluate on')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Dataset directory (optional)')

    args = parser.parse_args()

    print("=" * 70)
    print("Ternary CNN Evaluation")
    print("=" * 70)

    # ======================================================================
    # 1. Load Checkpoint
    # ======================================================================
    print("\n[1/4] Loading checkpoint...")

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("\nPlease train a model first using:")
        print("  python examples/train_cifar10_cnn.py")
        return

    with open(args.checkpoint, 'rb') as f:
        checkpoint = pickle.load(f)

    print(f"✓ Loaded checkpoint: {args.checkpoint}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Accuracy: {checkpoint['accuracy'] * 100:.2f}%")

    # ======================================================================
    # 2. Load Dataset
    # ======================================================================
    print("\n[2/4] Loading dataset...")

    augmentation = DataAugmentation(normalize=True)

    if args.dataset == 'cifar10':
        from ternary.cnn_data import CIFAR10Loader

        data_dir = args.data_dir or './data/cifar10'
        data_loader = CIFAR10Loader(data_dir=data_dir, augmentation=augmentation)

        try:
            data_loader.load()
            print("✓ Loaded CIFAR-10 dataset")
        except FileNotFoundError:
            print("Creating synthetic CIFAR-10 for demonstration...")
            num_test = 1000
            data_loader.X_test = np.random.randn(num_test, 3, 32, 32).astype(np.float32) * 0.2 + 0.5
            data_loader.X_test = data_loader.X_test.clip(0, 1)
            data_loader.y_test = np.random.randint(0, 10, num_test)

        num_classes = 10
        class_names = data_loader.class_names

    elif args.dataset == 'cifar100':
        from ternary.cnn_data import CIFAR100Loader

        data_dir = args.data_dir or './data/cifar100'
        data_loader = CIFAR100Loader(data_dir=data_dir, augmentation=augmentation)

        try:
            data_loader.load()
            print("✓ Loaded CIFAR-100 dataset")
        except FileNotFoundError:
            print("Creating synthetic CIFAR-100...")
            num_test = 1000
            data_loader.X_test = np.random.randn(num_test, 3, 32, 32).astype(np.float32) * 0.2 + 0.5
            data_loader.X_test = data_loader.X_test.clip(0, 1)
            data_loader.y_test = np.random.randint(0, 100, num_test)

        num_classes = 100
        class_names = None

    else:  # ImageNet
        data_dir = args.data_dir or './data/imagenet'
        data_loader = ImageNetLoader(data_dir=data_dir, augmentation=augmentation)

        print("Creating synthetic ImageNet subset...")
        data_loader.load_subset(num_samples=1000)

        num_classes = 1000
        class_names = None

    print(f"Test samples: {len(data_loader.X_test) if hasattr(data_loader, 'X_test') else len(data_loader.X_val)}")

    # ======================================================================
    # 3. Create Model and Load Weights
    # ======================================================================
    print("\n[3/4] Creating model...")

    # Determine model architecture from checkpoint
    state = checkpoint['model_state']
    architecture = state.get('architecture', 'resnet18')

    if 'resnet18' in architecture.lower():
        model = create_ternary_resnet18(num_classes=num_classes)
    elif 'vgg16' in architecture.lower():
        model = create_ternary_vgg16(num_classes=num_classes)
    else:
        print(f"Warning: Unknown architecture {architecture}, using ResNet-18")
        model = create_ternary_resnet18(num_classes=num_classes)

    # Load weights
    if 'layers' in state:
        for i, layer_state in enumerate(state['layers']):
            if i >= len(model.layers):
                break

            layer = model.layers[i]

            if 'weights_fp' in layer_state and hasattr(layer, 'weights_fp'):
                layer.weights_fp = layer_state['weights_fp']
                layer.bias_fp = layer_state['bias_fp']
                layer.quantize()

            if 'gamma' in layer_state and hasattr(layer, 'gamma'):
                layer.gamma = layer_state['gamma']
                layer.beta = layer_state['beta']
                layer.running_mean = layer_state['running_mean']
                layer.running_var = layer_state['running_var']

    print(f"✓ Loaded model: {model.__class__.__name__}")
    print_model_summary(model)

    # ======================================================================
    # 4. Evaluate
    # ======================================================================
    print("\n[4/4] Evaluating model...")

    metrics = evaluate_model(
        model,
        data_loader,
        batch_size=args.batch_size,
        split='test' if hasattr(data_loader, 'X_test') else 'val'
    )

    print_evaluation_results(metrics, class_names)

    # Model statistics
    stats = calculate_model_stats(model)

    print("\nModel Statistics:")
    print("-" * 70)
    print(f"Total parameters: {stats['total_params']:,}")
    print(f"Ternary memory: {stats['ternary_size_mb']:.2f} MB")
    print(f"Compression ratio: {stats['compression_ratio']:.1f}x")
    print(f"Sparsity: {stats.get('sparsity', 0) * 100:.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
