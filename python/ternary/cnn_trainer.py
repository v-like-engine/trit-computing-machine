"""
Training pipeline for ternary CNNs.

Provides trainer class with:
- Training loop with progress tracking
- Validation during training
- Top-1 and Top-5 accuracy metrics
- Model checkpointing
- Learning rate scheduling
- Training visualization
"""

import numpy as np
import time
import os
import pickle
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    momentum: float = 0.9

    # Learning rate schedule
    lr_schedule: str = 'step'  # 'step', 'cosine', 'constant'
    lr_decay_epochs: List[int] = None  # For step schedule
    lr_decay_factor: float = 0.1

    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4

    # Checkpointing
    save_dir: str = './checkpoints'
    save_best: bool = True
    save_every: int = 10  # Save every N epochs

    # Validation
    val_every: int = 1  # Validate every N epochs

    # Display
    print_every: int = 10  # Print every N batches

    def __post_init__(self):
        if self.lr_decay_epochs is None:
            self.lr_decay_epochs = [30, 60, 90]


class LearningRateScheduler:
    """Learning rate scheduling."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.base_lr = config.learning_rate

    def get_lr(self, epoch: int) -> float:
        """Get learning rate for current epoch."""
        if self.config.lr_schedule == 'constant':
            return self.base_lr

        elif self.config.lr_schedule == 'step':
            lr = self.base_lr
            for decay_epoch in self.config.lr_decay_epochs:
                if epoch >= decay_epoch:
                    lr *= self.config.lr_decay_factor
            return lr

        elif self.config.lr_schedule == 'cosine':
            # Cosine annealing
            return self.base_lr * (1 + np.cos(np.pi * epoch / self.config.epochs)) / 2

        else:
            raise ValueError(f"Unknown schedule: {self.config.lr_schedule}")


class Metrics:
    """Training metrics tracker."""

    def __init__(self):
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.val_top5_acc = []
        self.learning_rates = []
        self.epoch_times = []

    def update_train(self, loss: float, acc: float):
        """Update training metrics."""
        self.train_loss.append(loss)
        self.train_acc.append(acc)

    def update_val(self, loss: float, acc: float, top5_acc: Optional[float] = None):
        """Update validation metrics."""
        self.val_loss.append(loss)
        self.val_acc.append(acc)
        if top5_acc is not None:
            self.val_top5_acc.append(top5_acc)

    def update_lr(self, lr: float):
        """Update learning rate."""
        self.learning_rates.append(lr)

    def update_time(self, epoch_time: float):
        """Update epoch time."""
        self.epoch_times.append(epoch_time)

    def get_best_val_acc(self) -> float:
        """Get best validation accuracy."""
        if not self.val_acc:
            return 0.0
        return max(self.val_acc)

    def get_best_val_epoch(self) -> int:
        """Get epoch with best validation accuracy."""
        if not self.val_acc:
            return 0
        return int(np.argmax(self.val_acc))

    def summary(self) -> str:
        """Get metrics summary."""
        if not self.val_acc:
            return "No validation metrics yet"

        best_epoch = self.get_best_val_epoch()
        best_acc = self.val_acc[best_epoch]

        lines = [
            f"Best validation accuracy: {best_acc * 100:.2f}% (epoch {best_epoch + 1})",
            f"Final training loss: {self.train_loss[-1]:.4f}",
            f"Final training accuracy: {self.train_acc[-1] * 100:.2f}%",
        ]

        if self.val_top5_acc:
            best_top5 = self.val_top5_acc[best_epoch]
            lines.append(f"Best Top-5 accuracy: {best_top5 * 100:.2f}%")

        if self.epoch_times:
            avg_time = np.mean(self.epoch_times)
            lines.append(f"Average epoch time: {avg_time:.2f}s")

        return "\n".join(lines)


class TernaryCNNTrainer:
    """
    Trainer for ternary CNNs.

    Handles training loop, validation, checkpointing, and metrics.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        config: Optional[TrainingConfig] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()

        self.lr_scheduler = LearningRateScheduler(self.config)
        self.metrics = Metrics()

        # Early stopping
        self.best_val_acc = 0.0
        self.patience_counter = 0

        # Create checkpoint directory
        os.makedirs(self.config.save_dir, exist_ok=True)

    def train(self):
        """Run training loop."""
        print("=" * 70)
        print("Starting training")
        print("=" * 70)
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Training samples: {len(self.train_loader.X_train) if hasattr(self.train_loader, 'X_train') else 'N/A'}")
        print(f"Validation samples: {len(self.val_loader.X_val) if self.val_loader and hasattr(self.val_loader, 'X_val') else 'N/A'}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Initial learning rate: {self.config.learning_rate}")
        print("=" * 70)

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # Update learning rate
            current_lr = self.lr_scheduler.get_lr(epoch)
            self.model.learning_rate = current_lr
            self.metrics.update_lr(current_lr)

            # Training
            train_loss, train_acc = self._train_epoch(epoch)
            self.metrics.update_train(train_loss, train_acc)

            # Validation
            val_metrics_str = ""
            if self.val_loader and (epoch + 1) % self.config.val_every == 0:
                val_loss, val_acc, val_top5 = self._validate()
                self.metrics.update_val(val_loss, val_acc, val_top5)

                val_metrics_str = f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%"
                if val_top5 is not None:
                    val_metrics_str += f", Val Top-5: {val_top5 * 100:.2f}%"

                # Check for improvement
                if val_acc > self.best_val_acc + self.config.min_delta:
                    self.best_val_acc = val_acc
                    self.patience_counter = 0

                    # Save best model
                    if self.config.save_best:
                        self._save_checkpoint('best_model.pkl', epoch, val_acc)
                else:
                    self.patience_counter += 1

            # Epoch time
            epoch_time = time.time() - epoch_start
            self.metrics.update_time(epoch_time)

            # Print progress
            print(f"Epoch [{epoch + 1}/{self.config.epochs}] "
                  f"LR: {current_lr:.6f}, "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc * 100:.2f}%{val_metrics_str}, "
                  f"Time: {epoch_time:.2f}s")

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pkl', epoch, val_acc if self.val_loader else train_acc)

            # Early stopping
            if self.config.early_stopping and self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation accuracy: {self.best_val_acc * 100:.2f}%")
                break

        # Training complete
        print("\n" + "=" * 70)
        print("Training complete!")
        print("=" * 70)
        print(self.metrics.summary())
        print("=" * 70)

        return self.metrics

    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        batch_count = 0

        # Get batches
        batches = self.train_loader.get_batch(
            self.config.batch_size,
            split='train',
            shuffle=True,
            augment=True
        )

        for batch_idx, (x_batch, y_batch) in enumerate(batches):
            batch_count += 1

            # Forward pass
            logits = self.model.forward(x_batch, training=True)

            # Compute loss
            loss, grad_logits = self._compute_loss(logits, y_batch)

            # Backward pass
            gradients = self.model.backward(grad_logits)

            # Update weights
            self.model.update(gradients)

            # Compute accuracy
            predictions = np.argmax(logits, axis=1)
            correct = np.sum(predictions == y_batch)

            # Accumulate metrics
            total_loss += loss
            total_correct += correct
            total_samples += len(y_batch)

            # Print batch progress
            if (batch_idx + 1) % self.config.print_every == 0:
                batch_loss = total_loss / batch_count
                batch_acc = total_correct / total_samples
                print(f"  Batch [{batch_idx + 1}] Loss: {batch_loss:.4f}, Acc: {batch_acc * 100:.2f}%")

        avg_loss = total_loss / batch_count
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def _validate(self) -> Tuple[float, float, Optional[float]]:
        """Run validation."""
        total_loss = 0.0
        total_correct = 0
        total_top5_correct = 0
        total_samples = 0
        batch_count = 0

        # Get validation batches
        batches = self.val_loader.get_batch(
            self.config.batch_size,
            split='val' if hasattr(self.val_loader, 'X_val') else 'test',
            shuffle=False,
            augment=False
        )

        for x_batch, y_batch in batches:
            batch_count += 1

            # Forward pass (no training)
            logits = self.model.forward(x_batch, training=False)

            # Compute loss
            loss, _ = self._compute_loss(logits, y_batch)

            # Top-1 accuracy
            predictions = np.argmax(logits, axis=1)
            correct = np.sum(predictions == y_batch)

            # Top-5 accuracy (if applicable)
            top5_correct = 0
            if self.model.num_classes > 5:
                top5_preds = np.argsort(logits, axis=1)[:, -5:]
                for i, true_label in enumerate(y_batch):
                    if true_label in top5_preds[i]:
                        top5_correct += 1

            # Accumulate metrics
            total_loss += loss
            total_correct += correct
            total_top5_correct += top5_correct
            total_samples += len(y_batch)

        avg_loss = total_loss / batch_count
        avg_acc = total_correct / total_samples
        avg_top5_acc = total_top5_correct / total_samples if self.model.num_classes > 5 else None

        return avg_loss, avg_acc, avg_top5_acc

    def _compute_loss(self, logits: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute cross-entropy loss and gradients.

        Args:
            logits: (batch_size, num_classes)
            labels: (batch_size,)

        Returns:
            loss: Scalar loss
            grad_logits: Gradient w.r.t. logits
        """
        batch_size = logits.shape[0]

        # Softmax (numerically stable)
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Cross-entropy loss
        loss = 0.0
        for i in range(batch_size):
            loss -= np.log(probs[i, labels[i]] + 1e-8)
        loss /= batch_size

        # Gradient
        grad_logits = probs.copy()
        for i in range(batch_size):
            grad_logits[i, labels[i]] -= 1
        grad_logits /= batch_size

        return loss, grad_logits

    def _save_checkpoint(self, filename: str, epoch: int, accuracy: float):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.config.save_dir, filename)

        checkpoint = {
            'epoch': epoch,
            'accuracy': accuracy,
            'model_state': self._get_model_state(),
            'metrics': self.metrics,
            'config': self.config
        }

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"  Saved checkpoint: {checkpoint_path}")

    def _get_model_state(self) -> Dict:
        """Get model state for checkpointing."""
        state = {
            'architecture': self.model.architecture if hasattr(self.model, 'architecture') else None,
            'learning_rate': self.model.learning_rate,
            'threshold': self.model.threshold,
        }

        # Save layer weights
        if hasattr(self.model, 'layers'):
            state['layers'] = []
            for layer in self.model.layers:
                layer_state = {}

                if hasattr(layer, 'weights_fp'):
                    layer_state['weights_fp'] = layer.weights_fp
                    layer_state['bias_fp'] = layer.bias_fp

                if hasattr(layer, 'gamma'):
                    layer_state['gamma'] = layer.gamma
                    layer_state['beta'] = layer.beta
                    layer_state['running_mean'] = layer.running_mean
                    layer_state['running_var'] = layer.running_var

                state['layers'].append(layer_state)

        return state

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

        self._load_model_state(checkpoint['model_state'])
        self.metrics = checkpoint.get('metrics', Metrics())

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Epoch: {checkpoint['epoch']}, Accuracy: {checkpoint['accuracy'] * 100:.2f}%")

    def _load_model_state(self, state: Dict):
        """Load model state from checkpoint."""
        if 'learning_rate' in state:
            self.model.learning_rate = state['learning_rate']

        if 'threshold' in state:
            self.model.threshold = state['threshold']

        # Load layer weights
        if 'layers' in state and hasattr(self.model, 'layers'):
            for i, layer_state in enumerate(state['layers']):
                if i >= len(self.model.layers):
                    break

                layer = self.model.layers[i]

                if 'weights_fp' in layer_state and hasattr(layer, 'weights_fp'):
                    layer.weights_fp = layer_state['weights_fp']
                    layer.bias_fp = layer_state['bias_fp']
                    layer.quantize()

                if 'gamma' in layer_state and hasattr(layer, 'gamma'):
                    layer.gamma = layer_state['gamma']
                    layer.beta = layer_state['beta']
                    layer.running_mean = layer_state['running_mean']
                    layer.running_var = layer_state['running_var']


def calculate_model_stats(model) -> Dict:
    """
    Calculate comprehensive model statistics.

    Args:
        model: Ternary neural network model

    Returns:
        Dictionary with statistics
    """
    stats = {}

    # Total parameters
    total_params = 0
    ternary_params = 0

    if hasattr(model, 'layers'):
        for layer in model.layers:
            if hasattr(layer, 'weights_ternary'):
                # Convolutional or linear layer
                w_shape = layer.weights_ternary.shape
                params = np.prod(w_shape)

                if hasattr(layer, 'bias_ternary'):
                    params += layer.bias_ternary.size

                total_params += params
                ternary_params += params

            elif hasattr(layer, 'gamma'):
                # Batch norm layer
                total_params += layer.gamma.size * 2  # gamma + beta

    stats['total_params'] = int(total_params)
    stats['ternary_params'] = int(ternary_params)

    # Memory usage
    float32_size = total_params * 4  # 4 bytes per float32
    ternary_size = ternary_params * 0.27 / 8  # ~0.27 bits per ternary weight

    stats['float32_size_mb'] = float32_size / (1024 * 1024)
    stats['ternary_size_mb'] = ternary_size / (1024 * 1024)
    stats['compression_ratio'] = float32_size / ternary_size if ternary_size > 0 else 0

    # Sparsity
    if hasattr(model, 'get_sparsity'):
        stats['sparsity'] = model.get_sparsity()

    return stats


def print_model_summary(model):
    """Print model architecture summary."""
    print("\n" + "=" * 70)
    print("Model Summary")
    print("=" * 70)

    if hasattr(model, '__class__'):
        print(f"Model: {model.__class__.__name__}")

    if hasattr(model, 'architecture'):
        print(f"Architecture: {model.architecture}")

    # Layer-by-layer
    if hasattr(model, 'layers'):
        print("\nLayers:")
        print("-" * 70)

        for i, layer in enumerate(model.layers):
            layer_name = layer.__class__.__name__

            if hasattr(layer, 'weights_ternary'):
                w_shape = layer.weights_ternary.shape
                params = np.prod(w_shape)
                if hasattr(layer, 'bias_ternary'):
                    params += layer.bias_ternary.size

                print(f"  {i + 1}. {layer_name:30s} {str(w_shape):30s} {params:,} params")

            elif hasattr(layer, 'gamma'):
                params = layer.gamma.size * 2
                print(f"  {i + 1}. {layer_name:30s} {'':30s} {params:,} params")

            else:
                print(f"  {i + 1}. {layer_name:30s}")

    # Statistics
    stats = calculate_model_stats(model)

    print("\n" + "-" * 70)
    print(f"Total parameters: {stats['total_params']:,}")
    print(f"Ternary parameters: {stats['ternary_params']:,}")
    print(f"Float32 size: {stats['float32_size_mb']:.2f} MB")
    print(f"Ternary size: {stats['ternary_size_mb']:.2f} MB")
    print(f"Compression ratio: {stats['compression_ratio']:.1f}x")

    if 'sparsity' in stats:
        print(f"Sparsity: {stats['sparsity'] * 100:.1f}%")

    print("=" * 70 + "\n")
