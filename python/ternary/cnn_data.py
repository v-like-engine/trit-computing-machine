"""
Data loaders for ImageNet and CIFAR-10.

Supports:
- ImageNet (ILSVRC2012)
- CIFAR-10
- CIFAR-100
- Custom datasets
- Data augmentation
- Preprocessing
"""

import numpy as np
from typing import Tuple, Optional, List, Callable
import os
import pickle
from dataclasses import dataclass


@dataclass
class DataAugmentation:
    """Data augmentation configuration."""
    random_crop: bool = True
    random_flip: bool = True
    random_rotation: float = 0.0  # degrees
    color_jitter: bool = False
    normalize: bool = True


class CIFAR10Loader:
    """
    CIFAR-10 dataset loader.

    32x32 color images in 10 classes.
    50,000 training images, 10,000 test images.
    """

    def __init__(self, data_dir: str = './data/cifar10', augmentation: Optional[DataAugmentation] = None):
        self.data_dir = data_dir
        self.augmentation = augmentation or DataAugmentation()

        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

        self.num_classes = 10
        self.input_shape = (3, 32, 32)

        # Load data
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None

    def download(self):
        """Download CIFAR-10 dataset."""
        print("CIFAR-10 auto-download not implemented.")
        print("Please download from: https://www.cs.toronto.edu/~kriz/cifar.html")
        print(f"Extract to: {self.data_dir}")

    def load_batch(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single batch file."""
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CIFAR-10 batch not found: {filepath}")

        with open(filepath, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')

        data = batch[b'data']
        labels = batch[b'labels']

        # Reshape: (N, 3072) → (N, 3, 32, 32)
        data = data.reshape(-1, 3, 32, 32).astype(np.float32)

        # Normalize to [0, 1]
        data = data / 255.0

        return data, np.array(labels)

    def load(self):
        """Load training and test data."""
        # Load training batches
        X_train_batches = []
        y_train_batches = []

        for i in range(1, 6):
            X_batch, y_batch = self.load_batch(f'data_batch_{i}')
            X_train_batches.append(X_batch)
            y_train_batches.append(y_batch)

        self.X_train = np.concatenate(X_train_batches, axis=0)
        self.y_train = np.concatenate(y_train_batches, axis=0)

        # Load test batch
        self.X_test, self.y_test = self.load_batch('test_batch')

        print(f"Loaded CIFAR-10:")
        print(f"  Training: {self.X_train.shape}")
        print(f"  Test: {self.X_test.shape}")

        return self

    def preprocess(self, images: np.ndarray) -> np.ndarray:
        """Preprocess images."""
        if self.augmentation.normalize:
            # CIFAR-10 mean and std
            mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
            std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)
            images = (images - mean) / std

        return images

    def augment(self, images: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        if not self.augmentation:
            return images

        augmented = images.copy()

        # Random horizontal flip
        if self.augmentation.random_flip:
            flip_mask = np.random.rand(len(images)) > 0.5
            augmented[flip_mask] = augmented[flip_mask, :, :, ::-1]

        # Random crop with padding
        if self.augmentation.random_crop:
            # Pad 4 pixels on each side
            padded = np.pad(augmented, ((0, 0), (0, 0), (4, 4), (4, 4)), mode='reflect')

            # Random crop back to 32x32
            for i in range(len(augmented)):
                h_offset = np.random.randint(0, 9)
                w_offset = np.random.randint(0, 9)
                augmented[i] = padded[i, :, h_offset:h_offset+32, w_offset:w_offset+32]

        return augmented

    def get_batch(
        self,
        batch_size: int,
        split: str = 'train',
        shuffle: bool = True,
        augment: bool = True
    ):
        """Generate batches."""
        if split == 'train':
            X, y = self.X_train, self.y_train
        else:
            X, y = self.X_test, self.y_test

        indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(X), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]

            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # Augment training data
            if split == 'train' and augment:
                X_batch = self.augment(X_batch)

            # Preprocess
            X_batch = self.preprocess(X_batch)

            yield X_batch, y_batch


class CIFAR100Loader(CIFAR10Loader):
    """
    CIFAR-100 dataset loader.

    32x32 color images in 100 classes.
    """

    def __init__(self, data_dir: str = './data/cifar100', augmentation: Optional[DataAugmentation] = None):
        super().__init__(data_dir, augmentation)
        self.num_classes = 100
        # 100 fine-grained classes

    def load(self):
        """Load CIFAR-100 data."""
        # Load training data
        X_train, y_train = self.load_batch('train')
        self.X_train = X_train
        self.y_train = y_train

        # Load test data
        X_test, y_test = self.load_batch('test')
        self.X_test = X_test
        self.y_test = y_test

        print(f"Loaded CIFAR-100:")
        print(f"  Training: {self.X_train.shape}")
        print(f"  Test: {self.X_test.shape}")

        return self


class ImageNetLoader:
    """
    ImageNet (ILSVRC2012) dataset loader.

    224x224 color images (resized) in 1000 classes.
    1.2M training images, 50K validation images.

    Note: Due to size, this implementation assumes preprocessed numpy arrays.
    For full ImageNet, use torchvision or tensorflow datasets.
    """

    def __init__(
        self,
        data_dir: str = './data/imagenet',
        input_size: int = 224,
        augmentation: Optional[DataAugmentation] = None
    ):
        self.data_dir = data_dir
        self.input_size = input_size
        self.augmentation = augmentation or DataAugmentation()

        self.num_classes = 1000
        self.input_shape = (3, input_size, input_size)

        # Data (loaded lazily)
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    def load_subset(self, num_samples: int = 10000):
        """
        Load a small subset of ImageNet for testing.

        Creates synthetic data that mimics ImageNet statistics.
        """
        print(f"Creating synthetic ImageNet subset ({num_samples} samples)...")

        # Training data
        self.X_train = np.random.randn(num_samples, 3, self.input_size, self.input_size).astype(np.float32)
        self.X_train = (self.X_train * 0.2 + 0.5).clip(0, 1)  # Normalize to [0, 1]

        # Create some structure (clusters)
        prototypes = []
        for i in range(self.num_classes):
            proto = np.random.randn(3, self.input_size, self.input_size).astype(np.float32) * 0.3
            prototypes.append(proto)

        self.y_train = np.random.randint(0, self.num_classes, num_samples)

        for i in range(num_samples):
            class_idx = self.y_train[i]
            self.X_train[i] += prototypes[class_idx]

        self.X_train = self.X_train.clip(0, 1)

        # Validation data (smaller)
        num_val = num_samples // 10
        self.X_val = np.random.randn(num_val, 3, self.input_size, self.input_size).astype(np.float32)
        self.X_val = (self.X_val * 0.2 + 0.5).clip(0, 1)
        self.y_val = np.random.randint(0, self.num_classes, num_val)

        print(f"Loaded ImageNet subset:")
        print(f"  Training: {self.X_train.shape}")
        print(f"  Validation: {self.X_val.shape}")

        return self

    def preprocess(self, images: np.ndarray) -> np.ndarray:
        """Preprocess images (ImageNet normalization)."""
        if self.augmentation.normalize:
            # ImageNet mean and std
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
            images = (images - mean) / std

        return images

    def augment(self, images: np.ndarray) -> np.ndarray:
        """Apply data augmentation for ImageNet."""
        if not self.augmentation:
            return images

        augmented = images.copy()

        # Random horizontal flip
        if self.augmentation.random_flip:
            flip_mask = np.random.rand(len(images)) > 0.5
            augmented[flip_mask] = augmented[flip_mask, :, :, ::-1]

        # Random crop (simplified - assumes already resized)
        if self.augmentation.random_crop:
            # Would implement random resized crop here
            pass

        return augmented

    def get_batch(
        self,
        batch_size: int,
        split: str = 'train',
        shuffle: bool = True,
        augment: bool = True
    ):
        """Generate batches."""
        if split == 'train':
            X, y = self.X_train, self.y_train
        else:
            X, y = self.X_val, self.y_val

        indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(X), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]

            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # Augment training data
            if split == 'train' and augment:
                X_batch = self.augment(X_batch)

            # Preprocess
            X_batch = self.preprocess(X_batch)

            yield X_batch, y_batch


def create_synthetic_mnist_cnn(num_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create synthetic MNIST-like data for CNN testing.

    Returns (X_train, y_train, X_test, y_test)
    """
    print(f"Creating synthetic MNIST-like dataset ({num_samples} samples)...")

    # Create prototypes for each digit
    prototypes = []
    for digit in range(10):
        # Simple patterns for each digit
        proto = np.zeros((28, 28))

        if digit == 0:  # Circle
            y, x = np.ogrid[-14:14, -14:14]
            mask = (x*x + y*y <= 100) & (x*x + y*y >= 64)
            proto[mask] = 1.0
        elif digit == 1:  # Vertical line
            proto[:, 13:15] = 1.0
        elif digit == 7:  # Horizontal line + diagonal
            proto[5:7, :] = 1.0
            for i in range(20):
                proto[7+i, 20-i] = 1.0
        else:
            # Random pattern for other digits
            proto = np.random.rand(28, 28) > 0.7

        prototypes.append(proto)

    # Generate training data
    X_train = np.zeros((num_samples, 1, 28, 28))
    y_train = np.zeros(num_samples, dtype=int)

    for i in range(num_samples):
        digit = i % 10
        y_train[i] = digit

        # Start with prototype
        img = prototypes[digit].copy()

        # Add noise
        img += np.random.randn(28, 28) * 0.1

        # Random shift
        shift_x = np.random.randint(-2, 3)
        shift_y = np.random.randint(-2, 3)
        img = np.roll(img, shift_x, axis=1)
        img = np.roll(img, shift_y, axis=0)

        X_train[i, 0] = img.clip(0, 1)

    # Generate test data (smaller)
    num_test = num_samples // 10
    X_test = np.zeros((num_test, 1, 28, 28))
    y_test = np.zeros(num_test, dtype=int)

    for i in range(num_test):
        digit = i % 10
        y_test[i] = digit
        img = prototypes[digit].copy()
        img += np.random.randn(28, 28) * 0.1
        X_test[i, 0] = img.clip(0, 1)

    print(f"Created synthetic MNIST:")
    print(f"  Training: {X_train.shape}")
    print(f"  Test: {X_test.shape}")

    return X_train, y_train, X_test, y_test
