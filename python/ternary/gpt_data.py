"""
Data loaders and tokenization for Ternary GPT.

Supports:
- Character-level tokenization
- Simple word tokenization
- Text dataset loading
- Shakespeare, WikiText, custom text
"""

import numpy as np
from typing import List, Tuple, Optional, Iterator
import os
import re
from collections import Counter


class CharTokenizer:
    """
    Character-level tokenizer.

    Simple tokenizer that maps each unique character to an integer ID.
    Good for small datasets and demo purposes.
    """

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0

    def fit(self, text: str):
        """Build vocabulary from text."""
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return [self.char_to_id.get(ch, 0) for ch in text]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return ''.join([self.id_to_char.get(i, '?') for i in ids])

    def save(self, path: str):
        """Save tokenizer."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'char_to_id': self.char_to_id,
                'id_to_char': self.id_to_char,
                'vocab_size': self.vocab_size
            }, f)

    def load(self, path: str):
        """Load tokenizer."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.char_to_id = data['char_to_id']
            self.id_to_char = data['id_to_char']
            self.vocab_size = data['vocab_size']


class SimpleWordTokenizer:
    """
    Simple word-level tokenizer.

    Splits on whitespace and punctuation, with a fixed vocabulary.
    """

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}

        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.BOS_TOKEN = '<BOS>'
        self.EOS_TOKEN = '<EOS>'

    def fit(self, text: str):
        """Build vocabulary from text."""
        # Tokenize
        words = self._tokenize(text)

        # Count frequencies
        word_counts = Counter(words)

        # Take most common words
        vocab_words = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        vocab_words += [word for word, _ in word_counts.most_common(self.vocab_size - 4)]

        # Create mappings
        self.word_to_id = {word: i for i, word in enumerate(vocab_words)}
        self.id_to_word = {i: word for i, word in enumerate(vocab_words)}
        self.vocab_size = len(vocab_words)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple whitespace + punctuation splitting
        text = text.lower()
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        words = text.split()
        return words

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        words = self._tokenize(text)

        ids = []
        if add_special_tokens:
            ids.append(self.word_to_id[self.BOS_TOKEN])

        for word in words:
            ids.append(self.word_to_id.get(word, self.word_to_id[self.UNK_TOKEN]))

        if add_special_tokens:
            ids.append(self.word_to_id[self.EOS_TOKEN])

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        words = []
        for i in ids:
            word = self.id_to_word.get(i, self.UNK_TOKEN)

            if skip_special_tokens and word in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]:
                continue

            words.append(word)

        return ' '.join(words)


class TextDataset:
    """
    Text dataset for language modeling.

    Loads text and creates sequences for training.
    """

    def __init__(
        self,
        text: str,
        tokenizer,
        seq_length: int = 256,
        stride: int = None
    ):
        self.text = text
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride or seq_length  # Default: non-overlapping

        # Tokenize entire text
        self.token_ids = self.tokenizer.encode(text)

        # Create sequences
        self.sequences = self._create_sequences()

    def _create_sequences(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create (input, target) sequence pairs."""
        sequences = []

        for i in range(0, len(self.token_ids) - self.seq_length, self.stride):
            # Input: [i : i+seq_length]
            # Target: [i+1 : i+seq_length+1]
            input_seq = self.token_ids[i:i + self.seq_length]
            target_seq = self.token_ids[i + 1:i + self.seq_length + 1]

            if len(input_seq) == self.seq_length and len(target_seq) == self.seq_length:
                sequences.append((
                    np.array(input_seq),
                    np.array(target_seq)
                ))

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.sequences[idx]

    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate batches of sequences."""
        indices = np.arange(len(self.sequences))

        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]

            # Get sequences
            inputs = []
            targets = []

            for idx in batch_indices:
                inp, tgt = self.sequences[idx]
                inputs.append(inp)
                targets.append(tgt)

            yield np.array(inputs), np.array(targets)


class ShakespeareDataset:
    """
    Shakespeare text dataset.

    Loads tiny shakespeare dataset for character-level language modeling.
    """

    def __init__(self, data_dir: str = './data/shakespeare', seq_length: int = 256):
        self.data_dir = data_dir
        self.seq_length = seq_length

        # Try to load
        self.text_train = None
        self.text_val = None
        self.tokenizer = CharTokenizer()

    def download(self):
        """Instructions to download shakespeare dataset."""
        print("Shakespeare dataset not found.")
        print("Download from: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
        print(f"Save to: {self.data_dir}/input.txt")

    def load(self):
        """Load shakespeare text."""
        filepath = os.path.join(self.data_dir, 'input.txt')

        if not os.path.exists(filepath):
            # Create synthetic shakespeare-like text for demo
            print("Creating synthetic Shakespeare-like text for demo...")
            self.text_train = self._create_synthetic_text(50000)
            self.text_val = self._create_synthetic_text(5000)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            # Split train/val (90/10)
            split_idx = int(len(text) * 0.9)
            self.text_train = text[:split_idx]
            self.text_val = text[split_idx:]

        # Fit tokenizer on training text
        self.tokenizer.fit(self.text_train)

        print(f"Loaded Shakespeare dataset:")
        print(f"  Train: {len(self.text_train)} characters")
        print(f"  Val: {len(self.text_val)} characters")
        print(f"  Vocab size: {self.tokenizer.vocab_size}")

        return self

    def _create_synthetic_text(self, length: int) -> str:
        """Create synthetic shakespeare-like text."""
        words = [
            'thou', 'art', 'thee', 'thy', 'the', 'and', 'of', 'to', 'in', 'is',
            'that', 'not', 'with', 'his', 'for', 'be', 'this', 'what', 'all',
            'lord', 'king', 'love', 'good', 'shall', 'now', 'come', 'well',
            'speak', 'know', 'thee', 'like', 'my', 'your', 'upon', 'when'
        ]

        text = []
        for _ in range(length // 5):  # Approximate word length
            word = np.random.choice(words)
            text.append(word)

            # Add punctuation occasionally
            if np.random.rand() < 0.1:
                text.append(np.random.choice([',', '.', ';', '!', '?']))

        return ' '.join(text)[:length]

    def get_train_dataset(self) -> TextDataset:
        """Get training dataset."""
        return TextDataset(
            self.text_train,
            self.tokenizer,
            seq_length=self.seq_length
        )

    def get_val_dataset(self) -> TextDataset:
        """Get validation dataset."""
        return TextDataset(
            self.text_val,
            self.tokenizer,
            seq_length=self.seq_length
        )


def load_custom_text(filepath: str, seq_length: int = 256) -> Tuple[TextDataset, TextDataset, CharTokenizer]:
    """
    Load custom text file and create datasets.

    Args:
        filepath: Path to text file
        seq_length: Sequence length for training

    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        tokenizer: Fitted tokenizer
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split train/val
    split_idx = int(len(text) * 0.9)
    text_train = text[:split_idx]
    text_val = text[split_idx:]

    # Create tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(text_train)

    # Create datasets
    train_dataset = TextDataset(text_train, tokenizer, seq_length)
    val_dataset = TextDataset(text_val, tokenizer, seq_length)

    print(f"Loaded custom text:")
    print(f"  Train: {len(text_train)} characters")
    print(f"  Val: {len(text_val)} characters")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Train sequences: {len(train_dataset)}")
    print(f"  Val sequences: {len(val_dataset)}")

    return train_dataset, val_dataset, tokenizer


def create_synthetic_text_dataset(
    num_chars: int = 100000,
    seq_length: int = 256
) -> Tuple[TextDataset, TextDataset, CharTokenizer]:
    """
    Create synthetic text dataset for testing.

    Args:
        num_chars: Number of characters to generate
        seq_length: Sequence length

    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        tokenizer: Fitted tokenizer
    """
    # Generate random text from limited alphabet
    alphabet = 'abcdefghijklmnopqrstuvwxyz .,!?'
    text = ''.join(np.random.choice(list(alphabet), num_chars))

    # Split
    split_idx = int(len(text) * 0.9)
    text_train = text[:split_idx]
    text_val = text[split_idx:]

    # Tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(text_train)

    # Datasets
    train_dataset = TextDataset(text_train, tokenizer, seq_length)
    val_dataset = TextDataset(text_val, tokenizer, seq_length)

    print(f"Created synthetic text dataset:")
    print(f"  Train: {len(text_train)} characters")
    print(f"  Val: {len(text_val)} characters")
    print(f"  Vocab size: {tokenizer.vocab_size}")

    return train_dataset, val_dataset, tokenizer
