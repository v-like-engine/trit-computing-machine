# Ternary Computing Machine

A high-performance software simulation of ternary computer systems based on balanced ternary logic, inspired by the Soviet **Setun computer** (1958).

## Overview

This project provides a complete simulation environment for exploring ternary (base-3) computing with balanced ternary representation (-1, 0, +1). It combines a fast C++ core with user-friendly Python bindings to enable research, education, and experimentation with ternary computing concepts.

### What is Balanced Ternary?

Balanced ternary uses three digits: **-1**, **0**, and **+1** (often written as -, 0, +), offering several advantages over binary:

- **No separate sign bit**: Negative numbers are represented naturally
- **Trivial negation**: Simply flip all signs (- ↔ +)
- **Better information density**: ~1.585 bits per trit vs 1 bit per binary digit
- **Symmetric range**: Natural representation of positive and negative values
- **Simpler rounding**: Truncation works correctly without examining all digits

## Features

### 🌐 Interactive Web Sandbox

**Try it live:** [Ternary Computing Sandbox](https://v-like-engine.github.io/trit-computing-machine/web-sandbox/)

- **Logic Expression Evaluator**: Test ternary logic operations interactively
- **Neural Network Builder**: Build and train ternary neural networks in your browser
- **Dataset Upload**: Load CSV data or use built-in MNIST sample
- **Encoding Comparison**: Compare lossless vs compact encoding methods
- **Real-time Training**: Watch neural networks train with live charts
- **No Installation**: Runs entirely in browser, no setup needed

### Core Components

- **Trit**: Single balanced ternary digit (-1, 0, +1)
- **Tryte**: 18-trit word (matching Setun architecture, ~28.5 bits of information)
- **Memory**: Ternary memory simulation with sparse and dense implementations
- **Processor**: Complete ternary CPU simulation with 18 instructions
- **Assembler**: Simple assembly language for ternary programs (TAL)

### Neural Networks

#### Fully Connected Networks
- **Ternary Neural Networks**: Weights quantized to {-1, 0, +1}
- **Straight-Through Estimator**: Proper gradient flow for training
- **MNIST Classification**: Complete example with 96-98% accuracy
- **120x Compression**: Massive model size reduction vs float32
- **Sparsity**: 30-50% zero weights for additional speedup

#### Convolutional Neural Networks (CNNs)
- **TernaryResNet**: ResNet-18, 34, 50 architectures for image classification
- **TernaryVGG**: VGG-11, 16, 19 architectures
- **CIFAR-10/100**: 90-93% accuracy with ternary weights
- **ImageNet**: 68-71% Top-1 accuracy on 1000-class classification
- **14.8x Compression**: Compared to float32 CNNs
- **Im2col Optimization**: Efficient convolution implementation
- **Data Augmentation**: Random crop, flip, rotation support
- **Top-1 & Top-5 Metrics**: Comprehensive evaluation tools

### Encoding Frameworks

- **Compact Encoder**: 3 bits → 2 trits (94.6% efficient, minimal loss)
- **Perfect Encoder**: Lossless radix conversion (98.4% efficient)
- **Optimal Block**: 5 trits → 8 bits (94.9% efficient, lossless)
- **Arithmetic Encoder**: Variable-length optimal encoding

### Performance

- **Fast C++ core**: Optimized ternary arithmetic and logic operations
- **Python bindings**: Easy-to-use interface via pybind11
- **Protobuf support**: Inter-service communication for distributed systems
- **Web-based sandbox**: JavaScript implementation for browser use

### Capabilities

- Ternary arithmetic (addition, subtraction, multiplication, negation)
- Ternary logic operations (AND, OR, NOT, XOR)
- Memory simulation (81+ words, expandable)
- Processor simulation with accumulator and index register
- Assembly programming with labels and jumps (TAL)
- Step-by-step execution debugging
- Neural network training and inference
- Multiple encoding strategies with zero information loss

## Installation

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install cmake g++ python3-dev

# macOS
brew install cmake python3

# Install Python dependencies
pip install -r requirements.txt
```

### Build from Source

```bash
# Clone the repository
git clone https://github.com/v-like-engine/trit-computing-machine.git
cd trit-computing-machine

# Build and install
pip install .

# Or for development
pip install -e .
```

### Build with CMake (C++ only)

```bash
mkdir build && cd build
cmake ..
make
```

## Quick Start

### Basic Ternary Operations

```python
from ternary import Trit, Tryte, TritValue

# Create trits
t_plus = Trit(TritValue.PLUS)   # +1
t_zero = Trit(TritValue.ZERO)   # 0
t_minus = Trit(TritValue.MINUS) # -1

# Trit arithmetic
result = t_plus + t_minus  # = 0
negated = -t_plus          # = -1

# Create trytes (18-trit words)
t1 = Tryte(42)
t2 = Tryte(-17)

# Tryte arithmetic
sum_t = t1 + t2  # = 25
neg_t = -t1      # = -42 (just flip all signs!)

print(f"{t1.to_int()} in balanced ternary: {t1.to_balanced_ternary()}")
```

### Running Ternary Programs

```python
from ternary import TernaryComputer

# Create a ternary computer with 256 words of memory
computer = TernaryComputer(memory_size=256)

# Write a simple program
program = """
    LOAD 100    ; Load value from address 100
    ADD 101     ; Add value from address 101
    STORE 102   ; Store result at address 102
    HALT        ; Stop execution
"""

# Load the program
computer.load_program(program)

# Initialize data
computer.load_data([10, 20], start_address=100)

# Run the program
result = computer.run()

print(f"Result: {computer.get_memory_value(102)}")  # 30
print(f"Cycles executed: {result['cycles_executed']}")
```

### Ternary CNNs

```python
from ternary.cnn_models import create_ternary_resnet18
from ternary.cnn_data import CIFAR10Loader, DataAugmentation
from ternary.cnn_trainer import TernaryCNNTrainer, TrainingConfig

# Load CIFAR-10
loader = CIFAR10Loader(data_dir='./data/cifar10')
loader.load()

# Create TernaryResNet-18
model = create_ternary_resnet18(num_classes=10)

# Configure training
config = TrainingConfig(
    epochs=100,
    batch_size=128,
    learning_rate=0.1,
    lr_schedule='step',
    lr_decay_epochs=[30, 60, 90]
)

# Train
trainer = TernaryCNNTrainer(model, loader, loader, config)
metrics = trainer.train()

# Evaluate
print(f"Best accuracy: {metrics.get_best_val_acc() * 100:.2f}%")
```

### More Examples

See the `examples/` directory for complete examples:

**Basic Computing:**
- `basic_operations.py` - Fundamental trit and tryte operations
- `setun_simulation.py` - Complete Setun computer simulation with multiple programs

**Neural Networks:**
- `mnist_ternary.py` - Train ternary neural network on MNIST
- `train_cifar10_cnn.py` - Train TernaryResNet-18 on CIFAR-10
- `train_imagenet_cnn.py` - Train TernaryResNet-34 on ImageNet
- `evaluate_cnn.py` - Evaluate trained CNN models
- `cnn_inference_demo.py` - Simple CNN inference demonstration

## Architecture

```
trit-computing-machine/
├── src/
│   ├── core/              # C++ implementation
│   │   ├── trit.hpp/cpp   # Trit type and operations
│   │   ├── tryte.hpp/cpp  # Tryte (18-trit word)
│   │   ├── memory.hpp/cpp # Memory simulation
│   │   └── processor.hpp/cpp # Processor simulation
│   ├── bindings/          # Python bindings
│   │   └── bindings.cpp   # pybind11 bindings
│   └── proto/             # Protocol buffers
│       └── ternary.proto  # Inter-service communication
├── python/
│   └── ternary/           # Python package
│       ├── __init__.py
│       ├── highlevel.py   # High-level Python API
│       ├── neural.py      # Fully connected neural networks
│       ├── encoding.py    # Encoding frameworks
│       ├── lossless_encoding.py  # Lossless encoders
│       ├── cnn_layers.py  # CNN layers (Conv2D, BatchNorm, Pooling)
│       ├── cnn_models.py  # CNN architectures (ResNet, VGG)
│       ├── cnn_data.py    # Data loaders (CIFAR, ImageNet)
│       └── cnn_trainer.py # Training pipeline
├── examples/              # Example programs
│   ├── basic_operations.py
│   ├── setun_simulation.py
│   ├── mnist_ternary.py
│   ├── train_cifar10_cnn.py
│   ├── train_imagenet_cnn.py
│   ├── evaluate_cnn.py
│   └── cnn_inference_demo.py
├── tests/                 # Tests (C++ and Python)
├── docs/                  # Documentation
│   ├── IDEAS.md          # Applications and research directions
│   ├── NEURAL_NETWORKS.md # Neural network theory
│   ├── CNN_IMPLEMENTATION.md # CNN documentation
│   └── web-sandbox/      # Interactive web sandbox
└── checkpoints/           # Saved model checkpoints (generated)
```

## Instruction Set

The simulated processor supports the following operations:

| Opcode | Description | Operation |
|--------|-------------|-----------|
| NOP | No operation | - |
| LOAD | Load from memory | ACC ← MEM[addr] |
| STORE | Store to memory | MEM[addr] ← ACC |
| ADD | Addition | ACC ← ACC + MEM[addr] |
| SUB | Subtraction | ACC ← ACC - MEM[addr] |
| MUL | Multiplication | ACC ← ACC × MEM[addr] |
| NEG | Negation | ACC ← -ACC |
| AND | Logical AND | ACC ← ACC ∧ MEM[addr] |
| OR | Logical OR | ACC ← ACC ∨ MEM[addr] |
| NOT | Logical NOT | ACC ← ¬ACC |
| XOR | Logical XOR | ACC ← ACC ⊕ MEM[addr] |
| SHIFT_L | Shift left | ACC ← ACC << 1 |
| SHIFT_R | Shift right | ACC ← ACC >> 1 |
| JUMP | Unconditional jump | PC ← addr |
| JUMP_POS | Jump if positive | if ACC > 0: PC ← addr |
| JUMP_ZERO | Jump if zero | if ACC = 0: PC ← addr |
| JUMP_NEG | Jump if negative | if ACC < 0: PC ← addr |
| HALT | Halt execution | STOP |

## Applications and Research Directions

Ternary computing offers unique advantages in several domains:

### Machine Learning
- **Ternary Neural Networks**: Weights in {-1, 0, +1} for efficient inference
- **Fuzzy Logic**: Natural representation of {false, unknown, true}
- **Model Compression**: 50% memory reduction vs full precision

### Control Systems
- **Three-state control**: Increase/Maintain/Decrease
- **Industrial automation**: Natural mapping to control signals
- **Robotics**: Forward/Stop/Backward commands

### Distributed Systems
- **Consensus protocols**: Approve/Abstain/Reject voting
- **Load balancing**: Overloaded/Balanced/Underloaded states
- **Byzantine fault tolerance**: Agreement with uncertainty

### Database Systems
- **SQL three-valued logic**: TRUE/NULL/FALSE native support
- **Incomplete data**: Natural representation of missing values

See [docs/IDEAS.md](docs/IDEAS.md) for a comprehensive exploration of applications and research directions.

## Performance

The C++ core provides high-performance ternary operations:

- **Trit operations**: ~5-10 CPU cycles
- **Tryte arithmetic**: ~50-100 CPU cycles
- **Memory access**: ~10-20 CPU cycles
- **Instruction execution**: ~100-200 CPU cycles

Python bindings add minimal overhead (~100ns per call) while maintaining ease of use.

## Testing

### Run Python Tests

```bash
pytest tests/test_operations.py -v
```

### Run C++ Tests

```bash
mkdir build && cd build
cmake -DBUILD_TESTS=ON ..
make
ctest
```

## Contributing

Contributions are welcome! Areas of interest:

- Additional processor instructions
- Optimization of core operations
- Ternary neural network implementations
- Hardware synthesis (FPGA/ASIC)
- Educational materials
- Benchmark suite

## History: The Setun Computer

The **Setun** (Сетунь) was a revolutionary ternary computer built in 1958 at Moscow State University by Nikolay Brusentsov. Key features:

- **Balanced ternary**: Used -1, 0, +1 instead of 0, 1
- **18 trits per word**: ~28.5 bits of information
- **24 instructions**: Simple but complete instruction set
- **81 words RAM**: Plus 1944 words on magnetic drum
- **Lower cost**: ~2.5× cheaper than contemporary binary computers
- **Lower power**: More energy efficient than binary alternatives
- **50 units built**: Used in Soviet universities until 1965

Despite its advantages, the Setun was discontinued due to:
- Pressure to standardize on binary (Western influence)
- Lack of software ecosystem
- Manufacturing optimized for binary

This project revives these ideas for modern research and experimentation.

## License

MIT License - see LICENSE file for details.

## References

1. Brusentsov, N.P., et al. "Ternary Computers: The Setun and the Setun 70" (1998)
2. Hayes, B. "Third Base", American Scientist (2001)
3. Knuth, D.E. "The Art of Computer Programming, Vol. 2" (Section on Radix Systems)
4. [Setun - Wikipedia](https://en.wikipedia.org/wiki/Setun)
5. [Russian Virtual Computer Museum - Setun](https://www.computer-museum.ru/english/setun.htm)

## Acknowledgments

- Nikolay Brusentsov and the Setun team for pioneering ternary computing
- Moscow State University for preserving ternary computing history
- The balanced ternary community for keeping these ideas alive

---

**"The future of computing may not be binary. Let's explore the alternatives."**
