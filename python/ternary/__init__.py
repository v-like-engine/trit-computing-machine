"""
Ternary Computing Simulation Library

A Python library for simulating ternary (base-3) computer systems,
inspired by the Soviet Setun computer (1958).

This library provides:
- Balanced ternary arithmetic (-1, 0, +1)
- Ternary logic operations
- Memory simulation
- Basic processor simulation
- Ternary neural networks
- Efficient 3-bit to 2-trit encoding
- High-performance C++ core with Python bindings
"""

# Try to import C++ core (may not be built)
try:
    from ._ternary_core import (
        # Core types
        Trit, TritValue,
        Tryte,

        # Memory
        Memory, SparseMemory,

        # Processor
        Processor, Instruction, Opcode,
    )
    _has_core = True
except ImportError:
    _has_core = False
    # Provide dummy classes for graceful degradation
    Trit = None
    TritValue = None
    Tryte = None
    Memory = None
    SparseMemory = None
    Processor = None
    Instruction = None
    Opcode = None

# High-level wrappers (only if core is available)
if _has_core:
    try:
        from .highlevel import (
            # High-level wrappers
            TernaryComputer,
            TernaryAssembler,

            # Utilities
            int_to_balanced_ternary,
            balanced_ternary_to_int,
        )
        _has_highlevel = True
    except ImportError:
        _has_highlevel = False
        TernaryComputer = None
        TernaryAssembler = None
        int_to_balanced_ternary = None
        balanced_ternary_to_int = None
else:
    _has_highlevel = False
    TernaryComputer = None
    TernaryAssembler = None
    int_to_balanced_ternary = None
    balanced_ternary_to_int = None

# Neural network components
try:
    from .neural import (
        TernaryNeuralNetwork,
        TernaryLinear,
        TernaryActivation,
        TernaryConfig,
        ternary_quantize,
        train_step,
        evaluate,
    )
    _has_neural = True
except ImportError:
    _has_neural = False

# Encoding components
try:
    from .encoding import (
        TritEncoder,
        TryteEncoder,
    )
    _has_encoding = True
except ImportError:
    _has_encoding = False

# TAL compiler
try:
    from .tal import (
        TALCompiler,
    )
    _has_tal = True
except ImportError:
    _has_tal = False

__version__ = "0.2.0"
__author__ = "Ternary Computing Research"

__all__ = [
    # Core
    "Trit", "TritValue", "Tryte",
    "Memory", "SparseMemory",
    "Processor", "Instruction", "Opcode",

    # High-level
    "TernaryComputer",
    "TernaryAssembler",

    # Utilities
    "int_to_balanced_ternary",
    "balanced_ternary_to_int",
]

# Add optional components if available
if _has_neural:
    __all__.extend([
        "TernaryNeuralNetwork",
        "TernaryLinear",
        "TernaryActivation",
        "TernaryConfig",
        "ternary_quantize",
        "train_step",
        "evaluate",
    ])

if _has_encoding:
    __all__.extend([
        "TritEncoder",
        "TryteEncoder",
    ])

if _has_tal:
    __all__.extend([
        "TALCompiler",
    ])
