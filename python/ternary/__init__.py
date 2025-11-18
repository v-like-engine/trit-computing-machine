"""
Ternary Computing Simulation Library

A Python library for simulating ternary (base-3) computer systems,
inspired by the Soviet Setun computer (1958).

This library provides:
- Balanced ternary arithmetic (-1, 0, +1)
- Ternary logic operations
- Memory simulation
- Basic processor simulation
- High-performance C++ core with Python bindings
"""

from ._ternary_core import (
    # Core types
    Trit, TritValue,
    Tryte,

    # Memory
    Memory, SparseMemory,

    # Processor
    Processor, Instruction, Opcode,
)

from .highlevel import (
    # High-level wrappers
    TernaryComputer,
    TernaryAssembler,

    # Utilities
    int_to_balanced_ternary,
    balanced_ternary_to_int,
)

__version__ = "0.1.0"
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
