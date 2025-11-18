"""
High-level Python wrappers for ternary computing.

Provides user-friendly interfaces and utilities on top of the C++ core.
"""

from typing import List, Dict, Optional, Union
from ._ternary_core import (
    Trit, Tryte, Memory, SparseMemory, Processor, Instruction, Opcode
)


def int_to_balanced_ternary(value: int, width: int = 18) -> str:
    """
    Convert an integer to balanced ternary string representation.

    Args:
        value: Integer to convert
        width: Number of trits (default 18 for Tryte)

    Returns:
        Balanced ternary string (e.g., "+-00+")
    """
    tryte = Tryte(value)
    return tryte.to_balanced_ternary()


def balanced_ternary_to_int(trit_string: str) -> int:
    """
    Convert balanced ternary string to integer.

    Args:
        trit_string: Balanced ternary string (e.g., "+-00+")

    Returns:
        Integer value
    """
    tryte = Tryte(trit_string)
    return tryte.to_int()


class TernaryAssembler:
    """
    Assembler for ternary processor instructions.

    Provides a simple assembly language for the ternary processor.
    """

    OPCODE_MAP = {
        'NOP': Opcode.NOP,
        'LOAD': Opcode.LOAD,
        'STORE': Opcode.STORE,
        'ADD': Opcode.ADD,
        'SUB': Opcode.SUB,
        'MUL': Opcode.MUL,
        'NEG': Opcode.NEG,
        'AND': Opcode.AND,
        'OR': Opcode.OR,
        'NOT': Opcode.NOT,
        'XOR': Opcode.XOR,
        'SHIFT_L': Opcode.SHIFT_L,
        'SHIFT_R': Opcode.SHIFT_R,
        'JUMP': Opcode.JUMP,
        'JUMP_POS': Opcode.JUMP_POS,
        'JUMP_ZERO': Opcode.JUMP_ZERO,
        'JUMP_NEG': Opcode.JUMP_NEG,
        'HALT': Opcode.HALT,
    }

    def __init__(self):
        self.labels: Dict[str, int] = {}
        self.instructions: List[Instruction] = []
        self.data: Dict[int, Tryte] = {}

    def parse_line(self, line: str) -> Optional[Instruction]:
        """Parse a single line of assembly code."""
        # Remove comments and strip whitespace
        line = line.split(';')[0].strip()
        if not line:
            return None

        # Check for label
        if ':' in line:
            label, rest = line.split(':', 1)
            self.labels[label.strip()] = len(self.instructions)
            line = rest.strip()
            if not line:
                return None

        # Parse instruction
        parts = line.split()
        if not parts:
            return None

        opcode_str = parts[0].upper()
        if opcode_str not in self.OPCODE_MAP:
            raise ValueError(f"Unknown opcode: {opcode_str}")

        opcode = self.OPCODE_MAP[opcode_str]
        address = 0

        if len(parts) > 1:
            # Parse address (could be number or label)
            addr_str = parts[1]
            try:
                address = int(addr_str)
            except ValueError:
                # It's a label, will resolve later
                address = addr_str

        return Instruction(opcode, address if isinstance(address, int) else 0)

    def assemble(self, code: str) -> List[Tryte]:
        """
        Assemble code into trytes.

        Args:
            code: Assembly code as string

        Returns:
            List of trytes representing the program
        """
        self.labels.clear()
        self.instructions.clear()

        # First pass: parse instructions and collect labels
        lines = code.split('\n')
        temp_instructions = []

        for line in lines:
            inst = self.parse_line(line)
            if inst is not None:
                temp_instructions.append((inst, line))

        # Second pass: resolve labels
        for inst, line in temp_instructions:
            parts = line.split()
            if len(parts) > 1 and parts[1] in self.labels:
                inst.address = self.labels[parts[1]]
            self.instructions.append(inst)

        # Encode to trytes
        return [inst.encode() for inst in self.instructions]

    def assemble_to_memory(self, code: str, memory: Memory, start_address: int = 0):
        """
        Assemble code and load it into memory.

        Args:
            code: Assembly code as string
            memory: Memory object to load into
            start_address: Starting address (default 0)
        """
        trytes = self.assemble(code)
        for i, tryte in enumerate(trytes):
            memory.write(start_address + i, tryte)


class TernaryComputer:
    """
    High-level interface for a complete ternary computer system.

    Combines processor, memory, and provides easy-to-use methods
    for programming and running ternary programs.
    """

    def __init__(self, memory_size: int = 1024):
        """
        Initialize ternary computer.

        Args:
            memory_size: Size of memory in trytes (default 1024)
        """
        self.memory = Memory(memory_size)
        self.processor = Processor(self.memory)
        self.assembler = TernaryAssembler()

    def load_program(self, code: str, start_address: int = 0):
        """
        Load assembly program into memory.

        Args:
            code: Assembly code as string
            start_address: Starting address (default 0)
        """
        self.assembler.assemble_to_memory(code, self.memory, start_address)
        self.processor.program_counter = start_address

    def load_data(self, data: List[Union[int, Tryte]], start_address: int):
        """
        Load data into memory.

        Args:
            data: List of integers or Trytes
            start_address: Starting address
        """
        for i, value in enumerate(data):
            if isinstance(value, int):
                value = Tryte(value)
            self.memory.write(start_address + i, value)

    def run(self, max_cycles: int = 10000) -> Dict:
        """
        Run the program until halt or max cycles.

        Args:
            max_cycles: Maximum number of cycles to execute

        Returns:
            Dictionary with execution results
        """
        initial_cycles = self.processor.cycle_count
        self.processor.run(max_cycles)

        return {
            'cycles_executed': self.processor.cycle_count - initial_cycles,
            'total_cycles': self.processor.cycle_count,
            'halted': self.processor.halted,
            'accumulator': self.processor.accumulator.to_int(),
            'program_counter': self.processor.program_counter,
        }

    def step(self):
        """Execute one instruction."""
        self.processor.step()

    def reset(self):
        """Reset processor state."""
        self.processor.reset()

    def get_memory_value(self, address: int) -> int:
        """Get memory value as integer."""
        return self.memory.read(address).to_int()

    def set_memory_value(self, address: int, value: Union[int, Tryte]):
        """Set memory value."""
        if isinstance(value, int):
            value = Tryte(value)
        self.memory.write(address, value)

    def dump_memory(self, start: int = 0, length: int = 10) -> str:
        """Dump memory contents."""
        return self.memory.dump(start, length)

    def get_status(self) -> str:
        """Get processor status."""
        return self.processor.get_status()

    def __repr__(self):
        return (f"TernaryComputer(memory_size={self.memory.size}, "
                f"pc={self.processor.program_counter}, "
                f"cycles={self.processor.cycle_count})")
