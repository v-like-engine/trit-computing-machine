#!/usr/bin/env python3
"""
Setun computer simulation example.

Demonstrates a complete ternary computer system with memory and processor.
"""

from ternary import TernaryComputer, Memory, Processor, Instruction, Opcode


def example_add_two_numbers():
    """Simple program: add two numbers stored in memory."""
    print("=" * 60)
    print("EXAMPLE 1: Add Two Numbers")
    print("=" * 60)

    # Create computer with 256 words of memory
    computer = TernaryComputer(memory_size=256)

    # Write assembly program
    program = """
        ; Add two numbers from memory
        LOAD 100    ; Load first number
        ADD 101     ; Add second number
        STORE 102   ; Store result
        HALT        ; Stop
    """

    # Load program
    computer.load_program(program)

    # Store data: 42 at address 100, 17 at address 101
    computer.load_data([42, 17], start_address=100)

    print("\nInitial state:")
    print(f"  Memory[100] = {computer.get_memory_value(100)}")
    print(f"  Memory[101] = {computer.get_memory_value(101)}")
    print(f"  Memory[102] = {computer.get_memory_value(102)}")

    # Run program
    result = computer.run()

    print(f"\nExecution complete:")
    print(f"  Cycles: {result['cycles_executed']}")
    print(f"  Halted: {result['halted']}")
    print(f"  Accumulator: {result['accumulator']}")

    print(f"\nFinal state:")
    print(f"  Memory[102] = {computer.get_memory_value(102)} (should be 59)")


def example_fibonacci():
    """Calculate Fibonacci numbers."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Fibonacci Sequence")
    print("=" * 60)

    computer = TernaryComputer(memory_size=256)

    # Fibonacci program:
    # Compute first N Fibonacci numbers
    # Memory layout:
    #   100: N (count)
    #   101: current counter
    #   102: F(n-1)
    #   103: F(n)
    #   104+: results
    program = """
        ; Initialize
        LOAD 102      ; Load F(0) = 0
        STORE 104     ; Store first result
        LOAD 103      ; Load F(1) = 1
        STORE 105     ; Store second result

        ; Loop
    loop:
        LOAD 101      ; Load counter
        ADD 110       ; Add 1 (stored at 110)
        STORE 101     ; Save counter
        SUB 100       ; Subtract N
        JUMP_POS done ; If counter > N, done

        ; Calculate next Fibonacci
        LOAD 102      ; Load F(n-1)
        ADD 103       ; Add F(n)
        STORE 111     ; Store temporarily

        ; Shift values
        LOAD 103      ; F(n) becomes F(n-1)
        STORE 102
        LOAD 111      ; New value becomes F(n)
        STORE 103

        JUMP loop     ; Repeat

    done:
        LOAD 103      ; Load final result
        HALT
    """

    # Initialize data
    N = 10
    computer.set_memory_value(100, N)      # N = 10
    computer.set_memory_value(101, 1)      # counter = 1
    computer.set_memory_value(102, 0)      # F(n-1) = 0
    computer.set_memory_value(103, 1)      # F(n) = 1
    computer.set_memory_value(110, 1)      # constant 1

    computer.load_program(program)

    print(f"\nCalculating first {N} Fibonacci numbers...")

    result = computer.run(max_cycles=1000)

    print(f"\nExecution:")
    print(f"  Cycles: {result['cycles_executed']}")
    print(f"  Final Fibonacci number: {result['accumulator']}")

    # Show stored results
    print("\nFirst Fibonacci numbers stored in memory:")
    for i in range(min(N, 10)):
        addr = 104 + i
        val = computer.get_memory_value(addr)
        if val != 0 or i < 2:
            print(f"  F({i}) at [{addr}] = {val}")


def example_ternary_logic():
    """Demonstrate ternary logic operations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Ternary Logic")
    print("=" * 60)

    computer = TernaryComputer(memory_size=256)

    # Logic operations program
    program = """
        LOAD 100      ; Load first value
        AND 101       ; AND with second value
        STORE 102     ; Store result

        LOAD 100      ; Load first value
        OR 101        ; OR with second value
        STORE 103     ; Store result

        LOAD 100      ; Load first value
        NOT           ; NOT operation
        STORE 104     ; Store result

        LOAD 100      ; Load first value
        XOR 101       ; XOR with second value
        STORE 105     ; Store result

        HALT
    """

    # Test with balanced ternary values
    from ternary import Tryte

    val1 = Tryte("+-+0+-+0")  # Mixed pattern
    val2 = Tryte("++--++--")  # Another pattern

    computer.set_memory_value(100, val1)
    computer.set_memory_value(101, val2)

    computer.load_program(program)

    print(f"\nInput values:")
    print(f"  A = {val1.to_balanced_ternary()} ({val1.to_int()})")
    print(f"  B = {val2.to_balanced_ternary()} ({val2.to_int()})")

    result = computer.run()

    print(f"\nResults:")
    result_and = computer.memory.read(102)
    result_or = computer.memory.read(103)
    result_not = computer.memory.read(104)
    result_xor = computer.memory.read(105)

    print(f"  A AND B = {result_and.to_balanced_ternary()} ({result_and.to_int()})")
    print(f"  A OR  B = {result_or.to_balanced_ternary()} ({result_or.to_int()})")
    print(f"  NOT A   = {result_not.to_balanced_ternary()} ({result_not.to_int()})")
    print(f"  A XOR B = {result_xor.to_balanced_ternary()} ({result_xor.to_int()})")


def step_through_execution():
    """Demonstrate step-by-step execution."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Step-by-Step Execution")
    print("=" * 60)

    computer = TernaryComputer(memory_size=256)

    program = """
        LOAD 100
        ADD 101
        STORE 102
        NEG
        STORE 103
        HALT
    """

    computer.load_data([10, 20], start_address=100)
    computer.load_program(program)

    print("\nStepping through program:\n")

    step = 0
    while not computer.processor.halted and step < 10:
        print(f"Step {step}:")
        print(computer.get_status())
        computer.step()
        step += 1

    print("Final state:")
    print(computer.get_status())
    print(f"\nResults:")
    print(f"  Memory[102] = {computer.get_memory_value(102)} (10 + 20 = 30)")
    print(f"  Memory[103] = {computer.get_memory_value(103)} (-(30) = -30)")


if __name__ == "__main__":
    example_add_two_numbers()
    example_fibonacci()
    example_ternary_logic()
    step_through_execution()

    print("\n" + "=" * 60)
    print("Setun simulation examples complete!")
    print("=" * 60)
