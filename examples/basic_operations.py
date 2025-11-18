#!/usr/bin/env python3
"""
Basic operations with ternary computing.

Demonstrates fundamental trit and tryte operations.
"""

from ternary import Trit, Tryte, TritValue


def demonstrate_trits():
    """Show basic trit operations."""
    print("=" * 60)
    print("TRIT OPERATIONS")
    print("=" * 60)

    # Create trits
    t_minus = Trit(TritValue.MINUS)
    t_zero = Trit(TritValue.ZERO)
    t_plus = Trit(TritValue.PLUS)

    print(f"\nTrit values: {t_minus}, {t_zero}, {t_plus}")
    print(f"As integers: {t_minus.to_int()}, {t_zero.to_int()}, {t_plus.to_int()}")

    # Arithmetic
    print("\nArithmetic:")
    print(f"  {t_plus} + {t_plus} = {t_plus + t_plus}")
    print(f"  {t_plus} - {t_minus} = {t_plus - t_minus}")
    print(f"  {t_plus} * {t_minus} = {t_plus * t_minus}")
    print(f"  -{t_plus} = {-t_plus}")

    # Logic operations
    print("\nLogic operations:")
    print(f"  {t_plus} AND {t_minus} = {t_plus.logic_and(t_minus)}")
    print(f"  {t_plus} OR {t_minus} = {t_plus.logic_or(t_minus)}")
    print(f"  {t_zero} OR {t_plus} = {t_zero.logic_or(t_plus)}")
    print(f"  NOT {t_plus} = {t_plus.logic_not()}")


def demonstrate_trytes():
    """Show basic tryte operations."""
    print("\n" + "=" * 60)
    print("TRYTE OPERATIONS")
    print("=" * 60)

    # Create trytes
    t1 = Tryte(42)
    t2 = Tryte(-17)
    t3 = Tryte("+-0+")  # Balanced ternary string

    print(f"\nTryte from int: {t1.to_int()} = {t1.to_balanced_ternary()}")
    print(f"Tryte from int: {t2.to_int()} = {t2.to_balanced_ternary()}")
    print(f"Tryte from string: {t3.to_balanced_ternary()} = {t3.to_int()}")

    # Arithmetic
    print("\nArithmetic:")
    result_add = t1 + t2
    print(f"  {t1.to_int()} + {t2.to_int()} = {result_add.to_int()}")
    print(f"  In balanced ternary: {t1.to_balanced_ternary()} + {t2.to_balanced_ternary()} = {result_add.to_balanced_ternary()}")

    result_mul = Tryte(5) * Tryte(7)
    print(f"\n  5 * 7 = {result_mul.to_int()}")

    # Negation is easy in balanced ternary!
    negated = -t1
    print(f"\n  -{t1.to_int()} = {negated.to_int()}")
    print(f"  Simply flip signs: {t1.to_balanced_ternary()} -> {negated.to_balanced_ternary()}")

    # Logic operations
    print("\nLogic operations:")
    t_all_plus = Tryte("++++++++++++++++++")
    t_all_minus = Tryte("------------------")
    result_and = t_all_plus.logic_and(t_all_minus)
    print(f"  ALL+ AND ALL- = {result_and.to_balanced_ternary()}")

    # Shift operations
    print("\nShift operations:")
    t_shift = Tryte(9)  # = "++0" in balanced ternary
    print(f"  Original: {t_shift.to_int()} = {t_shift.to_balanced_ternary()}")
    print(f"  Shift left 1: {t_shift.shift_left(1).to_int()} = {t_shift.shift_left(1).to_balanced_ternary()}")
    print(f"  Shift right 1: {t_shift.shift_right(1).to_int()} = {t_shift.shift_right(1).to_balanced_ternary()}")

    # Range
    print(f"\nTryte range:")
    print(f"  Min value: {Tryte.min_value():,}")
    print(f"  Max value: {Tryte.max_value():,}")
    print(f"  Total range: {Tryte.max_value() - Tryte.min_value() + 1:,} values")


def demonstrate_balanced_ternary_advantages():
    """Show advantages of balanced ternary."""
    print("\n" + "=" * 60)
    print("BALANCED TERNARY ADVANTAGES")
    print("=" * 60)

    # No separate sign bit needed
    print("\n1. No separate sign bit:")
    values = [42, 0, -42]
    for val in values:
        t = Tryte(val)
        print(f"   {val:4d} = {t.to_balanced_ternary()}")

    # Easy negation
    print("\n2. Easy negation (just flip all signs):")
    t = Tryte(100)
    neg_t = -t
    print(f"   {t.to_int():4d} = {t.to_balanced_ternary()}")
    print(f"  {neg_t.to_int():4d} = {neg_t.to_balanced_ternary()}")

    # Rounding by truncation
    print("\n3. Rounding by truncation:")
    t = Tryte(12345)
    print(f"   Original: {t.to_int()} = {t.to_balanced_ternary()}")
    print("   (In binary, rounding requires checking all remaining bits)")

    # Three-way comparison in one operation
    print("\n4. Natural three-way comparison:")
    a = Tryte(10)
    b = Tryte(20)
    diff = a - b
    print(f"   {a.to_int()} - {b.to_int()} = {diff.to_int()}")
    if diff.to_int() < 0:
        print("   Result is negative, so a < b")


if __name__ == "__main__":
    demonstrate_trits()
    demonstrate_trytes()
    demonstrate_balanced_ternary_advantages()

    print("\n" + "=" * 60)
    print("Basic operations demonstration complete!")
    print("=" * 60)
