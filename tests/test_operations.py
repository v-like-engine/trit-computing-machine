#!/usr/bin/env python3
"""
Python tests for ternary operations.
"""

import pytest
from ternary import Trit, TritValue, Tryte, Memory, TernaryComputer


class TestTrit:
    def test_construction(self):
        t1 = Trit(TritValue.ZERO)
        assert t1.to_int() == 0

        t2 = Trit(TritValue.PLUS)
        assert t2.to_int() == 1

        t3 = Trit(TritValue.MINUS)
        assert t3.to_int() == -1

    def test_arithmetic(self):
        plus = Trit(TritValue.PLUS)
        minus = Trit(TritValue.MINUS)
        zero = Trit(TritValue.ZERO)

        assert (plus + zero).to_int() == 1
        assert (plus + minus).to_int() == 0
        assert (plus - minus).to_int() == 2

        assert (-plus).to_int() == -1
        assert (-minus).to_int() == 1

    def test_logic(self):
        plus = Trit(TritValue.PLUS)
        minus = Trit(TritValue.MINUS)
        zero = Trit(TritValue.ZERO)

        # AND is min
        assert plus.logic_and(minus).to_int() == -1
        assert plus.logic_and(zero).to_int() == 0

        # OR is max
        assert plus.logic_or(minus).to_int() == 1
        assert zero.logic_or(minus).to_int() == 0

        # NOT is negation
        assert plus.logic_not().to_int() == -1
        assert minus.logic_not().to_int() == 1


class TestTryte:
    def test_construction(self):
        t1 = Tryte()
        assert t1.to_int() == 0

        t2 = Tryte(42)
        assert t2.to_int() == 42

        t3 = Tryte(-17)
        assert t3.to_int() == -17

    def test_arithmetic(self):
        t1 = Tryte(10)
        t2 = Tryte(5)

        assert (t1 + t2).to_int() == 15
        assert (t1 - t2).to_int() == 5
        assert (t1 * t2).to_int() == 50

    def test_negation(self):
        t = Tryte(42)
        neg = -t
        assert neg.to_int() == -42

        double_neg = -neg
        assert double_neg.to_int() == 42

    def test_range(self):
        max_val = Tryte.max_value()
        min_val = Tryte.min_value()

        assert max_val > 0
        assert min_val < 0
        assert max_val == -min_val

        # 3^18 / 2 (rounded)
        assert max_val == 193710244

    def test_balanced_ternary_string(self):
        t = Tryte(0)
        s = t.to_balanced_ternary()
        assert len(s) == 18
        assert all(c in '0+-' for c in s)

    def test_comparison(self):
        t1 = Tryte(10)
        t2 = Tryte(20)
        t3 = Tryte(10)

        assert t1 < t2
        assert t2 > t1
        assert t1 == t3
        assert t1 != t2


class TestMemory:
    def test_basic_operations(self):
        mem = Memory(100)
        assert mem.size == 100

        # Write and read
        t = Tryte(42)
        mem.write(0, t)
        result = mem.read(0)
        assert result.to_int() == 42

    def test_clear(self):
        mem = Memory(10)
        mem.write(5, Tryte(42))
        mem.clear()
        assert mem.read(5).to_int() == 0

    def test_block_operations(self):
        mem = Memory(100)
        data = [Tryte(i) for i in range(10)]
        mem.write_block(20, data)

        block = mem.read_block(20, 10)
        assert len(block) == 10
        assert block[5].to_int() == 5


class TestTernaryComputer:
    def test_simple_program(self):
        computer = TernaryComputer(memory_size=256)

        program = """
            LOAD 100
            ADD 101
            STORE 102
            HALT
        """

        computer.load_program(program)
        computer.load_data([10, 20], start_address=100)

        result = computer.run()

        assert result['halted'] is True
        assert computer.get_memory_value(102) == 30

    def test_negation(self):
        computer = TernaryComputer(memory_size=256)

        program = """
            LOAD 100
            NEG
            STORE 101
            HALT
        """

        computer.load_program(program)
        computer.set_memory_value(100, 42)

        result = computer.run()

        assert result['halted'] is True
        assert computer.get_memory_value(101) == -42

    def test_logic_operations(self):
        computer = TernaryComputer(memory_size=256)

        program = """
            LOAD 100
            AND 101
            STORE 102
            HALT
        """

        computer.load_program(program)

        val1 = Tryte("++++++++")
        val2 = Tryte("--------")
        computer.set_memory_value(100, val1)
        computer.set_memory_value(101, val2)

        result = computer.run()

        assert result['halted'] is True
        result_val = computer.memory.read(102)
        # AND is min, so all should be minus
        assert '--------' in result_val.to_balanced_ternary()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
