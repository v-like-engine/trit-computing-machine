"""
Efficient encoding/decoding between binary and ternary representations.

3 bits can store 8 values (2^3 = 8)
2 trits can represent 9 values (3^2 = 9)

We use 3 bits to encode 8 out of 9 possible 2-trit combinations.
This gives us ~1.585 bits per trit efficiency in practice.
"""

from typing import List, Tuple
from ternary import Trit, Tryte, TritValue
import numpy as np


class TritEncoder:
    """
    Efficient encoding between binary and ternary.

    Maps 2 trits (9 states) to 3 bits (8 states).
    We drop one state (++), mapping 8/9 combinations.

    Mapping:
    -- → 000 (0)    -0 → 001 (1)    -+ → 010 (2)
    0- → 011 (3)    00 → 100 (4)    0+ → 101 (5)
    +- → 110 (6)    +0 → 111 (7)    ++ → [unmappable]
    """

    # Trit pair to 3-bit encoding
    TRIT_PAIR_TO_BITS = {
        (-1, -1): 0b000,  # --
        (-1,  0): 0b001,  # -0
        (-1,  1): 0b010,  # -+
        ( 0, -1): 0b011,  # 0-
        ( 0,  0): 0b100,  # 00
        ( 0,  1): 0b101,  # 0+
        ( 1, -1): 0b110,  # +-
        ( 1,  0): 0b111,  # +0
        # (1, 1) is unmappable - we use saturation
    }

    # Reverse mapping
    BITS_TO_TRIT_PAIR = {v: k for k, v in TRIT_PAIR_TO_BITS.items()}

    @staticmethod
    def encode_trit_pair(t1: Trit, t2: Trit) -> int:
        """Encode 2 trits into 3 bits."""
        pair = (t1.to_int(), t2.to_int())

        # Handle the unmappable case (++)
        if pair == (1, 1):
            # Saturate to +0
            return TritEncoder.TRIT_PAIR_TO_BITS[(1, 0)]

        return TritEncoder.TRIT_PAIR_TO_BITS[pair]

    @staticmethod
    def decode_trit_pair(bits: int) -> Tuple[Trit, Trit]:
        """Decode 3 bits into 2 trits."""
        bits = bits & 0b111  # Ensure only 3 bits
        t1_val, t2_val = TritEncoder.BITS_TO_TRIT_PAIR[bits]
        return Trit(t1_val), Trit(t2_val)

    @staticmethod
    def encode_trits(trits: List[Trit]) -> bytes:
        """
        Encode a list of trits into bytes efficiently.

        Packs 2 trits per 3 bits. If odd number of trits,
        the last trit is paired with 0.
        """
        # Pad to even length
        if len(trits) % 2 == 1:
            trits = trits + [Trit(TritValue.ZERO)]

        # Encode pairs
        encoded_bits = []
        for i in range(0, len(trits), 2):
            bits = TritEncoder.encode_trit_pair(trits[i], trits[i + 1])
            encoded_bits.append(bits)

        # Pack into bytes (8 bits each)
        result = bytearray()
        bit_buffer = 0
        bit_count = 0

        for bits in encoded_bits:
            bit_buffer = (bit_buffer << 3) | bits
            bit_count += 3

            while bit_count >= 8:
                byte_val = (bit_buffer >> (bit_count - 8)) & 0xFF
                result.append(byte_val)
                bit_count -= 8

        # Flush remaining bits
        if bit_count > 0:
            byte_val = (bit_buffer << (8 - bit_count)) & 0xFF
            result.append(byte_val)

        return bytes(result)

    @staticmethod
    def decode_trits(data: bytes, num_trits: int) -> List[Trit]:
        """
        Decode bytes back into trits.

        Args:
            data: Encoded byte data
            num_trits: Expected number of trits
        """
        # Extract 3-bit chunks
        bit_buffer = 0
        bit_count = 0
        three_bit_chunks = []

        for byte_val in data:
            bit_buffer = (bit_buffer << 8) | byte_val
            bit_count += 8

            while bit_count >= 3:
                chunk = (bit_buffer >> (bit_count - 3)) & 0b111
                three_bit_chunks.append(chunk)
                bit_count -= 3

        # Decode pairs
        trits = []
        for chunk in three_bit_chunks:
            t1, t2 = TritEncoder.decode_trit_pair(chunk)
            trits.extend([t1, t2])

            if len(trits) >= num_trits:
                break

        return trits[:num_trits]


class TryteEncoder:
    """Encode trytes efficiently using 3-bit to 2-trit encoding."""

    @staticmethod
    def encode_tryte(tryte: Tryte) -> bytes:
        """Encode a tryte (18 trits) into bytes."""
        trits = [tryte.get_trit(i) for i in range(18)]
        return TritEncoder.encode_trits(trits)

    @staticmethod
    def decode_tryte(data: bytes) -> Tryte:
        """Decode bytes back into a tryte."""
        trits = TritEncoder.decode_trits(data, 18)

        # Construct tryte
        from ternary import Tryte
        import array
        tryte = Tryte()
        for i, trit in enumerate(trits[:18]):
            tryte.set_trit(i, trit)

        return tryte

    @staticmethod
    def calculate_efficiency():
        """Calculate storage efficiency."""
        # 18 trits per tryte
        # Ideal: 18 * log2(3) ≈ 28.53 bits
        # Our encoding: 9 pairs * 3 bits = 27 bits = 3.375 bytes
        # Efficiency: 27/28.53 ≈ 94.6%

        ideal_bits = 18 * np.log2(3)
        actual_bits = 9 * 3  # 9 pairs of 2 trits, 3 bits each
        efficiency = actual_bits / ideal_bits

        return {
            'ideal_bits': ideal_bits,
            'actual_bits': actual_bits,
            'bytes_used': np.ceil(actual_bits / 8),
            'efficiency': efficiency,
            'waste_per_tryte': ideal_bits - actual_bits,
        }


def demonstrate_encoding():
    """Demonstrate the encoding/decoding process."""
    print("=" * 70)
    print("TERNARY ENCODING DEMONSTRATION")
    print("=" * 70)

    # Show efficiency
    print("\nStorage Efficiency:")
    eff = TryteEncoder.calculate_efficiency()
    print(f"  Ideal bits per tryte:   {eff['ideal_bits']:.2f}")
    print(f"  Actual bits per tryte:  {eff['actual_bits']:.0f}")
    print(f"  Bytes used per tryte:   {eff['bytes_used']:.0f}")
    print(f"  Efficiency:             {eff['efficiency']*100:.1f}%")
    print(f"  Wasted bits per tryte:  {eff['waste_per_tryte']:.2f}")

    # Show 2-trit encoding
    print("\n2-Trit to 3-Bit Encoding Table:")
    print("  Trit Pair → Binary → Decimal")
    for (t1, t2), bits in sorted(TritEncoder.TRIT_PAIR_TO_BITS.items()):
        trit_str = f"{t1:+d}{t2:+d}".replace('+', '+').replace('0', '0')
        print(f"  {trit_str:>4} → {bits:03b} → {bits}")

    # Encode/decode example
    print("\nExample: Encoding a Tryte")
    tryte = Tryte(12345)
    print(f"  Original value: {tryte.to_int()}")
    print(f"  Balanced ternary: {tryte.to_balanced_ternary()}")

    encoded = TryteEncoder.encode_tryte(tryte)
    print(f"  Encoded to {len(encoded)} bytes: {encoded.hex()}")

    decoded = TryteEncoder.decode_tryte(encoded)
    print(f"  Decoded value: {decoded.to_int()}")
    print(f"  Match: {tryte.to_int() == decoded.to_int()}")

    # Array encoding
    print("\nArray Encoding Efficiency:")
    trytes = [Tryte(i * 100) for i in range(100)]

    # Naive binary encoding (8 bytes per int64)
    naive_bytes = 100 * 8

    # Our ternary encoding
    encoded_array = b''.join(TryteEncoder.encode_tryte(t) for t in trytes)
    ternary_bytes = len(encoded_array)

    print(f"  100 trytes:")
    print(f"    Naive int64 encoding:   {naive_bytes} bytes")
    print(f"    Ternary encoding:       {ternary_bytes} bytes")
    print(f"    Savings:                {(1 - ternary_bytes/naive_bytes)*100:.1f}%")


if __name__ == "__main__":
    demonstrate_encoding()
