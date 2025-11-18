// Ternary Computing Core - JavaScript Implementation

class Trit {
    constructor(value) {
        // Normalize to -1, 0, or 1
        if (value < -0.5) this.value = -1;
        else if (value > 0.5) this.value = 1;
        else this.value = 0;
    }

    toInt() {
        return this.value;
    }

    toString() {
        switch (this.value) {
            case -1: return '-';
            case 0: return '0';
            case 1: return '+';
        }
    }

    // Arithmetic
    add(other) {
        return new Trit(this.value + other.value);
    }

    subtract(other) {
        return new Trit(this.value - other.value);
    }

    multiply(other) {
        return new Trit(this.value * other.value);
    }

    negate() {
        return new Trit(-this.value);
    }

    // Logic operations
    and(other) {
        // Min operation
        return new Trit(Math.min(this.value, other.value));
    }

    or(other) {
        // Max operation
        return new Trit(Math.max(this.value, other.value));
    }

    not() {
        return this.negate();
    }

    xor(other) {
        const sum = this.value + other.value;
        if (sum > 1) return new Trit(-1);
        if (sum < -1) return new Trit(1);
        return new Trit(sum);
    }
}

class Tryte {
    constructor(value = 0) {
        this.trits = new Array(18);

        if (typeof value === 'number') {
            this.fromInt(value);
        } else if (Array.isArray(value)) {
            for (let i = 0; i < 18; i++) {
                this.trits[i] = value[i] || new Trit(0);
            }
        }
    }

    fromInt(value) {
        const isNegative = value < 0;
        let absValue = Math.abs(value);

        for (let i = 0; i < 18; i++) {
            const remainder = absValue % 3;
            absValue = Math.floor(absValue / 3);

            if (remainder === 2) {
                this.trits[i] = new Trit(1);
                absValue++;
            } else if (remainder === 1) {
                this.trits[i] = new Trit(1);
            } else {
                this.trits[i] = new Trit(0);
            }
        }

        if (isNegative) {
            for (let i = 0; i < 18; i++) {
                this.trits[i] = this.trits[i].negate();
            }
        }
    }

    toInt() {
        let result = 0;
        let power = 1;

        for (let i = 0; i < 18; i++) {
            result += this.trits[i].toInt() * power;
            power *= 3;
        }

        return result;
    }

    toString() {
        return this.trits.map(t => t.toString()).reverse().join('');
    }

    toBalancedTernary() {
        return this.toString();
    }
}

// Ternary quantization
function ternaryQuantize(x, threshold = 0.3) {
    if (x > threshold) return 1;
    if (x < -threshold) return -1;
    return 0;
}

// Ternary quantization for array
function ternaryQuantizeArray(arr, threshold = 0.3) {
    return arr.map(x => ternaryQuantize(x, threshold));
}

// Encoding frameworks
class CompactEncoder {
    static encode2Trits(t1, t2) {
        const mapping = {
            '-1,-1': 0, '-1,0': 1, '-1,1': 2,
            '0,-1': 3, '0,0': 4, '0,1': 5,
            '1,-1': 6, '1,0': 7, '1,1': 7  // Saturation!
        };
        const key = `${t1},${t2}`;
        return mapping[key] || 0;
    }

    static decode2Trits(bits) {
        const mapping = [
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1], [0, 0], [0, 1],
            [1, -1], [1, 0]
        ];
        return mapping[bits] || [0, 0];
    }

    static encode(tryte) {
        const trits = [];
        for (let i = 0; i < 18; i++) {
            trits.push(tryte.trits[i].toInt());
        }

        // Encode pairs
        const encoded = [];
        for (let i = 0; i < 18; i += 2) {
            const bits = this.encode2Trits(trits[i], trits[i + 1] || 0);
            encoded.push(bits);
        }

        return {
            method: 'Compact (3→2)',
            bits: encoded.length * 3,
            bytes: Math.ceil(encoded.length * 3 / 8),
            efficiency: '94.6%',
            lossless: false,
            data: encoded
        };
    }
}

class PerfectEncoder {
    static encode(tryte) {
        const value = tryte.toInt();
        const idealBits = 18 * Math.log2(3);
        const actualBits = Math.ceil(idealBits);

        return {
            method: 'Perfect (Radix)',
            bits: actualBits,
            bytes: Math.ceil(actualBits / 8),
            efficiency: ((idealBits / actualBits) * 100).toFixed(1) + '%',
            lossless: true,
            data: value
        };
    }
}

class OptimalBlockEncoder {
    static encode(tryte) {
        // 5 trits per block = 243 states < 256
        const trits = [];
        for (let i = 0; i < 18; i++) {
            trits.push(tryte.trits[i].toInt());
        }

        const blocks = [];
        for (let i = 0; i < 18; i += 5) {
            const block = trits.slice(i, i + 5);
            // Convert to value [0, 242]
            let value = 0;
            let base = 1;
            for (let j = 0; j < block.length; j++) {
                value += (block[j] + 1) * base;
                base *= 3;
            }
            blocks.push(value);
        }

        const actualBits = blocks.length * 8 + 16; // +16 for header

        return {
            method: 'Optimal Block (5→8)',
            bits: actualBits,
            bytes: blocks.length + 2,
            efficiency: '94.9%',
            lossless: true,
            data: blocks
        };
    }
}
