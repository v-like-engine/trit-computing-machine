#include "tryte.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace ternary {

Tryte::Tryte() {
    trits_.fill(Trit(Trit::ZERO));
}

Tryte::Tryte(int64_t value) {
    trits_.fill(Trit(Trit::ZERO));

    // Convert to balanced ternary
    bool isNegative = value < 0;
    int64_t absValue = std::abs(value);

    for (size_t i = 0; i < TRIT_COUNT && absValue > 0; ++i) {
        int remainder = absValue % 3;
        absValue /= 3;

        // Handle balanced ternary: use -1, 0, 1 instead of 0, 1, 2
        if (remainder == 2) {
            trits_[i] = Trit(Trit::PLUS);
            absValue++; // Carry
        } else if (remainder == 1) {
            trits_[i] = Trit(Trit::PLUS);
        } else {
            trits_[i] = Trit(Trit::ZERO);
        }
    }

    if (isNegative) {
        *this = -*this;
    }
}

Tryte::Tryte(const std::string& tritString) {
    trits_.fill(Trit(Trit::ZERO));

    size_t len = std::min(tritString.length(), TRIT_COUNT);
    for (size_t i = 0; i < len; ++i) {
        // Read from right to left (LST at index 0)
        trits_[i] = Trit::fromChar(tritString[tritString.length() - 1 - i]);
    }
}

Tryte::Tryte(const std::array<Trit, TRIT_COUNT>& trits) : trits_(trits) {}

Trit Tryte::getTrit(size_t index) const {
    if (index >= TRIT_COUNT) {
        throw std::out_of_range("Trit index out of range");
    }
    return trits_[index];
}

void Tryte::setTrit(size_t index, Trit value) {
    if (index >= TRIT_COUNT) {
        throw std::out_of_range("Trit index out of range");
    }
    trits_[index] = value;
}

int64_t Tryte::toInt64() const {
    int64_t result = 0;
    int64_t power = 1;

    for (size_t i = 0; i < TRIT_COUNT; ++i) {
        result += trits_[i].toInt() * power;
        power *= 3;
    }

    return result;
}

std::string Tryte::toString() const {
    std::ostringstream oss;
    oss << "Tryte(" << toInt64() << ", " << toBalancedTernaryString() << ")";
    return oss.str();
}

std::string Tryte::toBalancedTernaryString() const {
    std::string result;
    for (int i = TRIT_COUNT - 1; i >= 0; --i) {
        result += trits_[i].toChar();
    }
    return result;
}

// Arithmetic operations
Tryte Tryte::operator+(const Tryte& other) const {
    Tryte result;
    Trit carry(Trit::ZERO);

    for (size_t i = 0; i < TRIT_COUNT; ++i) {
        // Add trits and carry
        int sum = trits_[i].toInt() + other.trits_[i].toInt() + carry.toInt();

        // Convert to balanced ternary with carry
        if (sum > 1) {
            result.trits_[i] = Trit(sum - 3);
            carry = Trit(Trit::PLUS);
        } else if (sum < -1) {
            result.trits_[i] = Trit(sum + 3);
            carry = Trit(Trit::MINUS);
        } else {
            result.trits_[i] = Trit(sum);
            carry = Trit(Trit::ZERO);
        }
    }

    return result;
}

Tryte Tryte::operator-(const Tryte& other) const {
    return *this + (-other);
}

Tryte Tryte::operator*(const Tryte& other) const {
    // Simple multiplication: convert to int64, multiply, convert back
    // For production, would implement proper ternary multiplication
    return Tryte(this->toInt64() * other.toInt64());
}

Tryte Tryte::operator-() const {
    Tryte result;
    for (size_t i = 0; i < TRIT_COUNT; ++i) {
        result.trits_[i] = -trits_[i];
    }
    return result;
}

// Logic operations
Tryte Tryte::logicAnd(const Tryte& other) const {
    Tryte result;
    for (size_t i = 0; i < TRIT_COUNT; ++i) {
        result.trits_[i] = trits_[i].logicAnd(other.trits_[i]);
    }
    return result;
}

Tryte Tryte::logicOr(const Tryte& other) const {
    Tryte result;
    for (size_t i = 0; i < TRIT_COUNT; ++i) {
        result.trits_[i] = trits_[i].logicOr(other.trits_[i]);
    }
    return result;
}

Tryte Tryte::logicNot() const {
    Tryte result;
    for (size_t i = 0; i < TRIT_COUNT; ++i) {
        result.trits_[i] = trits_[i].logicNot();
    }
    return result;
}

Tryte Tryte::logicXor(const Tryte& other) const {
    Tryte result;
    for (size_t i = 0; i < TRIT_COUNT; ++i) {
        result.trits_[i] = trits_[i].logicXor(other.trits_[i]);
    }
    return result;
}

// Shift operations
Tryte Tryte::shiftLeft(size_t positions) const {
    Tryte result;
    for (size_t i = positions; i < TRIT_COUNT; ++i) {
        result.trits_[i] = trits_[i - positions];
    }
    return result;
}

Tryte Tryte::shiftRight(size_t positions) const {
    Tryte result;
    for (size_t i = 0; i + positions < TRIT_COUNT; ++i) {
        result.trits_[i] = trits_[i + positions];
    }
    return result;
}

// Comparison
bool Tryte::operator==(const Tryte& other) const {
    return trits_ == other.trits_;
}

bool Tryte::operator!=(const Tryte& other) const {
    return !(*this == other);
}

bool Tryte::operator<(const Tryte& other) const {
    return toInt64() < other.toInt64();
}

bool Tryte::operator>(const Tryte& other) const {
    return toInt64() > other.toInt64();
}

bool Tryte::operator<=(const Tryte& other) const {
    return toInt64() <= other.toInt64();
}

bool Tryte::operator>=(const Tryte& other) const {
    return toInt64() >= other.toInt64();
}

std::ostream& operator<<(std::ostream& os, const Tryte& tryte) {
    os << tryte.toBalancedTernaryString();
    return os;
}

Tryte Tryte::fromInt64(int64_t value) {
    return Tryte(value);
}

Tryte Tryte::fromString(const std::string& str) {
    return Tryte(str);
}

int64_t Tryte::maxValue() {
    // 3^18 / 2 ≈ 193,710,244
    int64_t result = 0;
    int64_t power = 1;
    for (size_t i = 0; i < TRIT_COUNT; ++i) {
        result += power;
        power *= 3;
    }
    return result;
}

int64_t Tryte::minValue() {
    return -maxValue();
}

} // namespace ternary
