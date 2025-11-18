#include "trit.hpp"

namespace ternary {

Trit::Trit(int val) {
    if (val < -1) value_ = MINUS;
    else if (val > 1) value_ = PLUS;
    else value_ = static_cast<Value>(val);
}

char Trit::toChar() const {
    switch (value_) {
        case MINUS: return '-';
        case ZERO:  return '0';
        case PLUS:  return '+';
        default:    return '?';
    }
}

Trit Trit::fromChar(char c) {
    switch (c) {
        case '-': case 'T': return Trit(MINUS);
        case '0': case 'U': return Trit(ZERO);
        case '+': case '1': return Trit(PLUS);
        default: throw std::invalid_argument("Invalid trit character");
    }
}

Trit Trit::fromBool(bool b) {
    return Trit(b ? PLUS : MINUS);
}

// Arithmetic operations
Trit Trit::operator+(const Trit& other) const {
    int sum = toInt() + other.toInt();
    return Trit(sum);
}

Trit Trit::operator-(const Trit& other) const {
    int diff = toInt() - other.toInt();
    return Trit(diff);
}

Trit Trit::operator*(const Trit& other) const {
    int prod = toInt() * other.toInt();
    return Trit(prod);
}

Trit Trit::operator-() const {
    return Trit(-toInt());
}

// Logic operations
Trit Trit::logicAnd(const Trit& other) const {
    // Conjunction: returns minimum (most negative)
    return Trit(std::min(toInt(), other.toInt()));
}

Trit Trit::logicOr(const Trit& other) const {
    // Disjunction: returns maximum (most positive)
    return Trit(std::max(toInt(), other.toInt()));
}

Trit Trit::logicNot() const {
    return -(*this);
}

Trit Trit::logicXor(const Trit& other) const {
    // XOR: returns sum modulo 3 in balanced ternary
    int result = toInt() + other.toInt();
    if (result > 1) return Trit(MINUS);
    if (result < -1) return Trit(PLUS);
    return Trit(result);
}

std::ostream& operator<<(std::ostream& os, const Trit& trit) {
    os << trit.toChar();
    return os;
}

} // namespace ternary
