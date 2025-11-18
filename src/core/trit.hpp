#ifndef TRIT_HPP
#define TRIT_HPP

#include <iostream>
#include <string>
#include <stdexcept>

namespace ternary {

/**
 * @brief Balanced Ternary Digit (Trit)
 *
 * Represents a single trit with three possible values:
 * - MINUS (-1): negative/false
 * - ZERO  ( 0): neutral/unknown
 * - PLUS  (+1): positive/true
 *
 * This is the foundation of balanced ternary computing,
 * as used in the Soviet Setun computer (1958).
 */
class Trit {
public:
    enum Value : int8_t {
        MINUS = -1,
        ZERO = 0,
        PLUS = 1
    };

private:
    Value value_;

public:
    // Constructors
    Trit() : value_(ZERO) {}
    explicit Trit(int val);
    Trit(Value val) : value_(val) {}

    // Getters
    Value getValue() const { return value_; }
    int toInt() const { return static_cast<int>(value_); }
    char toChar() const;

    // Arithmetic operations
    Trit operator+(const Trit& other) const;
    Trit operator-(const Trit& other) const;
    Trit operator*(const Trit& other) const;
    Trit operator-() const; // Negation

    // Logic operations (ternary logic)
    Trit logicAnd(const Trit& other) const;  // Conjunction (min)
    Trit logicOr(const Trit& other) const;   // Disjunction (max)
    Trit logicNot() const;                    // Negation
    Trit logicXor(const Trit& other) const;  // Exclusive OR

    // Comparison
    bool operator==(const Trit& other) const { return value_ == other.value_; }
    bool operator!=(const Trit& other) const { return value_ != other.value_; }
    bool operator<(const Trit& other) const { return value_ < other.value_; }
    bool operator>(const Trit& other) const { return value_ > other.value_; }
    bool operator<=(const Trit& other) const { return value_ <= other.value_; }
    bool operator>=(const Trit& other) const { return value_ >= other.value_; }

    // Assignment
    Trit& operator=(const Trit& other) = default;

    // Stream output
    friend std::ostream& operator<<(std::ostream& os, const Trit& trit);

    // Static helpers
    static Trit fromChar(char c);
    static Trit fromBool(bool b); // true -> PLUS, false -> MINUS
};

} // namespace ternary

#endif // TRIT_HPP
