#ifndef TRYTE_HPP
#define TRYTE_HPP

#include "trit.hpp"
#include <array>
#include <vector>
#include <string>
#include <cstdint>

namespace ternary {

/**
 * @brief Tryte - 18 trit word (as in Setun computer)
 *
 * A tryte is a word consisting of 18 trits, matching the Setun architecture.
 * This provides a range of 3^18 = 387,420,489 possible values
 * (-193,710,244 to +193,710,244 in balanced ternary).
 *
 * Can be used for both data and instructions.
 */
class Tryte {
public:
    static constexpr size_t TRIT_COUNT = 18;

private:
    std::array<Trit, TRIT_COUNT> trits_;

public:
    // Constructors
    Tryte();
    explicit Tryte(int64_t value);
    explicit Tryte(const std::string& tritString);
    explicit Tryte(const std::array<Trit, TRIT_COUNT>& trits);

    // Getters/Setters
    Trit getTrit(size_t index) const;
    void setTrit(size_t index, Trit value);
    const std::array<Trit, TRIT_COUNT>& getTrits() const { return trits_; }

    // Conversion
    int64_t toInt64() const;
    std::string toString() const;
    std::string toBalancedTernaryString() const;

    // Arithmetic operations
    Tryte operator+(const Tryte& other) const;
    Tryte operator-(const Tryte& other) const;
    Tryte operator*(const Tryte& other) const;
    Tryte operator-() const; // Negation

    // Bitwise (tritwise) operations
    Tryte logicAnd(const Tryte& other) const;
    Tryte logicOr(const Tryte& other) const;
    Tryte logicNot() const;
    Tryte logicXor(const Tryte& other) const;

    // Shift operations
    Tryte shiftLeft(size_t positions) const;
    Tryte shiftRight(size_t positions) const;

    // Comparison
    bool operator==(const Tryte& other) const;
    bool operator!=(const Tryte& other) const;
    bool operator<(const Tryte& other) const;
    bool operator>(const Tryte& other) const;
    bool operator<=(const Tryte& other) const;
    bool operator>=(const Tryte& other) const;

    // Assignment
    Tryte& operator=(const Tryte& other) = default;

    // Stream output
    friend std::ostream& operator<<(std::ostream& os, const Tryte& tryte);

    // Static helpers
    static Tryte fromInt64(int64_t value);
    static Tryte fromString(const std::string& str);
    static int64_t maxValue();
    static int64_t minValue();
};

} // namespace ternary

#endif // TRYTE_HPP
