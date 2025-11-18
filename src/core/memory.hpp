#ifndef MEMORY_HPP
#define MEMORY_HPP

#include "tryte.hpp"
#include <vector>
#include <map>

namespace ternary {

/**
 * @brief Ternary Memory System
 *
 * Simulates memory for ternary computer systems.
 * Setun had 81 words of fast memory + 1944 words on magnetic drum.
 * This implementation is more flexible and modern.
 */
class Memory {
private:
    std::vector<Tryte> storage_;
    size_t size_;

public:
    // Constructors
    explicit Memory(size_t size);

    // Memory operations
    Tryte read(size_t address) const;
    void write(size_t address, const Tryte& value);

    // Bulk operations
    void clear();
    void fill(const Tryte& value);
    std::vector<Tryte> readBlock(size_t start, size_t length) const;
    void writeBlock(size_t start, const std::vector<Tryte>& data);

    // Getters
    size_t size() const { return size_; }
    const std::vector<Tryte>& getStorage() const { return storage_; }

    // Dump memory for debugging
    std::string dump(size_t start = 0, size_t length = 0) const;
};

/**
 * @brief Sparse Memory - for large address spaces
 *
 * Uses a map to store only non-zero values, efficient for sparse usage.
 */
class SparseMemory {
private:
    std::map<size_t, Tryte> storage_;
    Tryte defaultValue_;

public:
    SparseMemory();
    explicit SparseMemory(const Tryte& defaultValue);

    Tryte read(size_t address) const;
    void write(size_t address, const Tryte& value);
    void clear();

    size_t usedCells() const { return storage_.size(); }
};

} // namespace ternary

#endif // MEMORY_HPP
