#include "memory.hpp"
#include <sstream>
#include <iomanip>

namespace ternary {

// Memory implementation
Memory::Memory(size_t size) : size_(size) {
    storage_.resize(size, Tryte());
}

Tryte Memory::read(size_t address) const {
    if (address >= size_) {
        throw std::out_of_range("Memory address out of range");
    }
    return storage_[address];
}

void Memory::write(size_t address, const Tryte& value) {
    if (address >= size_) {
        throw std::out_of_range("Memory address out of range");
    }
    storage_[address] = value;
}

void Memory::clear() {
    std::fill(storage_.begin(), storage_.end(), Tryte());
}

void Memory::fill(const Tryte& value) {
    std::fill(storage_.begin(), storage_.end(), value);
}

std::vector<Tryte> Memory::readBlock(size_t start, size_t length) const {
    if (start + length > size_) {
        throw std::out_of_range("Memory block out of range");
    }
    return std::vector<Tryte>(storage_.begin() + start,
                               storage_.begin() + start + length);
}

void Memory::writeBlock(size_t start, const std::vector<Tryte>& data) {
    if (start + data.size() > size_) {
        throw std::out_of_range("Memory block out of range");
    }
    std::copy(data.begin(), data.end(), storage_.begin() + start);
}

std::string Memory::dump(size_t start, size_t length) const {
    if (length == 0) length = size_;
    if (start + length > size_) length = size_ - start;

    std::ostringstream oss;
    oss << "Memory Dump (addresses " << start << " to " << (start + length - 1) << "):\n";

    for (size_t i = start; i < start + length; ++i) {
        oss << std::setw(6) << i << ": "
            << storage_[i].toBalancedTernaryString()
            << " (" << std::setw(12) << storage_[i].toInt64() << ")\n";
    }

    return oss.str();
}

// SparseMemory implementation
SparseMemory::SparseMemory() : defaultValue_(Tryte()) {}

SparseMemory::SparseMemory(const Tryte& defaultValue)
    : defaultValue_(defaultValue) {}

Tryte SparseMemory::read(size_t address) const {
    auto it = storage_.find(address);
    if (it != storage_.end()) {
        return it->second;
    }
    return defaultValue_;
}

void SparseMemory::write(size_t address, const Tryte& value) {
    if (value == defaultValue_) {
        storage_.erase(address);
    } else {
        storage_[address] = value;
    }
}

void SparseMemory::clear() {
    storage_.clear();
}

} // namespace ternary
