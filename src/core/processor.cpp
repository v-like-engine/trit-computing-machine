#include "processor.hpp"
#include <sstream>
#include <iostream>

namespace ternary {

Processor::Processor(Memory& memory)
    : accumulator_(Tryte()),
      indexRegister_(Tryte()),
      programCounter_(0),
      memory_(memory),
      halted_(false),
      cycleCount_(0) {}

void Processor::reset() {
    accumulator_ = Tryte();
    indexRegister_ = Tryte();
    programCounter_ = 0;
    halted_ = false;
    cycleCount_ = 0;
}

size_t Processor::resolveAddress(size_t baseAddress) const {
    // Add index register value to base address (if index register is used)
    int64_t indexValue = indexRegister_.toInt64();
    int64_t effectiveAddress = static_cast<int64_t>(baseAddress) + indexValue;

    // Ensure address is within valid range
    if (effectiveAddress < 0) effectiveAddress = 0;
    if (static_cast<size_t>(effectiveAddress) >= memory_.size()) {
        effectiveAddress = memory_.size() - 1;
    }

    return static_cast<size_t>(effectiveAddress);
}

void Processor::step() {
    if (halted_ || programCounter_ >= memory_.size()) {
        halted_ = true;
        return;
    }

    // Fetch instruction
    Tryte instructionWord = memory_.read(programCounter_);
    Instruction inst = Instruction::decode(instructionWord);

    // Execute
    programCounter_++;
    executeInstruction(inst.opcode, inst.address);

    cycleCount_++;
}

void Processor::run(size_t maxCycles) {
    while (!halted_ && cycleCount_ < maxCycles) {
        step();
    }
}

void Processor::executeInstruction(Opcode opcode, size_t address) {
    size_t effectiveAddr = resolveAddress(address);

    switch (opcode) {
        case NOP:
            // Do nothing
            break;

        case LOAD:
            accumulator_ = memory_.read(effectiveAddr);
            break;

        case STORE:
            memory_.write(effectiveAddr, accumulator_);
            break;

        case ADD:
            accumulator_ = accumulator_ + memory_.read(effectiveAddr);
            break;

        case SUB:
            accumulator_ = accumulator_ - memory_.read(effectiveAddr);
            break;

        case MUL:
            accumulator_ = accumulator_ * memory_.read(effectiveAddr);
            break;

        case NEG:
            accumulator_ = -accumulator_;
            break;

        case AND:
            accumulator_ = accumulator_.logicAnd(memory_.read(effectiveAddr));
            break;

        case OR:
            accumulator_ = accumulator_.logicOr(memory_.read(effectiveAddr));
            break;

        case NOT:
            accumulator_ = accumulator_.logicNot();
            break;

        case XOR:
            accumulator_ = accumulator_.logicXor(memory_.read(effectiveAddr));
            break;

        case SHIFT_L:
            accumulator_ = accumulator_.shiftLeft(1);
            break;

        case SHIFT_R:
            accumulator_ = accumulator_.shiftRight(1);
            break;

        case JUMP:
            programCounter_ = effectiveAddr;
            break;

        case JUMP_POS:
            if (accumulator_.toInt64() > 0) {
                programCounter_ = effectiveAddr;
            }
            break;

        case JUMP_ZERO:
            if (accumulator_.toInt64() == 0) {
                programCounter_ = effectiveAddr;
            }
            break;

        case JUMP_NEG:
            if (accumulator_.toInt64() < 0) {
                programCounter_ = effectiveAddr;
            }
            break;

        case HALT:
            halted_ = true;
            break;

        default:
            // Unknown opcode, treat as NOP
            break;
    }
}

std::string Processor::getStatus() const {
    std::ostringstream oss;
    oss << "Processor Status:\n"
        << "  PC:    " << programCounter_ << "\n"
        << "  ACC:   " << accumulator_.toBalancedTernaryString()
        << " (" << accumulator_.toInt64() << ")\n"
        << "  IDX:   " << indexRegister_.toBalancedTernaryString()
        << " (" << indexRegister_.toInt64() << ")\n"
        << "  Cycles: " << cycleCount_ << "\n"
        << "  Halted: " << (halted_ ? "yes" : "no") << "\n";
    return oss.str();
}

// Instruction implementation
Tryte Instruction::encode() const {
    // Simplified encoding: opcode in high trits, address in low trits
    // In real Setun, encoding was more complex
    int64_t encoded = (static_cast<int64_t>(opcode) << 10) | (address & 0x3FF);
    return Tryte(encoded);
}

Instruction Instruction::decode(const Tryte& tryte) {
    int64_t value = tryte.toInt64();
    Processor::Opcode opcode = static_cast<Processor::Opcode>((value >> 10) & 0xFF);
    size_t address = value & 0x3FF;
    return Instruction(opcode, address);
}

std::string Instruction::toString() const {
    std::ostringstream oss;

    const char* opcodeNames[] = {
        "NOP", "LOAD", "STORE", "ADD", "SUB", "MUL", "NEG",
        "AND", "OR", "NOT", "XOR", "SHIFT_L", "SHIFT_R",
        "JUMP", "JUMP_POS", "JUMP_ZERO", "JUMP_NEG", "HALT"
    };

    if (opcode >= 0 && opcode <= Processor::HALT) {
        oss << opcodeNames[opcode];
    } else {
        oss << "UNKNOWN";
    }

    oss << " " << address;
    return oss.str();
}

} // namespace ternary
