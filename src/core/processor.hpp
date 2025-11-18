#ifndef PROCESSOR_HPP
#define PROCESSOR_HPP

#include "tryte.hpp"
#include "memory.hpp"
#include <functional>
#include <string>

namespace ternary {

/**
 * @brief Ternary Processor
 *
 * Simplified simulation of a ternary processor inspired by Setun.
 * Features:
 * - Accumulator register (ACC)
 * - Index register (IDX)
 * - Program counter (PC)
 * - Instruction set inspired by Setun's 24 instructions
 */
class Processor {
public:
    // Instruction opcodes (simplified)
    enum Opcode {
        NOP = 0,    // No operation
        LOAD,       // Load from memory to ACC
        STORE,      // Store ACC to memory
        ADD,        // Add memory to ACC
        SUB,        // Subtract memory from ACC
        MUL,        // Multiply ACC by memory
        NEG,        // Negate ACC
        AND,        // Logic AND
        OR,         // Logic OR
        NOT,        // Logic NOT
        XOR,        // Logic XOR
        SHIFT_L,    // Shift left
        SHIFT_R,    // Shift right
        JUMP,       // Unconditional jump
        JUMP_POS,   // Jump if ACC positive
        JUMP_ZERO,  // Jump if ACC zero
        JUMP_NEG,   // Jump if ACC negative
        HALT        // Halt execution
    };

private:
    Tryte accumulator_;      // ACC register
    Tryte indexRegister_;    // IDX register
    size_t programCounter_;  // PC
    Memory& memory_;         // Reference to memory
    bool halted_;           // Halt flag
    size_t cycleCount_;     // Instruction cycle counter

public:
    // Constructor
    explicit Processor(Memory& memory);

    // Register access
    Tryte getAccumulator() const { return accumulator_; }
    void setAccumulator(const Tryte& value) { accumulator_ = value; }

    Tryte getIndexRegister() const { return indexRegister_; }
    void setIndexRegister(const Tryte& value) { indexRegister_ = value; }

    size_t getProgramCounter() const { return programCounter_; }
    void setProgramCounter(size_t pc) { programCounter_ = pc; }

    size_t getCycleCount() const { return cycleCount_; }
    bool isHalted() const { return halted_; }

    // Execution
    void reset();
    void step();  // Execute one instruction
    void run(size_t maxCycles = 10000);  // Run until halt or max cycles

    // Instruction execution
    void executeInstruction(Opcode opcode, size_t address);

    // Status
    std::string getStatus() const;

private:
    size_t resolveAddress(size_t baseAddress) const;
};

/**
 * @brief Instruction decoder/encoder
 */
struct Instruction {
    Processor::Opcode opcode;
    size_t address;

    Instruction(Processor::Opcode op = Processor::NOP, size_t addr = 0)
        : opcode(op), address(addr) {}

    // Encode instruction to Tryte (simplified encoding)
    Tryte encode() const;

    // Decode Tryte to instruction
    static Instruction decode(const Tryte& tryte);

    std::string toString() const;
};

} // namespace ternary

#endif // PROCESSOR_HPP
