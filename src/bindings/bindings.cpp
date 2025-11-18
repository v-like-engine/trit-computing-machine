#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "../core/trit.hpp"
#include "../core/tryte.hpp"
#include "../core/memory.hpp"
#include "../core/processor.hpp"

namespace py = pybind11;
using namespace ternary;

PYBIND11_MODULE(_ternary_core, m) {
    m.doc() = "Ternary computing simulation core - Python bindings";

    // Trit class
    py::class_<Trit>(m, "Trit", "Balanced ternary digit (-1, 0, +1)")
        .def(py::init<>(), "Construct a Trit with value 0")
        .def(py::init<int>(), "Construct a Trit from an integer",
             py::arg("value"))
        .def(py::init<Trit::Value>(), "Construct a Trit from a Value enum",
             py::arg("value"))

        // Properties
        .def_property_readonly("value", &Trit::getValue, "Get the trit value")
        .def("to_int", &Trit::toInt, "Convert to integer (-1, 0, or 1)")
        .def("to_char", &Trit::toChar, "Convert to character (-, 0, or +)")

        // Arithmetic operations
        .def(py::self + py::self, "Add two trits")
        .def(py::self - py::self, "Subtract two trits")
        .def(py::self * py::self, "Multiply two trits")
        .def(-py::self, "Negate a trit")

        // Logic operations
        .def("logic_and", &Trit::logicAnd, "Ternary AND (conjunction)",
             py::arg("other"))
        .def("logic_or", &Trit::logicOr, "Ternary OR (disjunction)",
             py::arg("other"))
        .def("logic_not", &Trit::logicNot, "Ternary NOT (negation)")
        .def("logic_xor", &Trit::logicXor, "Ternary XOR",
             py::arg("other"))

        // Comparison
        .def(py::self == py::self, "Equal comparison")
        .def(py::self != py::self, "Not equal comparison")
        .def(py::self < py::self, "Less than comparison")
        .def(py::self > py::self, "Greater than comparison")
        .def(py::self <= py::self, "Less than or equal comparison")
        .def(py::self >= py::self, "Greater than or equal comparison")

        // String representation
        .def("__repr__", [](const Trit& t) {
            return "Trit(" + std::string(1, t.toChar()) + ")";
        })
        .def("__str__", [](const Trit& t) {
            return std::string(1, t.toChar());
        })

        // Static methods
        .def_static("from_char", &Trit::fromChar, "Create Trit from character",
                    py::arg("c"))
        .def_static("from_bool", &Trit::fromBool, "Create Trit from boolean",
                    py::arg("b"));

    // Trit::Value enum
    py::enum_<Trit::Value>(m, "TritValue", "Trit value enumeration")
        .value("MINUS", Trit::MINUS)
        .value("ZERO", Trit::ZERO)
        .value("PLUS", Trit::PLUS)
        .export_values();

    // Tryte class
    py::class_<Tryte>(m, "Tryte", "18-trit word (Setun architecture)")
        .def(py::init<>(), "Construct a Tryte with value 0")
        .def(py::init<int64_t>(), "Construct a Tryte from an integer",
             py::arg("value"))
        .def(py::init<const std::string&>(), "Construct a Tryte from a string",
             py::arg("trit_string"))

        // Getters/Setters
        .def("get_trit", &Tryte::getTrit, "Get trit at index",
             py::arg("index"))
        .def("set_trit", &Tryte::setTrit, "Set trit at index",
             py::arg("index"), py::arg("value"))

        // Conversion
        .def("to_int", &Tryte::toInt64, "Convert to 64-bit integer")
        .def("to_string", &Tryte::toString, "Convert to string representation")
        .def("to_balanced_ternary", &Tryte::toBalancedTernaryString,
             "Get balanced ternary string")

        // Arithmetic operations
        .def(py::self + py::self, "Add two trytes")
        .def(py::self - py::self, "Subtract two trytes")
        .def(py::self * py::self, "Multiply two trytes")
        .def(-py::self, "Negate a tryte")

        // Logic operations
        .def("logic_and", &Tryte::logicAnd, "Tritwise AND",
             py::arg("other"))
        .def("logic_or", &Tryte::logicOr, "Tritwise OR",
             py::arg("other"))
        .def("logic_not", &Tryte::logicNot, "Tritwise NOT")
        .def("logic_xor", &Tryte::logicXor, "Tritwise XOR",
             py::arg("other"))

        // Shift operations
        .def("shift_left", &Tryte::shiftLeft, "Shift left",
             py::arg("positions"))
        .def("shift_right", &Tryte::shiftRight, "Shift right",
             py::arg("positions"))

        // Comparison
        .def(py::self == py::self, "Equal comparison")
        .def(py::self != py::self, "Not equal comparison")
        .def(py::self < py::self, "Less than comparison")
        .def(py::self > py::self, "Greater than comparison")
        .def(py::self <= py::self, "Less than or equal comparison")
        .def(py::self >= py::self, "Greater than or equal comparison")

        // String representation
        .def("__repr__", [](const Tryte& t) {
            return "Tryte(" + std::to_string(t.toInt64()) + ", '" +
                   t.toBalancedTernaryString() + "')";
        })
        .def("__str__", &Tryte::toBalancedTernaryString)

        // Static methods
        .def_static("from_int", &Tryte::fromInt64, "Create Tryte from integer",
                    py::arg("value"))
        .def_static("from_string", &Tryte::fromString, "Create Tryte from string",
                    py::arg("str"))
        .def_static("max_value", &Tryte::maxValue, "Get maximum Tryte value")
        .def_static("min_value", &Tryte::minValue, "Get minimum Tryte value")

        // Class attributes
        .def_readonly_static("TRIT_COUNT", &Tryte::TRIT_COUNT,
                            "Number of trits in a tryte");

    // Memory class
    py::class_<Memory>(m, "Memory", "Ternary memory system")
        .def(py::init<size_t>(), "Construct memory with given size",
             py::arg("size"))

        // Memory operations
        .def("read", &Memory::read, "Read tryte from address",
             py::arg("address"))
        .def("write", &Memory::write, "Write tryte to address",
             py::arg("address"), py::arg("value"))

        // Bulk operations
        .def("clear", &Memory::clear, "Clear all memory to zero")
        .def("fill", &Memory::fill, "Fill memory with value",
             py::arg("value"))
        .def("read_block", &Memory::readBlock, "Read block of memory",
             py::arg("start"), py::arg("length"))
        .def("write_block", &Memory::writeBlock, "Write block of memory",
             py::arg("start"), py::arg("data"))

        // Properties
        .def_property_readonly("size", &Memory::size, "Get memory size")

        // Debug
        .def("dump", &Memory::dump, "Dump memory contents",
             py::arg("start") = 0, py::arg("length") = 0)

        // String representation
        .def("__repr__", [](const Memory& m) {
            return "Memory(size=" + std::to_string(m.size()) + ")";
        });

    // SparseMemory class
    py::class_<SparseMemory>(m, "SparseMemory", "Sparse ternary memory")
        .def(py::init<>(), "Construct sparse memory with zero default")
        .def(py::init<const Tryte&>(), "Construct sparse memory with default value",
             py::arg("default_value"))

        .def("read", &SparseMemory::read, "Read tryte from address",
             py::arg("address"))
        .def("write", &SparseMemory::write, "Write tryte to address",
             py::arg("address"), py::arg("value"))
        .def("clear", &SparseMemory::clear, "Clear all memory")
        .def("used_cells", &SparseMemory::usedCells,
             "Get number of used memory cells")

        .def("__repr__", [](const SparseMemory& m) {
            return "SparseMemory(used_cells=" +
                   std::to_string(m.usedCells()) + ")";
        });

    // Processor class
    py::class_<Processor>(m, "Processor", "Ternary processor simulation")
        .def(py::init<Memory&>(), "Construct processor with memory",
             py::arg("memory"))

        // Register access
        .def_property("accumulator",
                     &Processor::getAccumulator,
                     &Processor::setAccumulator,
                     "Accumulator register")
        .def_property("index_register",
                     &Processor::getIndexRegister,
                     &Processor::setIndexRegister,
                     "Index register")
        .def_property("program_counter",
                     &Processor::getProgramCounter,
                     &Processor::setProgramCounter,
                     "Program counter")

        // Read-only properties
        .def_property_readonly("cycle_count", &Processor::getCycleCount,
                              "Instruction cycle count")
        .def_property_readonly("halted", &Processor::isHalted,
                              "Check if processor is halted")

        // Execution
        .def("reset", &Processor::reset, "Reset processor state")
        .def("step", &Processor::step, "Execute one instruction")
        .def("run", &Processor::run, "Run until halt or max cycles",
             py::arg("max_cycles") = 10000)

        // Status
        .def("get_status", &Processor::getStatus, "Get processor status string")
        .def("__repr__", [](const Processor& p) {
            return "Processor(pc=" + std::to_string(p.getProgramCounter()) +
                   ", cycles=" + std::to_string(p.getCycleCount()) +
                   ", halted=" + (p.isHalted() ? "True" : "False") + ")";
        });

    // Processor::Opcode enum
    py::enum_<Processor::Opcode>(m, "Opcode", "Processor instruction opcodes")
        .value("NOP", Processor::NOP)
        .value("LOAD", Processor::LOAD)
        .value("STORE", Processor::STORE)
        .value("ADD", Processor::ADD)
        .value("SUB", Processor::SUB)
        .value("MUL", Processor::MUL)
        .value("NEG", Processor::NEG)
        .value("AND", Processor::AND)
        .value("OR", Processor::OR)
        .value("NOT", Processor::NOT)
        .value("XOR", Processor::XOR)
        .value("SHIFT_L", Processor::SHIFT_L)
        .value("SHIFT_R", Processor::SHIFT_R)
        .value("JUMP", Processor::JUMP)
        .value("JUMP_POS", Processor::JUMP_POS)
        .value("JUMP_ZERO", Processor::JUMP_ZERO)
        .value("JUMP_NEG", Processor::JUMP_NEG)
        .value("HALT", Processor::HALT)
        .export_values();

    // Instruction class
    py::class_<Instruction>(m, "Instruction", "Processor instruction")
        .def(py::init<>(), "Construct NOP instruction")
        .def(py::init<Processor::Opcode, size_t>(),
             "Construct instruction with opcode and address",
             py::arg("opcode"), py::arg("address") = 0)

        .def_readwrite("opcode", &Instruction::opcode, "Instruction opcode")
        .def_readwrite("address", &Instruction::address, "Instruction address")

        .def("encode", &Instruction::encode, "Encode instruction to tryte")
        .def_static("decode", &Instruction::decode, "Decode tryte to instruction",
                    py::arg("tryte"))

        .def("to_string", &Instruction::toString, "Get string representation")
        .def("__repr__", [](const Instruction& i) {
            return "Instruction(" + i.toString() + ")";
        })
        .def("__str__", &Instruction::toString);
}
