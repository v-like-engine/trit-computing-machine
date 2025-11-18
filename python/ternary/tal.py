"""
TAL - Ternary Assembly Language

Higher-level assembly with macros, functions, and conveniences.
Compiles down to basic ternary assembly.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ternary import Opcode


@dataclass
class TALMacro:
    """Macro definition."""
    name: str
    params: List[str]
    body: List[str]


class TALCompiler:
    """
    Compiler for Ternary Assembly Language (TAL).

    Features:
    - Functions with local variables
    - Macros
    - Stack operations
    - Arithmetic shortcuts
    - Comments and better formatting
    """

    def __init__(self):
        self.macros: Dict[str, TALMacro] = {}
        self.functions: Dict[str, int] = {}
        self.variables: Dict[str, int] = {}
        self.next_var_address = 200  # Start variables at address 200
        self.output_lines: List[str] = []

        # Built-in macros
        self._define_builtins()

    def _define_builtins(self):
        """Define built-in macros."""

        # PUSH - push accumulator to memory
        self.add_macro("PUSH", ["addr"], [
            "STORE {addr}"
        ])

        # POP - pop from memory to accumulator
        self.add_macro("POP", ["addr"], [
            "LOAD {addr}"
        ])

        # MOV - move between memory locations
        self.add_macro("MOV", ["src", "dst"], [
            "LOAD {src}",
            "STORE {dst}"
        ])

        # INC - increment memory location
        self.add_macro("INC", ["addr"], [
            "LOAD {addr}",
            "ADD __one__",
            "STORE {addr}"
        ])

        # DEC - decrement memory location
        self.add_macro("DEC", ["addr"], [
            "LOAD {addr}",
            "SUB __one__",
            "STORE {addr}"
        ])

        # CMP - compare two values (result in ACC)
        self.add_macro("CMP", ["addr1", "addr2"], [
            "LOAD {addr1}",
            "SUB {addr2}"
        ])

        # SWAP - swap two memory locations
        self.add_macro("SWAP", ["addr1", "addr2"], [
            "LOAD {addr1}",
            "STORE __temp__",
            "LOAD {addr2}",
            "STORE {addr1}",
            "LOAD __temp__",
            "STORE {addr2}"
        ])

        # CALL - function call (simplified, no stack)
        self.add_macro("CALL", ["func"], [
            "JUMP {func}"
        ])

        # RET - return from function
        self.add_macro("RET", [], [
            "JUMP __return__"
        ])

    def add_macro(self, name: str, params: List[str], body: List[str]):
        """Add a macro definition."""
        self.macros[name] = TALMacro(name, params, body)

    def expand_macro(self, name: str, args: List[str]) -> List[str]:
        """Expand a macro with given arguments."""
        if name not in self.macros:
            raise ValueError(f"Unknown macro: {name}")

        macro = self.macros[name]
        if len(args) != len(macro.params):
            raise ValueError(
                f"Macro {name} expects {len(macro.params)} args, got {len(args)}"
            )

        # Substitute parameters
        expanded = []
        for line in macro.body:
            for param, arg in zip(macro.params, args):
                line = line.replace(f"{{{param}}}", arg)
            expanded.append(line)

        return expanded

    def allocate_variable(self, name: str) -> int:
        """Allocate memory for a variable."""
        if name not in self.variables:
            self.variables[name] = self.next_var_address
            self.next_var_address += 1
        return self.variables[name]

    def compile(self, source: str) -> str:
        """
        Compile TAL source to basic assembly.

        Syntax:
        - .var name - declare variable
        - .macro name(param1, param2) - define macro
        - .endmacro - end macro
        - .function name - start function
        - .endfunction - end function
        - MACRO_NAME arg1, arg2 - call macro
        """
        lines = source.split('\n')
        output = []

        # Reserve addresses for special variables
        self.variables['__temp__'] = 190
        self.variables['__one__'] = 191
        self.variables['__return__'] = 192

        # Initialize constants
        output.append("; TAL Compiler Output")
        output.append("; Constants initialization")
        output.append("LOAD 0")
        output.append("STORE 190  ; __temp__")
        output.append("LOAD 0")
        output.append("ADD 1")  # This won't work directly, need data
        output.append("STORE 191  ; __one__")
        output.append("")

        in_macro = False
        in_function = False
        current_macro = None
        current_macro_params = []
        current_macro_body = []
        current_function = None

        for line_num, line in enumerate(lines, 1):
            # Remove comments
            line = line.split(';')[0].strip()
            if not line:
                continue

            # Directives
            if line.startswith('.'):
                parts = line[1:].split()
                directive = parts[0].lower()

                if directive == 'var':
                    # Variable declaration
                    var_name = parts[1]
                    addr = self.allocate_variable(var_name)
                    output.append(f"; Variable {var_name} at address {addr}")

                elif directive == 'macro':
                    # Start macro definition
                    in_macro = True
                    macro_def = ' '.join(parts[1:])
                    if '(' in macro_def:
                        name, params = macro_def.split('(')
                        params = params.rstrip(')').split(',')
                        current_macro = name.strip()
                        current_macro_params = [p.strip() for p in params if p.strip()]
                    else:
                        current_macro = macro_def.strip()
                        current_macro_params = []
                    current_macro_body = []

                elif directive == 'endmacro':
                    in_macro = False
                    self.add_macro(current_macro, current_macro_params, current_macro_body)
                    current_macro = None

                elif directive == 'function':
                    in_function = True
                    func_name = parts[1]
                    current_function = func_name
                    self.functions[func_name] = len(output)
                    output.append(f"{func_name}:")

                elif directive == 'endfunction':
                    in_function = False
                    current_function = None

                continue

            if in_macro:
                # Collect macro body
                current_macro_body.append(line)
                continue

            # Check if it's a macro call
            parts = line.split()
            if not parts:
                continue

            instruction = parts[0].upper()

            # Check for macro
            if instruction in self.macros:
                args = []
                if len(parts) > 1:
                    args = [arg.strip() for arg in ' '.join(parts[1:]).split(',')]

                # Expand macro
                expanded = self.expand_macro(instruction, args)
                for exp_line in expanded:
                    # Recursively process expanded lines
                    output.append(self._process_line(exp_line))
            else:
                # Regular instruction or label
                output.append(self._process_line(line))

        return '\n'.join(output)

    def _process_line(self, line: str) -> str:
        """Process a single line, substituting variable names."""
        # Replace variable names with addresses
        for var_name, addr in self.variables.items():
            if var_name in line and not line.startswith(';'):
                # Only replace whole words
                import re
                line = re.sub(r'\b' + var_name + r'\b', str(addr), line)

        return line


def tal_example():
    """Example TAL program."""
    return """
; TAL Example - Sum array elements

.var array_start
.var array_length
.var sum
.var counter
.var current_value

; Initialize variables
LOAD 0
STORE sum
STORE counter

; Set array parameters
LOAD 100
STORE array_start
LOAD 10
STORE array_length

; Main loop
loop_start:
    ; Load counter and check if done
    CMP counter, array_length
    JUMP_POS loop_end

    ; Calculate current address: array_start + counter
    LOAD array_start
    ADD counter
    STORE current_value

    ; Load array element and add to sum
    LOAD current_value
    ADD sum
    STORE sum

    ; Increment counter
    INC counter

    JUMP loop_start

loop_end:
    ; Result is in sum
    LOAD sum
    HALT

; Data section (would be loaded separately)
"""


if __name__ == "__main__":
    print("=" * 70)
    print("TAL - Ternary Assembly Language")
    print("=" * 70)

    compiler = TALCompiler()

    print("\nBuilt-in Macros:")
    for name in sorted(compiler.macros.keys()):
        macro = compiler.macros[name]
        params = ', '.join(macro.params)
        print(f"  {name}({params})")

    print("\n" + "=" * 70)
    print("Example Program")
    print("=" * 70)

    source = tal_example()
    print("\nSource:")
    print(source)

    print("\n" + "=" * 70)
    print("Compiled Output")
    print("=" * 70)
    compiled = compiler.compile(source)
    print(compiled)
