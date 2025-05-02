# Pylean: Lean4 in Python

Pylean is a Python implementation of the [Lean4 theorem prover](https://lean-lang.org/), designed to be used as a Python module through pip.

## Project Status

**Current Status: Beta**

The project has advanced to beta status with the following components fully functional:

- Complete Kernel module with expressions, environments, type checking, and inductive types
- Advanced parser supporting the full Lean4 syntax
- Comprehensive tactics system for interactive theorem proving
- Elaborator for converting parsed expressions to kernel expressions
- Type class system with instance resolution
- Command-line interface (CLI) with REPL
- LLVM backend with tail call optimization
- Pattern matching with exhaustiveness checking and optimization
- Foreign Function Interface (FFI) for C integration

The project is now ready for general use and feedback from the community.

## Installation

For development installation:

```bash
# Clone the repository
git clone https://github.com/yourusername/pylean.git
cd pylean

# Install in development mode
pip install -e .
# For development dependencies
pip install -e ".[dev]"
```

## Using the Command Line Interface

Pylean comes with a command-line interface that provides several ways to interact with the system:

```bash
# Start the interactive REPL
pylean

# Run a specific demo
pylean --demo pattern  # advanced pattern matching demo
pylean --demo tailcall  # tail call optimization demo
pylean --demo ffi  # foreign function interface demo
pylean --demo typeclass  # type class system demo
pylean --demo tactics  # tactics system demo

# Show version information
pylean --version
```

## Interactive REPL Commands

In the REPL, you can use the following commands:

```
:help             - Display help information
:quit, :exit      - Exit the REPL
:kernel           - Display kernel information
:env              - Display environment information
:tactic           - Enter tactic mode for theorem proving
:reduce <expr>    - Reduce an expression
:type <expr>      - Infer the type of an expression
:parse <expr>     - Parse an expression and show the syntax tree
```

## Using Pylean as a Library

```python
from pylean.kernel import (
    mk_const, mk_var, mk_app, mk_lambda, mk_pi, mk_sort, mk_match,
    mk_pattern, mk_alternative, Environment
)

# Create a simple environment
env = Environment()

# Define a natural number type
nat_type = mk_sort(0)
env = env.add_constant("Nat", nat_type)

# Define a simple function: double(n) = n + n
double_fn = mk_lambda(
    "n", mk_const("Nat"),
    mk_app(
        mk_app(mk_const("add"), mk_var(0)),  # n + ...
        mk_var(0)                           # ... + n
    )
)

# Add the function to the environment
env = env.add_def("double", double_fn.type, double_fn)

print(f"Function defined: double : {double_fn.type}")
```

For a more comprehensive example that demonstrates defining types, functions, pattern matching, theorem proving, and code generation, see `examples/comprehensive_usage_example.py`:

```bash
# Run the comprehensive example
python examples/comprehensive_usage_example.py
```

This example shows the full capabilities of Pylean, including:
- Creating inductive types (natural numbers)
- Defining recursive functions with pattern matching
- Pattern matching optimizations with exhaustiveness checking
- Interactive theorem proving with tactics
- LLVM code generation for efficient execution

For more examples, see the `examples` directory.

## Features Overview

- **Advanced Pattern Matching**
  - Exhaustiveness checking for pattern matching
  - Redundancy detection and elimination
  - Decision tree generation for efficient pattern matching
  - Nested pattern handling

- **Tail Call Optimization**
  - Automatic detection of tail recursive functions
  - Transformation of tail recursion into efficient loops
  - Performance optimization for recursive algorithms

- **Foreign Function Interface (FFI)**
  - Call C functions from Pylean
  - Pass data between Pylean and C code
  - Integrate with existing C libraries

- **Type Class System**
  - Define and use type classes for polymorphism
  - Instance resolution with priority based selection
  - Automatic instance derivation for common classes

- **Tactics System**
  - Interactive theorem proving
  - Step-by-step proof construction
  - Proof automation with tactics

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Recommended: a virtual environment

### Setting Up Development Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Project Structure

```
pylean/                    # Main package
├── __init__.py
├── cli.py                 # Command-line interface
├── elaborator.py          # Elaborator (SyntaxNode -> Expr)
├── parser/                # Parser implementation
│   ├── __init__.py
│   ├── core.py            # Core parsing functionality
│   ├── expr.py            # Expression parsing
│   └── pratt.py           # Pratt parsing implementation
├── kernel/                # Kernel implementation
│   ├── __init__.py
│   ├── expr.py            # Core expression system
│   ├── env.py             # Environment system
│   ├── typecheck.py       # Type checking system
│   ├── reduce.py          # Reduction and normalization
│   ├── pattern_opt.py     # Pattern matching optimizations
│   ├── typeclass.py       # Type class system
│   ├── eliminator.py      # Eliminator system
│   └── tactic_repl.py     # Tactics system
├── codegen/               # Code generation
│   ├── __init__.py
│   ├── backend.py         # Abstract backend interface
│   ├── c_backend.py       # C code generation
│   └── llvm_backend.py    # LLVM code generation with TCO
└── module/                # Module system
    ├── __init__.py
    ├── env.py             # Environment extensions for modules
    └── module.py          # Module implementation
```

## Examples

The `examples/` directory contains several demonstration files showing various features of Pylean:

- **comprehensive_usage_example.py**: Complete walkthrough of Pylean's capabilities
- **even_square_theorem.py**: Proves that if x is even, then x² is also even
- **pattern_matching_demo.py**: Demonstrates pattern matching functionality
- **tactics_demo.py**: Shows interactive theorem proving with tactics
- **llvm_compiler_demo.py**: Demonstrates code generation to LLVM

## Contributing

Contributions are welcome! Please open an issue first to discuss any significant changes.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details. 
