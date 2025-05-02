# Pylean Examples

This directory contains example programs demonstrating various features of Pylean.

## Available Examples

### Pattern Matching (pattern_matching_demo.py)

Demonstrates pattern matching on inductive types. The example defines:
- A simple natural number type with zero and successor constructors
- A recursive addition function using pattern matching
- Shows how the LLVM backend compiles pattern matching to efficient code

Run with:
```bash
python examples/pattern_matching_demo.py
```

### Advanced Pattern Matching (advanced_pattern_matching_demo.py)

Demonstrates advanced pattern matching optimizations:
- Exhaustiveness checking for pattern matching
- Redundancy detection and elimination
- Decision tree generation for efficient pattern matching
- Visual representation of pattern matching compilation

Run with:
```bash
python examples/advanced_pattern_matching_demo.py
```

### Foreign Function Interface (ffi_demo.py)

Demonstrates the FFI functionality for calling C code from Pylean:
- Creates a small C library with basic functions
- Declares these functions as external in Pylean
- Generates LLVM IR that can call the C functions
- Shows how to convert between Pylean and C types

Run with:
```bash
python examples/ffi_demo.py
```

Note: Running the demo on NixOS may require additional setup for C compilation.

### Tail Call Optimization (tail_call_optimization_demo.py)

Demonstrates the LLVM backend's tail call optimization capabilities:
- Implements factorial in two ways: naive recursion and tail recursion
- Shows how tail recursion is optimized into efficient loops
- Explains the performance benefits of tail call optimization

Run with:
```bash
python examples/tail_call_optimization_demo.py
```

## Adding New Examples

When creating new examples:
1. Place the example script in this directory
2. Update this README with a description of your example
3. Make sure the example includes proper documentation
4. Consider adding a section to the technical_log.txt documenting what the example demonstrates 