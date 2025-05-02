# Pylean Kernel Module

This module implements the core functionality of the Lean4 theorem prover in Python. It provides the foundation for the theorem-proving system, including expressions, environments, and type checking.

## Components

### Expression System (`expr.py`)

The expression module defines the core expression types for Lean4:

- **Variables**: De Bruijn indices for bound variables
- **Constants**: Named entities in the environment
- **Applications**: Function application expressions
- **Lambdas**: Function abstractions
- **Pi types**: Dependent function types
- **Let expressions**: Local definitions
- **Sort expressions**: Type universes (Type, Prop)
- **Metavariables**: Placeholders for unknown terms
- **Local constants**: Named local variables

### Utility Functions

The expression module also provides utility functions for manipulating expressions:

- **occurs_in**: Check if a variable occurs in an expression
- **lift**: Lift free variables in an expression
- **instantiate**: Substitute a term for a bound variable

## Planned Components

The following components are planned for future implementation:

- **Environment**: Store and manage declarations
- **Declaration**: Represent constants, definitions, axioms, etc.
- **Type Checker**: Check well-formedness and infer types
- **Reduction**: Perform computation and normalization
- **Elaboration**: Convert surface syntax to fully-typed expressions

## Usage

```python
from pylean.kernel import (
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi
)

# Create a constant
nat = mk_const("Nat")

# Create a variable
x = mk_var(0)

# Create a function application
app = mk_app(nat, x)

# Create a lambda expression
lambda_expr = mk_lambda("x", mk_sort(0), x)

# Create a dependent function type
pi_expr = mk_pi("x", nat, x)
```

## Design Principles

1. **Immutability**: All expression objects are immutable for thread safety and cache coherence.
2. **De Bruijn Indices**: Bound variables use de Bruijn indices to simplify substitution and equality checking.
3. **Pure Python**: The implementation is in pure Python for maximum compatibility and ease of installation.
4. **Lean4 Compatibility**: The design closely mirrors Lean4's own implementation for compatibility.
5. **Performance**: While not the primary goal, reasonable performance is maintained through careful implementation. 