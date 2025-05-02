#!/usr/bin/env python3
"""
Identity Function Proof Using Lambda Expressions in Pylean CLI.

This script demonstrates how to work with the identity function type proof
using lambda expressions which are supported by the parser.
"""

from pylean.cli import PyleanREPL
from pylean.kernel import (
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, ReductionStrategy, ReductionMode
)

def main():
    """Run the identity function proof demo."""
    print("Identity Function Proof Demo")
    print("==========================")
    
    # Create a CLI REPL
    repl = PyleanREPL()
    
    print("\n1. Lambda expression parsing (supported):")
    # Parse a lambda expression representing the identity function
    repl.cmd_parse("λ x : Nat, x")
    
    print("\n2. Reducing the identity function application:")
    # Apply the identity function to a value and reduce
    repl.cmd_reduce("(λ x : Nat, x) 0")
    
    print("\n3. Using the identity function with types:")
    
    # Create a kernel instance for programmatic operations
    kernel = Kernel()
    
    # Programmatically create the identity function term: λ (A : Type), λ (x : A), x
    univ_type = mk_sort(1)  # Type 1
    inner_lambda = mk_lambda("x", mk_var(0), mk_var(0))
    id_term = mk_lambda("A", univ_type, inner_lambda)
    
    # Infer the type of the term
    try:
        id_type = kernel.infer_type(id_term)
        print(f"\nInferred type of identity function: {id_type}")
        print(f"Identity function term: {id_term}")
        
        # Register the identity function in the environment using the inferred type
        kernel = kernel.add_definition("id", id_type, id_term)
        print(f"Successfully added identity function 'id' to environment")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n4. Workaround for proving Pi types since they're not directly parseable:")
    print("   - Use programmatic construction with kernel API (as demonstrated)")
    print("   - When working interactively, focus on lambda expressions for implementation")
    print("   - Use the kernel's infer_type method to get the correct type syntax")
    
    print("\n5. Alternative with simpler expressions:")
    # Parse a simple typed lambda expression
    print("Lambda expression for a function that adds 1 to a number:")
    repl.cmd_parse("λ x : Nat, x + 1")
    
    print("\nReducing a function application:")
    repl.cmd_reduce("(λ x : Nat, x + 1) 5")
    
    print("\nIdentity function proof demo completed.")

if __name__ == "__main__":
    main() 