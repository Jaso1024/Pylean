#!/usr/bin/env python3
"""
Test script for Pi types in tactic mode using the kernel API directly.
"""

from pylean.kernel import (
    Name, Level, Expr, ExprKind,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, ReductionStrategy, ReductionMode
)
from pylean.kernel.tactic_repl import start_tactic_repl

def main():
    """Create a Pi type directly and start the tactic REPL."""
    print("Testing Pi types with tactic REPL")
    
    # Create a new kernel with standard environment
    kernel = Kernel()
    
    # Create the identity function type: Π (A : Type), A -> A
    # First, create the inner function type: A -> A 
    # In dependent type theory, A -> A is syntactic sugar for Π (x : A), A
    inner_type = mk_pi("x", mk_var(0), mk_var(1))  # Π (x : A), A
    
    # Now create the outer Pi type: Π (A : Type), inner_type
    id_type = mk_pi("A", mk_sort(0), inner_type)  # Π (A : Type), Π (x : A), A
    
    print(f"Created identity function type: {id_type}")
    
    # Start tactic REPL for proving the identity function type
    print("\nStarting tactic REPL...")
    start_tactic_repl(kernel, id_type)

if __name__ == "__main__":
    main() 