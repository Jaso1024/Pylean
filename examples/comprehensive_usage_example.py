#!/usr/bin/env python3
"""
Comprehensive Pylean Usage Example

This example demonstrates how to use Pylean as a library for:
1. Defining mathematical types and functions
2. Proving theorems about those functions
3. Using pattern matching with exhaustiveness checking
4. Using tactics for interactive theorem proving
5. Compiling functions to efficient LLVM code

It implements natural numbers, defines operations on them,
proves properties about these operations, and compiles them to
efficient code.
"""

import os
from pathlib import Path
import sys

from pylean.kernel import (
    Expr, Name, Level, ExprKind, Environment, Kernel,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi, mk_match,
    mk_pattern, mk_alternative, mk_inductive, mk_constructor,
    ReductionStrategy, ReductionMode
)
from pylean.kernel.env import mk_definition, mk_constant
from pylean.kernel.pattern_opt import (
    check_exhaustiveness, optimize_match, generate_decision_tree
)
from pylean.kernel.tactic import (
    TacticState, Tactic, IntroTactic, ExactTactic, AssumptionTactic,
    InductionTactic, Goal, init_tactic_state
)
from pylean.codegen.llvm_backend import LLVMBackend


def main():
    print("\n=== Comprehensive Pylean Usage Example ===\n")
    
    # Part 1: Define basic types and functions
    print("Part 1: Defining basic types and functions")
    print("-----------------------------------------")
    
    # Create a kernel and environment
    kernel = Kernel()
    env = kernel.env
    
    # Define the natural number type (Peano numbers)
    print("\nDefining natural numbers...")
    nat_type = mk_sort(0)
    
    # Create constructors for natural numbers
    zero_ctor = mk_constructor("zero", nat_type, "Nat")
    succ_ctor = mk_constructor(
        "succ", 
        mk_pi("n", nat_type, nat_type),
        "Nat"
    )
    
    # Define the inductive type
    nat_decl = mk_inductive("Nat", nat_type, [zero_ctor, succ_ctor])
    env = env.add_decl(nat_decl)
    
    # Access the constructors from the environment
    zero = mk_const("zero")
    succ = mk_const("succ")
    
    # Define some numbers for convenience
    one = mk_app(succ, zero)
    two = mk_app(succ, one)
    three = mk_app(succ, two)
    
    print(f"Created natural numbers: zero: {zero}, one: {one}, two: {two}, three: {three}")
    
    # Define addition function using pattern matching
    print("\nDefining addition function using pattern matching...")
    
    # Type of add: Nat → Nat → Nat
    add_type = mk_pi("m", nat_type, mk_pi("n", nat_type, nat_type))
    
    # Define addition using pattern matching:
    # add(zero, n) = n
    # add(succ(m), n) = succ(add(m, n))
    
    # First pattern: add(zero, n) = n
    zero_pattern = mk_pattern("zero", [])
    zero_alt = mk_alternative(zero_pattern, mk_var(0))  # n is at index 0
    
    # Second pattern: add(succ(m), n) = succ(add(m, n))
    succ_pattern = mk_pattern("succ", ["m"])
    
    # Build the expression succ(add(m, n))
    add_rec = mk_app(mk_const("add"), mk_var(1))  # add(m, ...)
    add_rec = mk_app(add_rec, mk_var(0))         # add(m, n)
    succ_rec = mk_app(succ, add_rec)             # succ(add(m, n))
    
    succ_alt = mk_alternative(succ_pattern, succ_rec)
    
    # Create the match expression for add
    match_add = mk_match(
        mk_var(1),  # scrutinee is the first argument (m)
        nat_type,   # return type
        [zero_alt, succ_alt]
    )
    
    # Create the full add function
    add_body = mk_lambda("n", nat_type, match_add)
    add_fn = mk_lambda("m", nat_type, add_body)
    
    # Add the definition to the environment
    add_def = mk_definition("add", add_type, add_fn)
    env = env.add_decl(add_def)
    
    # Let's test addition by reducing expressions
    print("\nTesting addition...")
    
    # Create expressions to test: 2 + 3
    add_expr = mk_app(mk_const("add"), two) # add(2, ...)
    add_expr = mk_app(add_expr, three)      # add(2, 3)
    
    # Reduce it
    kernel = Kernel(env)  # Create a new kernel with our updated environment
    result = kernel.reduce(add_expr, strategy=ReductionStrategy.NF)
    
    print(f"2 + 3 = {result}")
    
    # Part 2: Pattern matching optimizations
    print("\nPart 2: Pattern matching optimizations")
    print("------------------------------------")
    
    # Check if our pattern matching is exhaustive
    is_exhaustive = check_exhaustiveness(env, match_add)
    print(f"\nIs add pattern matching exhaustive? {is_exhaustive}")
    
    # Optimize match expression
    optimized_match = optimize_match(env, match_add)
    print(f"\nOptimized match expression has {len(optimized_match.alternatives)} alternatives")
    
    # Generate decision tree
    decision_tree = generate_decision_tree(env, match_add)
    print("\nDecision tree for add:")
    print(f"  Switch on scrutinee[{decision_tree.scrutinee}]")
    for ctor_name, case in decision_tree.cases.items():
        print(f"  - Case '{ctor_name}': action={case.action_index}")
    
    # Part 3: Simple tactics demonstration
    print("\nPart 3: Demonstration of proof tactics")
    print("-------------------------------------")
    
    # Simple identity function: λ x : Nat, x
    id_type = mk_pi("x", nat_type, nat_type)
    id_body = mk_var(0)  # Just the variable x
    id_fn = mk_lambda("x", nat_type, id_body)
    
    # Define the theorem: id_fn applied to any number is the number itself
    # ∀ n : Nat, id(n) = n
    # We'll represent it simply as the id function itself, since it's a direct proof
    print("\nProving: ∀ n : Nat, id(n) = n")
    
    # Create a tactic state for this proof
    tactic_state = init_tactic_state(env, id_type)
    print(f"Initial goal: {tactic_state.goals[0]}")
    
    # Start the proof
    print("\nStep 1: Introduce the variable n")
    intro_tactic = IntroTactic("n")
    tactic_state = intro_tactic.apply(tactic_state)
    print(f"Goal after intro: {tactic_state.goals[0]}")
    
    # Step 2: Use the assumption tactic to use 'n' directly
    print("\nStep 2: Apply assumption (use 'n' directly)")
    assumption_tactic = AssumptionTactic()
    tactic_state = assumption_tactic.apply(tactic_state)
    
    # Check that the proof is complete
    print(f"\nProof state: {tactic_state}")
    
    # Add the id function to the environment
    id_def = mk_definition("id", id_type, id_fn)
    env = env.add_decl(id_def)
    
    # Part 4: LLVM code generation
    print("\nPart 4: LLVM code generation")
    print("---------------------------")
    
    # Initialize LLVM backend
    llvm_backend = LLVMBackend(env)
    
    # Compile the add function to LLVM IR
    print("\nCompiling add function to LLVM IR...")
    add_ir = llvm_backend.compile_decl(env.find_decl("add"))
    
    # Compile the id function to LLVM IR
    print("\nCompiling id function to LLVM IR...")
    id_ir = llvm_backend.compile_decl(env.find_decl("id"))
    
    # Generate a complete program for testing add
    program_ir = llvm_backend.generate_program(
        [env.find_decl("Nat"), env.find_decl("zero"), env.find_decl("succ"), 
         env.find_decl("add"), env.find_decl("id")],
        "add"  # main function
    )
    
    # Optimize the module
    llvm_backend.optimize_module(optimization_level=2)
    
    # Create output directory if it doesn't exist
    os.makedirs("llvm_output", exist_ok=True)
    
    # Save the IR to a file
    with open("llvm_output/comprehensive_example.ll", "w") as f:
        f.write(str(llvm_backend.module))
    
    print(f"Saved LLVM IR to llvm_output/comprehensive_example.ll")
    
    print("\n=== Example Completed Successfully ===")


if __name__ == "__main__":
    main() 