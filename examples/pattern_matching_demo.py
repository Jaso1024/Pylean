#!/usr/bin/env python
"""
Demo of pattern matching in Pylean LLVM backend.

This program demonstrates how pattern matching is handled in the LLVM backend,
including recursive functions with pattern matching on inductive types.
"""

from pylean.kernel import (
    Environment, Name, Level,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    mk_inductive, mk_constructor, mk_definition,
    Pattern, Alternative, mk_pattern, mk_alternative, mk_match
)
from pylean.codegen import LLVMBackend


def main():
    # Create a fresh environment
    env = Environment()
    
    # Define the natural numbers (Peano numbers)
    # Inductive nat : Type :=
    # | zero : nat
    # | succ : nat -> nat
    
    # Define the nat type
    nat_name = Name.from_string("nat")
    nat_type = mk_sort(0)  # Type 0
    
    # Define the constructors
    # zero : nat
    zero_name = Name.from_string("zero")
    zero_type = mk_const(nat_name)
    zero_constructor = mk_constructor(zero_name, zero_type, nat_name)
    
    # succ : nat -> nat
    succ_name = Name.from_string("succ")
    succ_type = mk_pi(Name.from_string("n"), mk_const(nat_name), mk_const(nat_name))
    succ_constructor = mk_constructor(succ_name, succ_type, nat_name)
    
    # Create the inductive type
    nat_inductive = mk_inductive(nat_name, nat_type, [zero_constructor, succ_constructor])
    
    # Add to environment
    env = env.add_decl(nat_inductive)
    
    # Define addition using pattern matching
    # def add (a b : nat) : nat :=
    # match a with
    # | zero => b
    # | succ n => succ (add n b)
    
    # First, create the type of add
    # add : nat -> nat -> nat
    add_name = Name.from_string("add")
    add_type = mk_pi(
        Name.from_string("a"), 
        mk_const(nat_name),
        mk_pi(
            Name.from_string("b"), 
            mk_const(nat_name),
            mk_const(nat_name)
        )
    )
    
    # Now create the body of add using pattern matching
    # We need to define add recursively, which is a bit tricky
    # First, we create a forward reference to add itself
    add_ref = mk_const(add_name)
    
    # Match on the first argument 'a' (#0 in de Bruijn indices)
    # The second argument 'b' will be #1 in outer scope, #0 in match scope
    
    # Case 1: zero => b
    zero_pattern = mk_pattern(zero_name)
    zero_case = mk_alternative(zero_pattern, mk_var(0))  # Return b
    
    # Case 2: succ n => succ (add n b)
    # 'n' will be bound by the pattern as field #0
    succ_pattern = mk_pattern(succ_name, ["n"])
    
    # Recursive call: add n b
    # n is field #0 from pattern, b is #0 from outer scope
    rec_call = mk_app(mk_app(add_ref, mk_var(0)), mk_var(1))
    
    # Wrap in succ
    succ_case_expr = mk_app(mk_const(succ_name), rec_call)
    succ_case = mk_alternative(succ_pattern, succ_case_expr)
    
    # Create the match expression
    match_expr = mk_match(
        mk_var(0),                 # First argument 'a'
        mk_const(nat_name),        # Result type is nat
        [zero_case, succ_case]     # Pattern match cases
    )
    
    # Wrap in lambdas to bind the arguments
    # λb. λa. match a with ...
    add_body = mk_lambda(
        Name.from_string("b"),
        mk_const(nat_name),
        mk_lambda(
            Name.from_string("a"),
            mk_const(nat_name),
            match_expr
        )
    )
    
    # Define 'add'
    add_def = mk_definition(add_name, add_type, add_body)
    env = env.add_decl(add_def)
    
    # Create a simple test function to compute add 2 3
    # def test_add : nat := add (succ (succ zero)) (succ (succ (succ zero)))
    test_name = Name.from_string("test_add")
    test_type = mk_const(nat_name)
    
    # Build the expression for add 2 3
    # First, create 2 (succ (succ zero))
    two = mk_app(
        mk_const(succ_name),
        mk_app(mk_const(succ_name), mk_const(zero_name))
    )
    
    # Create 3 (succ (succ (succ zero)))
    three = mk_app(
        mk_const(succ_name),
        mk_app(
            mk_const(succ_name),
            mk_app(mk_const(succ_name), mk_const(zero_name))
        )
    )
    
    # add 2 3
    test_expr = mk_app(mk_app(mk_const(add_name), two), three)
    
    # Define the test function
    test_def = mk_definition(test_name, test_type, test_expr)
    env = env.add_decl(test_def)
    
    # Create LLVM backend
    llvm_backend = LLVMBackend(env)
    
    # Generate LLVM IR program
    llvm_ir = llvm_backend.generate_program([add_def, test_def], "test_add")
    
    # Print the LLVM IR program
    print(llvm_ir)
    
    # Optimize the module
    llvm_backend.optimize_module(optimization_level=2)
    
    # Print optimized IR
    print("\n// OPTIMIZED IR:")
    print(str(llvm_backend.module))
    
    # Emit object file
    output_file = "llvm_output/pattern_matching_demo.o"
    llvm_backend.emit_object_code(output_file)
    print(f"\nEmitted object file to {output_file}")
    
    
if __name__ == "__main__":
    main() 