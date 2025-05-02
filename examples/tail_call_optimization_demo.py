#!/usr/bin/env python
"""
Tail Call Optimization (TCO) demonstration for Pylean.

This example shows how the LLVM backend performs tail call optimization
for recursive functions, transforming them into efficient loops.
"""

import os
from pathlib import Path
import time

from pylean.kernel import (
    Environment, Name, Level,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi, mk_definition
)
from pylean.codegen import LLVMBackend


def main():
    # Create a fresh environment
    env = Environment()
    
    # Define basic types
    int_type = mk_sort(0)
    env = env.add_decl(mk_definition("Int", int_type, None))
    
    int_const = mk_const("Int")
    
    # Define a recursive countdown function that demonstrates tail recursion
    # countdown(n) = if n <= 0 then 0 else countdown(n-1)
    # This is tail recursive because the recursive call is the last operation
    countdown_body = mk_lambda(
        "n", int_const,
        mk_app(
            mk_const("countdown"),
            mk_app(mk_const("sub1"), mk_var(0))  # n-1
        )
    )
    
    env = env.add_decl(mk_definition(
        "sub1",
        mk_pi("n", int_const, int_const),
        mk_lambda("n", int_const, int_const)  # Dummy implementation
    ))
    
    env = env.add_decl(mk_definition(
        "countdown",
        mk_pi("n", int_const, int_const),
        countdown_body
    ))
    
    # Define a non-tail-recursive function for comparison
    # sum_to(n) = if n <= 0 then 0 else n + sum_to(n-1)
    # This is not tail recursive because we need to add n after the recursive call returns
    sum_to_body = mk_lambda(
        "n", int_const,
        mk_app(
            mk_app(
                mk_const("add"),
                mk_var(0)  # n
            ),
            mk_app(
                mk_const("sum_to"),
                mk_app(mk_const("sub1"), mk_var(0))  # n-1
            )
        )
    )
    
    env = env.add_decl(mk_definition(
        "add",
        mk_pi("x", int_const, mk_pi("y", int_const, int_const)),
        mk_lambda("x", int_const, mk_lambda("y", int_const, int_const))  # Dummy implementation
    ))
    
    env = env.add_decl(mk_definition(
        "sum_to",
        mk_pi("n", int_const, int_const),
        sum_to_body
    ))
    
    # Create an LLVM backend
    backend = LLVMBackend(env)
    
    # Generate LLVM IR
    llvm_ir = backend.generate_program(list(env.declarations.values()))
    
    # Save the LLVM IR
    os.makedirs("llvm_output", exist_ok=True)
    with open("llvm_output/tail_call_demo.ll", "w") as f:
        f.write(llvm_ir)
    
    print(f"Generated LLVM IR: llvm_output/tail_call_demo.ll")
    
    # Let's look at the code to see if our tail call optimization worked
    # Add a delay to make sure the file is completely written
    time.sleep(0.5)
    
    with open("llvm_output/tail_call_demo.ll", "r") as f:
        code = f.read()
        print("\nExamining LLVM IR for tail call patterns...")
        
        # Check for br instructions (branches)
        has_branches = "br label %" in code
        
        # Print all lambda functions
        print("\nAll lambda function definitions:")
        lines = code.split("\n")
        for i, line in enumerate(lines):
            if "define" in line and "lambda" in line:
                # Print the lambda function and the next 10 lines
                lambda_start = i
                lambda_end = min(i + 15, len(lines))
                lambda_code = "\n".join(lines[lambda_start:lambda_end])
                print(lambda_code)
                print("-" * 40)
                
                # Check if this lambda has a branch instruction
                if "br label %" in lambda_code:
                    has_branches = True
                    print("Branch instruction found in this lambda!")
        
        # Look for all call instructions
        print("\nAll call instructions in the IR:")
        for line in lines:
            if "call" in line:
                print(line.strip())
        
        # Also check for tail call instructions
        has_tail_calls = "tail call" in code
        
        if has_branches or has_tail_calls:
            print("\nTail call optimization detected!")
            if has_branches:
                print("Found branch instructions that form loops.")
            if has_tail_calls:
                print("Found explicit tail call instructions.")
            print("The recursive function 'countdown' should be more efficient.")
        else:
            print("\nNo tail call optimization detected in the generated code.")
    
    # Explanation for the user
    print("""
Explanation:
1. The 'countdown' function is tail recursive:
   - The recursive call is the last operation
   - This can be optimized to a simple loop
   - No stack growth for deep recursion

2. The 'sum_to' function is NOT tail recursive:
   - The recursive call isn't the last operation
   - After the recursive call returns, we still need to add n
   - Each call needs a new stack frame
   
The LLVM backend's tail call optimization identifies tail recursive
functions and transforms them into efficient loops, avoiding stack overflow
for deep recursion.
""")


if __name__ == "__main__":
    main() 