"""
LLVM Compiler Demo for Pylean.

This example demonstrates how to use the Pylean LLVM compiler backend
to generate LLVM IR and executable code from Pylean expressions.
"""

import os
import subprocess
from pylean.kernel import (
    Expr, ExprKind, Name, 
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, Context, Declaration, DeclKind
)
from pylean.codegen import (
    BackendType, compile_expr, compile_decl, compile_program, 
    emit_to_file, emit_object_code, LLVMBackend
)


def main():
    """Run the LLVM compiler demo."""
    print("Pylean LLVM Compiler Demo")
    print("=========================")
    
    # Create a kernel with a standard environment
    kernel = Kernel()
    env = kernel.env
    
    # Add some declarations to the environment
    print("\nDefining some constants and functions...")
    
    # Define Natural numbers (Nat)
    kernel = kernel.add_constant("Nat", mk_sort(0))
    
    # Define constructors for Nat
    kernel = kernel.add_constant("zero", mk_const("Nat"))
    
    # Define succ : Nat -> Nat
    succ_type = mk_pi("n", mk_const("Nat"), mk_const("Nat"))
    kernel = kernel.add_constant("succ", succ_type)
    
    # Define add : Nat -> Nat -> Nat
    add_type = mk_pi(
        "m", mk_const("Nat"),
        mk_pi("n", mk_const("Nat"), mk_const("Nat"))
    )
    kernel = kernel.add_constant("add", add_type)
    
    # Define a simple function: double(n) = add n n
    double_type = mk_pi("n", mk_const("Nat"), mk_const("Nat"))
    n = mk_var(0)  # The bound variable 'n'
    add = mk_const("add")
    add_n = mk_app(add, n)
    double_body = mk_app(add_n, n)  # add n n
    double_def = mk_lambda("n", mk_const("Nat"), double_body)
    kernel = kernel.add_definition("double", double_type, double_def)
    
    # Get updated environment
    env = kernel.env
    
    # Print our declarations
    print("\nDefined declarations:")
    for name, decl in env.declarations.items():
        if name in ["Nat", "zero", "succ", "add", "double"]:
            print(f"  {name} : {decl.type}")
    
    # Create an LLVM backend
    llvm_backend = LLVMBackend(env)
    
    # Compile the program
    print("\nCompiling to LLVM IR...")
    decls = [env.find_decl(name) for name in ["Nat", "zero", "succ", "add", "double"]]
    llvm_ir = compile_program(decls, env, main_fn="double", backend_type=BackendType.LLVM)
    
    # Save the IR to a file
    ir_filename = "double.ll"
    emit_to_file(llvm_ir, ir_filename)
    print(f"Wrote LLVM IR to {ir_filename}")
    
    # Emit object code
    obj_filename = "double.o"
    print(f"\nGenerating object code to {obj_filename}...")
    emit_object_code(llvm_backend, obj_filename)
    
    # Show how to use the generated object file
    print("\nTo compile the object file into an executable:")
    print(f"  clang {obj_filename} -o double_prog")
    
    # Try to compile the program if clang is available
    try:
        print("\nTrying to compile the object file...")
        subprocess.run(["clang", obj_filename, "-o", "double_prog"], check=True)
        print("Successfully compiled to executable 'double_prog'")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Could not compile object file. Please make sure clang is installed.")
    
    print("\nLLVM Compiler Demo Completed!")


if __name__ == "__main__":
    main() 