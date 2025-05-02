"""
Compiler Demo for Pylean.

This example demonstrates how to use the Pylean compiler
to generate executable code from Pylean expressions.
"""

from pylean.kernel import (
    Expr, ExprKind, Name, 
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, Context, Declaration, DeclKind
)
from pylean.codegen import (
    BackendType, compile_expr, compile_decl, compile_program, emit_to_file
)


def main():
    """Run the compiler demo."""
    print("Pylean Compiler Demo")
    print("===================")
    
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
    for name, decl in env.decls.items():
        print(f"  {name} : {kernel.infer_type(mk_const(name))}")
    
    # Compile individual expressions
    print("\nCompiling individual expressions:")
    
    # Compile the zero constant
    zero_expr = mk_const("zero")
    zero_code = compile_expr(zero_expr, env)
    print(f"\nzero expression:")
    print(f"Python: {zero_expr}")
    print(f"C code: {zero_code}")
    
    # Compile the double function application
    double_expr = mk_app(mk_const("double"), mk_const("zero"))
    double_code = compile_expr(double_expr, env)
    print(f"\ndouble(zero) expression:")
    print(f"Python: {double_expr}")
    print(f"C code: {double_code}")
    
    # Compile a full program
    print("\nCompiling a complete program:")
    decls = [env.decls[name] for name in ["Nat", "zero", "succ", "add", "double"]]
    program_code = compile_program(decls, env, main_fn="double")
    
    print(f"\nGenerated C program:")
    print("----------------------")
    print(program_code)
    
    # Write the program to a file
    output_file = "double.c"
    emit_to_file(program_code, output_file)
    print(f"\nWrote C program to {output_file}")
    
    print("\nCompiler Demo Completed!")


if __name__ == "__main__":
    main() 