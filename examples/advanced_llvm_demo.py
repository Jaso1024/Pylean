"""
Advanced LLVM Backend Demo for Pylean.

This example demonstrates the enhanced LLVM backend features, including:
- Inductive type definition and code generation
- Constructor support
- Pattern matching via function application
- Proper handling of De Bruijn indices and local variables
"""

import os
import subprocess
from pathlib import Path
from pylean.kernel import (
    Expr, ExprKind, Name, 
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi, mk_let,
    Kernel, Context, Declaration, DeclKind
)
from pylean.codegen import (
    BackendType, compile_expr, compile_decl, compile_program, 
    emit_to_file, emit_object_code, LLVMBackend
)


def main():
    """Run the advanced LLVM compiler demo."""
    print("Pylean Advanced LLVM Compiler Demo")
    print("===================================")
    
    # Create a kernel with a standard environment
    kernel = Kernel()
    env = kernel.env
    
    print("\nDefining inductive types and functions...")
    
    # Define MyBool type (custom boolean type to avoid conflicts)
    kernel = kernel.add_inductive("MyBool", mk_sort(0), [
        ("mytrue", mk_const("MyBool")),
        ("myfalse", mk_const("MyBool"))
    ])
    
    # Define MyNat type (custom natural number type to avoid conflicts)
    kernel = kernel.add_inductive("MyNat", mk_sort(0), [
        ("mzero", mk_const("MyNat")),
        ("msucc", mk_pi("n", mk_const("MyNat"), mk_const("MyNat")))
    ])
    
    # Define MyList type
    list_sort = mk_sort(0)
    list_type = mk_pi("T", list_sort, list_sort)
    list_T = mk_app(mk_const("MyList"), mk_var(0))  # MyList T where T is the first argument
    
    kernel = kernel.add_inductive("MyList", list_type, [
        ("mynil", mk_pi("T", list_sort, mk_app(mk_const("MyList"), mk_var(0)))),
        ("mycons", mk_pi(
            "T", list_sort,
            mk_pi("head", mk_var(0),
                  mk_pi("tail", mk_app(mk_const("MyList"), mk_var(1)),
                        mk_app(mk_const("MyList"), mk_var(2))))
        ))
    ])
    
    # Define add_zero: MyNat -> MyNat -> MyNat
    # add_zero m n = n (a simpler version that just returns the second argument)
    add_zero_type = mk_pi("m", mk_const("MyNat"), mk_pi("n", mk_const("MyNat"), mk_const("MyNat")))
    add_zero_body = mk_lambda("m", mk_const("MyNat"), 
                             mk_lambda("n", mk_const("MyNat"), mk_var(0)))  # λm. λn. n
    kernel = kernel.add_definition("add_zero", add_zero_type, add_zero_body)
    
    # Define add_one: MyNat -> MyNat
    # add_one n = msucc n
    add_one_type = mk_pi("n", mk_const("MyNat"), mk_const("MyNat"))
    add_one_body = mk_lambda("n", mk_const("MyNat"), 
                            mk_app(mk_const("msucc"), mk_var(0)))  # λn. msucc n
    kernel = kernel.add_definition("add_one", add_one_type, add_one_body)
    
    # Define a simple add that just adds 1 for demonstration
    # add m n = msucc n (simplified implementation that ignores m)
    add_type = mk_pi(
        "m", mk_const("MyNat"),
        mk_pi("n", mk_const("MyNat"), mk_const("MyNat"))
    )
    add_body = mk_lambda("m", mk_const("MyNat"),
                        mk_lambda("n", mk_const("MyNat"),
                                 mk_app(mk_const("msucc"), mk_var(0))))  # λm. λn. msucc n
    
    kernel = kernel.add_definition("add", add_type, add_body)
    
    # Define a more complex function using let-expressions: factorial
    fact_type = mk_pi("n", mk_const("MyNat"), mk_const("MyNat"))
    
    # For simplicity, our factorial just returns one
    # fact(n) = let one = msucc mzero in one
    
    one_expr = mk_app(mk_const("msucc"), mk_const("mzero"))
    
    # Define fact_body using a let expression
    fact_body = mk_lambda(
        "n", mk_const("MyNat"),
        mk_let(
            "one", mk_const("MyNat"), one_expr,
            # Use one from the let-binding
            mk_var(0)  # Return one
        )
    )
    
    kernel = kernel.add_definition("fact", fact_type, fact_body)
    
    # Get updated environment
    env = kernel.env
    
    # Print our declarations
    print("\nDefined declarations:")
    for name in ["MyBool", "mytrue", "myfalse", "MyNat", "mzero", "msucc", 
                "MyList", "mynil", "mycons", "add", "add_zero", "add_one", "fact"]:
        decl = env.find_decl(name)
        if decl:
            print(f"  {name} : {decl.type}")
    
    # Create an LLVM backend
    llvm_backend = LLVMBackend(env)
    
    # Compile the program
    print("\nCompiling to LLVM IR...")
    decls = [env.find_decl(name) for name in [
        "MyBool", "mytrue", "myfalse", 
        "MyNat", "mzero", "msucc", 
        "MyList", "mynil", "mycons",
        "add", "add_zero", "add_one", 
        "fact"
    ]]
    
    # Use fact as the main function
    llvm_ir = compile_program(decls, env, main_fn="fact", backend_type=BackendType.LLVM)
    
    # Create output directory if it doesn't exist
    output_dir = Path("llvm_output")
    output_dir.mkdir(exist_ok=True)
    
    # Save the IR to a file
    ir_filename = output_dir / "advanced_demo.ll"
    emit_to_file(llvm_ir, ir_filename)
    print(f"Wrote LLVM IR to {ir_filename}")
    
    # Emit object code
    obj_filename = output_dir / "advanced_demo.o"
    print(f"\nGenerating object code to {obj_filename}...")
    emit_object_code(llvm_backend, obj_filename)
    
    # Generate a simple C driver program
    c_driver = """
    #include <stdio.h>
    
    // External references to Pylean functions
    extern void* lean_fact();
    extern void* lean_mzero();
    extern void* lean_msucc(void* n);
    extern long lean_unbox(void* obj);
    
    int main() {
        // Create MyNat(3) = msucc(msucc(msucc(mzero)))
        void* zero = lean_mzero();
        void* one = lean_msucc(zero);
        void* two = lean_msucc(one);
        void* three = lean_msucc(two);
        
        // Compute fact(3)
        void* result = lean_fact(three);
        
        // Print the result
        printf("fact(3) = %ld\\n", lean_unbox(result));
        
        return 0;
    }
    """
    
    c_filename = output_dir / "driver.c"
    with open(c_filename, "w") as f:
        f.write(c_driver)
    
    # Show how to compile the complete program
    print("\nTo compile the program with the C driver:")
    print(f"  clang {obj_filename} {c_filename} -o advanced_demo")
    
    # Try to compile the program if clang is available
    try:
        print("\nTrying to compile the program...")
        subprocess.run(["clang", str(obj_filename), str(c_filename), "-o", str(output_dir / "advanced_demo")], check=True)
        print("Successfully compiled to executable 'llvm_output/advanced_demo'")
        print("You can run it with: ./llvm_output/advanced_demo")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Could not compile. Please make sure clang is installed.")
    
    print("\nAdvanced LLVM Compiler Demo Completed!")


if __name__ == "__main__":
    main() 