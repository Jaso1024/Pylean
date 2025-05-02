"""
Namespace Demo for Pylean.

This example demonstrates how to use the Pylean module system
with namespaces to organize code.
"""

import os
from pylean.kernel import (
    Expr, ExprKind, Name, 
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, Context, Environment
)
from pylean.module import (
    Module, ModuleData, Import, save_module, load_module, import_module, open_namespace
)
from pylean.module.module import Namespace


def create_math_module(name: str) -> Module:
    """
    Create a math module with various namespaces.
    
    Args:
        name: The module name
        
    Returns:
        A math module with namespaces
    """
    # Create a kernel with a standard environment
    kernel = Kernel()
    
    # Create a module
    module = Module(name)
    
    # Natural numbers in the core namespace
    
    # Define Natural numbers (Nat)
    kernel = kernel.add_constant("Nat", mk_sort(0))
    
    # Define zero and one
    kernel = kernel.add_constant("zero", mk_const("Nat"))
    kernel = kernel.add_constant("one", mk_app(mk_const("succ"), mk_const("zero")))
    
    # Add constants to the module in the core namespace
    module.add_declaration(kernel.env.find_decl("Nat"))
    module.add_declaration(kernel.env.find_decl("zero"))
    module.add_declaration(kernel.env.find_decl("one"))
    
    # Arithmetic operations in the arith namespace
    
    # Define succ : Nat -> Nat
    succ_type = mk_pi("n", mk_const("Nat"), mk_const("Nat"))
    kernel = kernel.add_constant("succ", succ_type)
    
    # Define add : Nat -> Nat -> Nat
    add_type = mk_pi(
        "m", mk_const("Nat"),
        mk_pi("n", mk_const("Nat"), mk_const("Nat"))
    )
    kernel = kernel.add_constant("add", add_type)
    
    # Define mul : Nat -> Nat -> Nat
    mul_type = mk_pi(
        "m", mk_const("Nat"),
        mk_pi("n", mk_const("Nat"), mk_const("Nat"))
    )
    kernel = kernel.add_constant("mul", mul_type)
    
    # Add operations to the module in the arith namespace
    module.add_declaration(kernel.env.find_decl("succ"), namespace="arith")
    module.add_declaration(kernel.env.find_decl("add"), namespace="arith")
    module.add_declaration(kernel.env.find_decl("mul"), namespace="arith")
    
    # Theorems about arithmetic in the arith.theorems namespace
    
    # Define add_zero : ∀ n, add zero n = n
    add_zero_type = mk_pi(
        "n", mk_const("Nat"),
        mk_app(
            mk_app(
                mk_app(
                    mk_const("Eq"),
                    mk_const("Nat")
                ),
                mk_app(mk_app(mk_const("add"), mk_const("zero")), mk_var(0))
            ),
            mk_var(0)
        )
    )
    kernel = kernel.add_constant("add_zero", add_zero_type)
    
    # Define add_succ : ∀ m n, add (succ m) n = succ (add m n)
    add_succ_type = mk_pi(
        "m", mk_const("Nat"),
        mk_pi(
            "n", mk_const("Nat"),
            mk_app(
                mk_app(
                    mk_app(
                        mk_const("Eq"),
                        mk_const("Nat")
                    ),
                    mk_app(
                        mk_app(
                            mk_const("add"),
                            mk_app(mk_const("succ"), mk_var(1))
                        ),
                        mk_var(0)
                    )
                ),
                mk_app(
                    mk_const("succ"),
                    mk_app(mk_app(mk_const("add"), mk_var(1)), mk_var(0))
                )
            )
        )
    )
    kernel = kernel.add_constant("add_succ", add_succ_type)
    
    # Add theorems to the module in the arith.theorems namespace
    module.add_declaration(kernel.env.find_decl("add_zero"), namespace="arith.theorems")
    module.add_declaration(kernel.env.find_decl("add_succ"), namespace="arith.theorems")
    
    # Complex numbers in the complex namespace
    
    # Define Complex
    kernel = kernel.add_constant("Complex", mk_sort(0))
    
    # Define real and imaginary parts
    real_type = mk_pi("z", mk_const("Complex"), mk_const("Nat"))
    kernel = kernel.add_constant("real", real_type)
    
    imag_type = mk_pi("z", mk_const("Complex"), mk_const("Nat"))
    kernel = kernel.add_constant("imag", imag_type)
    
    # Add complex number definitions to the module in the complex namespace
    module.add_declaration(kernel.env.find_decl("Complex"), namespace="complex")
    module.add_declaration(kernel.env.find_decl("real"), namespace="complex")
    module.add_declaration(kernel.env.find_decl("imag"), namespace="complex")
    
    return module


def create_using_module(name: str) -> Module:
    """
    Create a module that uses the math module.
    
    Args:
        name: The module name
        
    Returns:
        A module that imports and uses the math module
    """
    # Create a kernel with a standard environment
    kernel = Kernel()
    
    # Create a module
    module = Module(name)
    
    # Import the math module
    module.add_import(Import("math"))
    
    # Import the math.arith namespace with an alias
    module.add_import(Import("math.arith", alias="arith"))
    
    # Open the math.arith.theorems namespace
    module.add_import(Import("math.arith.theorems", is_explicit=True))
    
    # Define a function using imported names
    double_type = mk_pi("n", mk_const("Nat"), mk_const("Nat"))
    n = mk_var(0)  # The bound variable 'n'
    add = mk_const("add")
    add_n = mk_app(add, n)
    double_body = mk_app(add_n, n)  # add n n
    double_def = mk_lambda("n", mk_const("Nat"), double_body)
    kernel = kernel.add_definition("double", double_type, double_def)
    
    # Add the definition to the module
    module.add_declaration(kernel.env.find_decl("double"))
    
    return module


def main():
    """Run the namespace demo."""
    print("Pylean Namespace Demo")
    print("====================")
    
    # Create a temporary directory for modules
    modules_dir = "temp_modules"
    os.makedirs(modules_dir, exist_ok=True)
    
    # Create and save the math module
    print("\nCreating math module...")
    math_module = create_math_module("math")
    math_module_path = os.path.join(modules_dir, "math.plm")
    save_module(math_module, math_module_path)
    print(f"Saved math module to {math_module_path}")
    
    # Print math module contents by namespace
    print("\nMath module declarations by namespace:")
    
    # Root namespace
    print("\nRoot namespace:")
    for name, _ in math_module.get_declarations_in_namespace().items():
        print(f"  {name}")
    
    # Arith namespace
    print("\nArith namespace:")
    for name, _ in math_module.get_declarations_in_namespace("arith").items():
        print(f"  {name}")
    
    # Arith.theorems namespace
    print("\nArith.theorems namespace:")
    for name, _ in math_module.get_declarations_in_namespace("arith.theorems").items():
        print(f"  {name}")
    
    # Complex namespace
    print("\nComplex namespace:")
    for name, _ in math_module.get_declarations_in_namespace("complex").items():
        print(f"  {name}")
    
    # Create and save the using module
    print("\nCreating using module...")
    using_module = create_using_module("using")
    using_module_path = os.path.join(modules_dir, "using.plm")
    save_module(using_module, using_module_path)
    print(f"Saved using module to {using_module_path}")
    
    # Print using module imports
    print("\nUsing module imports:")
    for imp in using_module.get_imports():
        imports_str = "all" if imp.imports is None else ", ".join(imp.imports)
        alias_str = f" as {imp.alias}" if imp.alias else ""
        explicit_str = " (open)" if imp.is_explicit else ""
        print(f"  import {imp.module_name}{alias_str}{explicit_str} ({imports_str})")
    
    # Create a kernel with a standard environment
    kernel = Kernel()
    env = kernel.env
    
    # Import the using module (which will also import math)
    print("\nImporting using module into environment...")
    try:
        new_env, module = import_module(env, "using", search_paths=[modules_dir])
        
        # Demonstrate accessing declarations through different paths
        print("\nAccessing declarations:")
        
        # Direct access from the root namespace
        try:
            nat_type = new_env.find_decl("Nat")
            print(f"  Nat (direct): {nat_type is not None}")
        except:
            print("  Failed to access Nat directly")
        
        # Access through the math module
        try:
            nat_type = new_env.find_decl("math.Nat")
            print(f"  math.Nat (qualified): {nat_type is not None}")
        except:
            print("  Failed to access math.Nat")
        
        # Access through the arith alias
        try:
            add_type = new_env.find_decl("arith.add")
            print(f"  arith.add (alias): {add_type is not None}")
        except:
            print("  Failed to access arith.add")
        
        # Access from the opened theorems namespace
        try:
            add_zero_type = new_env.find_decl("add_zero")
            print(f"  add_zero (opened): {add_zero_type is not None}")
        except:
            print("  Failed to access add_zero directly")
        
        # Fully qualified access
        try:
            add_zero_type = new_env.find_decl("math.arith.theorems.add_zero")
            print(f"  math.arith.theorems.add_zero (fully qualified): {add_zero_type is not None}")
        except:
            print("  Failed to access math.arith.theorems.add_zero")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    print("\nNamespace Demo Completed!")
    
    # Clean up
    try:
        os.remove(math_module_path)
        os.remove(using_module_path)
        os.rmdir(modules_dir)
    except:
        pass


if __name__ == "__main__":
    main() 