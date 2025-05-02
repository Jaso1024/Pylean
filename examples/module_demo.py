"""
Module System Demo for Pylean.

This example demonstrates how to use the Pylean module system
to define, import, and use modules.
"""

import os
from pylean.kernel import (
    Expr, ExprKind, Name, 
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, Context, Environment
)
from pylean.module import (
    Module, ModuleData, Import, save_module, load_module, import_module
)


def create_demo_module(name: str) -> Module:
    """
    Create a demo module with some declarations.
    
    Args:
        name: The module name
        
    Returns:
        A demo module
    """
    # Create a kernel with a standard environment
    kernel = Kernel()
    
    # Create a module
    module = Module(name)
    
    # Add some declarations to the kernel
    if name == "Nat":
        # Define Natural numbers (Nat)
        kernel = kernel.add_constant("Nat", mk_sort(0))
        
        # Define constructors for Nat
        kernel = kernel.add_constant("zero", mk_const("Nat"))
        
        # Define succ : Nat -> Nat
        succ_type = mk_pi("n", mk_const("Nat"), mk_const("Nat"))
        kernel = kernel.add_constant("succ", succ_type)
        
        # Add the declarations to the module
        nat_decl = kernel.env.get_decl("Nat")
        zero_decl = kernel.env.get_decl("zero")
        succ_decl = kernel.env.get_decl("succ")
        
        module.add_declaration(nat_decl)
        module.add_declaration(zero_decl)
        module.add_declaration(succ_decl)
        
    elif name == "NatOps":
        # This module imports Nat and adds operations on Nat
        module.add_import(Import("Nat"))
        
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
        
        # Add the declarations to the module
        add_decl = kernel.env.get_decl("add")
        mul_decl = kernel.env.get_decl("mul")
        
        module.add_declaration(add_decl)
        module.add_declaration(mul_decl)
    
    return module


def main():
    """Run the module system demo."""
    print("Pylean Module System Demo")
    print("========================")
    
    # Create a temporary directory for modules
    modules_dir = "temp_modules"
    os.makedirs(modules_dir, exist_ok=True)
    
    # Create and save the Nat module
    print("\nCreating Nat module...")
    nat_module = create_demo_module("Nat")
    nat_module_path = os.path.join(modules_dir, "Nat.plm")
    save_module(nat_module, nat_module_path)
    print(f"Saved Nat module to {nat_module_path}")
    
    # Print Nat module contents
    print("\nNat module declarations:")
    for name, decl in nat_module.get_all_declarations().items():
        print(f"  {name} : {decl.type}")
    
    # Create and save the NatOps module
    print("\nCreating NatOps module...")
    natops_module = create_demo_module("NatOps")
    natops_module_path = os.path.join(modules_dir, "NatOps.plm")
    save_module(natops_module, natops_module_path)
    print(f"Saved NatOps module to {natops_module_path}")
    
    # Print NatOps module contents
    print("\nNatOps module declarations:")
    for name, decl in natops_module.get_all_declarations().items():
        print(f"  {name} : {decl.type}")
    
    print("\nNatOps module imports:")
    for imp in natops_module.get_imports():
        imports_str = "all" if imp.imports is None else ", ".join(imp.imports)
        alias_str = f" as {imp.alias}" if imp.alias else ""
        print(f"  import {imp.module_name}{alias_str} ({imports_str})")
    
    # Create a kernel with a standard environment
    kernel = Kernel()
    env = kernel.env
    
    # Import the NatOps module (which will also import Nat)
    print("\nImporting NatOps module into environment...")
    try:
        new_env, module = import_module(env, "NatOps", search_paths=[modules_dir])
        
        # Use the imported declarations
        print("\nAvailable declarations after import:")
        for name in sorted(new_env.decls.keys()):
            if name in ["Nat", "zero", "succ", "add", "mul"]:
                print(f"  {name}")
        
        # Create an example expression using imported declarations
        zero = mk_const("zero")
        succ = mk_const("succ")
        add = mk_const("add")
        
        # Build expression add(succ(zero), zero)
        succ_zero = mk_app(succ, zero)
        expr = mk_app(mk_app(add, succ_zero), zero)
        
        print("\nExample expression using imported declarations:")
        print(f"  {expr}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    print("\nModule System Demo Completed!")
    
    # Clean up
    try:
        os.remove(nat_module_path)
        os.remove(natops_module_path)
        os.rmdir(modules_dir)
    except:
        pass


if __name__ == "__main__":
    main() 