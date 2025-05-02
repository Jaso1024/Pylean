#!/usr/bin/env python
"""
Foreign Function Interface (FFI) demonstration for Pylean.

This example shows how to call external C functions from Pylean code
using the FFI mechanism.
"""

import os
import sys
import ctypes
from pathlib import Path

from pylean.kernel import (
    Environment, Name, Level,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi, mk_extern, mk_extern_decl
)
from pylean.codegen import LLVMBackend

# Create a tiny C library for testing
C_SOURCE = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to add two integers
int add_numbers(int a, int b) {
    return a + b;
}

// Function to print a string
void print_message(const char* message) {
    printf("%s\\n", message);
}

// Function to allocate memory and return a string
char* create_greeting(const char* name) {
    char* result = (char*)malloc(strlen(name) + 20);
    sprintf(result, "Hello, %s!", name);
    return result;
}
"""

def create_c_library():
    """Create a small C library for testing FFI."""
    os.makedirs("llvm_output", exist_ok=True)
    with open("llvm_output/ffi_test.c", "w") as f:
        f.write(C_SOURCE)
    
    # Check if gcc is available (for NixOS)
    compiler = "gcc"
    if os.system("which gcc > /dev/null") != 0:
        # Try clang as an alternative
        if os.system("which clang > /dev/null") != 0:
            print("Neither GCC nor Clang found. Please make sure you have a C compiler installed.")
            # Create a dummy for demo purposes
            print("Creating a dummy library for demonstration.")
            with open("llvm_output/libffi_test.so", "w") as f:
                f.write("# Dummy library for demo")
            return "llvm_output/libffi_test.so"
        compiler = "clang"
    
    print(f"Using {compiler} to compile C library")
    
    # Compile to shared library
    compile_cmd = f"{compiler} -fPIC -shared -o llvm_output/libffi_test.so llvm_output/ffi_test.c"
    print(f"Running: {compile_cmd}")
    os.system(compile_cmd)
    
    # Make sure it exists
    if not os.path.exists("llvm_output/libffi_test.so"):
        print("Failed to compile C library. Creating a dummy for demo purposes.")
        with open("llvm_output/libffi_test.so", "w") as f:
            f.write("# Dummy library for demo")
    
    print("Created C library: llvm_output/libffi_test.so")
    return "llvm_output/libffi_test.so"


def main():
    # First, create the C library
    lib_path = create_c_library()
    
    # Create a fresh environment
    env = Environment()
    
    # Add basic types to the environment
    int_type = mk_sort(0)  # Type 0
    env = env.add_decl(mk_extern_decl(
        "Int", [], int_type
    ))
    
    string_type = mk_sort(0)  # Type 0
    env = env.add_decl(mk_extern_decl(
        "String", [], string_type
    ))
    
    unit_type = mk_sort(0)  # Type 0 (Unit type for void functions)
    env = env.add_decl(mk_extern_decl(
        "Unit", [], unit_type
    ))
    
    # Declare external functions from our C library
    
    # add_numbers: Int -> Int -> Int
    int_const = mk_const("Int")
    env = env.add_decl(mk_extern_decl(
        "add_numbers", 
        [int_const, int_const],
        int_const,
        "add_numbers"
    ))
    
    # print_message: String -> Unit
    string_const = mk_const("String")
    unit_const = mk_const("Unit")
    env = env.add_decl(mk_extern_decl(
        "print_message", 
        [string_const],
        unit_const,
        "print_message"
    ))
    
    # create_greeting: String -> String
    env = env.add_decl(mk_extern_decl(
        "create_greeting", 
        [string_const],
        string_const,
        "create_greeting"
    ))
    
    # Create a sample function that uses the external functions
    sample_fn = mk_lambda(
        "x", int_const,
        mk_lambda(
            "y", int_const,
            mk_app(
                mk_app(
                    mk_extern("add_numbers", [int_const, int_const], int_const),
                    mk_var(1)  # x
                ),
                mk_var(0)  # y
            )
        )
    )
    
    # Add the function to the environment
    env = env.add_decl(mk_extern_decl(
        "sample_add",
        [int_const, int_const],
        int_const
    ))
    
    # Create an LLVM backend
    backend = LLVMBackend(env)
    
    # Generate LLVM IR for our declarations
    llvm_ir = backend.generate_program(list(env.declarations.values()), "sample_add")
    
    # Save the LLVM IR
    os.makedirs("llvm_output", exist_ok=True)
    with open("llvm_output/ffi_demo.ll", "w") as f:
        f.write(llvm_ir)
    
    print(f"Generated LLVM IR: llvm_output/ffi_demo.ll")
    
    # Compile to an object file
    backend.emit_object_code("llvm_output/ffi_demo.o")
    
    print(f"Generated object file: llvm_output/ffi_demo.o")
    
    # Link with our C library to create an executable
    # Check which compiler to use
    compiler = "gcc"
    if os.system("which gcc > /dev/null") != 0:
        if os.system("which clang > /dev/null") != 0:
            print("Skipping linking step - no compiler available")
            return
        compiler = "clang"
    
    link_cmd = f"{compiler} -o llvm_output/ffi_demo llvm_output/ffi_demo.o -L./llvm_output -lffi_test"
    print(f"Running: {link_cmd}")
    os.system(link_cmd)
    
    if not os.path.exists("llvm_output/ffi_demo"):
        print("Failed to link executable")
        return
    
    print(f"Created executable: llvm_output/ffi_demo")
    
    # Run the executable
    print("\nRunning the demo:")
    os.environ["LD_LIBRARY_PATH"] = "./llvm_output"
    
    # Check if we can run it
    if os.path.exists("llvm_output/ffi_demo"):
        os.system("./llvm_output/ffi_demo")
    else:
        print("Executable not found. Demo cannot be run.")


if __name__ == "__main__":
    main() 