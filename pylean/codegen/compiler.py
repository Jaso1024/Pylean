"""
Main compiler module for Pylean.

This module provides the high-level functions for compiling
Pylean expressions and declarations to executable code.
"""

from typing import Dict, List, Optional, Union

from pylean.kernel import Expr, Environment, Declaration
from pylean.codegen.backend import Backend, BackendType
from pylean.codegen.c_backend import CBackend
from pylean.codegen.llvm_backend import LLVMBackend


def get_backend(env: Environment, backend_type: BackendType = BackendType.C) -> Backend:
    """
    Get the appropriate backend for the given type.
    
    Args:
        env: The environment containing declarations
        backend_type: The type of backend to use
        
    Returns:
        A backend instance
    """
    if backend_type == BackendType.C:
        return CBackend(env)
    elif backend_type == BackendType.LLVM:
        return LLVMBackend(env)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def compile_expr(expr: Expr, env: Environment, backend_type: BackendType = BackendType.C) -> str:
    """
    Compile an expression to the target language.
    
    Args:
        expr: The expression to compile
        env: The environment containing declarations
        backend_type: The type of backend to use
        
    Returns:
        The compiled code as a string
    """
    backend = get_backend(env, backend_type)
    return backend.compile_expr(expr)


def compile_decl(decl: Declaration, env: Environment, backend_type: BackendType = BackendType.C) -> str:
    """
    Compile a declaration to the target language.
    
    Args:
        decl: The declaration to compile
        env: The environment containing declarations
        backend_type: The type of backend to use
        
    Returns:
        The compiled code as a string
    """
    backend = get_backend(env, backend_type)
    return backend.compile_decl(decl)


def compile_program(decls: List[Declaration], env: Environment, 
                   main_fn: Optional[str] = None,
                   backend_type: BackendType = BackendType.C) -> str:
    """
    Compile a complete program from a list of declarations.
    
    Args:
        decls: The declarations to compile
        env: The environment containing declarations
        main_fn: Optional name of the main function
        backend_type: The type of backend to use
        
    Returns:
        The compiled program as a string
    """
    backend = get_backend(env, backend_type)
    program = backend.generate_program(decls, main_fn)
    
    # Apply backend-specific optimizations
    if backend_type == BackendType.LLVM and isinstance(backend, LLVMBackend):
        backend.optimize_module(optimization_level=2)
        return str(backend.module)
    
    return program


def emit_to_file(code: str, filename: str) -> None:
    """
    Write compiled code to a file.
    
    Args:
        code: The compiled code
        filename: The path to the output file
    """
    with open(filename, 'w') as f:
        f.write(code)


def emit_object_code(backend: Backend, filename: str) -> None:
    """
    Emit object code for backends that support it.
    
    Args:
        backend: The backend to use
        filename: The path to the output file
    """
    if isinstance(backend, LLVMBackend):
        backend.emit_object_code(filename)
    else:
        raise ValueError(f"Backend {type(backend).__name__} does not support object code emission") 