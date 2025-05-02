"""
Pylean code generation module.

This module implements code generation from Pylean expressions to
executable code, supporting both C and LLVM backends.
"""

from pylean.codegen.backend import Backend, BackendType
from pylean.codegen.c_backend import CBackend
from pylean.codegen.llvm_backend import LLVMBackend
from pylean.codegen.compiler import compile_expr, compile_decl, compile_program, emit_to_file, emit_object_code

__all__ = [
    'Backend',
    'BackendType',
    'CBackend',
    'LLVMBackend',
    'compile_expr',
    'compile_decl',
    'compile_program',
    'emit_to_file',
    'emit_object_code',
] 