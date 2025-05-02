"""
Backend interface for code generation.

This module defines the abstract Backend class that all code generation
backends must implement.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

from pylean.kernel import Expr, Environment, Declaration


class BackendType(Enum):
    """Enum for the different backend types."""
    C = "c"
    LLVM = "llvm"


class Backend(ABC):
    """
    Abstract base class for code generation backends.
    
    A backend is responsible for converting Pylean expressions
    to the target language (e.g., C or LLVM IR).
    """
    
    def __init__(self, env: Environment):
        """
        Initialize the backend.
        
        Args:
            env: The environment containing declarations
        """
        self.env = env
        self.emitted_decls: Set[str] = set()
        
    @abstractmethod
    def compile_expr(self, expr: Expr) -> str:
        """
        Compile an expression to the target language.
        
        Args:
            expr: The expression to compile
            
        Returns:
            The compiled code as a string
        """
        pass
    
    @abstractmethod
    def compile_decl(self, decl: Declaration) -> str:
        """
        Compile a declaration to the target language.
        
        Args:
            decl: The declaration to compile
            
        Returns:
            The compiled code as a string
        """
        pass
    
    @abstractmethod
    def generate_program(self, decls: List[Declaration], main_fn: Optional[str] = None) -> str:
        """
        Generate a complete program from a list of declarations.
        
        Args:
            decls: The declarations to include in the program
            main_fn: Optional name of the main function
            
        Returns:
            The complete program code as a string
        """
        pass
    
    def should_generate_code(self, decl: Declaration) -> bool:
        """
        Determine if code should be generated for a declaration.
        
        Args:
            decl: The declaration to check
            
        Returns:
            True if code should be generated, False otherwise
        """
        # Don't generate code for propositions (proofs)
        # Don't generate code for type formers
        # Don't generate code for inductive type declarations (only constructors)
        # This is a simplified version of Lean's shouldGenerateCode
        return True 