"""
Kernel module for the Pylean theorem prover.

This module integrates the environment, expressions, and type checking systems
to provide a complete Lean4-compatible kernel implementation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, cast

from pylean.kernel.expr import (
    Expr, Name, ExprKind, SortExpr, ConstExpr, AppExpr, LambdaExpr, PiExpr,
    mk_sort, mk_const, mk_app, mk_lambda, mk_pi
)
from pylean.kernel.env import (
    Environment, Declaration, DeclKind,
    AxiomDecl, DefinitionDecl, TheoremDecl, OpaqueDecl, ConstantDecl, InductiveDecl,
    mk_axiom, mk_definition, mk_theorem, mk_opaque, mk_constant, mk_inductive, mk_constructor
)
from pylean.kernel.typecheck import (
    Context, TypeError, infer_type, check_type, ensure_type_is_valid, type_check_declaration
)
from pylean.kernel.reduce import (
    reduce, is_def_eq, ReductionStrategy, ReductionMode
)


class KernelException(Exception):
    """Base exception for kernel errors."""
    pass


class TypeCheckException(KernelException):
    """Exception raised for type checking errors in the kernel."""
    pass


class NameAlreadyExistsException(KernelException):
    """Exception raised when a name already exists in the environment."""
    pass


class Kernel:
    """
    Main kernel class that provides an interface to the theorem prover.
    
    This class manages the environment and provides methods for adding
    declarations, type checking expressions, and evaluating expressions.
    """
    
    def __init__(self, env: Optional[Environment] = None):
        """Initialize the kernel with an optional environment."""
        from pylean.kernel.env import mk_std_env
        self.env = env if env is not None else mk_std_env()
    
    def add_axiom(self, name: Union[Name, str], type_expr: Expr, 
                 universe_params: List[Name] = None) -> Kernel:
        """
        Add an axiom to the environment.
        
        Args:
            name: The name of the axiom
            type_expr: The type of the axiom
            universe_params: Universe parameters (if any)
        
        Returns:
            A new Kernel instance with the axiom added
        
        Raises:
            NameAlreadyExistsException: If the name already exists
            TypeCheckException: If the type is not well-formed
        """
        if self.env.find_decl(name) is not None:
            raise NameAlreadyExistsException(f"Declaration '{name}' already exists")
        
        try:
            # Create an empty context
            ctx = Context()
            # Check that the type is well-formed
            ensure_type_is_valid(self.env, ctx, type_expr)
            
            # Create the axiom declaration
            axiom = mk_axiom(name, type_expr, universe_params)
            # Create a new environment with the axiom added
            new_env = self.env.add_decl(axiom)
            # Return a new kernel with the updated environment
            return Kernel(new_env)
        
        except TypeError as e:
            raise TypeCheckException(f"Type error in axiom '{name}': {e}")
    
    def add_definition(self, name: Union[Name, str], type_expr: Expr, value: Expr,
                      universe_params: List[Name] = None) -> Kernel:
        """
        Add a definition to the environment.
        
        Args:
            name: The name of the definition
            type_expr: The type of the definition
            value: The value of the definition
            universe_params: Universe parameters (if any)
        
        Returns:
            A new Kernel instance with the definition added
        
        Raises:
            NameAlreadyExistsException: If the name already exists
            TypeCheckException: If the type or value is not well-formed
        """
        if self.env.find_decl(name) is not None:
            raise NameAlreadyExistsException(f"Declaration '{name}' already exists")
        
        try:
            # Create an empty context
            ctx = Context()
            # Check that the type is well-formed
            ensure_type_is_valid(self.env, ctx, type_expr)
            # Check that the value has the declared type
            check_type(self.env, ctx, value, type_expr)
            
            # Create the definition declaration
            definition = mk_definition(name, type_expr, value, universe_params)
            # Create a new environment with the definition added
            new_env = self.env.add_decl(definition)
            # Return a new kernel with the updated environment
            return Kernel(new_env)
        
        except TypeError as e:
            raise TypeCheckException(f"Type error in definition '{name}': {e}")
    
    def add_theorem(self, name: Union[Name, str], type_expr: Expr, proof: Expr,
                   universe_params: List[Name] = None) -> Kernel:
        """
        Add a theorem to the environment.
        
        Args:
            name: The name of the theorem
            type_expr: The type of the theorem
            proof: The proof of the theorem
            universe_params: Universe parameters (if any)
        
        Returns:
            A new Kernel instance with the theorem added
        
        Raises:
            NameAlreadyExistsException: If the name already exists
            TypeCheckException: If the type or proof is not well-formed
        """
        if self.env.find_decl(name) is not None:
            raise NameAlreadyExistsException(f"Declaration '{name}' already exists")
        
        try:
            # Create an empty context
            ctx = Context()
            # Check that the type is well-formed
            ensure_type_is_valid(self.env, ctx, type_expr)
            # Check that the proof has the declared type
            check_type(self.env, ctx, proof, type_expr)
            
            # Create the theorem declaration
            theorem = mk_theorem(name, type_expr, proof, universe_params)
            # Create a new environment with the theorem added
            new_env = self.env.add_decl(theorem)
            # Return a new kernel with the updated environment
            return Kernel(new_env)
        
        except TypeError as e:
            raise TypeCheckException(f"Type error in theorem '{name}': {e}")
    
    def add_constant(self, name: Union[Name, str], type_expr: Expr,
                    universe_params: List[Name] = None) -> Kernel:
        """
        Add a constant to the environment.
        
        Args:
            name: The name of the constant
            type_expr: The type of the constant
            universe_params: Universe parameters (if any)
        
        Returns:
            A new Kernel instance with the constant added
        
        Raises:
            NameAlreadyExistsException: If the name already exists
            TypeCheckException: If the type is not well-formed
        """
        if self.env.find_decl(name) is not None:
            raise NameAlreadyExistsException(f"Declaration '{name}' already exists")
        
        try:
            # Create an empty context
            ctx = Context()
            # Check that the type is well-formed
            ensure_type_is_valid(self.env, ctx, type_expr)
            
            # Create the constant declaration
            constant = mk_constant(name, type_expr, universe_params)
            # Create a new environment with the constant added
            new_env = self.env.add_decl(constant)
            # Return a new kernel with the updated environment
            return Kernel(new_env)
        
        except TypeError as e:
            raise TypeCheckException(f"Type error in constant '{name}': {e}")
    
    def add_inductive(self, name: Union[Name, str], type_expr: Expr,
                     constructors: List[Tuple[Union[Name, str], Expr]],
                     universe_params: List[Name] = None) -> Kernel:
        """
        Add an inductive type to the environment.
        
        Args:
            name: The name of the inductive type
            type_expr: The type of the inductive type
            constructors: List of (name, type) pairs for constructors
            universe_params: Universe parameters (if any)
        
        Returns:
            A new Kernel instance with the inductive type added
        
        Raises:
            NameAlreadyExistsException: If the name already exists but not as a compatible constant
            TypeCheckException: If the type is not well-formed
        """
        # Check if the name already exists in the environment
        existing_decl = self.env.find_decl(name)
        
        # Special handling for existing declarations
        if existing_decl is not None:
            # If it's already an inductive type, raise an exception
            if existing_decl.kind == DeclKind.INDUCTIVE:
                raise NameAlreadyExistsException(f"Inductive type '{name}' already exists")
            
            # If it's a constant with the same type, we can proceed
            if existing_decl.kind == DeclKind.CONSTANT:
                # Check that the existing constant has the same type
                existing_type = self.env.get_type(name)
                if not is_def_eq(existing_type, type_expr, self.env):
                    raise TypeCheckException(
                        f"Existing constant '{name}' has incompatible type with the inductive type")
            else:
                # Not a constant or inductive, so raise an exception
                raise NameAlreadyExistsException(
                    f"Declaration '{name}' already exists and is not a constant")
        
        try:
            # Create an empty context
            ctx = Context()
            
            # Check that the type is well-formed
            ensure_type_is_valid(self.env, ctx, type_expr)
            
            # Create constructor declarations
            constructor_decls = []
            
            # First create a temporary environment with the inductive type added as a constant if it doesn't exist
            temp_env = self.env
            if existing_decl is None:
                from pylean.kernel.env import mk_constant
                temp_const = mk_constant(name, type_expr, universe_params)
                temp_env = temp_env.add_decl(temp_const)
            
            # Check that constructors are well-formed using the temporary environment
            for ctor_name, ctor_type in constructors:
                # Check that the constructor name doesn't exist
                if self.env.find_decl(ctor_name) is not None:
                    raise NameAlreadyExistsException(
                        f"Constructor '{ctor_name}' already exists")
                
                # Check that the constructor type is well-formed in the environment with the type available
                ensure_type_is_valid(temp_env, ctx, ctor_type)
                
                # Create the constructor declaration
                ctor = mk_constructor(ctor_name, ctor_type, name)
                constructor_decls.append(ctor)
            
            # Now create the inductive type declaration
            from pylean.kernel.env import mk_inductive
            inductive = mk_inductive(name, type_expr, constructor_decls, universe_params)
            
            # Add the inductive type to the environment
            new_env = self.env.add_decl(inductive)
            
            # Return a new kernel with the updated environment
            return Kernel(new_env)
        
        except TypeError as e:
            raise TypeCheckException(f"Type error in inductive type '{name}': {e}")
    
    def infer_type(self, expr: Expr, ctx: Optional[Context] = None) -> Expr:
        """
        Infer the type of an expression.
        
        Args:
            expr: The expression to type check
            ctx: Optional context (empty by default)
        
        Returns:
            The type of the expression
        
        Raises:
            TypeCheckException: If the expression is not well-typed
        """
        try:
            if ctx is None:
                ctx = Context()
            return infer_type(self.env, ctx, expr)
        except TypeError as e:
            raise TypeCheckException(f"Type error: {e}")
    
    def check_type(self, expr: Expr, expected_type: Expr, 
                  ctx: Optional[Context] = None) -> None:
        """
        Check that an expression has the expected type.
        
        Args:
            expr: The expression to check
            expected_type: The expected type
            ctx: Optional context (empty by default)
        
        Raises:
            TypeCheckException: If the expression does not have the expected type
        """
        try:
            if ctx is None:
                ctx = Context()
            check_type(self.env, ctx, expr, expected_type)
        except TypeError as e:
            raise TypeCheckException(f"Type error: {e}")
    
    def check_declaration(self, name: Union[Name, str]) -> bool:
        """
        Check that a declaration is well-typed.
        
        Args:
            name: The name of the declaration to check
        
        Returns:
            True if the declaration is well-typed, False otherwise
        """
        return type_check_declaration(self.env, name)
    
    def is_def_eq(self, expr1: Expr, expr2: Expr) -> bool:
        """
        Check if two expressions are definitionally equal.
        
        Args:
            expr1: The first expression
            expr2: The second expression
        
        Returns:
            True if the expressions are definitionally equal, False otherwise
        """
        return is_def_eq(expr1, expr2, self.env)
    
    def reduce(self, expr: Expr, strategy: ReductionStrategy = ReductionStrategy.WHNF) -> Expr:
        """
        Reduce an expression to its normal form.
        
        Args:
            expr: The expression to reduce
            strategy: The reduction strategy to use
        
        Returns:
            The reduced expression
        """
        return reduce(expr, self.env, strategy, ReductionMode.ALL)
    
    def normalize(self, expr: Expr) -> Expr:
        """
        Normalize an expression to its full normal form.
        
        Args:
            expr: The expression to normalize
        
        Returns:
            The normalized expression
        """
        return reduce(expr, self.env, ReductionStrategy.NF, ReductionMode.ALL)
    
    def get_environment(self) -> Environment:
        """
        Get the current environment.
        
        Returns:
            The current environment
        """
        return self.env
    
    def create_child_kernel(self) -> Kernel:
        """
        Create a new kernel with a child environment.
        
        Returns:
            A new kernel with a child environment
        """
        child_env = self.env.create_child()
        return Kernel(child_env) 