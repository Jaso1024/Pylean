"""
Type checking module for the Pylean kernel.

This module provides functions for type checking expressions
in the context of an environment and local context.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, cast

from pylean.kernel.expr import (
    Expr, Name, ExprKind, 
    VarExpr, SortExpr, ConstExpr, AppExpr, LambdaExpr, PiExpr, LetExpr, LocalExpr, MatchExpr,
    Alternative, Pattern, occurs_in, instantiate, lift, mk_sort, mk_pi, mk_app, ExternExpr
)
from pylean.kernel.env import Environment
from pylean.kernel.reduce import is_def_eq, reduce, ReductionStrategy, ReductionMode


@dataclass
class Context:
    """
    A typing context for local variables.
    
    This represents the local context during type checking, 
    tracking the types of bound variables.
    """
    types: List[Expr] = field(default_factory=list)
    names: List[Name] = field(default_factory=list)
    
    def extend(self, name: Name, type_expr: Expr) -> Context:
        """
        Extend the context with a new variable.
        
        Returns a new context with the variable added.
        """
        new_ctx = Context(self.types.copy(), self.names.copy())
        new_ctx.types.insert(0, type_expr)  # Add at the beginning (de Bruijn index 0)
        new_ctx.names.insert(0, name)
        return new_ctx
    
    def copy(self) -> Context:
        """
        Create a copy of this context.
        
        Returns a new context with the same types and names.
        """
        return Context(self.types.copy(), self.names.copy())
    
    def lookup_type(self, idx: int) -> Optional[Expr]:
        """
        Get the type of a variable by de Bruijn index.
        
        Returns None if the index is out of bounds.
        """
        if 0 <= idx < len(self.types):
            return self.types[idx]
        return None
    
    def lookup_name(self, idx: int) -> Optional[Name]:
        """
        Get the name of a variable by de Bruijn index.
        
        Returns None if the index is out of bounds.
        """
        if 0 <= idx < len(self.names):
            return self.names[idx]
        return None


class TypeError(Exception):
    """Exception raised for type checking errors."""
    pass


def infer_type(env: Environment, ctx: Context, expr: Expr) -> Expr:
    """
    Infer the type of an expression in the given environment and context.
    
    Args:
        env: The environment containing declarations
        ctx: The local context
        expr: The expression to type check
    
    Returns:
        The type of the expression
    
    Raises:
        TypeError: If the expression is not well-typed
    """
    if expr.kind == ExprKind.VAR:
        # Variable case: look up in context
        var_expr = cast(VarExpr, expr)
        var_type = ctx.lookup_type(var_expr.idx)
        if var_type is None:
            raise TypeError(f"Variable #{var_expr.idx} not found in context")
        return var_type
    
    elif expr.kind == ExprKind.SORT:
        # Sort case: Type i has type Type (i+1)
        sort_expr = cast(SortExpr, expr)
        level_val = 0
        if hasattr(sort_expr.level, 'param') and sort_expr.level.param is not None:
            level_val = sort_expr.level.param
        return mk_sort(level_val + 1)
    
    elif expr.kind == ExprKind.CONST:
        # Constant case: look up in environment
        const_expr = cast(ConstExpr, expr)
        const_type = env.get_type(const_expr.name)
        if const_type is None:
            raise TypeError(f"Constant {const_expr.name} not found in environment")
        # TODO: Handle universe level instantiation
        return const_type
    
    elif expr.kind == ExprKind.APP:
        # Application case: check function and argument
        app_expr = cast(AppExpr, expr)
        fn_type = infer_type(env, ctx, app_expr.fn)
        
        # Reduce the function type to WHNF
        fn_type = reduce(fn_type, env, ReductionStrategy.WHNF, ReductionMode.ALL)
        
        # Ensure function type is a Pi type
        if fn_type.kind != ExprKind.PI:
            raise TypeError(f"Expected function type, got {fn_type}")
        
        pi_type = cast(PiExpr, fn_type)
        
        # Check argument type
        arg_type = infer_type(env, ctx, app_expr.arg)
        if not is_type_convertible(env, ctx, arg_type, pi_type.type):
            raise TypeError(f"Argument type mismatch: expected {pi_type.type}, got {arg_type}")
        
        # Return the result type with the argument substituted
        return instantiate(pi_type.body, app_expr.arg)
    
    elif expr.kind == ExprKind.LAMBDA:
        # Lambda case: introduce variable and check body
        lambda_expr = cast(LambdaExpr, expr)
        
        # Check that the domain type is well-formed
        ensure_type_is_valid(env, ctx, lambda_expr.type)
        
        # Extend context with the new variable
        new_ctx = ctx.extend(lambda_expr.name, lambda_expr.type)
        
        # Infer the type of the body
        body_type = infer_type(env, new_ctx, lambda_expr.body)
        
        # Construct the function type
        return mk_pi(lambda_expr.name, lambda_expr.type, body_type, lambda_expr.binder_info)
    
    elif expr.kind == ExprKind.PI:
        # Pi case: check domain and codomain
        pi_expr = cast(PiExpr, expr)
        
        # Check that the domain type is well-formed
        domain_type = infer_type(env, ctx, pi_expr.type)
        if domain_type.kind != ExprKind.SORT:
            raise TypeError(f"Expected sort, got {domain_type}")
        
        # Extend context with the new variable
        new_ctx = ctx.extend(pi_expr.name, pi_expr.type)
        
        # Check that the codomain is well-formed
        codomain_type = infer_type(env, new_ctx, pi_expr.body)
        if codomain_type.kind != ExprKind.SORT:
            raise TypeError(f"Expected sort, got {codomain_type}")
        
        # The type of a Pi type is the max of the domain and codomain sorts
        domain_sort = cast(SortExpr, domain_type)
        codomain_sort = cast(SortExpr, codomain_type)
        
        # TODO: Implement proper universe level calculation
        # For now, just take the maximum
        domain_level = 0
        codomain_level = 0
        if hasattr(domain_sort.level, 'param') and domain_sort.level.param is not None:
            domain_level = domain_sort.level.param
        if hasattr(codomain_sort.level, 'param') and codomain_sort.level.param is not None:
            codomain_level = codomain_sort.level.param
        
        return mk_sort(max(domain_level, codomain_level))
    
    elif expr.kind == ExprKind.LET:
        # Let case: check definition and body
        let_expr = cast(LetExpr, expr)
        
        # Check the definition type
        def_type = infer_type(env, ctx, let_expr.value)
        if not is_type_convertible(env, ctx, def_type, let_expr.type):
            raise TypeError(f"Let definition type mismatch: expected {let_expr.type}, got {def_type}")
        
        # Extend context with the new variable
        new_ctx = ctx.extend(let_expr.name, let_expr.type)
        
        # Infer the type of the body
        body_type = infer_type(env, new_ctx, let_expr.body)
        
        # Substitute the definition in the body type
        return instantiate(body_type, let_expr.value)
    
    elif expr.kind == ExprKind.LOCAL:
        # Local constant case
        local_expr = cast(LocalExpr, expr)
        return local_expr.type
    
    elif expr.kind == ExprKind.MATCH:
        # Match case: check scrutinee and alternatives
        match_expr = cast(MatchExpr, expr)
        
        # Infer the type of the scrutinee
        scrutinee_type = infer_type(env, ctx, match_expr.scrutinee)
        scrutinee_type = reduce(scrutinee_type, env, ReductionStrategy.WHNF, ReductionMode.ALL)
        
        # For each alternative, check that:
        # 1. The pattern constructor is valid for the scrutinee type
        # 2. The pattern has the correct number of fields
        # 3. The alternative body has the expected type
        
        # Get the inductive type name from the scrutinee type
        inductive_name = None
        curr_type = scrutinee_type
        while curr_type.kind == ExprKind.APP:
            curr_type = cast(AppExpr, curr_type).fn
        
        if curr_type.kind == ExprKind.CONST:
            inductive_name = cast(ConstExpr, curr_type).name
        
        if inductive_name is None:
            raise TypeError(f"Scrutinee type {scrutinee_type} is not an inductive type")
        
        # Get the inductive type information from the environment
        inductive_info = env.get_inductive_info(inductive_name)
        if inductive_info is None:
            raise TypeError(f"Inductive type {inductive_name} not found in environment")
        
        # Check each alternative
        for alt in match_expr.alternatives:
            # Check that the constructor is valid for this inductive type
            constructor_info = env.get_constructor_info(alt.pattern.constructor)
            if constructor_info is None:
                raise TypeError(f"Constructor {alt.pattern.constructor} not found in environment")
            
            if constructor_info.get('inductive_type') != str(inductive_name):
                raise TypeError(f"Constructor {alt.pattern.constructor} doesn't belong to type {inductive_name}")
            
            # Check that the pattern has the correct number of fields
            constructor_arity = constructor_info.get('arity', 0)
            if len(alt.pattern.fields) != constructor_arity:
                raise TypeError(f"Constructor {alt.pattern.constructor} expects {constructor_arity} fields, but pattern has {len(alt.pattern.fields)}")
            
            # Create a new context with the pattern fields
            alt_ctx = ctx.copy()
            for field_name in alt.pattern.fields:
                # For simplicity, we assume all fields have the same type as the scrutinee
                # In a full implementation, we would use the correct field types
                alt_ctx = alt_ctx.extend(Name.from_string(field_name), scrutinee_type)
            
            # Check that the alternative body has the expected type
            alt_type = infer_type(env, alt_ctx, alt.expr)
            if not is_type_convertible(env, alt_ctx, alt_type, match_expr.type):
                raise TypeError(f"Alternative type mismatch: expected {match_expr.type}, got {alt_type}")
        
        # The type of the match expression is already stored in the expression
        return match_expr.type
    
    elif expr.kind == ExprKind.EXTERN:
        # External function declaration
        extern_expr = cast(ExternExpr, expr)
        
        # Check that all parameter types and return type are valid types
        for param_type in extern_expr.param_types:
            ensure_type_is_valid(env, ctx, param_type)
        
        ensure_type_is_valid(env, ctx, extern_expr.return_type)
        
        # Construct a Pi type: (x1 : T1) -> ... -> (xn : Tn) -> R
        # Start with the return type
        result_type = extern_expr.return_type
        
        # Work backwards through parameter types
        for i, param_type in enumerate(reversed(extern_expr.param_types)):
            # Create a temporary name for each parameter
            param_name = Name.from_string(f"arg{i}")
            result_type = mk_pi(param_name, param_type, result_type)
        
        return result_type
    
    else:
        raise TypeError(f"Unknown expression kind: {expr.kind}")


def ensure_type_is_valid(env: Environment, ctx: Context, type_expr: Expr) -> None:
    """
    Ensure that a type expression is valid (has type Sort).
    
    Args:
        env: The environment
        ctx: The local context
        type_expr: The type expression to check
    
    Raises:
        TypeError: If the type is not valid
    """
    # Special handling for variables - they must have a type that is a sort
    if type_expr.kind == ExprKind.VAR:
        var_expr = cast(VarExpr, type_expr)
        var_type = ctx.lookup_type(var_expr.idx)
        if var_type is None:
            raise TypeError(f"Variable #{var_expr.idx} not found in context")
            
        # Check if the variable's type is a sort
        var_type_of_type = infer_type(env, ctx, var_type)
        if var_type_of_type.kind != ExprKind.SORT:
            raise TypeError(f"Variable type must be a sort, got {var_type}")
    else:
        # For other expressions, check if their type is a sort
        try:
            type_of_type = infer_type(env, ctx, type_expr)
            
            # Reduce to WHNF
            type_of_type = reduce(type_of_type, env, ReductionStrategy.WHNF, ReductionMode.ALL)
            
            if type_of_type.kind != ExprKind.SORT:
                raise TypeError(f"Expected sort, got {type_of_type}")
        except TypeError as e:
            # Re-raise any type errors that occur during inference
            raise TypeError(f"Invalid type expression: {e}")


def check_type(env: Environment, ctx: Context, expr: Expr, expected_type: Expr) -> None:
    """
    Check that an expression has the expected type.
    
    Args:
        env: The environment
        ctx: The local context
        expr: The expression to check
        expected_type: The expected type
    
    Raises:
        TypeError: If the expression does not have the expected type
    """
    actual_type = infer_type(env, ctx, expr)
    if not is_type_convertible(env, ctx, actual_type, expected_type):
        raise TypeError(f"Type mismatch: expected {expected_type}, got {actual_type}")


def is_type_convertible(env: Environment, ctx: Context, type1: Expr, type2: Expr) -> bool:
    """
    Check if two types are convertible (equal up to definitional equality).
    
    Args:
        env: The environment
        ctx: The local context
        type1: The first type
        type2: The second type
    
    Returns:
        True if the types are convertible, False otherwise
    """
    # Use the reduction system's definitional equality check
    return is_def_eq(type1, type2, env)


def type_check_declaration(env: Environment, decl_name: Union[Name, str]) -> bool:
    """
    Type check a declaration in the environment.
    
    Args:
        env: The environment
        decl_name: The name of the declaration to check
    
    Returns:
        True if the declaration is well-typed, False otherwise
    """
    decl = env.find_decl(decl_name)
    if decl is None:
        return False
    
    try:
        # Create an empty context
        ctx = Context()
        
        # Check that the declaration type is well-formed
        ensure_type_is_valid(env, ctx, decl.type)
        
        # For definitions, check that the value has the declared type
        from pylean.kernel.env import DefinitionDecl, TheoremDecl, OpaqueDecl
        if isinstance(decl, (DefinitionDecl, TheoremDecl, OpaqueDecl)):
            check_type(env, ctx, decl.value, decl.type)
        
        return True
    except TypeError:
        return False


def is_type(env: Environment, ctx: Context, expr: Expr) -> bool:
    """
    Check if an expression is a valid type.
    
    A valid type is either a Sort or a Pi type where the codomain is a valid type.
    
    Args:
        env: The environment containing declarations
        ctx: The local context
        expr: The expression to check
    
    Returns:
        True if the expression is a valid type, False otherwise
    """
    try:
        # Infer the type of the expression
        expr_type = infer_type(env, ctx, expr)
        
        # A valid type has a Sort as its type
        return expr_type.kind == ExprKind.SORT
    except TypeError:
        return False 