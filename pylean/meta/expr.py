"""
Metaprogramming expression module.

This module implements the core types and functions for
manipulating expressions at compile time in Pylean.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Generic, Union

from pylean.kernel import (
    Expr, ExprKind, Name, Declaration, Environment, Context, Kernel,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi
)

# Type variable for the MetaM monad
T = TypeVar('T')


@dataclass
class MetaState:
    """
    State for metaprogramming operations.
    
    This represents the current state of a metaprogramming
    operation, including the environment and local context.
    """
    env: Environment
    ctx: Context
    locals: Dict[str, Expr] = None
    
    def __post_init__(self):
        if self.locals is None:
            self.locals = {}


@dataclass
class MetaResult(Generic[T]):
    """
    Result of a metaprogramming operation.
    
    This includes both the result value and the updated state.
    """
    value: T
    state: MetaState


class MetaM(Generic[T]):
    """
    Monad for metaprogramming operations.
    
    This represents a computation that manipulates expressions
    at compile time, with access to the environment and context.
    """
    
    def __init__(self, run_fn: Callable[[MetaState], MetaResult[T]]):
        """
        Initialize a metaprogramming monad.
        
        Args:
            run_fn: The function to run the monad
        """
        self.run = run_fn
    
    def bind(self, f: Callable[[T], 'MetaM[Any]']) -> 'MetaM[Any]':
        """
        Bind a function to the result of this monad.
        
        Args:
            f: The function to bind
            
        Returns:
            A new monad with the function applied
        """
        def run_fn(state: MetaState) -> MetaResult[Any]:
            # Run the current monad
            result = self.run(state)
            # Run the function on the result
            return f(result.value).run(result.state)
        
        return MetaM(run_fn)
    
    def map(self, f: Callable[[T], Any]) -> 'MetaM[Any]':
        """
        Map a function over the result of this monad.
        
        Args:
            f: The function to map
            
        Returns:
            A new monad with the function applied
        """
        def run_fn(state: MetaState) -> MetaResult[Any]:
            # Run the current monad
            result = self.run(state)
            # Apply the function to the result
            return MetaResult(f(result.value), result.state)
        
        return MetaM(run_fn)
    
    @staticmethod
    def pure(value: T) -> 'MetaM[T]':
        """
        Create a monad with a pure value.
        
        Args:
            value: The value to use
            
        Returns:
            A new monad with the value
        """
        def run_fn(state: MetaState) -> MetaResult[T]:
            return MetaResult(value, state)
        
        return MetaM(run_fn)
    
    @staticmethod
    def get_env() -> 'MetaM[Environment]':
        """
        Get the current environment.
        
        Returns:
            A monad with the environment
        """
        def run_fn(state: MetaState) -> MetaResult[Environment]:
            return MetaResult(state.env, state)
        
        return MetaM(run_fn)
    
    @staticmethod
    def get_ctx() -> 'MetaM[Context]':
        """
        Get the current context.
        
        Returns:
            A monad with the context
        """
        def run_fn(state: MetaState) -> MetaResult[Context]:
            return MetaResult(state.ctx, state)
        
        return MetaM(run_fn)
    
    @staticmethod
    def set_env(env: Environment) -> 'MetaM[None]':
        """
        Set the current environment.
        
        Args:
            env: The new environment
            
        Returns:
            A monad with no result
        """
        def run_fn(state: MetaState) -> MetaResult[None]:
            new_state = MetaState(env, state.ctx, state.locals)
            return MetaResult(None, new_state)
        
        return MetaM(run_fn)
    
    @staticmethod
    def set_ctx(ctx: Context) -> 'MetaM[None]':
        """
        Set the current context.
        
        Args:
            ctx: The new context
            
        Returns:
            A monad with no result
        """
        def run_fn(state: MetaState) -> MetaResult[None]:
            new_state = MetaState(state.env, ctx, state.locals)
            return MetaResult(None, new_state)
        
        return MetaM(run_fn)
    
    @staticmethod
    def get_local(name: str) -> 'MetaM[Optional[Expr]]':
        """
        Get a local variable.
        
        Args:
            name: The name of the variable
            
        Returns:
            A monad with the variable expression or None
        """
        def run_fn(state: MetaState) -> MetaResult[Optional[Expr]]:
            return MetaResult(state.locals.get(name), state)
        
        return MetaM(run_fn)
    
    @staticmethod
    def set_local(name: str, expr: Expr) -> 'MetaM[None]':
        """
        Set a local variable.
        
        Args:
            name: The name of the variable
            expr: The expression to set
            
        Returns:
            A monad with no result
        """
        def run_fn(state: MetaState) -> MetaResult[None]:
            new_locals = state.locals.copy()
            new_locals[name] = expr
            new_state = MetaState(state.env, state.ctx, new_locals)
            return MetaResult(None, new_state)
        
        return MetaM(run_fn)
    
    @staticmethod
    def infer_type(expr: Expr) -> 'MetaM[Expr]':
        """
        Infer the type of an expression.
        
        Args:
            expr: The expression to check
            
        Returns:
            A monad with the type expression
        """
        def run_fn(state: MetaState) -> MetaResult[Expr]:
            # Create a temporary kernel for type checking
            kernel = Kernel(state.env)
            
            # Infer the type in the current context
            type_expr = kernel.infer_type(expr, state.ctx)
            
            return MetaResult(type_expr, state)
        
        return MetaM(run_fn)
    
    @staticmethod
    def add_decl(decl: Declaration) -> 'MetaM[Environment]':
        """
        Add a declaration to the environment.
        
        Args:
            decl: The declaration to add
            
        Returns:
            A monad with the updated environment
        """
        def run_fn(state: MetaState) -> MetaResult[Environment]:
            # Create a temporary kernel for adding the declaration
            kernel = Kernel(state.env)
            
            # Add the declaration to get a new environment
            new_env = kernel.env.add_decl(decl)
            
            # Update the state with the new environment
            new_state = MetaState(new_env, state.ctx, state.locals)
            
            return MetaResult(new_env, new_state)
        
        return MetaM(run_fn)


class MetaExpr:
    """
    Metaprogramming expression builder.
    
    This provides a more convenient interface for constructing
    and manipulating expressions at compile time.
    """
    
    @staticmethod
    def var(idx: int) -> MetaM[Expr]:
        """
        Create a variable expression.
        
        Args:
            idx: The De Bruijn index
            
        Returns:
            A monad with the variable expression
        """
        return MetaM.pure(mk_var(idx))
    
    @staticmethod
    def sort(level: int = 0) -> MetaM[Expr]:
        """
        Create a sort expression.
        
        Args:
            level: The sort level
            
        Returns:
            A monad with the sort expression
        """
        return MetaM.pure(mk_sort(level))
    
    @staticmethod
    def const(name: Union[str, Name]) -> MetaM[Expr]:
        """
        Create a constant expression.
        
        Args:
            name: The constant name
            
        Returns:
            A monad with the constant expression
        """
        return MetaM.pure(mk_const(name))
    
    @staticmethod
    def app(fn: Expr, arg: Expr) -> MetaM[Expr]:
        """
        Create an application expression.
        
        Args:
            fn: The function expression
            arg: The argument expression
            
        Returns:
            A monad with the application expression
        """
        return MetaM.pure(mk_app(fn, arg))
    
    @staticmethod
    def lambda_expr(name: str, type_expr: Expr, body: Expr) -> MetaM[Expr]:
        """
        Create a lambda expression.
        
        Args:
            name: The variable name
            type_expr: The variable type
            body: The body expression
            
        Returns:
            A monad with the lambda expression
        """
        return MetaM.pure(mk_lambda(name, type_expr, body))
    
    @staticmethod
    def pi(name: str, type_expr: Expr, body: Expr) -> MetaM[Expr]:
        """
        Create a Pi expression.
        
        Args:
            name: The variable name
            type_expr: The variable type
            body: The body expression
            
        Returns:
            A monad with the Pi expression
        """
        return MetaM.pure(mk_pi(name, type_expr, body))


def run_meta(meta: MetaM[T], env: Environment, ctx: Optional[Context] = None) -> T:
    """
    Run a metaprogramming operation.
    
    Args:
        meta: The metaprogramming operation
        env: The environment to use
        ctx: The context to use, or None for an empty context
        
    Returns:
        The result of the operation
    """
    # Create an initial state
    state = MetaState(env, ctx or Context())
    
    # Run the metaprogramming operation
    result = meta.run(state)
    
    # Return the result value
    return result.value 