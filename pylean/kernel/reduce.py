"""
Reduction module for the Pylean kernel.

This module provides functions for normalizing and evaluating
expressions through various reduction strategies.
"""

from __future__ import annotations
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union, cast

from pylean.kernel.expr import (
    Expr, Name, ExprKind,
    VarExpr, SortExpr, ConstExpr, AppExpr, LambdaExpr, PiExpr, LetExpr, MatchExpr,
    Alternative, Pattern, lift, instantiate, mk_app, mk_lambda, mk_let, mk_match
)
from pylean.kernel.env import Environment, DefinitionDecl, OpaqueDecl, TheoremDecl


class ReductionStrategy(Enum):
    """Strategies for reducing expressions."""
    WHNF = auto()    # Weak Head Normal Form (lazy)
    NF = auto()      # Full Normal Form (eager)


class ReductionMode(Enum):
    """Modes of reduction."""
    BETA = auto()    # Beta reduction only (λx. e) a => e[a/x]
    ZETA = auto()    # Let expansion only (let x := v in e) => e[v/x]
    DELTA = auto()   # Definition unfolding
    ALL = auto()     # All reductions


def is_stuck(expr: Expr) -> bool:
    """
    Check if an expression is "stuck" - cannot be reduced further.
    
    Args:
        expr: The expression to check
    
    Returns:
        True if the expression is stuck, False otherwise
    """
    if expr.kind == ExprKind.APP:
        app_expr = cast(AppExpr, expr)
        # If the function is a lambda, the application can be reduced (not stuck)
        if app_expr.fn.kind == ExprKind.LAMBDA:
            return False
        # Otherwise, it's stuck if the function is stuck
        return is_stuck(app_expr.fn)
    
    # Match expressions can be reduced if the scrutinee is a constructor application
    if expr.kind == ExprKind.MATCH:
        match_expr = cast(MatchExpr, expr)
        # Reduce the scrutinee first
        scrutinee = match_expr.scrutinee
        if scrutinee.kind == ExprKind.APP:
            # If scrutinee is a constructor application, match is not stuck
            app_scrutinee = cast(AppExpr, scrutinee)
            fn = app_scrutinee.fn
            while fn.kind == ExprKind.APP:
                fn = cast(AppExpr, fn).fn
            if fn.kind == ExprKind.CONST:
                return False
        return is_stuck(scrutinee)
    
    # Atomic expressions and binding expressions are stuck
    return (expr.kind in (ExprKind.SORT, ExprKind.VAR, ExprKind.LOCAL, ExprKind.META) or
            (expr.kind == ExprKind.CONST) or
            (expr.kind == ExprKind.LAMBDA) or
            (expr.kind == ExprKind.PI))


def reduce_beta(fn: LambdaExpr, arg: Expr) -> Expr:
    """
    Perform beta reduction: (λx. e) a => e[a/x].
    
    Args:
        fn: The lambda expression (function)
        arg: The argument expression
    
    Returns:
        The reduced expression
    """
    return instantiate(fn.body, arg)


def reduce_zeta(let_expr: LetExpr) -> Expr:
    """
    Perform zeta reduction: (let x := v in e) => e[v/x].
    
    Args:
        let_expr: The let expression
    
    Returns:
        The reduced expression
    """
    return instantiate(let_expr.body, let_expr.value)


def reduce_delta(env: Environment, const_expr: ConstExpr) -> Optional[Expr]:
    """
    Perform delta reduction: unfold a definition.
    
    Args:
        env: The environment
        const_expr: The constant expression to unfold
    
    Returns:
        The unfolded definition, or None if the constant is not a definition
    """
    decl = env.find_decl(const_expr.name)
    if decl is None:
        return None
    
    if isinstance(decl, (DefinitionDecl, TheoremDecl)):
        # Unfold the definition
        # TODO: Handle universe level instantiation
        return decl.value
    
    # Not a definition or opaque
    return None


def whnf(expr: Expr, env: Environment, mode: ReductionMode = ReductionMode.ALL) -> Expr:
    """
    Compute the Weak Head Normal Form of an expression.
    
    Reduces only the head of the expression until it reaches a "stuck" form.
    
    Args:
        expr: The expression to reduce
        env: The environment
        mode: The reduction mode to use
    
    Returns:
        The expression in Weak Head Normal Form
    """
    # Base case: already in WHNF
    if is_stuck(expr):
        return expr
    
    # Reduce based on expression kind
    if expr.kind == ExprKind.APP:
        app_expr = cast(AppExpr, expr)
        # Reduce the function first
        fn = whnf(app_expr.fn, env, mode)
        
        # If the head is a lambda, perform beta reduction
        if fn.kind == ExprKind.LAMBDA and (mode in (ReductionMode.BETA, ReductionMode.ALL)):
            lambda_expr = cast(LambdaExpr, fn)
            # Perform beta reduction and continue reducing
            reduced = reduce_beta(lambda_expr, app_expr.arg)
            return whnf(reduced, env, mode)
        
        # If fn was reduced but is not a lambda, update the application
        if fn is not app_expr.fn:
            return mk_app(fn, app_expr.arg)
        
        # No further reduction possible at the head
        return app_expr
    
    elif expr.kind == ExprKind.LET and (mode in (ReductionMode.ZETA, ReductionMode.ALL)):
        # Perform zeta reduction and continue reducing
        let_expr = cast(LetExpr, expr)
        reduced = reduce_zeta(let_expr)
        return whnf(reduced, env, mode)
    
    elif expr.kind == ExprKind.CONST and (mode in (ReductionMode.DELTA, ReductionMode.ALL)):
        # Try to unfold the definition
        const_expr = cast(ConstExpr, expr)
        unfolded = reduce_delta(env, const_expr)
        if unfolded is not None:
            return whnf(unfolded, env, mode)
        return expr
    
    elif expr.kind == ExprKind.MATCH:
        match_expr = cast(MatchExpr, expr)
        
        # Reduce the scrutinee to WHNF
        scrutinee = whnf(match_expr.scrutinee, env, mode)
        
        # If scrutinee is a constructor application, select the matching pattern
        if scrutinee.kind == ExprKind.APP:
            # Find the constructor name (head of the application)
            constructor_expr = scrutinee
            constructor_args = []
            
            # Collect all arguments and find the base constructor
            while constructor_expr.kind == ExprKind.APP:
                app_expr = cast(AppExpr, constructor_expr)
                constructor_args.insert(0, app_expr.arg)  # Insert at beginning to preserve order
                constructor_expr = app_expr.fn
            
            # Check if we have a constructor constant
            if constructor_expr.kind == ExprKind.CONST:
                const_expr = cast(ConstExpr, constructor_expr)
                constructor_name = const_expr.name
                
                # Look for a matching pattern in the alternatives
                for alt in match_expr.alternatives:
                    if alt.pattern.constructor == constructor_name:
                        # Found a matching pattern
                        
                        # Check if the number of fields matches the number of arguments
                        if len(alt.pattern.fields) != len(constructor_args):
                            continue  # Skip this pattern
                        
                        # Create the result expression by instantiating the pattern fields
                        # with the constructor arguments, starting from the innermost binding
                        result = alt.expr
                        for i, arg in enumerate(reversed(constructor_args)):
                            result = instantiate(result, arg)
                        
                        # Continue reducing the result
                        return whnf(result, env, mode)
            
            # No matching pattern found, or not a constructor application
            if scrutinee != match_expr.scrutinee:
                # If scrutinee was reduced, update the match expression
                return mk_match(scrutinee, match_expr.type, match_expr.alternatives)
    
    # No further reduction possible
    return expr


def nf(expr: Expr, env: Environment, mode: ReductionMode = ReductionMode.ALL) -> Expr:
    """
    Compute the Normal Form of an expression.
    
    Fully reduces the expression, including subexpressions.
    
    Args:
        expr: The expression to reduce
        env: The environment
        mode: The reduction mode to use
    
    Returns:
        The expression in Normal Form
    """
    # First, reduce to WHNF
    expr = whnf(expr, env, mode)
    
    # Then, recursively normalize subexpressions
    if expr.kind == ExprKind.APP:
        app_expr = cast(AppExpr, expr)
        # Head is already in WHNF, normalize the argument
        fn = nf(app_expr.fn, env, mode)  # Normalize the function part too
        arg = nf(app_expr.arg, env, mode)
        
        # Create new application with normalized parts
        result = mk_app(fn, arg)
        # If the function is now a lambda (after normalization), reduce it further
        if fn.kind == ExprKind.LAMBDA and (mode in (ReductionMode.BETA, ReductionMode.ALL)):
            lambda_expr = cast(LambdaExpr, fn)
            return nf(reduce_beta(lambda_expr, arg), env, mode)
        return result
    
    elif expr.kind == ExprKind.LAMBDA:
        lambda_expr = cast(LambdaExpr, expr)
        # Normalize the domain and body
        domain = nf(lambda_expr.type, env, mode)
        body = nf(lambda_expr.body, env, mode)
        
        # If domain or body were reduced, update the lambda
        if domain is not lambda_expr.type or body is not lambda_expr.body:
            return mk_lambda(lambda_expr.name, domain, body, lambda_expr.binder_info)
        
        return lambda_expr
    
    elif expr.kind == ExprKind.PI:
        pi_expr = cast(PiExpr, expr)
        # Normalize the domain and codomain
        domain = nf(pi_expr.type, env, mode)
        codomain = nf(pi_expr.body, env, mode)
        
        # If domain or codomain were reduced, update the pi
        if domain is not pi_expr.type or codomain is not pi_expr.body:
            return mk_pi(pi_expr.name, domain, codomain, pi_expr.binder_info)
        
        return pi_expr
    
    elif expr.kind == ExprKind.LET:
        let_expr = cast(LetExpr, expr)
        # We'd normally normalize type, value, and body, but for let expressions,
        # we typically would have already reduced the let via whnf, 
        # so this would be a different expression now.
        if (mode in (ReductionMode.ZETA, ReductionMode.ALL)):
            # Perform zeta reduction and normalize the result
            reduced = reduce_zeta(let_expr)
            return nf(reduced, env, mode)
        else:
            # If we're not doing zeta reduction, normalize the parts
            type_expr = nf(let_expr.type, env, mode)
            value = nf(let_expr.value, env, mode)
            body = nf(let_expr.body, env, mode)
            
            if (type_expr is not let_expr.type or 
                value is not let_expr.value or 
                body is not let_expr.body):
                return mk_let(let_expr.name, type_expr, value, body)
            
            return let_expr
    
    elif expr.kind == ExprKind.MATCH:
        match_expr = cast(MatchExpr, expr)
        # If we get here, the match couldn't be fully reduced in whnf
        # So we normalize its scrutinee and alternatives
        
        scrutinee = nf(match_expr.scrutinee, env, mode)
        new_alternatives = []
        
        for alt in match_expr.alternatives:
            # Normalize the alternative expression
            new_expr = nf(alt.expr, env, mode)
            if new_expr != alt.expr:
                new_alt = Alternative(alt.pattern, new_expr)
                new_alternatives.append(new_alt)
            else:
                new_alternatives.append(alt)
        
        # If something was normalized, create a new match expression
        if (scrutinee != match_expr.scrutinee or 
            any(new_alt != old_alt for new_alt, old_alt in zip(new_alternatives, match_expr.alternatives))):
            return mk_match(scrutinee, match_expr.type, new_alternatives)
        
        return expr
    
    # No further reduction possible
    return expr


def is_def_eq(expr1: Expr, expr2: Expr, env: Environment) -> bool:
    """
    Check if two expressions are definitionally equal.
    
    Expressions are definitionally equal if they reduce to alpha-equivalent terms.
    
    Args:
        expr1: The first expression
        expr2: The second expression
        env: The environment
    
    Returns:
        True if the expressions are definitionally equal, False otherwise
    """
    # Normalize both expressions to ensure complete reduction
    # Use ALL reduction mode to ensure maximum reduction
    norm1 = nf(expr1, env, ReductionMode.ALL)
    norm2 = nf(expr2, env, ReductionMode.ALL)
    
    # Check for syntactic equality of the normalized forms
    # This is a simplified check that doesn't account for alpha-equivalence
    # A full implementation would include proper alpha-equivalence checking
    return expr_equals(norm1, norm2)


def expr_equals(expr1: Expr, expr2: Expr) -> bool:
    """
    Check if two expressions are syntactically equal.
    
    Args:
        expr1: The first expression
        expr2: The second expression
    
    Returns:
        True if the expressions are equal, False otherwise
    """
    if expr1.kind != expr2.kind:
        return False
    
    if expr1.kind == ExprKind.VAR:
        var1 = cast(VarExpr, expr1)
        var2 = cast(VarExpr, expr2)
        return var1.idx == var2.idx
    
    elif expr1.kind == ExprKind.SORT:
        sort1 = cast(SortExpr, expr1)
        sort2 = cast(SortExpr, expr2)
        return str(sort1.level) == str(sort2.level)  # Simplified check
    
    elif expr1.kind == ExprKind.CONST:
        const1 = cast(ConstExpr, expr1)
        const2 = cast(ConstExpr, expr2)
        return (str(const1.name) == str(const2.name) and
                len(const1.levels) == len(const2.levels) and
                all(str(l1) == str(l2) for l1, l2 in zip(const1.levels, const2.levels)))
    
    elif expr1.kind == ExprKind.APP:
        app1 = cast(AppExpr, expr1)
        app2 = cast(AppExpr, expr2)
        return expr_equals(app1.fn, app2.fn) and expr_equals(app1.arg, app2.arg)
    
    elif expr1.kind == ExprKind.LAMBDA:
        lambda1 = cast(LambdaExpr, expr1)
        lambda2 = cast(LambdaExpr, expr2)
        return (expr_equals(lambda1.type, lambda2.type) and
                expr_equals(lambda1.body, lambda2.body))
    
    elif expr1.kind == ExprKind.PI:
        pi1 = cast(PiExpr, expr1)
        pi2 = cast(PiExpr, expr2)
        return (expr_equals(pi1.type, pi2.type) and
                expr_equals(pi1.body, pi2.body))
    
    elif expr1.kind == ExprKind.LET:
        let1 = cast(LetExpr, expr1)
        let2 = cast(LetExpr, expr2)
        return (expr_equals(let1.type, let2.type) and
                expr_equals(let1.value, let2.value) and
                expr_equals(let1.body, let2.body))
    
    return False


def reduce(expr: Expr, env: Environment, 
          strategy: ReductionStrategy = ReductionStrategy.WHNF,
          mode: ReductionMode = ReductionMode.ALL) -> Expr:
    """
    Reduce an expression using the specified strategy and mode.
    
    Args:
        expr: The expression to reduce
        env: The environment
        strategy: The reduction strategy to use
        mode: The reduction mode to use
    
    Returns:
        The reduced expression
    """
    if strategy == ReductionStrategy.WHNF:
        return whnf(expr, env, mode)
    else:  # NF
        return nf(expr, env, mode)


def beta_reduce(expr: Expr, env: Environment) -> Expr:
    """
    Perform a single beta reduction step on the expression.
    
    Beta reduction is the process of applying a lambda expression to an argument,
    replacing the bound variable with the argument in the body.
    
    Args:
        expr: The expression to reduce
        env: The environment
        
    Returns:
        The reduced expression
    """
    if expr.kind != ExprKind.APP:
        return expr
    
    app_expr = cast(AppExpr, expr)
    fn = app_expr.fn
    arg = app_expr.arg
    
    # Check if the function is a lambda
    if fn.kind == ExprKind.LAMBDA:
        lambda_expr = cast(LambdaExpr, fn)
        # Substitute the argument into the lambda body
        return instantiate(lambda_expr.body, arg)
    
    return expr


def unfold(expr: Expr, env: Environment, unfold_constants: Set[Name] = None) -> Expr:
    """
    Unfold definitions in the expression.
    
    Unfolding replaces constants with their definitions from the environment.
    
    Args:
        expr: The expression to unfold
        env: The environment
        unfold_constants: Optional set of constant names to unfold
        
    Returns:
        The unfolded expression
    """
    if expr.kind != ExprKind.CONST:
        return expr
    
    const_expr = cast(ConstExpr, expr)
    
    # Check if the constant should be unfolded
    if unfold_constants is not None and const_expr.name not in unfold_constants:
        return expr
    
    # Get the definition from the environment
    value = env.get_value(const_expr.name)
    if value is None:
        return expr
    
    # TODO: Handle universe level instantiation
    return value 