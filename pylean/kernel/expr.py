"""
Expression module for the Pylean kernel.

This module defines the core expression types for Lean4, including:
- Constants
- Variables
- Lambdas
- Applications
- Types
- Let expressions
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union, cast


class ExprKind(Enum):
    """Kinds of expressions in Lean4."""
    VAR = auto()        # Variable: x
    SORT = auto()       # Type universe: Type, Prop
    CONST = auto()      # Constant: f
    APP = auto()        # Application: f x
    LAMBDA = auto()     # Lambda abstraction: λx, t
    PI = auto()         # Dependent function type: Πx, t
    LET = auto()        # Let expression: let x := v; t
    META = auto()       # Metavariable: ?m
    LOCAL = auto()      # Local constant: x (with unique ID)
    MATCH = auto()      # Pattern matching: match e with | p₁ => e₁ | ... | pₙ => eₙ
    EXTERN = auto()     # External function declaration: extern "C" f(args) -> ret


@dataclass(frozen=True)
class Name:
    """Represents a hierarchical name in Lean4."""
    parts: Tuple[str, ...] = field(default_factory=tuple)
    
    @staticmethod
    def from_string(s: str) -> Name:
        """Create a name from a string with dot notation."""
        if not s:
            return Name()
        return Name(tuple(s.split('.')))
    
    def __str__(self) -> str:
        """Convert name to string representation."""
        return '.'.join(self.parts)
    
    def append(self, part: str) -> Name:
        """Create a new name by appending a part."""
        return Name(self.parts + (part,))
    
    def is_anonymous(self) -> bool:
        """Check if this is an anonymous name."""
        return len(self.parts) == 0


@dataclass(frozen=True)
class Level:
    """Represents a universe level in Lean4."""
    # For simplicity, we'll just track a name for now
    name: Name
    param: Optional[int] = None
    
    def __str__(self) -> str:
        """Convert level to string representation."""
        if self.name.is_anonymous() and self.param is not None:
            return str(self.param)
        result = str(self.name)
        if self.param is not None:
            result += f"+{self.param}"
        return result


@dataclass(frozen=True)
class Expr:
    """Base class for all Lean expressions."""
    kind: ExprKind


@dataclass(frozen=True)
class VarExpr(Expr):
    """A variable expression (de Bruijn index)."""
    idx: int  # de Bruijn index
    
    def __init__(self, idx: int):
        """Initialize with a de Bruijn index."""
        super().__init__(ExprKind.VAR)
        object.__setattr__(self, 'idx', idx)
    
    def __str__(self) -> str:
        """String representation of a variable."""
        return f"#{self.idx}"


@dataclass(frozen=True)
class SortExpr(Expr):
    """A sort expression (Type, Prop)."""
    level: Level
    
    def __init__(self, level: Level):
        """Initialize with a universe level."""
        super().__init__(ExprKind.SORT)
        object.__setattr__(self, 'level', level)
    
    def __str__(self) -> str:
        """String representation of a sort."""
        if str(self.level) == "0":
            return "Prop"
        return f"Type {self.level}"


@dataclass(frozen=True)
class ConstExpr(Expr):
    """A constant expression (named entity in environment)."""
    name: Name
    levels: Tuple[Level, ...] = field(default_factory=tuple)
    
    def __init__(self, name: Name, levels: Tuple[Level, ...] = ()):
        """Initialize with a name and universe levels."""
        super().__init__(ExprKind.CONST)
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'levels', levels)
    
    def __str__(self) -> str:
        """String representation of a constant."""
        if not self.levels:
            return str(self.name)
        levels_str = ", ".join(str(level) for level in self.levels)
        return f"{self.name}.{{{levels_str}}}"


@dataclass(frozen=True)
class AppExpr(Expr):
    """An application expression (function application)."""
    fn: Expr
    arg: Expr
    
    def __init__(self, fn: Expr, arg: Expr):
        """Initialize with function and argument expressions."""
        super().__init__(ExprKind.APP)
        object.__setattr__(self, 'fn', fn)
        object.__setattr__(self, 'arg', arg)
    
    def __str__(self) -> str:
        """String representation of an application."""
        fn_str = str(self.fn)
        if self.fn.kind in (ExprKind.LAMBDA, ExprKind.PI, ExprKind.LET):
            fn_str = f"({fn_str})"
        arg_str = str(self.arg)
        if self.arg.kind in (ExprKind.LAMBDA, ExprKind.PI, ExprKind.LET, ExprKind.APP):
            arg_str = f"({arg_str})"
        return f"{fn_str} {arg_str}"


@dataclass(frozen=True)
class BinderInfo:
    """Information about a binder (explicit/implicit/etc.)."""
    is_implicit: bool = False
    is_strict_implicit: bool = False
    is_inst_implicit: bool = False
    
    def is_explicit(self) -> bool:
        """Check if this is an explicit binder."""
        return not (self.is_implicit or self.is_strict_implicit or self.is_inst_implicit)


@dataclass(frozen=True)
class LambdaExpr(Expr):
    """A lambda expression (function abstraction)."""
    name: Name
    type: Expr
    body: Expr
    binder_info: BinderInfo = field(default_factory=BinderInfo)
    
    def __init__(self, name: Name, type_expr: Expr, body: Expr, 
                binder_info: BinderInfo = None):
        """Initialize with name, type, and body expressions."""
        super().__init__(ExprKind.LAMBDA)
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'type', type_expr)
        object.__setattr__(self, 'body', body)
        object.__setattr__(self, 'binder_info', 
                          binder_info if binder_info else BinderInfo())
    
    def __str__(self) -> str:
        """String representation of a lambda expression."""
        prefix = "λ"
        if self.binder_info.is_implicit:
            return f"{prefix}{{{self.name} : {self.type}}}, {self.body}"
        if self.binder_info.is_strict_implicit:
            return f"{prefix}{{{self.name} : {self.type}}}, {self.body}"
        if self.binder_info.is_inst_implicit:
            return f"{prefix}[{self.name} : {self.type}], {self.body}"
        return f"{prefix}({self.name} : {self.type}), {self.body}"


@dataclass(frozen=True)
class PiExpr(Expr):
    """A pi expression (dependent function type)."""
    name: Name
    type: Expr
    body: Expr
    binder_info: BinderInfo = field(default_factory=BinderInfo)
    
    def __init__(self, name: Name, type_expr: Expr, body: Expr, 
                binder_info: BinderInfo = None):
        """Initialize with name, type, and body expressions."""
        super().__init__(ExprKind.PI)
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'type', type_expr)
        object.__setattr__(self, 'body', body)
        object.__setattr__(self, 'binder_info', 
                          binder_info if binder_info else BinderInfo())
    
    def __str__(self) -> str:
        """String representation of a pi expression."""
        # Handle simple function types
        if self.name.is_anonymous() or not occurs_in(0, self.body):
            return f"{self.type} → {instantiate(self.body, VarExpr(0))}"
        
        prefix = "Π"
        if self.binder_info.is_implicit:
            return f"{prefix}{{{self.name} : {self.type}}}, {self.body}"
        if self.binder_info.is_strict_implicit:
            return f"{prefix}{{{self.name} : {self.type}}}, {self.body}"
        if self.binder_info.is_inst_implicit:
            return f"{prefix}[{self.name} : {self.type}], {self.body}"
        return f"{prefix}({self.name} : {self.type}), {self.body}"


@dataclass(frozen=True)
class LetExpr(Expr):
    """A let expression (local definition)."""
    name: Name
    type: Expr
    value: Expr
    body: Expr
    
    def __init__(self, name: Name, type_expr: Expr, value: Expr, body: Expr):
        """Initialize with name, type, value, and body expressions."""
        super().__init__(ExprKind.LET)
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'type', type_expr)
        object.__setattr__(self, 'value', value)
        object.__setattr__(self, 'body', body)
    
    def __str__(self) -> str:
        """String representation of a let expression."""
        return f"let {self.name} : {self.type} := {self.value}; {self.body}"


@dataclass(frozen=True)
class MetaExpr(Expr):
    """A metavariable expression (placeholder)."""
    name: Name
    
    def __init__(self, name: Name):
        """Initialize with a name."""
        super().__init__(ExprKind.META)
        object.__setattr__(self, 'name', name)
    
    def __str__(self) -> str:
        """String representation of a metavariable."""
        return f"?{self.name}"


@dataclass(frozen=True)
class LocalExpr(Expr):
    """A local constant expression (named local variable)."""
    name: Name
    type: Expr
    binder_info: BinderInfo = field(default_factory=BinderInfo)
    
    def __init__(self, name: Name, type_expr: Expr, 
                binder_info: BinderInfo = None):
        """Initialize with name and type expression."""
        super().__init__(ExprKind.LOCAL)
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'type', type_expr)
        object.__setattr__(self, 'binder_info', 
                          binder_info if binder_info else BinderInfo())
    
    def __str__(self) -> str:
        """String representation of a local constant."""
        return str(self.name)


@dataclass(frozen=True)
class Pattern:
    """Represents a pattern in a match expression."""
    constructor: Name  # Constructor name
    fields: Tuple[str, ...]  # Field names (for binding)
    
    def __str__(self) -> str:
        """String representation of a pattern."""
        if not self.fields:
            return str(self.constructor)
        fields_str = ", ".join(self.fields)
        return f"{self.constructor} {fields_str}"


@dataclass(frozen=True)
class Alternative:
    """Represents a match alternative (pattern => expression)."""
    pattern: Pattern
    expr: Expr
    
    def __str__(self) -> str:
        """String representation of a match alternative."""
        return f"{self.pattern} => {self.expr}"


@dataclass(frozen=True)
class MatchExpr(Expr):
    """A match expression for pattern matching."""
    scrutinee: Expr  # Expression being matched
    type: Expr       # Type of the match expression
    alternatives: Tuple[Alternative, ...]  # Match alternatives
    
    def __init__(self, scrutinee: Expr, type_expr: Expr, alternatives: Tuple[Alternative, ...]):
        """Initialize with scrutinee, type, and alternatives."""
        super().__init__(ExprKind.MATCH)
        object.__setattr__(self, 'scrutinee', scrutinee)
        object.__setattr__(self, 'type', type_expr)
        object.__setattr__(self, 'alternatives', alternatives)
    
    def __str__(self) -> str:
        """String representation of a match expression."""
        scr_str = str(self.scrutinee)
        if self.scrutinee.kind in (ExprKind.LAMBDA, ExprKind.PI, ExprKind.LET, ExprKind.APP, ExprKind.MATCH):
            scr_str = f"({scr_str})"
        alts_str = " | ".join(str(alt) for alt in self.alternatives)
        return f"match {scr_str} with {alts_str}"


@dataclass(frozen=True)
class ExternExpr(Expr):
    """An external function declaration (for FFI)."""
    name: Name          # Name of the external function
    param_types: Tuple[Expr, ...]  # Parameter types
    return_type: Expr   # Return type
    c_name: str         # Name in C (can be different from Lean name)
    
    def __init__(self, name: Name, param_types: Tuple[Expr, ...], 
                return_type: Expr, c_name: str = None):
        """Initialize with name, parameter types, and return type."""
        super().__init__(ExprKind.EXTERN)
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'param_types', param_types)
        object.__setattr__(self, 'return_type', return_type)
        object.__setattr__(self, 'c_name', c_name or str(name))
    
    def __str__(self) -> str:
        """String representation of an external function declaration."""
        params_str = ", ".join(str(t) for t in self.param_types)
        return f'extern "{self.c_name}" ({params_str}) -> {self.return_type}'


def mk_var(idx: int) -> VarExpr:
    """Create a variable expression with the given de Bruijn index."""
    return VarExpr(idx)


def mk_sort(level: Union[Level, int, str]) -> SortExpr:
    """Create a sort expression with the given level."""
    if isinstance(level, int):
        return SortExpr(Level(Name(), level))
    if isinstance(level, str):
        return SortExpr(Level(Name.from_string(level)))
    return SortExpr(level)


def mk_const(name: Union[Name, str], levels: List[Level] = None) -> ConstExpr:
    """Create a constant expression with the given name and levels."""
    if isinstance(name, str):
        name = Name.from_string(name)
    return ConstExpr(name, tuple(levels) if levels else ())


def mk_app(fn: Expr, arg: Expr) -> AppExpr:
    """Create an application expression with the given function and argument."""
    return AppExpr(fn, arg)


def mk_lambda(name: Union[Name, str], type_expr: Expr, body: Expr, 
             binder_info: BinderInfo = None) -> LambdaExpr:
    """Create a lambda expression with the given name, type, and body."""
    if isinstance(name, str):
        name = Name.from_string(name)
    return LambdaExpr(name, type_expr, body, binder_info)


def mk_pi(name: Union[Name, str], type_expr: Expr, body: Expr,
         binder_info: BinderInfo = None) -> PiExpr:
    """Create a pi expression with the given name, type, and body."""
    if isinstance(name, str):
        name = Name.from_string(name)
    return PiExpr(name, type_expr, body, binder_info)


def mk_let(name: Union[Name, str], type_expr: Expr, value: Expr, 
          body: Expr) -> LetExpr:
    """Create a let expression with the given name, type, value, and body."""
    if isinstance(name, str):
        name = Name.from_string(name)
    return LetExpr(name, type_expr, value, body)


def mk_meta(name: Union[Name, str]) -> MetaExpr:
    """Create a metavariable expression with the given name."""
    if isinstance(name, str):
        name = Name.from_string(name)
    return MetaExpr(name)


def mk_local(name: Union[Name, str], type_expr: Expr,
            binder_info: BinderInfo = None) -> LocalExpr:
    """Create a local constant expression with the given name and type."""
    if isinstance(name, str):
        name = Name.from_string(name)
    return LocalExpr(name, type_expr, binder_info)


def mk_match(scrutinee: Expr, type_expr: Expr, 
             alternatives: List[Alternative]) -> MatchExpr:
    """Create a match expression.
    
    Args:
        scrutinee: Expression to match on
        type_expr: Type of the entire match expression
        alternatives: List of match alternatives
        
    Returns:
        A match expression
    """
    return MatchExpr(scrutinee, type_expr, tuple(alternatives))


def mk_pattern(constructor: Union[Name, str], fields: List[str] = None) -> Pattern:
    """Create a pattern for match expressions.
    
    Args:
        constructor: Constructor name
        fields: Field names for binding
        
    Returns:
        A pattern
    """
    if isinstance(constructor, str):
        constructor = Name.from_string(constructor)
    return Pattern(constructor, tuple(fields or []))


def mk_alternative(pattern: Pattern, expr: Expr) -> Alternative:
    """Create a match alternative.
    
    Args:
        pattern: Pattern to match
        expr: Expression to evaluate if pattern matches
        
    Returns:
        A match alternative
    """
    return Alternative(pattern, expr)


def mk_extern(name: Union[Name, str], param_types: List[Expr], 
             return_type: Expr, c_name: str = None) -> ExternExpr:
    """
    Create an external function declaration for FFI.
    
    Args:
        name: Lean name for the function
        param_types: Types of the parameters
        return_type: Return type
        c_name: Optional C name (defaults to the Lean name)
        
    Returns:
        An external function declaration
    """
    if isinstance(name, str):
        name = Name.from_string(name)
    return ExternExpr(name, tuple(param_types), return_type, c_name)


def occurs_in(idx: int, expr: Expr) -> bool:
    """
    Check if the variable with given de Bruijn index occurs in the expression.
    
    Args:
        idx: De Bruijn index to check for
        expr: Expression to search in
        
    Returns:
        True if the variable occurs in the expression, False otherwise
    """
    if expr.kind == ExprKind.VAR:
        var_expr = cast(VarExpr, expr)
        return var_expr.idx == idx
    elif expr.kind == ExprKind.SORT or expr.kind == ExprKind.CONST or expr.kind == ExprKind.META:
        return False
    elif expr.kind == ExprKind.APP:
        app_expr = cast(AppExpr, expr)
        return occurs_in(idx, app_expr.fn) or occurs_in(idx, app_expr.arg)
    elif expr.kind == ExprKind.LAMBDA or expr.kind == ExprKind.PI:
        lambda_expr = cast(Union[LambdaExpr, PiExpr], expr)
        return occurs_in(idx, lambda_expr.type) or occurs_in(idx + 1, lambda_expr.body)
    elif expr.kind == ExprKind.LET:
        let_expr = cast(LetExpr, expr)
        return occurs_in(idx, let_expr.type) or occurs_in(idx, let_expr.value) or occurs_in(idx + 1, let_expr.body)
    elif expr.kind == ExprKind.LOCAL:
        return False
    elif expr.kind == ExprKind.MATCH:
        match_expr = cast(MatchExpr, expr)
        if occurs_in(idx, match_expr.scrutinee) or occurs_in(idx, match_expr.type):
            return True
        # For each alternative, check if the variable occurs
        # Increase the index by the number of bound fields for each pattern
        for alt in match_expr.alternatives:
            # Adjust index based on number of bound fields in the pattern
            adjusted_idx = idx + len(alt.pattern.fields)
            if occurs_in(adjusted_idx, alt.expr):
                return True
        return False
    elif expr.kind == ExprKind.EXTERN:
        extern_expr = cast(ExternExpr, expr)
        # Check in parameter types and return type
        return (any(occurs_in(idx, t) for t in extern_expr.param_types) or 
                occurs_in(idx, extern_expr.return_type))
    else:
        # Should never happen with a complete pattern match
        raise ValueError(f"Unexpected expression kind: {expr.kind}")


def lift(expr: Expr, n: int, d: int) -> Expr:
    """
    Lift free variables in an expression.
    
    This increases the de Bruijn indices of free variables by n,
    where a variable is considered free if its index is >= d.
    
    Args:
        expr: Expression to lift
        n: Amount to increase indices by
        d: Threshold for free variables
        
    Returns:
        Lifted expression
    """
    if expr.kind == ExprKind.VAR:
        var_expr = cast(VarExpr, expr)
        if var_expr.idx >= d:
            return mk_var(var_expr.idx + n)
        else:
            return expr
    elif expr.kind == ExprKind.SORT or expr.kind == ExprKind.CONST or expr.kind == ExprKind.META:
        return expr
    elif expr.kind == ExprKind.APP:
        app_expr = cast(AppExpr, expr)
        new_fn = lift(app_expr.fn, n, d)
        new_arg = lift(app_expr.arg, n, d)
        if new_fn == app_expr.fn and new_arg == app_expr.arg:
            return expr
        else:
            return mk_app(new_fn, new_arg)
    elif expr.kind == ExprKind.LAMBDA:
        lambda_expr = cast(LambdaExpr, expr)
        new_type = lift(lambda_expr.type, n, d)
        new_body = lift(lambda_expr.body, n, d + 1)
        if new_type == lambda_expr.type and new_body == lambda_expr.body:
            return expr
        else:
            return mk_lambda(lambda_expr.name, new_type, new_body, lambda_expr.binder_info)
    elif expr.kind == ExprKind.PI:
        pi_expr = cast(PiExpr, expr)
        new_type = lift(pi_expr.type, n, d)
        new_body = lift(pi_expr.body, n, d + 1)
        if new_type == pi_expr.type and new_body == pi_expr.body:
            return expr
        else:
            return mk_pi(pi_expr.name, new_type, new_body, pi_expr.binder_info)
    elif expr.kind == ExprKind.LET:
        let_expr = cast(LetExpr, expr)
        new_type = lift(let_expr.type, n, d)
        new_value = lift(let_expr.value, n, d)
        new_body = lift(let_expr.body, n, d + 1)
        if new_type == let_expr.type and new_value == let_expr.value and new_body == let_expr.body:
            return expr
        else:
            return mk_let(let_expr.name, new_type, new_value, new_body)
    elif expr.kind == ExprKind.LOCAL:
        return expr
    elif expr.kind == ExprKind.MATCH:
        match_expr = cast(MatchExpr, expr)
        new_scrutinee = lift(match_expr.scrutinee, n, d)
        new_type = lift(match_expr.type, n, d)
        
        # Lift alternatives
        new_alternatives = []
        for alt in match_expr.alternatives:
            # The pattern doesn't contain expressions, only field names
            # For the alternative expression, adjust d based on the pattern fields
            new_alt_expr = lift(alt.expr, n, d + len(alt.pattern.fields))
            if new_alt_expr != alt.expr:
                new_alternatives.append(mk_alternative(alt.pattern, new_alt_expr))
            else:
                new_alternatives.append(alt)
        
        if (new_scrutinee == match_expr.scrutinee and 
            new_type == match_expr.type and 
            all(new_alt == old_alt for new_alt, old_alt in zip(new_alternatives, match_expr.alternatives))):
            return expr
        else:
            return mk_match(new_scrutinee, new_type, new_alternatives)
    elif expr.kind == ExprKind.EXTERN:
        extern_expr = cast(ExternExpr, expr)
        # Lift parameter types and return type
        new_param_types = tuple(lift(t, n, d) for t in extern_expr.param_types)
        new_return_type = lift(extern_expr.return_type, n, d)
        
        if (all(new_t == old_t for new_t, old_t in zip(new_param_types, extern_expr.param_types)) and
            new_return_type == extern_expr.return_type):
            return expr
        else:
            return mk_extern(extern_expr.name, list(new_param_types), new_return_type, extern_expr.c_name)
    else:
        # Should never happen with a complete pattern match
        raise ValueError(f"Unexpected expression kind: {expr.kind}")


def instantiate(expr: Expr, subst: Expr, idx: int = 0) -> Expr:
    """
    Instantiate a variable in an expression.
    
    This replaces the variable with de Bruijn index `idx` with `subst`.
    
    Args:
        expr: Expression to instantiate in
        subst: Expression to substitute
        idx: De Bruijn index to replace
        
    Returns:
        Instantiated expression
    """
    if expr.kind == ExprKind.VAR:
        var_expr = cast(VarExpr, expr)
        if var_expr.idx == idx:
            return subst
        elif var_expr.idx > idx:
            # Decrement the index since we're removing a variable
            return mk_var(var_expr.idx - 1)
        else:
            return expr
    elif expr.kind == ExprKind.SORT or expr.kind == ExprKind.CONST or expr.kind == ExprKind.META:
        return expr
    elif expr.kind == ExprKind.APP:
        app_expr = cast(AppExpr, expr)
        new_fn = instantiate(app_expr.fn, subst, idx)
        new_arg = instantiate(app_expr.arg, subst, idx)
        if new_fn == app_expr.fn and new_arg == app_expr.arg:
            return expr
        else:
            return mk_app(new_fn, new_arg)
    elif expr.kind == ExprKind.LAMBDA:
        lambda_expr = cast(LambdaExpr, expr)
        new_type = instantiate(lambda_expr.type, subst, idx)
        # For the body, we need to lift the substitution and increment the index
        lifted_subst = lift(subst, 1, 0)
        new_body = instantiate(lambda_expr.body, lifted_subst, idx + 1)
        if new_type == lambda_expr.type and new_body == lambda_expr.body:
            return expr
        else:
            return mk_lambda(lambda_expr.name, new_type, new_body, lambda_expr.binder_info)
    elif expr.kind == ExprKind.PI:
        pi_expr = cast(PiExpr, expr)
        new_type = instantiate(pi_expr.type, subst, idx)
        # For the body, we need to lift the substitution and increment the index
        lifted_subst = lift(subst, 1, 0)
        new_body = instantiate(pi_expr.body, lifted_subst, idx + 1)
        if new_type == pi_expr.type and new_body == pi_expr.body:
            return expr
        else:
            return mk_pi(pi_expr.name, new_type, new_body, pi_expr.binder_info)
    elif expr.kind == ExprKind.LET:
        let_expr = cast(LetExpr, expr)
        new_type = instantiate(let_expr.type, subst, idx)
        new_value = instantiate(let_expr.value, subst, idx)
        # For the body, we need to lift the substitution and increment the index
        lifted_subst = lift(subst, 1, 0)
        new_body = instantiate(let_expr.body, lifted_subst, idx + 1)
        if new_type == let_expr.type and new_value == let_expr.value and new_body == let_expr.body:
            return expr
        else:
            return mk_let(let_expr.name, new_type, new_value, new_body)
    elif expr.kind == ExprKind.LOCAL:
        return expr
    elif expr.kind == ExprKind.MATCH:
        match_expr = cast(MatchExpr, expr)
        new_scrutinee = instantiate(match_expr.scrutinee, subst, idx)
        new_type = instantiate(match_expr.type, subst, idx)
        
        # Instantiate alternatives
        new_alternatives = []
        for alt in match_expr.alternatives:
            # For each alternative, we need to lift the substitution by the number of
            # pattern fields (which become bound variables)
            num_fields = len(alt.pattern.fields)
            alt_subst = lift(subst, num_fields, 0)
            # The idx also needs to be adjusted by the number of bound variables
            alt_idx = idx + num_fields
            new_alt_expr = instantiate(alt.expr, alt_subst, alt_idx)
            
            if new_alt_expr != alt.expr:
                new_alternatives.append(mk_alternative(alt.pattern, new_alt_expr))
            else:
                new_alternatives.append(alt)
        
        if (new_scrutinee == match_expr.scrutinee and 
            new_type == match_expr.type and 
            all(new_alt == old_alt for new_alt, old_alt in zip(new_alternatives, match_expr.alternatives))):
            return expr
        else:
            return mk_match(new_scrutinee, new_type, new_alternatives)
    elif expr.kind == ExprKind.EXTERN:
        extern_expr = cast(ExternExpr, expr)
        # Instantiate parameter types and return type
        new_param_types = tuple(instantiate(t, subst, idx) for t in extern_expr.param_types)
        new_return_type = instantiate(extern_expr.return_type, subst, idx)
        
        if (all(new_t == old_t for new_t, old_t in zip(new_param_types, extern_expr.param_types)) and
            new_return_type == extern_expr.return_type):
            return expr
        else:
            return mk_extern(extern_expr.name, list(new_param_types), new_return_type, extern_expr.c_name)
    else:
        # Should never happen with a complete pattern match
        raise ValueError(f"Unexpected expression kind: {expr.kind}") 