"""
Environment module for the Pylean kernel.

This module defines the environment system for managing definitions, constants,
axioms, and other declarations in Lean4.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union, cast, Any

from pylean.kernel.expr import (
    Expr, Name, Level, ExprKind, 
    ConstExpr, SortExpr, PiExpr, mk_const, mk_sort, mk_pi, mk_var
)


class DeclKind(Enum):
    """Kinds of declarations in Lean4."""
    AX = auto()        # Axiom
    DEF = auto()       # Definition
    THEOREM = auto()   # Theorem
    OPAQUE = auto()    # Opaque definition
    CONSTANT = auto()  # Constant
    INDUCTIVE = auto() # Inductive type
    EXTERN = auto()    # External function


class Declaration:
    """Base class for all declarations in the environment."""
    def __init__(self, name: Name, kind: DeclKind, type_expr: Expr, 
                universe_params: Tuple[Name, ...] = ()):
        self.name = name
        self.kind = kind
        self.type = type_expr
        self.universe_params = universe_params


class AxiomDecl(Declaration):
    """An axiom declaration."""
    
    def __init__(self, name: Name, type_expr: Expr, 
                universe_params: Tuple[Name, ...] = ()):
        """Initialize an axiom declaration."""
        super().__init__(name, DeclKind.AX, type_expr, universe_params)


class DefinitionDecl(Declaration):
    """A definition declaration."""
    
    def __init__(self, name: Name, type_expr: Expr, value: Expr,
                universe_params: Tuple[Name, ...] = ()):
        """Initialize a definition declaration."""
        super().__init__(name, DeclKind.DEF, type_expr, universe_params)
        self.value = value


class TheoremDecl(Declaration):
    """A theorem declaration."""
    
    def __init__(self, name: Name, type_expr: Expr, proof: Expr,
                universe_params: Tuple[Name, ...] = ()):
        """Initialize a theorem declaration."""
        super().__init__(name, DeclKind.THEOREM, type_expr, universe_params)
        self.proof = proof


class OpaqueDecl(Declaration):
    """An opaque definition declaration."""
    
    def __init__(self, name: Name, type_expr: Expr, value: Expr,
                universe_params: Tuple[Name, ...] = ()):
        """Initialize an opaque definition declaration."""
        super().__init__(name, DeclKind.OPAQUE, type_expr, universe_params)
        self.value = value


class ConstantDecl(Declaration):
    """A constant declaration."""
    
    def __init__(self, name: Name, type_expr: Expr,
                universe_params: Tuple[Name, ...] = ()):
        """Initialize a constant declaration."""
        super().__init__(name, DeclKind.CONSTANT, type_expr, universe_params)


class InductiveDecl(Declaration):
    """An inductive type declaration."""
    
    def __init__(self, name: Name, type_expr: Expr, 
                constructors: Tuple[ConstructorDecl, ...],
                universe_params: Tuple[Name, ...] = ()):
        """Initialize an inductive type declaration."""
        super().__init__(name, DeclKind.INDUCTIVE, type_expr, universe_params)
        self.constructors = constructors


class ConstructorDecl:
    """A constructor declaration for an inductive type."""
    
    def __init__(self, name: Name, type_expr: Expr, inductive_name: Name):
        """Initialize a constructor declaration."""
        self.name = name
        self.type = type_expr
        self.inductive_name = inductive_name


class ExternDecl(Declaration):
    """An external function declaration (for FFI)."""
    
    def __init__(self, name: Name, type_expr: Expr, param_types: Tuple[Expr, ...],
                return_type: Expr, c_name: str = None,
                universe_params: Tuple[Name, ...] = ()):
        """Initialize an external function declaration."""
        super().__init__(name, DeclKind.EXTERN, type_expr, universe_params)
        self.param_types = param_types
        self.return_type = return_type
        self.c_name = c_name or str(name)


@dataclass
class Environment:
    """The environment containing all declarations."""
    declarations: Dict[str, Declaration] = field(default_factory=dict)
    parent_env: Optional[Environment] = None
    constructor_info: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    inductive_info: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def add_decl(self, decl: Declaration) -> Environment:
        """
        Add a declaration to the environment.
        
        Returns a new environment with the declaration added.
        The original environment is not modified.
        """
        # Create a new environment with the same parent
        new_env = Environment(self.declarations.copy(), self.parent_env, 
                            self.constructor_info.copy(), self.inductive_info.copy())
        
        # Add the declaration to the new environment
        new_env.declarations[str(decl.name)] = decl
        
        # Special handling for inductive declarations - add constructors as constants
        if isinstance(decl, InductiveDecl):
            # Add inductive type info
            inductive_name_str = str(decl.name)
            new_env.inductive_info[inductive_name_str] = {
                'name': inductive_name_str,
                'num_constructors': len(decl.constructors),
                'constructors': [str(c.name) for c in decl.constructors],
                'type': decl.type
            }
            
            # Add each constructor
            for idx, constructor in enumerate(decl.constructors):
                # Create a constant declaration for the constructor
                const_decl = ConstantDecl(constructor.name, constructor.type)
                # Add it to the environment
                const_name_str = str(constructor.name)
                new_env.declarations[const_name_str] = const_decl
                
                # Determine arity (number of fields)
                curr_type = constructor.type
                arity = 0
                while curr_type.kind == ExprKind.PI:
                    arity += 1
                    curr_type = cast(PiExpr, curr_type).body
                
                # Add constructor info
                new_env.constructor_info[const_name_str] = {
                    'name': const_name_str,
                    'inductive_type': inductive_name_str,
                    'arity': arity,
                    'tag': idx,  # Use the index as the constructor tag
                    'type': constructor.type
                }
            
            # Generate eliminators (recursor and induction principle) for the inductive type
            try:
                from pylean.kernel.eliminator import generate_eliminators
                new_env = generate_eliminators(new_env, decl)
            except ImportError:
                # If eliminator module is not available, continue without generating eliminators
                pass
                
        return new_env
    
    def find_decl(self, name: Union[Name, str]) -> Optional[Declaration]:
        """Find a declaration by name."""
        name_str = str(name)
        if name_str in self.declarations:
            return self.declarations[name_str]
        # If not found in this environment, try parent
        if self.parent_env:
            return self.parent_env.find_decl(name)
        return None
    
    def is_constant(self, name: Union[Name, str]) -> bool:
        """Check if a name refers to a constant in the environment."""
        decl = self.find_decl(name)
        return decl is not None and decl.kind in (
            DeclKind.CONSTANT, DeclKind.DEF, DeclKind.THEOREM, 
            DeclKind.AX, DeclKind.OPAQUE
        )
    
    def is_inductive(self, name: Union[Name, str]) -> bool:
        """Check if a name refers to an inductive type in the environment."""
        decl = self.find_decl(name)
        return decl is not None and decl.kind == DeclKind.INDUCTIVE
    
    def get_type(self, name: Union[Name, str]) -> Optional[Expr]:
        """Get the type of a declaration by name."""
        decl = self.find_decl(name)
        return decl.type if decl else None
    
    def get_value(self, name: Union[Name, str]) -> Optional[Expr]:
        """Get the value of a definition by name."""
        decl = self.find_decl(name)
        if decl and isinstance(decl, (DefinitionDecl, OpaqueDecl)):
            return decl.value
        return None
    
    def get_constructors(self, name: Union[Name, str]) -> Tuple[ConstructorDecl, ...]:
        """Get the constructors of an inductive type by name."""
        decl = self.find_decl(name)
        if decl and isinstance(decl, InductiveDecl):
            return decl.constructors
        return ()
    
    def get_inductive_info(self, name: Union[Name, str]) -> Optional[Dict[str, Any]]:
        """
        Get information about an inductive type.
        
        Args:
            name: The name of the inductive type
            
        Returns:
            A dictionary with information about the inductive type, or None if not found
        """
        name_str = str(name)
        if name_str in self.inductive_info:
            return self.inductive_info[name_str]
        # If not found in this environment, try parent
        if self.parent_env:
            return self.parent_env.get_inductive_info(name)
        return None
    
    def get_constructor_info(self, name: Union[Name, str]) -> Optional[Dict[str, Any]]:
        """
        Get information about a constructor.
        
        Args:
            name: The name of the constructor
            
        Returns:
            A dictionary with information about the constructor, or None if not found
        """
        name_str = str(name)
        if name_str in self.constructor_info:
            return self.constructor_info[name_str]
        # If not found in this environment, try parent
        if self.parent_env:
            return self.parent_env.get_constructor_info(name)
        return None
    
    def create_child(self) -> Environment:
        """Create a child environment."""
        return Environment(parent_env=self, 
                        constructor_info=self.constructor_info.copy(),
                        inductive_info=self.inductive_info.copy())


# Factory functions for creating declarations

def mk_axiom(name: Union[Name, str], type_expr: Expr, 
            universe_params: List[Name] = None) -> AxiomDecl:
    """Create an axiom declaration."""
    if isinstance(name, str):
        name = Name.from_string(name)
    return AxiomDecl(name, type_expr, 
                    tuple(universe_params) if universe_params else ())


def mk_definition(name: Union[Name, str], type_expr: Expr, value: Expr,
                universe_params: List[Name] = None) -> DefinitionDecl:
    """Create a definition declaration."""
    if isinstance(name, str):
        name = Name.from_string(name)
    return DefinitionDecl(name, type_expr, value,
                        tuple(universe_params) if universe_params else ())


def mk_theorem(name: Union[Name, str], type_expr: Expr, proof: Expr,
              universe_params: List[Name] = None) -> TheoremDecl:
    """Create a theorem declaration."""
    if isinstance(name, str):
        name = Name.from_string(name)
    return TheoremDecl(name, type_expr, proof,
                      tuple(universe_params) if universe_params else ())


def mk_opaque(name: Union[Name, str], type_expr: Expr, value: Expr,
             universe_params: List[Name] = None) -> OpaqueDecl:
    """Create an opaque definition declaration."""
    if isinstance(name, str):
        name = Name.from_string(name)
    return OpaqueDecl(name, type_expr, value,
                     tuple(universe_params) if universe_params else ())


def mk_constant(name: Union[Name, str], type_expr: Expr,
               universe_params: List[Name] = None) -> ConstantDecl:
    """Create a constant declaration."""
    if isinstance(name, str):
        name = Name.from_string(name)
    return ConstantDecl(name, type_expr,
                       tuple(universe_params) if universe_params else ())


def mk_constructor(name: Union[Name, str], type_expr: Expr, 
                  inductive_name: Union[Name, str]) -> ConstructorDecl:
    """Create a constructor declaration."""
    if isinstance(name, str):
        name = Name.from_string(name)
    if isinstance(inductive_name, str):
        inductive_name = Name.from_string(inductive_name)
    return ConstructorDecl(name, type_expr, inductive_name)


def mk_inductive(name: Union[Name, str], type_expr: Expr, 
                constructors: List[ConstructorDecl],
                universe_params: List[Name] = None) -> InductiveDecl:
    """Create an inductive type declaration."""
    if isinstance(name, str):
        name = Name.from_string(name)
    return InductiveDecl(name, type_expr, tuple(constructors),
                        tuple(universe_params) if universe_params else ())


def mk_extern_decl(name: Union[Name, str], param_types: List[Expr],
                  return_type: Expr, c_name: str = None,
                  universe_params: List[Name] = None) -> ExternDecl:
    """
    Create an external function declaration for FFI.
    
    Args:
        name: The name of the external function
        param_types: Types of the parameters
        return_type: Return type of the function
        c_name: Name in C (defaults to the Lean name)
        universe_params: Universe parameters
        
    Returns:
        An external function declaration
    """
    from pylean.kernel.expr import mk_pi
    
    if isinstance(name, str):
        name = Name.from_string(name)
    
    # Create function type from parameter types and return type
    type_expr = return_type
    for i, param_type in enumerate(reversed(param_types)):
        param_name = Name.from_string(f"arg{i}")
        type_expr = mk_pi(param_name, param_type, type_expr)
    
    return ExternDecl(name, type_expr, tuple(param_types), return_type, c_name,
                     tuple(universe_params) if universe_params else ())


# Standard environment functions

def mk_std_env() -> Environment:
    """Create a standard environment with basic definitions."""
    env = Environment()
    
    # Add basic universe levels
    prop = mk_sort(0)
    type0 = mk_sort(1)
    
    # Add Prop and Type constants
    prop_decl = mk_constant("Prop", prop)
    type_decl = mk_constant("Type", type0)
    
    env = env.add_decl(prop_decl)
    env = env.add_decl(type_decl)
    
    # Add Nat type and constructors
    nat_decl = mk_constant("Nat", type0)
    env = env.add_decl(nat_decl)
    
    nat_type = mk_const("Nat")
    
    # Add zero constructor
    zero_decl = mk_constant("Nat.zero", nat_type)
    env = env.add_decl(zero_decl)
    
    # Add successor constructor: Nat.succ : Nat -> Nat
    succ_type = mk_pi("_", nat_type, nat_type)
    succ_decl = mk_constant("Nat.succ", succ_type)
    env = env.add_decl(succ_decl)
    
    # Add Nat operations
    # Nat.add : Nat -> Nat -> Nat
    add_type = mk_pi("a", nat_type, mk_pi("b", nat_type, nat_type))
    add_decl = mk_constant("Nat.add", add_type)
    env = env.add_decl(add_decl)
    
    # Nat.sub : Nat -> Nat -> Nat
    sub_type = mk_pi("a", nat_type, mk_pi("b", nat_type, nat_type))
    sub_decl = mk_constant("Nat.sub", sub_type)
    env = env.add_decl(sub_decl)
    
    # Nat.mul : Nat -> Nat -> Nat
    mul_type = mk_pi("a", nat_type, mk_pi("b", nat_type, nat_type))
    mul_decl = mk_constant("Nat.mul", mul_type)
    env = env.add_decl(mul_decl)
    
    # Nat.div : Nat -> Nat -> Nat
    div_type = mk_pi("a", nat_type, mk_pi("b", nat_type, nat_type))
    div_decl = mk_constant("Nat.div", div_type)
    env = env.add_decl(div_decl)
    
    # Add Bool type
    bool_decl = mk_constant("Bool", type0)
    env = env.add_decl(bool_decl)
    
    bool_type = mk_const("Bool")
    
    # Add true and false constructors
    true_decl = mk_constant("true", bool_type)
    false_decl = mk_constant("false", bool_type)
    env = env.add_decl(true_decl)
    env = env.add_decl(false_decl)
    
    # Add Bool operations
    # Bool.not : Bool -> Bool
    not_type = mk_pi("_", bool_type, bool_type)
    not_decl = mk_constant("Bool.not", not_type)
    env = env.add_decl(not_decl)
    
    # Add other operations
    
    # Eq : A -> A -> Prop (polymorphic equality)
    # First create a type parameter for Eq
    type_param = mk_sort(1)
    eq_type = mk_pi("A", type_param, mk_pi("a", mk_var(0), mk_pi("b", mk_var(1), prop)))
    eq_decl = mk_constant("Eq", eq_type)
    env = env.add_decl(eq_decl)
    
    # Lt and Gt (for natural numbers)
    lt_type = mk_pi("a", nat_type, mk_pi("b", nat_type, prop))
    lt_decl = mk_constant("Lt", lt_type)
    env = env.add_decl(lt_decl)
    
    gt_type = mk_pi("a", nat_type, mk_pi("b", nat_type, prop))
    gt_decl = mk_constant("Gt", gt_type)
    env = env.add_decl(gt_decl)
    
    return env 