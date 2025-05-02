"""
Pattern matching optimization module.

This module provides advanced pattern matching optimizations, including:
1. Exhaustiveness checking for pattern matching
2. Redundancy checking for pattern matching
3. Decision tree generation for efficient pattern matching
4. Nested pattern handling
5. Pattern guard optimization
"""

from typing import Dict, List, Optional, Set, Tuple, Union, Any, cast
from dataclasses import dataclass
from enum import Enum, auto

from pylean.kernel import (
    Expr, ExprKind, Environment, Declaration, DeclKind,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi, mk_match,
    MatchExpr, Alternative, Pattern
)


class PatternKind(Enum):
    """Kind of pattern in the optimization representation."""
    CONSTRUCTOR = auto()  # Constructor pattern: C(x, y, z)
    WILDCARD = auto()     # Wildcard pattern: _
    LITERAL = auto()      # Literal pattern: 42, "hello", etc.
    OR = auto()           # Or pattern: p1 | p2
    NESTED = auto()       # Nested pattern: C(D(x), y)
    GUARD = auto()        # Guard pattern: p when e


@dataclass
class ExpandedPattern:
    """Expanded pattern for optimization."""
    kind: PatternKind
    constructor: Optional[str] = None
    fields: List["ExpandedPattern"] = None
    variables: List[str] = None
    guard: Optional[Expr] = None
    sub_patterns: List["ExpandedPattern"] = None  # For OR patterns
    literal_value: Any = None
    
    def __post_init__(self):
        if self.fields is None:
            self.fields = []
        if self.variables is None:
            self.variables = []
        if self.sub_patterns is None:
            self.sub_patterns = []


@dataclass
class DecisionTree:
    """Decision tree for pattern matching compilation."""
    class NodeKind(Enum):
        """Kind of decision tree node."""
        SWITCH = auto()  # Switch on a constructor
        LEAF = auto()    # Leaf node with an action
        GUARD = auto()   # Guard node with a condition
    
    kind: NodeKind
    scrutinee: Optional[int] = None  # Index of scrutinee (0 for main, >0 for fields)
    cases: Dict[str, "DecisionTree"] = None  # For SWITCH nodes
    default: Optional["DecisionTree"] = None  # Default case for SWITCH
    action_index: Optional[int] = None  # For LEAF nodes (index of alternative)
    guard: Optional[Expr] = None  # For GUARD nodes
    success: Optional["DecisionTree"] = None  # For GUARD nodes (if guard succeeds)
    failure: Optional["DecisionTree"] = None  # For GUARD nodes (if guard fails)
    
    def __post_init__(self):
        if self.cases is None:
            self.cases = {}


class PatternOptimizer:
    """Pattern matching optimizer."""
    
    def __init__(self, env: Environment):
        """
        Initialize the pattern optimizer.
        
        Args:
            env: The environment
        """
        self.env = env
        self.constructors: Dict[str, Dict[str, Any]] = {}
        self.inductive_types: Dict[str, Dict[str, Any]] = {}
        
        # Parse all constructor information from the environment
        self._load_constructor_info()
    
    def _load_constructor_info(self) -> None:
        """Load constructor information from the environment."""
        for name, decl in self.env.declarations.items():
            if decl.kind == DeclKind.INDUCTIVE:
                # Get inductive type info
                inductive_info = self.env.get_inductive_info(name)
                if inductive_info:
                    self.inductive_types[name] = inductive_info
                    
                    # Get constructor info for each constructor
                    for ctor_name in inductive_info.get('constructors', []):
                        ctor_info = self.env.get_constructor_info(ctor_name)
                        if ctor_info:
                            self.constructors[ctor_name] = ctor_info
    
    def expand_pattern(self, pattern: Pattern) -> ExpandedPattern:
        """
        Expand a pattern for optimization.
        
        Args:
            pattern: The pattern to expand
            
        Returns:
            An expanded pattern
        """
        constructor_name = str(pattern.constructor)
        
        # Handle wildcard patterns
        if constructor_name == "_":
            return ExpandedPattern(
                kind=PatternKind.WILDCARD,
                variables=list(pattern.fields)
            )
        
        # Handle literal patterns
        if constructor_name.isdigit() or constructor_name.startswith('"'):
            return ExpandedPattern(
                kind=PatternKind.LITERAL,
                literal_value=constructor_name,
                variables=list(pattern.fields)
            )
        
        # Handle constructor patterns
        return ExpandedPattern(
            kind=PatternKind.CONSTRUCTOR,
            constructor=constructor_name,
            variables=list(pattern.fields)
        )
    
    def check_exhaustiveness(self, match_expr: MatchExpr) -> bool:
        """
        Check if a match expression is exhaustive.
        
        Args:
            match_expr: The match expression to check
            
        Returns:
            True if the match is exhaustive, False otherwise
        """
        # Get the type of the scrutinee
        scrutinee_type = None
        if match_expr.scrutinee.kind == ExprKind.CONST:
            scrutinee_type = str(match_expr.scrutinee.name)
        elif hasattr(match_expr.scrutinee, 'type'):
            scrutinee_type = str(match_expr.scrutinee.type)
        
        # If we can't determine the type, assume it's not exhaustive
        if not scrutinee_type or scrutinee_type not in self.inductive_types:
            return False
        
        # Get all constructors for the type
        inductive_info = self.inductive_types[scrutinee_type]
        all_constructors = set(inductive_info.get('constructors', []))
        
        # Get constructors covered by the patterns
        covered_constructors = set()
        has_wildcard = False
        
        for alt in match_expr.alternatives:
            pattern = alt.pattern
            constructor_name = str(pattern.constructor)
            
            if constructor_name == "_":
                # Wildcard pattern covers everything
                has_wildcard = True
                break
            else:
                covered_constructors.add(constructor_name)
        
        # Check if all constructors are covered
        return has_wildcard or covered_constructors == all_constructors
    
    def check_redundancy(self, match_expr: MatchExpr) -> List[int]:
        """
        Check for redundant patterns in a match expression.
        
        Args:
            match_expr: The match expression to check
            
        Returns:
            List of indices of redundant patterns
        """
        redundant = []
        covered_constructors = set()
        has_wildcard = False
        
        for i, alt in enumerate(match_expr.alternatives):
            pattern = alt.pattern
            constructor_name = str(pattern.constructor)
            
            if has_wildcard:
                # Everything after a wildcard is redundant
                redundant.append(i)
            elif constructor_name == "_":
                has_wildcard = True
            elif constructor_name in covered_constructors:
                # Duplicate constructor
                redundant.append(i)
            else:
                covered_constructors.add(constructor_name)
        
        return redundant
    
    def optimize_match(self, match_expr: MatchExpr) -> MatchExpr:
        """
        Optimize a match expression.
        
        Args:
            match_expr: The match expression to optimize
            
        Returns:
            The optimized match expression
        """
        # First, check for redundant patterns
        redundant = self.check_redundancy(match_expr)
        
        if redundant:
            # Remove redundant patterns
            new_alternatives = []
            for i, alt in enumerate(match_expr.alternatives):
                if i not in redundant:
                    new_alternatives.append(alt)
            
            return mk_match(
                match_expr.scrutinee,
                match_expr.type,
                new_alternatives
            )
        
        # No redundancy found, return the original expression
        return match_expr
    
    def generate_decision_tree(self, match_expr: MatchExpr) -> DecisionTree:
        """
        Generate a decision tree for efficient pattern matching.
        
        Args:
            match_expr: The match expression
            
        Returns:
            A decision tree for the match expression
        """
        expanded_patterns = []
        
        # Expand all patterns
        for alt in match_expr.alternatives:
            expanded_patterns.append(self.expand_pattern(alt.pattern))
        
        # Generate the decision tree
        return self._build_decision_tree(expanded_patterns, 0, list(range(len(match_expr.alternatives))))
    
    def _build_decision_tree(self, 
                            patterns: List[ExpandedPattern], 
                            scrutinee_idx: int, 
                            alt_indices: List[int]) -> DecisionTree:
        """
        Build a decision tree for the given patterns.
        
        Args:
            patterns: The patterns to match
            scrutinee_idx: Index of the scrutinee (0 for main, >0 for fields)
            alt_indices: Indices of the alternatives in the original match
            
        Returns:
            A decision tree
        """
        if not patterns:
            # No patterns, so no matches
            return None
        
        # Check for a wildcard pattern
        wildcard_idx = None
        for i, pattern in enumerate(patterns):
            if pattern.kind == PatternKind.WILDCARD:
                wildcard_idx = i
                break
        
        if len(patterns) == 1 or wildcard_idx is not None:
            # Only one pattern or found a wildcard, so this is a leaf
            return DecisionTree(
                kind=DecisionTree.NodeKind.LEAF,
                action_index=alt_indices[wildcard_idx if wildcard_idx is not None else 0]
            )
        
        # Group patterns by constructor
        constructor_groups: Dict[str, List[Tuple[int, ExpandedPattern]]] = {}
        for i, pattern in enumerate(patterns):
            if pattern.kind == PatternKind.CONSTRUCTOR:
                constructor = pattern.constructor
                if constructor not in constructor_groups:
                    constructor_groups[constructor] = []
                constructor_groups[constructor].append((i, pattern))
        
        if not constructor_groups:
            # No constructor patterns, just use the first pattern
            return DecisionTree(
                kind=DecisionTree.NodeKind.LEAF,
                action_index=alt_indices[0]
            )
        
        # Create a switch node
        switch_node = DecisionTree(
            kind=DecisionTree.NodeKind.SWITCH,
            scrutinee=scrutinee_idx,
            cases={}
        )
        
        # Build cases for each constructor
        for constructor, patterns_with_idx in constructor_groups.items():
            sub_patterns = []
            sub_alt_indices = []
            
            for i, pattern in patterns_with_idx:
                sub_patterns.append(pattern)
                sub_alt_indices.append(alt_indices[i])
            
            # Recursively build the decision tree for this constructor
            switch_node.cases[constructor] = self._build_decision_tree(
                sub_patterns,
                scrutinee_idx + 1,  # Next scrutinee is a field
                sub_alt_indices
            )
        
        # If we have a wildcard pattern, use it as the default case
        if wildcard_idx is not None:
            switch_node.default = DecisionTree(
                kind=DecisionTree.NodeKind.LEAF,
                action_index=alt_indices[wildcard_idx]
            )
        
        return switch_node
    
    def optimize_nested_patterns(self, pattern: Pattern) -> Pattern:
        """
        Optimize nested patterns by flattening them.
        
        Args:
            pattern: The pattern to optimize
            
        Returns:
            The optimized pattern
        """
        # In a real implementation, this would handle nested patterns by
        # transforming them into simpler patterns with additional matches
        # For this demonstration, we'll just return the original pattern
        return pattern
    
    def optimize_guards(self, match_expr: MatchExpr) -> MatchExpr:
        """
        Optimize pattern guards by moving them to where they're needed.
        
        Args:
            match_expr: The match expression to optimize
            
        Returns:
            The optimized match expression
        """
        # In a real implementation, this would handle guard expressions
        # For this demonstration, we'll just return the original expression
        return match_expr


def check_exhaustiveness(env: Environment, match_expr: MatchExpr) -> bool:
    """
    Check if a match expression is exhaustive.
    
    Args:
        env: The environment
        match_expr: The match expression to check
        
    Returns:
        True if the match is exhaustive, False otherwise
    """
    optimizer = PatternOptimizer(env)
    return optimizer.check_exhaustiveness(match_expr)


def optimize_match(env: Environment, match_expr: MatchExpr) -> MatchExpr:
    """
    Optimize a match expression.
    
    Args:
        env: The environment
        match_expr: The match expression to optimize
        
    Returns:
        The optimized match expression
    """
    optimizer = PatternOptimizer(env)
    return optimizer.optimize_match(match_expr)


def generate_decision_tree(env: Environment, match_expr: MatchExpr) -> DecisionTree:
    """
    Generate a decision tree for efficient pattern matching.
    
    Args:
        env: The environment
        match_expr: The match expression
        
    Returns:
        A decision tree for the match expression
    """
    optimizer = PatternOptimizer(env)
    return optimizer.generate_decision_tree(match_expr) 