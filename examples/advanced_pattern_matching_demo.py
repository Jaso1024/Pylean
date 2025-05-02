#!/usr/bin/env python
"""
Advanced Pattern Matching Optimization demonstration for Pylean.

This example demonstrates the advanced pattern matching optimizations:
1. Exhaustiveness checking for pattern matching
2. Redundancy checking for pattern matching 
3. Decision tree generation for efficient pattern matching
"""

import os
from pathlib import Path
import sys

from pylean.kernel import (
    Environment, Name, Level,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi, mk_match,
    mk_pattern, mk_alternative, mk_inductive, mk_constructor
)
from pylean.kernel.pattern_opt import (
    check_exhaustiveness, optimize_match, generate_decision_tree,
    PatternOptimizer, DecisionTree
)


def print_decision_tree(tree: DecisionTree, indent: int = 0):
    """Print a decision tree with proper indentation."""
    indent_str = "  " * indent
    
    if tree is None:
        print(f"{indent_str}None")
        return
    
    if tree.kind == DecisionTree.NodeKind.LEAF:
        print(f"{indent_str}LEAF: action={tree.action_index}")
    elif tree.kind == DecisionTree.NodeKind.SWITCH:
        print(f"{indent_str}SWITCH on scrutinee[{tree.scrutinee}]:")
        for ctor, subtree in tree.cases.items():
            print(f"{indent_str}  case {ctor}:")
            print_decision_tree(subtree, indent + 2)
        if tree.default:
            print(f"{indent_str}  default:")
            print_decision_tree(tree.default, indent + 2)
    elif tree.kind == DecisionTree.NodeKind.GUARD:
        print(f"{indent_str}GUARD: {tree.guard}")
        print(f"{indent_str}  if true:")
        print_decision_tree(tree.success, indent + 2)
        print(f"{indent_str}  if false:")
        print_decision_tree(tree.failure, indent + 2)


def main():
    # Create an environment with basic inductive types
    env = Environment()
    
    # Define a simple boolean type
    bool_type = mk_sort(0)
    bool_true = mk_constructor("true", bool_type, "Bool")
    bool_false = mk_constructor("false", bool_type, "Bool")
    env = env.add_decl(mk_inductive(
        "Bool", bool_type, [bool_true, bool_false]
    ))
    
    # Define a simple option type
    option_type = mk_sort(0)
    option_some = mk_constructor(
        "some", mk_pi("x", mk_sort(0), option_type), "Option"
    )
    option_none = mk_constructor("none", option_type, "Option")
    env = env.add_decl(mk_inductive(
        "Option", option_type, [option_some, option_none]
    ))
    
    # Define a simple list type
    list_type = mk_sort(0)
    list_cons = mk_constructor(
        "cons", 
        mk_pi("head", mk_sort(0), 
              mk_pi("tail", list_type, list_type)), 
        "List"
    )
    list_nil = mk_constructor("nil", list_type, "List")
    env = env.add_decl(mk_inductive(
        "List", list_type, [list_cons, list_nil]
    ))
    
    # Define a tree type
    tree_type = mk_sort(0)
    tree_node = mk_constructor(
        "node", 
        mk_pi("value", mk_sort(0), 
              mk_pi("left", tree_type, 
                    mk_pi("right", tree_type, tree_type))),
        "Tree"
    )
    tree_leaf = mk_constructor(
        "leaf", 
        mk_pi("value", mk_sort(0), tree_type),
        "Tree"
    )
    env = env.add_decl(mk_inductive(
        "Tree", tree_type, [tree_node, tree_leaf]
    ))
    
    print("\n=== Advanced Pattern Matching Optimization Demo ===\n")
    
    # Example 1: Exhaustiveness checking
    print("Example 1: Exhaustiveness Checking")
    print("-----------------------------------\n")
    
    # Create a match expression for boolean type (exhaustive)
    bool_match = mk_match(
        mk_const("Bool"),  # scrutinee
        mk_sort(0),        # return type
        [
            mk_alternative(
                mk_pattern("true"),
                mk_const("result_true")
            ),
            mk_alternative(
                mk_pattern("false"),
                mk_const("result_false")
            )
        ]
    )
    
    # Check if the match is exhaustive
    is_exhaustive = check_exhaustiveness(env, bool_match)
    print(f"Boolean match with true/false patterns is exhaustive: {is_exhaustive}\n")
    
    # Create a match expression for boolean type (non-exhaustive)
    bool_match_non_exhaustive = mk_match(
        mk_const("Bool"),  # scrutinee
        mk_sort(0),        # return type
        [
            mk_alternative(
                mk_pattern("true"),
                mk_const("result_true")
            )
        ]
    )
    
    # Check if the match is exhaustive
    is_exhaustive = check_exhaustiveness(env, bool_match_non_exhaustive)
    print(f"Boolean match with only true pattern is exhaustive: {is_exhaustive}\n")
    
    # Create a match expression with wildcard (always exhaustive)
    bool_match_wildcard = mk_match(
        mk_const("Bool"),  # scrutinee
        mk_sort(0),        # return type
        [
            mk_alternative(
                mk_pattern("true"),
                mk_const("result_true")
            ),
            mk_alternative(
                mk_pattern("_"),  # wildcard
                mk_const("result_wildcard")
            )
        ]
    )
    
    # Check if the match is exhaustive
    is_exhaustive = check_exhaustiveness(env, bool_match_wildcard)
    print(f"Boolean match with true/wildcard patterns is exhaustive: {is_exhaustive}\n")
    
    # Example 2: Redundancy checking and optimization
    print("\nExample 2: Redundancy Checking")
    print("----------------------------\n")
    
    # Create a match expression with redundant patterns
    redundant_match = mk_match(
        mk_const("Bool"),  # scrutinee
        mk_sort(0),        # return type
        [
            mk_alternative(
                mk_pattern("true"),
                mk_const("result_true")
            ),
            mk_alternative(
                mk_pattern("_"),  # wildcard
                mk_const("result_wildcard")
            ),
            mk_alternative(
                mk_pattern("false"),  # redundant after wildcard
                mk_const("result_false")
            )
        ]
    )
    
    print("Original match expression with redundancy:")
    for i, alt in enumerate(redundant_match.alternatives):
        print(f"  Pattern {i}: {alt.pattern.constructor}")
    
    # Optimize the match expression
    optimized_match = optimize_match(env, redundant_match)
    
    print("\nOptimized match expression:")
    for i, alt in enumerate(optimized_match.alternatives):
        print(f"  Pattern {i}: {alt.pattern.constructor}")
    
    # Example 3: Decision tree generation
    print("\nExample 3: Decision Tree Generation")
    print("--------------------------------\n")
    
    # Create a more complex match expression for list type
    list_match = mk_match(
        mk_const("List"),  # scrutinee
        mk_sort(0),        # return type
        [
            mk_alternative(
                mk_pattern("nil"),
                mk_const("empty_result")
            ),
            mk_alternative(
                mk_pattern("cons", ["head", "tail"]),
                mk_const("cons_result")
            )
        ]
    )
    
    # Generate a decision tree
    decision_tree = generate_decision_tree(env, list_match)
    
    print("Decision tree for list match:")
    print_decision_tree(decision_tree)
    
    # Create a match expression for option type
    option_match = mk_match(
        mk_const("Option"),  # scrutinee
        mk_sort(0),          # return type
        [
            mk_alternative(
                mk_pattern("some", ["value"]),
                mk_const("some_result")
            ),
            mk_alternative(
                mk_pattern("none"),
                mk_const("none_result")
            )
        ]
    )
    
    # Generate a decision tree
    decision_tree = generate_decision_tree(env, option_match)
    
    print("\nDecision tree for option match:")
    print_decision_tree(decision_tree)
    
    # Create a match expression for tree type
    tree_match = mk_match(
        mk_const("Tree"),  # scrutinee
        mk_sort(0),        # return type
        [
            mk_alternative(
                mk_pattern("node", ["value", "left", "right"]),
                mk_const("node_result")
            ),
            mk_alternative(
                mk_pattern("leaf", ["value"]),
                mk_const("leaf_result")
            )
        ]
    )
    
    # Generate a decision tree
    decision_tree = generate_decision_tree(env, tree_match)
    
    print("\nDecision tree for tree match:")
    print_decision_tree(decision_tree)
    
    print("\nBenefits of These Optimizations:")
    print("--------------------------------")
    print("1. Exhaustiveness checking prevents runtime errors by ensuring all")
    print("   possible values are handled by the match expression.")
    print("2. Redundancy checking eliminates dead code and improves performance.")
    print("3. Decision tree generation produces efficient code by minimizing")
    print("   the number of tests needed to determine which pattern matches.")
    print("4. These optimizations enable the compiler to generate faster code")
    print("   and provide better error messages to the user.")


if __name__ == "__main__":
    main() 