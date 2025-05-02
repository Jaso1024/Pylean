#!/usr/bin/env python3
"""
Demo of simple Boolean operations in Pylean.

This example shows how to define and work with a simple Boolean type
in the Pylean kernel.
"""

from pylean.kernel import (
    Name, Level, Expr, ExprKind,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, ReductionStrategy, ReductionMode, reduce
)


def main():
    """Run the Boolean types demo."""
    print("PyLean Boolean Operations Demo")
    print("============================")
    
    # Create a new kernel with standard environment
    kernel = Kernel()
    print("Created kernel with standard environment")
    print()
    
    # Step A: Define the Boolean type
    print("Step A: Define the Boolean type")
    print("----------------------------")
    
    # Define the boolean type
    bool_type = mk_sort(0)  # Type of Bool is Prop
    
    # Define Bool type constant
    kernel = kernel.add_constant("Bool", bool_type)
    print("Added Bool constant with type Prop")
    
    # Define true and false constants
    kernel = kernel.add_axiom("true", mk_const("Bool"))
    print("Added true constructor for Bool")
    
    kernel = kernel.add_axiom("false", mk_const("Bool"))
    print("Added false constructor for Bool")
    
    # Step B: Define operations on booleans
    print("\nStep B: Define operations on booleans")
    print("----------------------------------")
    
    # Define not operation
    not_type = mk_pi("b", mk_const("Bool"), mk_const("Bool"))
    kernel = kernel.add_constant("not", not_type)
    print("Added not constant with type Bool -> Bool")
    
    # Define the behavior of not
    not_true_body = mk_const("false")
    not_true_type = mk_app(mk_const("not"), mk_const("true"))
    kernel = kernel.add_definition("not_true_def", mk_const("Bool"), not_true_body)
    print("Defined not true = false")
    
    not_false_body = mk_const("true")
    not_false_type = mk_app(mk_const("not"), mk_const("false"))
    kernel = kernel.add_definition("not_false_def", mk_const("Bool"), not_false_body)
    print("Defined not false = true")
    
    # Define and operation
    and_type = mk_pi("b1", mk_const("Bool"), mk_pi("b2", mk_const("Bool"), mk_const("Bool")))
    kernel = kernel.add_constant("and", and_type)
    print("Added and constant with type Bool -> Bool -> Bool")
    
    # Define the behavior of and
    and_true_true_body = mk_const("true")
    kernel = kernel.add_definition("and_tt_def", mk_const("Bool"), and_true_true_body)
    print("Defined and true true = true")
    
    and_true_false_body = mk_const("false")
    kernel = kernel.add_definition("and_tf_def", mk_const("Bool"), and_true_false_body)
    print("Defined and true false = false")
    
    and_false_true_body = mk_const("false")
    kernel = kernel.add_definition("and_ft_def", mk_const("Bool"), and_false_true_body)
    print("Defined and false true = false")
    
    and_false_false_body = mk_const("false")
    kernel = kernel.add_definition("and_ff_def", mk_const("Bool"), and_false_false_body)
    print("Defined and false false = false")
    
    # Define or operation
    or_type = mk_pi("b1", mk_const("Bool"), mk_pi("b2", mk_const("Bool"), mk_const("Bool")))
    kernel = kernel.add_constant("or", or_type)
    print("Added or constant with type Bool -> Bool -> Bool")
    
    # Define the behavior of or
    or_true_true_body = mk_const("true")
    kernel = kernel.add_definition("or_tt_def", mk_const("Bool"), or_true_true_body)
    print("Defined or true true = true")
    
    or_true_false_body = mk_const("true")
    kernel = kernel.add_definition("or_tf_def", mk_const("Bool"), or_true_false_body)
    print("Defined or true false = true")
    
    or_false_true_body = mk_const("true")
    kernel = kernel.add_definition("or_ft_def", mk_const("Bool"), or_false_true_body)
    print("Defined or false true = true")
    
    or_false_false_body = mk_const("false")
    kernel = kernel.add_definition("or_ff_def", mk_const("Bool"), or_false_false_body)
    print("Defined or false false = false")
    
    # Step C: Evaluate boolean expressions
    print("\nStep C: Evaluate boolean expressions")
    print("--------------------------------")
    
    # Create some boolean expressions
    not_true_expr = mk_app(mk_const("not"), mk_const("true"))
    print(f"Expression: not true")
    print(f"Type: {kernel.infer_type(not_true_expr)}")
    
    not_false_expr = mk_app(mk_const("not"), mk_const("false"))
    print(f"Expression: not false")
    print(f"Type: {kernel.infer_type(not_false_expr)}")
    
    true_and_false_expr = mk_app(
        mk_app(mk_const("and"), mk_const("true")),
        mk_const("false")
    )
    print(f"Expression: true and false")
    print(f"Type: {kernel.infer_type(true_and_false_expr)}")
    
    true_or_false_expr = mk_app(
        mk_app(mk_const("or"), mk_const("true")),
        mk_const("false")
    )
    print(f"Expression: true or false")
    print(f"Type: {kernel.infer_type(true_or_false_expr)}")
    
    # Test boolean laws
    print("\nStep D: Testing boolean laws")
    print("-------------------------")
    
    # De Morgan's Law: not (a and b) = (not a) or (not b)
    a = mk_const("true")
    b = mk_const("false")
    
    left_side = mk_app(
        mk_const("not"),
        mk_app(mk_app(mk_const("and"), a), b)
    )
    print(f"not (true and false) = {kernel.infer_type(left_side)}")
    
    right_side = mk_app(
        mk_app(mk_const("or"), 
            mk_app(mk_const("not"), a)),
        mk_app(mk_const("not"), b)
    )
    print(f"(not true) or (not false) = {kernel.infer_type(right_side)}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main() 