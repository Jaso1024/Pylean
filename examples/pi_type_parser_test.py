#!/usr/bin/env python3
"""
Test for Pi type parsing in Pylean.
"""

from pylean.kernel import (
    Name, Level, Expr, ExprKind,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, ReductionStrategy, ReductionMode, reduce
)
from pylean.parser.expr import parse_expression
from pylean.elaborator import elaborate, ElaborationContext

def main():
    """
    Test parsing and elaboration of Pi types.
    """
    print("PyLean Pi Type Parser Test")
    print("=========================")
    
    # Create a new kernel with standard environment
    kernel = Kernel()
    print("Created kernel with standard environment")
    print()
    
    # Create an elaboration context
    elab_context = ElaborationContext(kernel.env)
    print("Created elaboration context")
    print()
    
    # Test parsing a simple Pi type
    print("Test 1: Parse a simple Pi type")
    print("----------------------------")
    
    pi_input = "forall x:Type, Type"
    print(f"Input: {pi_input}")
    
    try:
        # Parse the Pi type
        pi_syntax = parse_expression(pi_input)
        print(f"Parsed syntax: {pi_syntax}")
        
        # Elaborate to kernel expression
        pi_expr = elaborate(pi_syntax, elab_context)
        print(f"Elaborated expression: {pi_expr}")
        
        if pi_expr.kind == ExprKind.PI:
            print("✓ Success: Expression was correctly parsed as a Pi type")
        else:
            print(f"× Failed: Expression was parsed as {pi_expr.kind} (expected PI)")
    except Exception as e:
        print(f"× Error: {e}")
    
    print()
    
    # Test parsing a more complex Pi type
    print("Test 2: Parse a nested Pi type")
    print("---------------------------")
    
    nested_pi_input = "forall A:Type, forall x:A, A"
    print(f"Input: {nested_pi_input}")
    
    try:
        # Parse the nested Pi type
        nested_pi_syntax = parse_expression(nested_pi_input)
        print(f"Parsed syntax: {nested_pi_syntax}")
        
        # Elaborate to kernel expression
        nested_pi_expr = elaborate(nested_pi_syntax, elab_context)
        print(f"Elaborated expression: {nested_pi_expr}")
        
        if nested_pi_expr.kind == ExprKind.PI:
            print("✓ Success: Expression was correctly parsed as a Pi type")
        else:
            print(f"× Failed: Expression was parsed as {nested_pi_expr.kind} (expected PI)")
    except Exception as e:
        print(f"× Error: {e}")
    
    print()
    
    # Test with Unicode Pi symbol
    print("Test 3: Parse Pi type with Unicode Pi symbol")
    print("----------------------------------------")
    
    unicode_pi_input = "Π x:Type, Type"
    print(f"Input: {unicode_pi_input}")
    
    try:
        # Parse the Pi type with Unicode
        unicode_pi_syntax = parse_expression(unicode_pi_input)
        print(f"Parsed syntax: {unicode_pi_syntax}")
        
        # Elaborate to kernel expression
        unicode_pi_expr = elaborate(unicode_pi_syntax, elab_context)
        print(f"Elaborated expression: {unicode_pi_expr}")
        
        if unicode_pi_expr.kind == ExprKind.PI:
            print("✓ Success: Expression was correctly parsed as a Pi type")
        else:
            print(f"× Failed: Expression was parsed as {unicode_pi_expr.kind} (expected PI)")
    except Exception as e:
        print(f"× Error: {e}")
    
    print()

    # Test with parenthesized variables
    print("Test 4: Parse Pi type with parenthesized variables")
    print("----------------------------------------------")
    
    paren_pi_input = "forall (x:Type), Type"
    print(f"Input: {paren_pi_input}")
    
    try:
        # Parse the Pi type with parentheses
        paren_pi_syntax = parse_expression(paren_pi_input)
        print(f"Parsed syntax: {paren_pi_syntax}")
        
        # Elaborate to kernel expression
        paren_pi_expr = elaborate(paren_pi_syntax, elab_context)
        print(f"Elaborated expression: {paren_pi_expr}")
        
        if paren_pi_expr.kind == ExprKind.PI:
            print("✓ Success: Expression was correctly parsed as a Pi type")
        else:
            print(f"× Failed: Expression was parsed as {paren_pi_expr.kind} (expected PI)")
    except Exception as e:
        print(f"× Error: {e}")
    
    print()
    
    # Test with a complex return type
    print("Test 5: Parse Pi type with arrow in the body")
    print("------------------------------------------")
    
    complex_pi_input = "Type -> Type -> Type"
    print(f"Input: {complex_pi_input}")
    
    try:
        # Parse the Pi type with arrow
        complex_pi_syntax = parse_expression(complex_pi_input)
        print(f"Parsed syntax: {complex_pi_syntax}")
        
        # Elaborate to kernel expression
        complex_pi_expr = elaborate(complex_pi_syntax, elab_context)
        print(f"Elaborated expression: {complex_pi_expr}")
        
        if complex_pi_expr.kind == ExprKind.PI:
            print("✓ Success: Expression was correctly parsed as a Pi type")
        else:
            print(f"× Failed: Expression was parsed as {complex_pi_expr.kind} (expected PI)")
    except Exception as e:
        print(f"× Error: {e}")
    
    print()
    
    # Test with combined forall and arrow notation
    print("Test 6: Parse Pi type with forall and arrow combined")
    print("------------------------------------------------")
    
    mixed_pi_input = "forall A:Type, A -> A"
    print(f"Input: {mixed_pi_input}")
    
    try:
        # Parse the Pi type with arrow
        mixed_pi_syntax = parse_expression(mixed_pi_input)
        print(f"Parsed syntax: {mixed_pi_syntax}")
        
        # Elaborate to kernel expression
        mixed_pi_expr = elaborate(mixed_pi_syntax, elab_context)
        print(f"Elaborated expression: {mixed_pi_expr}")
        
        if mixed_pi_expr.kind == ExprKind.PI:
            print("✓ Success: Expression was correctly parsed as a Pi type")
        else:
            print(f"× Failed: Expression was parsed as {mixed_pi_expr.kind} (expected PI)")
    except Exception as e:
        print(f"× Error: {e}")
    
    print()
    
    print("Pi Type Parser Test completed.")

if __name__ == "__main__":
    main() 