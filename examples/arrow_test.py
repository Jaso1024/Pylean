#!/usr/bin/env python3
"""
Test for arrow type notation in Pylean.
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
    Test parsing and elaboration of arrow notation.
    """
    print("PyLean Arrow Notation Test")
    print("=========================")
    
    # Create a new kernel with standard environment
    kernel = Kernel()
    print("Created kernel with standard environment")
    print()
    
    # Create an elaboration context
    elab_context = ElaborationContext(kernel.env)
    print("Created elaboration context")
    print()
    
    # Test parsing a simple arrow type
    print("Test 1: Parse a simple arrow type")
    print("------------------------------")
    
    arrow_input = "Type -> Type"
    print(f"Input: {arrow_input}")
    
    try:
        # Parse the arrow type
        arrow_syntax = parse_expression(arrow_input)
        print(f"Parsed syntax: {arrow_syntax}")
        
        # Elaborate to kernel expression
        arrow_expr = elaborate(arrow_syntax, elab_context)
        print(f"Elaborated expression: {arrow_expr}")
        
        if arrow_expr.kind == ExprKind.PI:
            print("✓ Success: Expression was correctly parsed as a Pi type")
        else:
            print(f"× Failed: Expression was parsed as {arrow_expr.kind} (expected PI)")
    except Exception as e:
        print(f"× Error: {e}")
    
    print()
    
    # Test parsing a Unicode arrow type
    print("Test 2: Parse with Unicode arrow")
    print("-----------------------------")
    
    unicode_arrow_input = "Type → Type"
    print(f"Input: {unicode_arrow_input}")
    
    try:
        # Parse the Unicode arrow type
        unicode_arrow_syntax = parse_expression(unicode_arrow_input)
        print(f"Parsed syntax: {unicode_arrow_syntax}")
        
        # Elaborate to kernel expression
        unicode_arrow_expr = elaborate(unicode_arrow_syntax, elab_context)
        print(f"Elaborated expression: {unicode_arrow_expr}")
        
        if unicode_arrow_expr.kind == ExprKind.PI:
            print("✓ Success: Expression was correctly parsed as a Pi type")
        else:
            print(f"× Failed: Expression was parsed as {unicode_arrow_expr.kind} (expected PI)")
    except Exception as e:
        print(f"× Error: {e}")
    
    print()
    
    print("Arrow Notation Test completed.")

if __name__ == "__main__":
    main() 