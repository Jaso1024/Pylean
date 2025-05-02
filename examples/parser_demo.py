#!/usr/bin/env python3
"""
Demo of the Pylean parser and kernel integration.

This example shows how to parse Lean expressions and convert them
to kernel expressions for type checking and evaluation.
"""

from pylean.kernel import (
    Name, Level, Expr, ExprKind,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, ReductionStrategy, ReductionMode, reduce
)
from pylean.parser.core import (
    ParserContext, ParserState, InputContext,
    SyntaxNode, Parser, whitespace, token_fn
)
from pylean.parser.expr import (
    parse_expression, ident_expr, number_expr, function_call
)


def main():
    """Run the parser integration demo."""
    print("PyLean Parser Integration Demo")
    print("============================")
    
    # Create a new kernel with standard environment
    kernel = Kernel()
    print("Created kernel with standard environment")
    print()
    
    # Step A: Define basic types in the kernel
    print("Step A: Define basic types in the kernel")
    print("------------------------------------")
    
    nat_type = mk_sort(0)
    kernel = kernel.add_constant("Nat", nat_type)
    print("Added Nat type")
    
    bool_type = mk_sort(0)
    kernel = kernel.add_constant("Bool", bool_type)
    print("Added Bool type")
    
    kernel = kernel.add_axiom("true", mk_const("Bool"))
    print("Added true constructor")
    
    kernel = kernel.add_axiom("false", mk_const("Bool"))
    print("Added false constructor")
    
    # Step B: Manual parsing demonstration
    print("\nStep B: Manual parsing demonstration")
    print("--------------------------------")
    
    # Manually create a parser context and state for parsing
    input_str = "f(x) + 42"
    input_ctx = InputContext(input=input_str, file_name="<input>")
    context = ParserContext(input_ctx=input_ctx)
    state = ParserState(context=context)
    
    print(f"Input: {input_str}")
    
    # Parse a simple identifier token
    print("\nParsing a simple identifier:")
    state.pos = 0  # Reset position
    id_parser = ident_expr()
    result_state = id_parser(context, state)
    
    if not result_state.has_error() and len(result_state.stx_stack) > 0:
        id_node = result_state.stx_stack[-1]
        print(f"Parsed identifier: {id_node.kind}")
    else:
        print(f"Failed to parse identifier: {result_state.error}")
    
    # Parse a number
    print("\nParsing a number:")
    input_str = "42"
    input_ctx = InputContext(input=input_str, file_name="<input>")
    context = ParserContext(input_ctx=input_ctx)
    state = ParserState(context=context)
    
    num_parser = number_expr()
    result_state = num_parser(context, state)
    
    if not result_state.has_error() and len(result_state.stx_stack) > 0:
        num_node = result_state.stx_stack[-1]
        print(f"Parsed number: {num_node.kind}")
    else:
        print(f"Failed to parse number: {result_state.error}")
    
    # Step C: Convert parsed expressions to kernel expressions
    print("\nStep C: Create kernel expressions")
    print("------------------------------")
    
    # Skip the intermediate parsing steps for now, and directly create kernel expressions
    print("\nCreating Nat constant:")
    kernel_expr = mk_const("Nat")
    print(f"Kernel expression: {kernel_expr}")
    print(f"Type: {kernel.infer_type(kernel_expr)}")
    
    print("\nCreating 'true' constant:")
    kernel_expr = mk_const("true")
    print(f"Kernel expression: {kernel_expr}")
    print(f"Type: {kernel.infer_type(kernel_expr)}")
    
    # Step D: Type check kernel expressions
    print("\nStep D: Type check kernel expressions")
    print("--------------------------------")
    
    # Add some functions for type checking
    not_type = mk_pi("b", mk_const("Bool"), mk_const("Bool"))
    kernel = kernel.add_constant("not", not_type)
    print("Added 'not' function")
    
    add_type = mk_pi("n", mk_const("Nat"), mk_pi("m", mk_const("Nat"), mk_const("Nat")))
    kernel = kernel.add_constant("add", add_type)
    print("Added 'add' function")
    
    print("\nType checking 'not true':")
    try:
        # Create the kernel expression
        not_true_expr = mk_app(mk_const("not"), mk_const("true"))
        print(f"Expression: not true")
        print(f"Type: {kernel.infer_type(not_true_expr)}")
    except Exception as e:
        print(f"Type error: {e}")
    
    print("\nType checking 'add 1 2':")
    try:
        # Add number constants
        kernel = kernel.add_axiom("1", mk_const("Nat"))
        kernel = kernel.add_axiom("2", mk_const("Nat"))
        
        # Create the expression
        add_expr = mk_app(mk_app(mk_const("add"), mk_const("1")), mk_const("2"))
        print(f"Expression: add 1 2")
        print(f"Type: {kernel.infer_type(add_expr)}")
    except Exception as e:
        print(f"Type error: {e}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main() 