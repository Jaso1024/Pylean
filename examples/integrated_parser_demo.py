#!/usr/bin/env python3
"""
Enhanced demo of Pylean parser and kernel integration.

This example shows a more complete integration between the parser and kernel,
with conversion from parsed syntax to kernel expressions for a wider range 
of Lean syntax constructs.
"""

from pylean.kernel import (
    Name, Level, Expr, ExprKind,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, ReductionStrategy, ReductionMode, reduce
)
from pylean.parser.core import (
    ParserContext, ParserState, InputContext,
    SyntaxNode, Parser, SyntaxKind, 
    whitespace, token_fn, choice_fn, sequence_fn
)
from pylean.parser.expr import (
    parse_expression, ident_expr, number_expr, function_call,
    parens_expr, arrow_expr, type_expr, lambda_expr
)


def convert_syntax_to_expr(kernel: Kernel, node: SyntaxNode) -> Expr:
    """
    Convert a syntax node to a kernel expression.
    
    Args:
        kernel: The kernel instance
        node: The syntax node to convert
        
    Returns:
        The corresponding kernel expression
    """
    if node.kind == "ident":
        # Identifier - convert to constant or variable
        name = node.value
        try:
            # Try to find the name in the environment
            kernel.infer_type(mk_const(name))
            return mk_const(name)
        except Exception:
            # If not found, assume it's a variable (for simplicity)
            return mk_var(0, name)  # This is a simplification
    
    elif node.kind == "number":
        # Number literal - convert to a constant
        # In a real implementation, we'd convert numbers to their representations
        value = int(node.value)
        if value == 0:
            return mk_const("zero")
        
        # For simplicity, only handle small numbers
        result = mk_const("zero")
        for _ in range(value):
            result = mk_app(mk_const("succ"), result)
        return result
    
    elif node.kind == "app":
        # Function application - f(x)
        if len(node.children) == 2:
            fn = convert_syntax_to_expr(kernel, node.children[0])
            arg = convert_syntax_to_expr(kernel, node.children[1])
            return mk_app(fn, arg)
        else:
            raise ValueError(f"Invalid application: {node}")
    
    elif node.kind == "lambda":
        # Lambda expression - λx, body
        if len(node.children) == 3:
            var_name = node.children[0].value
            var_type = convert_syntax_to_expr(kernel, node.children[1])
            body = convert_syntax_to_expr(kernel, node.children[2])
            return mk_lambda(var_name, var_type, body)
        else:
            raise ValueError(f"Invalid lambda: {node}")
    
    elif node.kind == "pi":
        # Pi type - Π(x: A), B
        if len(node.children) == 3:
            var_name = node.children[0].value
            var_type = convert_syntax_to_expr(kernel, node.children[1])
            body_type = convert_syntax_to_expr(kernel, node.children[2])
            return mk_pi(var_name, var_type, body_type)
        else:
            raise ValueError(f"Invalid pi type: {node}")
    
    elif node.kind == "type":
        # Type literal - Type or Prop
        if node.value.lower() == "prop":
            return mk_sort(0)  # Prop is Sort 0
        else:
            # Extract universe level if present, e.g., Type 2
            level = 1  # Default is Type 1
            if len(node.children) > 0 and node.children[0].kind == "number":
                level = int(node.children[0].value)
            return mk_sort(level)
    
    elif node.kind == "parens":
        # Parenthesized expression - (expr)
        if len(node.children) == 1:
            return convert_syntax_to_expr(kernel, node.children[0])
        else:
            raise ValueError(f"Invalid parenthesized expression: {node}")
    
    elif node.kind == "binary":
        # Binary operations like a + b
        if len(node.children) == 3:
            left = convert_syntax_to_expr(kernel, node.children[0])
            op = node.children[1].value
            right = convert_syntax_to_expr(kernel, node.children[2])
            
            # Convert operator to function application
            if op == "+":
                # Assuming add function exists
                return mk_app(mk_app(mk_const("add"), left), right)
            elif op == "*":
                # Assuming mul function exists
                return mk_app(mk_app(mk_const("mul"), left), right)
            elif op == "→" or op == "->":
                # Function type - A → B is sugar for Π(_: A), B
                return mk_pi("_", left, right)
            else:
                raise ValueError(f"Unsupported binary operator: {op}")
        else:
            raise ValueError(f"Invalid binary expression: {node}")
    
    else:
        raise ValueError(f"Unsupported syntax node kind: {node.kind}")


def main():
    """Run the enhanced parser integration demo."""
    print("PyLean Enhanced Parser Integration Demo")
    print("=====================================")
    
    # Create a new kernel with standard environment
    kernel = Kernel()
    print("Created kernel with standard environment")
    print()
    
    # Step 1: Set up basic types and operations
    print("Step 1: Set up basic types and operations")
    print("--------------------------------------")
    
    # Define Nat type
    nat_type = mk_sort(0)
    kernel = kernel.add_constant("Nat", nat_type)
    print("Added Nat type")
    
    kernel = kernel.add_axiom("zero", mk_const("Nat"))
    print("Added zero constructor for Nat")
    
    succ_type = mk_pi("n", mk_const("Nat"), mk_const("Nat"))
    kernel = kernel.add_constant("succ", succ_type)
    print("Added succ constructor for Nat")
    
    # Define addition
    add_type = mk_pi("n", mk_const("Nat"), mk_pi("m", mk_const("Nat"), mk_const("Nat")))
    kernel = kernel.add_constant("add", add_type)
    print("Added add operation for Nat")
    
    # Define multiplication
    mul_type = mk_pi("n", mk_const("Nat"), mk_pi("m", mk_const("Nat"), mk_const("Nat")))
    kernel = kernel.add_constant("mul", mul_type)
    print("Added mul operation for Nat")
    
    # Define Bool type
    bool_type = mk_sort(0)
    kernel = kernel.add_constant("Bool", bool_type)
    print("Added Bool type")
    
    kernel = kernel.add_axiom("true", mk_const("Bool"))
    print("Added true constructor for Bool")
    
    kernel = kernel.add_axiom("false", mk_const("Bool"))
    print("Added false constructor for Bool")
    
    # Step 2: Parse and convert simple expressions
    print("\nStep 2: Parse and convert simple expressions")
    print("------------------------------------------")
    
    # Parse an identifier
    input_str = "zero"
    print(f"\nParsing identifier: '{input_str}'")
    input_ctx = InputContext(input=input_str, file_name="<input>")
    context = ParserContext(input_ctx=input_ctx)
    state = ParserState(context=context)
    
    id_parser = ident_expr()
    result_state = id_parser(context, state)
    
    if not result_state.has_error() and len(result_state.stx_stack) > 0:
        id_node = result_state.stx_stack[-1]
        print(f"Parsed identifier: {id_node.kind} = {id_node.value}")
        
        # Convert to kernel expression
        kernel_expr = convert_syntax_to_expr(kernel, id_node)
        print(f"Kernel expression: {kernel_expr}")
        print(f"Type: {kernel.infer_type(kernel_expr)}")
    else:
        print(f"Failed to parse identifier: {result_state.error}")
    
    # Parse a number
    input_str = "2"
    print(f"\nParsing number: '{input_str}'")
    input_ctx = InputContext(input=input_str, file_name="<input>")
    context = ParserContext(input_ctx=input_ctx)
    state = ParserState(context=context)
    
    num_parser = number_expr()
    result_state = num_parser(context, state)
    
    if not result_state.has_error() and len(result_state.stx_stack) > 0:
        num_node = result_state.stx_stack[-1]
        print(f"Parsed number: {num_node.kind} = {num_node.value}")
        
        # Convert to kernel expression
        kernel_expr = convert_syntax_to_expr(kernel, num_node)
        print(f"Kernel expression: {kernel_expr}")
        print(f"Type: {kernel.infer_type(kernel_expr)}")
    else:
        print(f"Failed to parse number: {result_state.error}")
    
    # Step 3: Parse and convert function applications
    print("\nStep 3: Parse and convert function applications")
    print("---------------------------------------------")
    
    # Parse a function application
    input_str = "succ(zero)"
    print(f"\nParsing function application: '{input_str}'")
    input_ctx = InputContext(input=input_str, file_name="<input>")
    context = ParserContext(input_ctx=input_ctx)
    state = ParserState(context=context)
    
    app_parser = function_call()
    result_state = app_parser(context, state)
    
    if not result_state.has_error() and len(result_state.stx_stack) > 0:
        app_node = result_state.stx_stack[-1]
        print(f"Parsed function application: {app_node.kind}")
        
        # Convert to kernel expression
        try:
            kernel_expr = convert_syntax_to_expr(kernel, app_node)
            print(f"Kernel expression: {kernel_expr}")
            print(f"Type: {kernel.infer_type(kernel_expr)}")
        except Exception as e:
            print(f"Error converting to kernel expression: {e}")
    else:
        print(f"Failed to parse function application: {result_state.error}")
    
    # Step 4: Parse and convert complex expressions
    print("\nStep 4: Parse a complete expression with the full parser")
    print("-----------------------------------------------------")
    
    # Parse a more complex expression using the full expression parser
    input_str = "add(2, mul(1, 2))"
    print(f"\nParsing complex expression: '{input_str}'")
    
    try:
        result = parse_expression(input_str)
        print(f"Parsed expression: {result.kind}")
        
        # Convert to kernel expression
        kernel_expr = convert_syntax_to_expr(kernel, result)
        print(f"Kernel expression: {kernel_expr}")
        print(f"Type: {kernel.infer_type(kernel_expr)}")
    except Exception as e:
        print(f"Error parsing or converting expression: {e}")
    
    # Step 5: Demonstrate function type parsing
    print("\nStep 5: Demonstrate function type parsing")
    print("---------------------------------------")
    
    # Parse a function type
    input_str = "Nat -> Nat"
    print(f"\nParsing function type: '{input_str}'")
    
    try:
        result = parse_expression(input_str)
        print(f"Parsed function type: {result.kind}")
        
        # Convert to kernel expression
        kernel_expr = convert_syntax_to_expr(kernel, result)
        print(f"Kernel expression: {kernel_expr}")
        print(f"Type: {kernel.infer_type(kernel_expr)}")
    except Exception as e:
        print(f"Error parsing or converting function type: {e}")
    
    # Step 6: Parse and type check a lambda expression
    print("\nStep 6: Parse and type check a lambda expression")
    print("---------------------------------------------")
    
    # Parse a lambda expression
    input_str = "λ(x: Nat), succ(x)"
    print(f"\nParsing lambda expression: '{input_str}'")
    
    try:
        result = parse_expression(input_str)
        print(f"Parsed lambda expression: {result.kind}")
        
        # Convert to kernel expression
        kernel_expr = convert_syntax_to_expr(kernel, result)
        print(f"Kernel expression: {kernel_expr}")
        print(f"Type: {kernel.infer_type(kernel_expr)}")
    except Exception as e:
        print(f"Error parsing or converting lambda expression: {e}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main() 