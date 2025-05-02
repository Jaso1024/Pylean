"""
Expression parsing module for Pylean.

This module implements specific parsers for different expression constructs
and sets up the Pratt parsing tables for expression parsing.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any

from pylean.parser.core import (
    Parser, ParserState, ParserContext, SyntaxNode,
    check_prec, set_lhs_prec, and_then,
    whitespace, InputContext, peek_token_node,
    is_id_first, is_id_rest
)

# Import these as callable functions
from pylean.parser.core import symbol, number_literal, token_fn, node, ident
from pylean.parser.pratt import (
    PrattParsingTables, TokenMap, pratt_parser
)

# Kernel and Elaboration imports
from pylean.kernel.expr import Expr # For return type hint
from pylean.kernel.env import mk_std_env, Environment
from pylean.elaborator import elaborate, ElaborationContext


# --- Expression Node Kinds ---

# Basic literals and identifiers
IDENT_EXPR = "ident_expr"
NUM_EXPR = "num_expr"
PAREN_EXPR = "paren_expr"

# Unary operators
UNARY_MINUS = "unary_minus"
UNARY_NOT = "unary_not"

# Binary operators
BINARY_OP = "binary_op"  # Generic binary operation
ADD_EXPR = "add_expr"
SUB_EXPR = "sub_expr"
MUL_EXPR = "mul_expr"
DIV_EXPR = "div_expr"
EQ_EXPR = "eq_expr"
LT_EXPR = "lt_expr"
GT_EXPR = "gt_expr"

# Function application
APP_EXPR = "app_expr"
FIELD_ACCESS = "field_access"

# Lambda and Pi expressions
LAMBDA_EXPR = "lambda_expr"
PI_EXPR = "pi_expr"

# Generic expression
EXPR = "expr"

# Constants for expression kinds
ARROW_EXPR = "arrow_expr"  # Non-dependent function type (A -> B)


# --- Leading (Prefix/Primary) Parsers ---

def function_call() -> Parser:
    """Parse a function call expression like f(x)."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        # First try to parse an identifier (the function name)
        state1 = ident_expr()(context, state)
        if state1.has_error():
            return state1
        
        # Make sure we have a function name on the stack
        if len(state1.stx_stack) == 0:
            return state1.make_error("Expected function name")
        
        func_node = state1.stx_stack.pop()
        
        # Skip whitespace between function name and opening parenthesis
        state2 = state1.clone()
        while not state2.is_eof() and state2.current_char().isspace():
            state2.pos += 1
        
        # Check if there's an open parenthesis
        if state2.is_eof() or state2.current_char() != '(':
            # Not a function call, just put the identifier back
            state1.stx_stack.append(func_node)
            return state1
        
        # Consume the opening parenthesis
        state2.pos += 1
        open_paren = SyntaxNode(kind="symbol", value="(")
        
        # Skip whitespace after the opening parenthesis
        while not state2.is_eof() and state2.current_char().isspace():
            state2.pos += 1
        
        # Parse the argument expression
        expr_parser_instance = parse_expr(0)  # Start with lowest precedence for argument
        arg_state = expr_parser_instance(context, state2)
        if arg_state.has_error():
            # If parsing argument fails, put back function node
            state1.stx_stack.append(func_node)
            return arg_state
        
        # Make sure we have an argument on the stack
        if len(arg_state.stx_stack) == 0:
            state1.stx_stack.append(func_node)
            return arg_state.make_error("Expected argument expression")
        
        arg_node = arg_state.stx_stack.pop()
        
        # Skip whitespace before the closing parenthesis
        while not arg_state.is_eof() and arg_state.current_char().isspace():
            arg_state.pos += 1
        
        # Check for the closing parenthesis
        if arg_state.is_eof() or arg_state.current_char() != ')':
            state1.stx_stack.append(func_node)
            arg_state.stx_stack.append(arg_node)
            return arg_state.make_error("Expected closing parenthesis ')'")
        
        # Consume the closing parenthesis
        arg_state.pos += 1
        close_paren = SyntaxNode(kind="symbol", value=")")
        
        # Create function application node
        app_node = SyntaxNode(kind=APP_EXPR, children=[func_node, arg_node])
        
        # Add to syntax stack
        arg_state.stx_stack.append(app_node)
        
        # Set LHS precedence (applications have high precedence)
        arg_state.lhs_prec = 180
        
        return arg_state
    
    return Parser(fn)


def ident_expr() -> Parser:
    """Parse an identifier as an expression."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        # Save starting position
        start_pos = state.pos
        
        # Try to find identifier token using token_fn
        token_parser = token_fn()
        token_state = token_parser(context, state)
        
        if token_state.has_error():
            return token_state
        
        # Check if we got an identifier token
        if len(token_state.stx_stack) == 0:
            return state.make_error("Expected identifier token")
        
        token = token_state.stx_stack.pop()
        
        if token.kind != "ident":
            # Put the token back and report error
            token_state.stx_stack.append(token)
            return state.make_error(f"Expected identifier, got {token.kind}")
        
        # Create an identifier expression node
        ident_expr_node = SyntaxNode(kind=IDENT_EXPR, children=[token])
        
        # Add to stack
        token_state.stx_stack.append(ident_expr_node)
        
        # Set precedence (atoms have high precedence)
        token_state.lhs_prec = 200
        
        return token_state
    
    return Parser(fn)


def number_expr() -> Parser:
    """Parse a number literal as an expression."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        # Save starting position
        start_pos = state.pos
        
        # Try to find number token using token_fn
        token_parser = token_fn()
        token_state = token_parser(context, state)
        
        if token_state.has_error():
            return token_state
        
        # Check if we got a number token
        if len(token_state.stx_stack) == 0:
            return state.make_error("Expected number token")
        
        token = token_state.stx_stack.pop()
        
        if token.kind != "num_lit":
            # Put the token back and report error
            token_state.stx_stack.append(token)
            return state.make_error(f"Expected number, got {token.kind}")
        
        # Create a number expression node
        num_expr_node = SyntaxNode(kind=NUM_EXPR, children=[token])
        
        # Add to stack
        token_state.stx_stack.append(num_expr_node)
        
        # Set precedence (atoms have high precedence)
        token_state.lhs_prec = 200
        
        return token_state
    
    return Parser(fn)


def paren_expr() -> Parser:
    """Parse a parenthesized expression like (expr)."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        # Save starting position
        start_pos = state.pos
        
        # Check for opening parenthesis
        if state.is_eof() or state.current_char() != '(':
            return state.make_error("Expected '('")
        
        # Create opening parenthesis token
        open_paren = SyntaxNode(kind="symbol", value="(")
        
        # Consume the opening parenthesis
        state.pos += 1
        
        # Skip whitespace
        while not state.is_eof() and state.current_char().isspace():
            state.pos += 1
        
        # Parse inner expression
        inner_expr_parser = parse_expr(0)  # Start with lowest precedence inside parentheses
        inner_state = inner_expr_parser(context, state)
        
        if inner_state.has_error():
            return inner_state
        
        # Make sure we have an expression
        if len(inner_state.stx_stack) == 0:
            return inner_state.make_error("Expected expression inside parentheses")
        
        inner_expr = inner_state.stx_stack.pop()
        
        # Skip whitespace
        while not inner_state.is_eof() and inner_state.current_char().isspace():
            inner_state.pos += 1
        
        # Check for closing parenthesis
        if inner_state.is_eof() or inner_state.current_char() != ')':
            inner_state.stx_stack.append(inner_expr)  # Put back inner expression
            return inner_state.make_error("Expected ')'")
        
        # Create closing parenthesis token
        close_paren = SyntaxNode(kind="symbol", value=")")
        
        # Consume the closing parenthesis
        inner_state.pos += 1
        
        # Create parenthesized expression node
        paren_expr_node = SyntaxNode(kind=PAREN_EXPR, children=[open_paren, inner_expr, close_paren])
        
        # Add to stack
        inner_state.stx_stack.append(paren_expr_node)
        
        # Set precedence (parenthesized expressions have high precedence like atoms)
        inner_state.lhs_prec = 200
        
        return inner_state
    
    return Parser(fn)


def unary_minus() -> Parser:
    """Parse a unary minus expression like -expr."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        # Save the position of the minus sign
        start_pos = state.pos
        
        # Check for minus sign
        if state.is_eof() or state.current_char() != '-':
            return state.make_error("Expected '-'")
        
        # Create the minus token
        minus_token = SyntaxNode(kind="symbol", value="-")
        
        # Consume the minus sign
        state.pos += 1
        
        # Skip whitespace after minus
        while not state.is_eof() and state.current_char().isspace():
            state.pos += 1
        
        # Parse the expression to negate with high precedence (stronger than binary operators)
        # Unary minus binds more tightly than binary operators
        expr_parser_instance = parse_expr(150)  # Higher precedence for unary operators
        expr_state = expr_parser_instance(context, state)
        if expr_state.has_error():
            return expr_state
        
        # Get the expression to negate
        if len(expr_state.stx_stack) == 0:
            return expr_state.make_error("Expected expression after '-'")
        
        expr_node = expr_state.stx_stack.pop()
        
        # Create unary minus node
        unary_minus_node = SyntaxNode(kind=UNARY_MINUS, children=[minus_token, expr_node])
        
        # Add to syntax stack
        expr_state.stx_stack.append(unary_minus_node)
        
        # Set LHS precedence to match unary operators
        expr_state.lhs_prec = 150
        
        return expr_state
    
    return Parser(fn)


# --- Trailing (Infix/Postfix) Parsers ---

def binary_op(op_symbol: str, kind: str, prec: int) -> Parser:
    """Parse a binary operator expression like 'a + b'."""
    def fn(context: Any, state: ParserState) -> ParserState:
        # For binary operators in Pratt parsing, context is the left-hand side SyntaxNode
        # And state is the current parsing state after consuming the left-hand side
        
        # Save the current position and stack size (for error recovery)
        start_pos = state.pos
        stack_size = len(state.stx_stack)
        
        # Get left-hand side from the stack
        if len(state.stx_stack) == 0:
            return state.make_error(f"Missing left-hand operand for '{op_symbol}'")
        
        lhs = state.stx_stack.pop()
        
        # Skip whitespace before operator
        input_text = state.context.input_ctx.input
        current_pos = state.pos
        
        # Skip whitespace manually
        while current_pos < len(input_text) and input_text[current_pos].isspace():
            current_pos += 1
        
        # Check if we're at EOF
        if current_pos >= len(input_text):
            # Put back the lhs and return error
            state.stx_stack.append(lhs)
            return state.make_error("Unexpected end of input")
        
        # Now check if the operator matches
        if current_pos + len(op_symbol) <= len(input_text):
            # Check if the operator matches
            if input_text[current_pos:current_pos + len(op_symbol)] == op_symbol:
                # Found the operator, advance past it
                state.pos = current_pos + len(op_symbol)
                
                # Create operator token
                op_token = SyntaxNode(kind="symbol", value=op_symbol)
                
                # Skip whitespace after operator
                while state.pos < len(input_text) and input_text[state.pos].isspace():
                    state.pos += 1
                
                # Check if we're at EOF
                if state.pos >= len(input_text):
                    # Put back the lhs and return error
                    state.stx_stack.append(lhs)
                    return state.make_error(f"Missing right-hand operand for '{op_symbol}'")
                
                # Parse the right-hand side expression
                rhs_prec = prec + 1  # Left associative
                rhs_state = parse_expr(rhs_prec)(context, state)
                
                if rhs_state.has_error():
                    # If right side parsing failed, put back the lhs and return error
                    state.stx_stack.append(lhs)
                    return rhs_state
                
                # Check if we have a right-hand side
                if len(rhs_state.stx_stack) == 0:
                    # Put back the lhs and return error
                    state.stx_stack.append(lhs)
                    return rhs_state.make_error(f"Missing right-hand operand for '{op_symbol}'")
                
                # Get the right-hand side
                rhs = rhs_state.stx_stack.pop()
                
                # Create binary operation node
                bin_op_node = SyntaxNode(kind=kind, children=[lhs, op_token, rhs])
                
                # Add the binary operation node to the stack
                rhs_state.stx_stack.append(bin_op_node)
                
                # Set LHS precedence
                rhs_state.lhs_prec = prec
                
                return rhs_state
            else:
                # Operator doesn't match, put back the lhs and return error
                state.stx_stack.append(lhs)
                return state.make_error(f"Expected '{op_symbol}' operator")
        else:
            # Not enough characters left for the operator
            state.stx_stack.append(lhs)
            return state.make_error("Unexpected end of input")
    
    return Parser(fn)


def app_expr() -> Parser:
    """Parse a function application: f(x)."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        # The precedence of function application
        app_prec = 180  # Higher precedence for function application
        
        # Need to get the function expression from outside
        if not state.stx_stack:
            return state.make_error("Expected function for application")
        
        # Check that we have sufficient precedence to continue
        if state.lhs_prec > app_prec:
            return state.make_error("Insufficient precedence for function application")
        
        # Take the function expression from the stack
        func = state.stx_stack.pop()
        
        # We need to ensure the function node is an expression
        if func.kind not in ['expr', 'ident_expr', IDENT_EXPR, PAREN_EXPR]:
            if func.kind == 'ident':
                # Convert ident to ident_expr
                func = SyntaxNode(kind=IDENT_EXPR, children=[func])
        
        # Parse the argument list in parentheses
        state1 = (
            symbol("(") >>
            parse_expr(0) >>  # Reset precedence inside parentheses
            symbol(")")
        )(context, state)
        
        if state1.has_error():
            # Restore function and propagate error
            state.stx_stack.append(func)
            return state1
        
        # Reconstruct as a function application
        if len(state1.stx_stack) >= 3:
            # Pop the nodes in reverse order
            close_paren = state1.stx_stack.pop()
            arg = state1.stx_stack.pop()
            open_paren = state1.stx_stack.pop()
            
            # Create an application node
            app_node = SyntaxNode(
                kind=APP_EXPR,
                children=[func, arg]
            )
            state1.stx_stack.append(app_node)
            # Set left-hand side precedence
            state1.lhs_prec = app_prec
        else:
            # Something went wrong, put back function
            state1.stx_stack.append(func)
            state1.make_error("Failed to parse function arguments")
        
        return state1
    
    return Parser(fn)


def lambda_expr() -> Parser:
    """Parse a lambda expression like λx:Type, x or λ(x:Type), x."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        # Save starting position
        start_pos = state.pos
        
        # Check for lambda character or keyword
        if state.is_eof():
            return state.make_error("Expected lambda expression")
        
        # Try to match λ character or "lambda" keyword
        if state.current_char() == 'λ':
            # Consume the lambda character
            state = state.advance(1)
        elif state.remaining_length() >= 6 and state.text[state.pos:state.pos+6] == 'lambda':
            # Consume the lambda keyword
            state = state.advance(6)
        else:
            # Not a lambda expression
            return state.make_error("Expected lambda expression")
        
        # Skip whitespace after lambda
        while not state.is_eof() and state.current_char().isspace():
            state = state.advance(1)
        
        # Check if we have a parenthesized binding (x:Type)
        has_parens = False
        if not state.is_eof() and state.current_char() == '(':
            has_parens = True
            state = state.advance(1)
            
            # Skip whitespace after opening parenthesis
            while not state.is_eof() and state.current_char().isspace():
                state = state.advance(1)
        
        # Parse the variable name (identifier)
        if state.is_eof() or not is_id_first(state.current_char()):
            return state.make_error("Expected identifier in lambda binding")
        
        var_name_start = state.pos
        state = state.advance(1)  # Consume first character
        
        # Consume rest of variable name
        while not state.is_eof() and is_id_rest(state.current_char()):
            state = state.advance(1)
        
        # Create variable name node
        var_name = state.text[var_name_start:state.pos]
        var_name_node = SyntaxNode(kind="ident", value=var_name)
        
        # Skip whitespace after variable name
        while not state.is_eof() and state.current_char().isspace():
            state = state.advance(1)
        
        # Check for the type annotation separator (:)
        if state.is_eof() or state.current_char() != ':':
            return state.make_error("Expected ':' after variable name in lambda expression")
        
        # Consume the colon
        state = state.advance(1)
        
        # Skip whitespace after colon
        while not state.is_eof() and state.current_char().isspace():
            state = state.advance(1)
        
        # Parse the variable type
        if state.is_eof() or not is_id_first(state.current_char()):
            return state.make_error("Expected type identifier in lambda binding")
        
        type_name_start = state.pos
        state = state.advance(1)  # Consume first character
        
        # Consume rest of type name
        while not state.is_eof() and is_id_rest(state.current_char()):
            state = state.advance(1)
        
        # Create type node
        type_name = state.text[type_name_start:state.pos]
        type_node = SyntaxNode(kind="ident_expr", children=[SyntaxNode(kind="ident", value=type_name)])
        
        # If we had opening parenthesis, check for closing parenthesis
        if has_parens:
            # Skip whitespace before closing parenthesis
            while not state.is_eof() and state.current_char().isspace():
                state = state.advance(1)
                
            if state.is_eof() or state.current_char() != ')':
                return state.make_error("Expected ')' after variable type in lambda expression")
            
            # Consume the closing parenthesis
            state = state.advance(1)
        
        # Skip whitespace before comma
        while not state.is_eof() and state.current_char().isspace():
            state = state.advance(1)
        
        # Check for the body separator (,)
        if state.is_eof() or state.current_char() != ',':
            return state.make_error("Expected ',' after variable type in lambda expression")
        
        # Consume the comma
        state = state.advance(1)
        
        # Skip whitespace after comma
        while not state.is_eof() and state.current_char().isspace():
            state = state.advance(1)
        
        # Parse the lambda body
        if state.is_eof():
            return state.make_error("Expected body expression after comma in lambda expression")
        
        # For simple lambda expressions, parse a simple identifier or expression
        if is_id_first(state.current_char()):
            # Simple identifier body
            body_start = state.pos
            state = state.advance(1)  # Consume first character
            
            # Consume rest of identifier
            while not state.is_eof() and is_id_rest(state.current_char()):
                state = state.advance(1)
            
            # Create body node as identifier expression
            body_name = state.text[body_start:state.pos]
            body_node = SyntaxNode(kind="ident_expr", children=[SyntaxNode(kind="ident", value=body_name)])
            
            # Check for possible addition after identifier (e.g., "x + 1")
            if not state.is_eof() and state.current_char() == '+':
                # We found an addition after the identifier
                # Consume the + sign
                state = state.advance(1)
                
                # Skip whitespace after + sign
                while not state.is_eof() and state.current_char().isspace():
                    state = state.advance(1)
                
                # Check for a number or identifier after +
                if state.is_eof():
                    return state.make_error("Expected expression after '+' in lambda body")
                
                if state.current_char().isdigit():
                    # Parse number
                    num_start = state.pos
                    state = state.advance(1)  # Consume first digit
                    
                    # Consume rest of number
                    while not state.is_eof() and state.current_char().isdigit():
                        state = state.advance(1)
                    
                    # Create number node
                    num_str = state.text[num_start:state.pos]
                    num_node = SyntaxNode(kind="num_expr", children=[SyntaxNode(kind="num_lit", value=num_str)])
                    
                    # Create addition node
                    body_node = SyntaxNode(
                        kind=ADD_EXPR,
                        children=[body_node, SyntaxNode(kind="operator", value="+"), num_node]
                    )
                elif is_id_first(state.current_char()):
                    # Parse identifier
                    rhs_start = state.pos
                    state = state.advance(1)  # Consume first character
                    
                    # Consume rest of identifier
                    while not state.is_eof() and is_id_rest(state.current_char()):
                        state = state.advance(1)
                    
                    # Create identifier node
                    rhs_name = state.text[rhs_start:state.pos]
                    rhs_node = SyntaxNode(kind="ident_expr", children=[SyntaxNode(kind="ident", value=rhs_name)])
                    
                    # Create addition node
                    body_node = SyntaxNode(
                        kind=ADD_EXPR,
                        children=[body_node, SyntaxNode(kind="operator", value="+"), rhs_node]
                    )
                else:
                    return state.make_error(f"Unexpected character in lambda body: '{state.current_char()}'")
        else:
            # Try to parse a more complex expression
            body_parser = parse_expr(0)
            body_state = body_parser(context, state)
            
            if body_state.has_error():
                return body_state
            
            if len(body_state.stx_stack) == 0:
                return body_state.make_error("Failed to parse body in lambda expression")
            
            # Get the body expression
            body_node = body_state.stx_stack.pop()
            
            # Update the state
            state = body_state
        
        # Create lambda expression node
        lambda_node = SyntaxNode(
            kind=LAMBDA_EXPR,
            children=[var_name_node, type_node, body_node]
        )
        
        # Push to stack
        state.stx_stack.append(lambda_node)
        
        return state
    
    return Parser(fn)


def pi_expr() -> Parser:
    """Parse a Pi type expression like Π x:Type, Type or forall x:Type, Type."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        # Save starting position
        start_pos = state.pos
        
        # Check for Pi character or keyword
        if state.is_eof():
            return state.make_error("Expected Pi expression")
        
        # Try to match Π character or "forall" keyword
        if state.current_char() == 'Π':
            # Consume the Pi character
            state = state.advance(1)
        elif state.remaining_length() >= 6 and state.text[state.pos:state.pos+6] == 'forall':
            # Consume the forall keyword
            state = state.advance(6)
        else:
            # Not a Pi expression
            return state.make_error("Expected Pi expression")
        
        # Skip whitespace after Pi
        while not state.is_eof() and state.current_char().isspace():
            state = state.advance(1)
        
        # Check if we have a parenthesized binding (x:Type)
        has_parens = False
        if not state.is_eof() and state.current_char() == '(':
            has_parens = True
            state = state.advance(1)
            
            # Skip whitespace after opening parenthesis
            while not state.is_eof() and state.current_char().isspace():
                state = state.advance(1)
        
        # Parse the variable name (identifier)
        if state.is_eof() or not is_id_first(state.current_char()):
            return state.make_error("Expected identifier in Pi binding")
        
        var_name_start = state.pos
        state = state.advance(1)  # Consume first character
        
        # Consume rest of variable name
        while not state.is_eof() and is_id_rest(state.current_char()):
            state = state.advance(1)
        
        # Create variable name node
        var_name = state.text[var_name_start:state.pos]
        var_name_node = SyntaxNode(kind="ident", value=var_name)
        
        # Skip whitespace after variable name
        while not state.is_eof() and state.current_char().isspace():
            state = state.advance(1)
        
        # Check for the type annotation separator (:)
        if state.is_eof() or state.current_char() != ':':
            return state.make_error("Expected ':' after variable name in Pi expression")
        
        # Consume the colon
        state = state.advance(1)
        
        # Skip whitespace after colon
        while not state.is_eof() and state.current_char().isspace():
            state = state.advance(1)
        
        # Parse the variable type
        type_parser = parse_expr(0)  # Use a full expression parser for the type
        type_state = type_parser(context, state)
        
        if type_state.has_error():
            return type_state
        
        if len(type_state.stx_stack) == 0:
            return type_state.make_error("Failed to parse type in Pi expression")
        
        # Get the type expression
        type_node = type_state.stx_stack.pop()
        
        # Update state
        state = type_state
        
        # If we had opening parenthesis, check for closing parenthesis
        if has_parens:
            # Skip whitespace before closing parenthesis
            while not state.is_eof() and state.current_char().isspace():
                state = state.advance(1)
                
            if state.is_eof() or state.current_char() != ')':
                return state.make_error("Expected ')' after variable type in Pi expression")
            
            # Consume the closing parenthesis
            state = state.advance(1)
        
        # Skip whitespace before comma
        while not state.is_eof() and state.current_char().isspace():
            state = state.advance(1)
        
        # Check for the body separator (,)
        if state.is_eof() or state.current_char() != ',':
            return state.make_error("Expected ',' after variable type in Pi expression")
        
        # Consume the comma
        state = state.advance(1)
        
        # Skip whitespace after comma
        while not state.is_eof() and state.current_char().isspace():
            state = state.advance(1)
        
        # Parse the body type (the return type)
        if state.is_eof():
            return state.make_error("Expected body type after comma in Pi expression")
        
        # Check if the rest of the input contains an arrow
        body_text = state.text[state.pos:]
        
        # If the body contains an arrow, create a special scope for parsing it
        if "->" in body_text or "→" in body_text:
            # First, get the variable identifier
            body_start = state.pos  # Save the body start position
            
            # Extract the entire rest of the input as the body
            body_input = state.text[body_start:].strip()
            
            # Create a special context with variable binding
            # We'll parse the expression in a nested parse call
            try:
                # Parse the body using the full parse_expression logic
                body_node = parse_expression(body_input)
                
                # Create Pi type expression node
                pi_node = SyntaxNode(
                    kind=PI_EXPR,
                    children=[var_name_node, type_node, body_node]
                )
                
                # Push to stack
                state.stx_stack.append(pi_node)
                
                # Advance past the entire body
                state.pos = len(state.text)
                
                return state
            except (SyntaxError, ValueError) as e:
                # Continue with normal parsing if special handling fails
                print(f"Special arrow parsing failed: {e}")
                # Reset position
                state.pos = body_start
        
        # Parse a full expression for the body type (regular path)
        body_parser = parse_expr(0)
        body_state = body_parser(context, state)
        
        if body_state.has_error():
            return body_state
        
        if len(body_state.stx_stack) == 0:
            return body_state.make_error("Failed to parse body type in Pi expression")
        
        # Get the body type expression
        body_node = body_state.stx_stack.pop()
        
        # Update the state
        state = body_state
        
        # Create Pi type expression node
        pi_node = SyntaxNode(
            kind=PI_EXPR,
            children=[var_name_node, type_node, body_node]
        )
        
        # Push to stack
        state.stx_stack.append(pi_node)
        
        return state
    
    return Parser(fn)


def arrow_expr() -> Parser:
    """Parse arrow notation for non-dependent function types like A -> B."""
    def fn(context: Any, state: ParserState) -> ParserState:
        # For binary operators in Pratt parsing, context is the left-hand side SyntaxNode
        # And state is the current parsing state after consuming the left-hand side
        
        # Save the current position and stack size (for error recovery)
        start_pos = state.pos
        stack_size = len(state.stx_stack)
        
        # Check for -> or →
        if state.is_eof():
            return state
        
        arrow_found = False
        arrow_len = 0
        
        # Check for Unicode arrow →
        if state.current_char() == '→':
            arrow_found = True
            arrow_len = 1
        # Check for ASCII arrow ->
        elif state.current_char() == '-' and state.remaining_length() >= 2 and state.text[state.pos + 1] == '>':
            arrow_found = True
            arrow_len = 2
        
        if not arrow_found:
            return state
            
        # Check precedence - arrow has lower precedence than arithmetic operators
        arrow_prec = 90  # Lower than comparison operators
        if arrow_prec <= state.lhs_prec:
            return state
        
        # Make sure we have a valid LHS
        if len(state.stx_stack) == 0:
            return state.make_error("Expected expression before arrow")
        
        # Get the LHS from the stack
        lhs = state.stx_stack.pop()
        
        # Consume the arrow token
        state = state.advance(arrow_len)
        
        # Skip whitespace after arrow
        while not state.is_eof() and state.current_char().isspace():
            state = state.advance(1)
        
        # Parse the right-hand side with higher precedence
        rhs_parser = parse_expr(arrow_prec)
        state = rhs_parser(context, state)
        
        if state.has_error():
            return state
        
        if len(state.stx_stack) == 0:
            return state.make_error("Expected expression after arrow")
        
        # Get the RHS from the stack
        rhs = state.stx_stack.pop()
        
        # Create a pi expression for non-dependent function type
        # A -> B is equivalent to Π(_:A), B
        # Use a dummy variable name
        dummy_name = SyntaxNode(kind="ident", value="_")
        
        # Create pi type expression node
        pi_node = SyntaxNode(
            kind=PI_EXPR,
            children=[dummy_name, lhs, rhs]
        )
        
        # Push to stack
        state.stx_stack.append(pi_node)
        
        # Set precedence for chaining
        state.lhs_prec = arrow_prec
        
        return state
    
    return Parser(fn)


# --- Expression Parsing Tables ---

def create_expr_tables() -> PrattParsingTables:
    """Create the Pratt parsing tables for the expression parser."""
    tables = PrattParsingTables()
    
    # Setup prefix parsers (for "leading" expressions)
    tables.prefix[TokenMap.IDENT] = ident_expr()
    tables.prefix[TokenMap.NUMBER] = number_expr()
    tables.prefix["("] = paren_expr()
    tables.prefix["-"] = unary_minus()
    tables.prefix["!"] = node(UNARY_NOT)
    tables.prefix["λ"] = lambda_expr()
    tables.prefix["lambda"] = lambda_expr()
    tables.prefix["Π"] = pi_expr()
    tables.prefix["forall"] = pi_expr()
    
    # Setup infix parsers (for "led" expressions)
    # Multiplication and division have higher precedence than addition and subtraction
    tables.infix["*"] = binary_op("*", MUL_EXPR, 120)
    tables.infix["/"] = binary_op("/", DIV_EXPR, 120)
    tables.infix["+"] = binary_op("+", ADD_EXPR, 110)
    tables.infix["-"] = binary_op("-", SUB_EXPR, 110)
    
    # Comparison operators have lower precedence than arithmetic
    tables.infix["=="] = binary_op("==", EQ_EXPR, 100)
    tables.infix["<"] = binary_op("<", LT_EXPR, 100)
    tables.infix[">"] = binary_op(">", GT_EXPR, 100)
    
    # Function arrow type has even lower precedence
    tables.infix["->"] = arrow_expr()
    tables.infix["→"] = arrow_expr()
    
    # Important: setup function call parser with high precedence
    tables.juxtaposition_handler = app_expr()
    
    return tables


# Expression tables singleton
_expr_tables = None


def get_expr_tables() -> PrattParsingTables:
    """Get the expression parsing tables, creating them if needed."""
    global _expr_tables
    if _expr_tables is None:
        _expr_tables = create_expr_tables()
    return _expr_tables


# Add remaining_length method to ParserState
# This is a monkey patch to add it to the class without modifying the core module
def remaining_length(self) -> int:
    """Returns the number of characters remaining in the input."""
    return len(self.context.input_ctx.input) - self.pos

def advance(self, n: int) -> 'ParserState':
    """Advance the position by n characters."""
    self.pos += n
    return self

# Add the methods to ParserState
ParserState.remaining_length = remaining_length
ParserState.advance = advance

# Define text property to access the input text
ParserState.text = property(lambda self: self.context.input_ctx.input)


# --- Expression Parsing Entry Point ---

def parse_expr(precedence: int = 0) -> Parser:
    """
    Parse an expression with the given minimum precedence.
    
    Args:
        precedence: Minimum precedence level for the expression
    
    Returns:
        A parser that parses an expression
    """
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        # Consume leading whitespace but don't fail if there isn't any
        ws_parser = whitespace()
        ws_state = ws_parser(context, state)
        if not ws_state.has_error():
            state = ws_state
        
        state0 = state.clone()  # Make sure we have a clean copy
        
        # Set the minimum precedence for this expression
        state0.lhs_prec = precedence
        
        if state0.is_eof():
            return state0.make_error("Expected expression, got end of input")
        
        # First, check what kind of token we have
        current_char = state0.current_char()
        input_text = state0.text
        
        # Result state after parsing the primary expression
        state1 = None

        # Special handling for lambda expressions
        if current_char == 'λ' or (state0.remaining_length() >= 6 and input_text[state0.pos:state0.pos+6] == 'lambda'):
            # Lambda expression parsing
            lambda_parser = lambda_expr()
            state1 = lambda_parser(context, state0)
        # Special handling for Pi types
        elif current_char == 'Π' or (state0.remaining_length() >= 6 and input_text[state0.pos:state0.pos+6] == 'forall'):
            # Pi type parsing
            pi_parser = pi_expr()
            state1 = pi_parser(context, state0)
        elif current_char == '(':
            # Parenthesized expression
            paren_parser = paren_expr()
            state1 = paren_parser(context, state0)
        elif current_char == '-':
            # Unary minus
            minus_parser = unary_minus()
            state1 = minus_parser(context, state0)
        elif current_char.isdigit():
            # Number
            num_parser = number_expr()
            state1 = num_parser(context, state0)
        elif is_id_first(current_char):
            # Identifier
            ident_parser = ident_expr()
            state1 = ident_parser(context, state0)
        else:
            # Couldn't recognize a valid expression start
            return state0.make_error(f"Unexpected character: '{current_char}'")
        
        if state1.has_error():
            return state1
        
        # Now try to parse binary operators and function applications
        state2 = state1.clone()
        while not state2.is_eof():
            # Skip whitespace
            while not state2.is_eof() and state2.current_char().isspace():
                state2.pos += 1
            
            if state2.is_eof():
                break
            
            # Check for binary operator
            binary_op_found = False
            
            # Check for + operator
            if state2.current_char() == '+':
                binary_op_found = True
                # Get the left operand from the stack
                if len(state2.stx_stack) == 0:
                    return state2.make_error("Missing left operand for '+'")
                left = state2.stx_stack.pop()
                
                # Consume the + sign
                state2.pos += 1
                
                # Skip whitespace after operator
                while not state2.is_eof() and state2.current_char().isspace():
                    state2.pos += 1
                
                # Parse the right operand
                right_expr = parse_expr(110)(context, state2)
                if right_expr.has_error():
                    return right_expr
                
                # Get right operand
                if len(right_expr.stx_stack) == 0:
                    return right_expr.make_error("Missing right operand for '+'")
                right = right_expr.stx_stack.pop()
                
                # Create binary operation node
                add_node = SyntaxNode(
                    kind=ADD_EXPR,
                    children=[left, SyntaxNode(kind="operator", value="+"), right]
                )
                
                # Push result to stack
                right_expr.stx_stack.append(add_node)
                state2 = right_expr
            
            # Check for - operator (but not unary minus)
            elif state2.current_char() == '-' and len(state2.stx_stack) > 0:
                binary_op_found = True
                # Get the left operand from the stack
                if len(state2.stx_stack) == 0:
                    return state2.make_error("Missing left operand for '-'")
                left = state2.stx_stack.pop()
                
                # Consume the - sign
                state2.pos += 1
                
                # Skip whitespace after operator
                while not state2.is_eof() and state2.current_char().isspace():
                    state2.pos += 1
                
                # Parse the right operand
                right_expr = parse_expr(110)(context, state2)
                if right_expr.has_error():
                    return right_expr
                
                # Get right operand
                if len(right_expr.stx_stack) == 0:
                    return right_expr.make_error("Missing right operand for '-'")
                right = right_expr.stx_stack.pop()
                
                # Create binary operation node
                sub_node = SyntaxNode(
                    kind=SUB_EXPR,
                    children=[left, SyntaxNode(kind="operator", value="-"), right]
                )
                
                # Push result to stack
                right_expr.stx_stack.append(sub_node)
                state2 = right_expr
            
            # Check for * operator
            elif state2.current_char() == '*':
                binary_op_found = True
                # Get the left operand from the stack
                if len(state2.stx_stack) == 0:
                    return state2.make_error("Missing left operand for '*'")
                left = state2.stx_stack.pop()
                
                # Consume the * sign
                state2.pos += 1
                
                # Skip whitespace after operator
                while not state2.is_eof() and state2.current_char().isspace():
                    state2.pos += 1
                
                # Parse the right operand
                right_expr = parse_expr(120)(context, state2)
                if right_expr.has_error():
                    return right_expr
                
                # Get right operand
                if len(right_expr.stx_stack) == 0:
                    return right_expr.make_error("Missing right operand for '*'")
                right = right_expr.stx_stack.pop()
                
                # Create binary operation node
                mul_node = SyntaxNode(
                    kind=MUL_EXPR,
                    children=[left, SyntaxNode(kind="operator", value="*"), right]
                )
                
                # Push result to stack
                right_expr.stx_stack.append(mul_node)
                state2 = right_expr
            
            # Check for / operator
            elif state2.current_char() == '/':
                binary_op_found = True
                # Get the left operand from the stack
                if len(state2.stx_stack) == 0:
                    return state2.make_error("Missing left operand for '/'")
                left = state2.stx_stack.pop()
                
                # Consume the / sign
                state2.pos += 1
                
                # Skip whitespace after operator
                while not state2.is_eof() and state2.current_char().isspace():
                    state2.pos += 1
                
                # Parse the right operand
                right_expr = parse_expr(120)(context, state2)
                if right_expr.has_error():
                    return right_expr
                
                # Get right operand
                if len(right_expr.stx_stack) == 0:
                    return right_expr.make_error("Missing right operand for '/'")
                right = right_expr.stx_stack.pop()
                
                # Create binary operation node
                div_node = SyntaxNode(
                    kind=DIV_EXPR,
                    children=[left, SyntaxNode(kind="operator", value="/"), right]
                )
                
                # Push result to stack
                right_expr.stx_stack.append(div_node)
                state2 = right_expr
            
            # Check for function application
            elif state2.current_char() == '(' and len(state2.stx_stack) > 0:
                binary_op_found = True
                # Get the function from the stack
                if len(state2.stx_stack) == 0:
                    return state2.make_error("Missing function for application")
                func = state2.stx_stack.pop()
                
                # Consume the opening parenthesis
                state2.pos += 1
                
                # Skip whitespace after opening parenthesis
                while not state2.is_eof() and state2.current_char().isspace():
                    state2.pos += 1
                
                # Parse the argument
                arg_expr = parse_expr(0)(context, state2)
                if arg_expr.has_error():
                    return arg_expr
                
                # Get argument
                if len(arg_expr.stx_stack) == 0:
                    return arg_expr.make_error("Missing argument for function application")
                arg = arg_expr.stx_stack.pop()
                
                # Skip whitespace before closing parenthesis
                while not arg_expr.is_eof() and arg_expr.current_char().isspace():
                    arg_expr.pos += 1
                
                # Expect closing parenthesis
                if arg_expr.is_eof() or arg_expr.current_char() != ')':
                    return arg_expr.make_error("Expected ')' after function argument")
                
                # Consume the closing parenthesis
                arg_expr.pos += 1
                
                # Create function application node
                app_node = SyntaxNode(
                    kind=APP_EXPR,
                    children=[func, arg]
                )
                
                # Push result to stack
                arg_expr.stx_stack.append(app_node)
                state2 = arg_expr
            
            # If no binary operation found, we're done
            if not binary_op_found:
                break
            
        return state2
    
    return Parser(fn)


def try_parse_binary_operators(context: ParserContext, state: ParserState, precedence: int) -> Optional[ParserState]:
    """Try to parse a binary operator and right-hand side operand."""
    # Make a copy of the original state in case we need to backtrack
    original_state = state.clone()
    
    # Skip any whitespace before checking for an operator
    ws_parser = whitespace()
    ws_state = ws_parser(context, state)
    if not ws_state.has_error():
        state = ws_state
        
    if state.is_eof():
        return None
    
    # Try to recognize binary operators
    operators = {
        "+": (ADD_EXPR, 110),  # Addition, precedence 110
        "-": (SUB_EXPR, 110),  # Subtraction, precedence 110
        "*": (MUL_EXPR, 120),  # Multiplication, precedence 120
        "/": (DIV_EXPR, 120),  # Division, precedence 120
        "==": (EQ_EXPR, 100),  # Equality, precedence 100
        "<": (LT_EXPR, 100),   # Less than, precedence 100
        ">": (GT_EXPR, 100),   # Greater than, precedence 100
        "->": (ARROW_EXPR, 90), # Function arrow, precedence 90
        "→": (ARROW_EXPR, 90),  # Unicode function arrow, precedence 90
    }
    
    # First, try to match multi-character operators
    for op_str, (kind, op_prec) in operators.items():
        if len(op_str) > 1 and state.remaining_length() >= len(op_str):
            if state.text[state.pos:state.pos+len(op_str)] == op_str:
                # Check precedence
                if op_prec <= precedence:
                    return None
                
                # We found a valid operator with higher precedence
                # Get the LHS from the stack
                if len(state.stx_stack) == 0:
                    return None
                
                lhs = state.stx_stack.pop()
                
                # Consume the operator
                state = state.advance(len(op_str))
                
                # Create an operator token node
                op_token = SyntaxNode(kind="operator", value=op_str)
                
                # Skip whitespace after the operator
                ws_state = ws_parser(context, state)
                if not ws_state.has_error():
                    state = ws_state
                
                # Special case for arrow operator
                if kind == ARROW_EXPR:
                    # Parse the RHS expression with higher precedence
                    rhs_parser = parse_expr(op_prec)
                    state = rhs_parser(context, state)
                    
                    if state.has_error():
                        return state
                    
                    if len(state.stx_stack) == 0:
                        return state.make_error(f"Expected expression after '{op_str}'")
                    
                    rhs = state.stx_stack.pop()
                    
                    # Create a pi expression for non-dependent function type
                    # A -> B is equivalent to Π(_:A), B
                    # Use a dummy variable name
                    dummy_name = SyntaxNode(kind="ident", value="_")
                    
                    # Create pi type expression node
                    pi_node = SyntaxNode(
                        kind=PI_EXPR,
                        children=[dummy_name, lhs, rhs]
                    )
                    
                    # Add to stack
                    state.stx_stack.append(pi_node)
                else:
                    # Parse the RHS expression with higher precedence for left-associative operators
                    rhs_parser = parse_expr(op_prec)
                    state = rhs_parser(context, state)
                    
                    if state.has_error():
                        return state
                    
                    if len(state.stx_stack) == 0:
                        return state.make_error(f"Expected expression after '{op_str}'")
                    
                    rhs = state.stx_stack.pop()
                    
                    # Create binary operation node with the operator token in the middle
                    binary_op_node = SyntaxNode(kind=kind, children=[lhs, op_token, rhs])
                    
                    # Add to stack
                    state.stx_stack.append(binary_op_node)
                
                # Set LHS precedence
                state.lhs_prec = op_prec
                
                return state
    
    # Then, try single-character operators
    current_char = state.current_char()
    for op_str, (kind, op_prec) in operators.items():
        if len(op_str) == 1 and current_char == op_str:
            # Check precedence
            if op_prec <= precedence:
                return None
            
            # We found a valid operator with higher precedence
            # Get the LHS from the stack
            if len(state.stx_stack) == 0:
                return None
            
            lhs = state.stx_stack.pop()
            
            # Consume the operator
            state = state.advance(1)
            
            # Create an operator token node
            op_token = SyntaxNode(kind="operator", value=op_str)
            
            # Skip whitespace after the operator
            ws_state = ws_parser(context, state)
            if not ws_state.has_error():
                state = ws_state
            
            # Special case for arrow operator
            if kind == ARROW_EXPR:
                # Parse the RHS expression with higher precedence
                rhs_parser = parse_expr(op_prec)
                state = rhs_parser(context, state)
                
                if state.has_error():
                    return state
                
                if len(state.stx_stack) == 0:
                    return state.make_error(f"Expected expression after '{op_str}'")
                
                rhs = state.stx_stack.pop()
                
                # Create a pi expression for non-dependent function type
                # A -> B is equivalent to Π(_:A), B
                # Use a dummy variable name
                dummy_name = SyntaxNode(kind="ident", value="_")
                
                # Create pi type expression node
                pi_node = SyntaxNode(
                    kind=PI_EXPR,
                    children=[dummy_name, lhs, rhs]
                )
                
                # Add to stack
                state.stx_stack.append(pi_node)
            else:
                # Parse the RHS expression with higher precedence for left-associative operators
                rhs_parser = parse_expr(op_prec)
                state = rhs_parser(context, state)
                
                if state.has_error():
                    return state
                
                if len(state.stx_stack) == 0:
                    return state.make_error(f"Expected expression after '{op_str}'")
                
                rhs = state.stx_stack.pop()
                
                # Create binary operation node with the operator token in the middle
                binary_op_node = SyntaxNode(kind=kind, children=[lhs, op_token, rhs])
                
                # Add to stack
                state.stx_stack.append(binary_op_node)
            
            # Set LHS precedence
            state.lhs_prec = op_prec
            
            return state
    
    # No binary operator found
    return None


def parse_expression(input_str: str, precedence: int = 0) -> SyntaxNode:
    """
    Parse a Lean expression string into a SyntaxNode.
    
    Args:
        input_str: The input string to parse
        precedence: Minimum precedence level for the expression
    
    Returns:
        The parsed SyntaxNode.
    
    Raises:
        SyntaxError: If parsing fails.
        ValueError: If parsing results in an unexpected state.
    """
    context = ParserContext(InputContext(input_str, "<input>"), prec=precedence)
    state = ParserState(context)
    
    # Check if this is a forall or Pi expression first
    if input_str.startswith("forall ") or input_str.startswith("Π "):
        # Try regular parsing first, which should handle forall directly
        state = parse_expr(precedence)(context, state)
        
        if not state.has_error() and len(state.stx_stack) == 1:
            # Successfully parsed as a forall/Pi expression
            return state.stx_stack[0]
    
    # Next try to handle simple arrow expressions: A -> B (A → B)
    arrow_idx = input_str.find('->')
    unicode_arrow_idx = input_str.find('→')
    
    if arrow_idx >= 0 or unicode_arrow_idx >= 0:
        # Found arrow syntax, try to parse as a function type
        # Determine where the arrow is
        arrow_pos = arrow_idx if arrow_idx >= 0 else float('inf')
        unicode_pos = unicode_arrow_idx if unicode_arrow_idx >= 0 else float('inf')
        pos = min(arrow_pos, unicode_pos)
        
        if pos > 0:  # Make sure we have a left-hand side
            # Parse the left-hand side
            left_input = input_str[:pos].strip()
            try:
                left_node = parse_expression(left_input)
                
                # Parse the right-hand side
                arrow_len = 2 if pos == arrow_pos else 1  # -> is 2 chars, → is 1 char
                right_input = input_str[pos + arrow_len:].strip()
                try:
                    right_node = parse_expression(right_input)
                    
                    # Create Pi type node with dummy var
                    dummy_name = SyntaxNode(kind="ident", value="_")
                    pi_node = SyntaxNode(
                        kind=PI_EXPR,
                        children=[dummy_name, left_node, right_node]
                    )
                    return pi_node
                except (SyntaxError, ValueError):
                    # Continue with normal parsing if right side fails
                    pass
            except (SyntaxError, ValueError):
                # Continue with normal parsing if left side fails
                pass
    
    # Reset state for regular parsing
    state = ParserState(context)
    
    # Run expression parser
    state = parse_expr(precedence)(context, state)
    
    # Check for parsing errors - but handle certain common cases
    if state.has_error():
        # If we have a generic "end of input" error at the final position, that's okay
        # This is a normal condition when the parser reaches the end of input
        if state.error and state.error.message == "end of input" and state.pos >= len(input_str.rstrip()):
            # We reached the end naturally, not an error
            # Clear the error if we have nodes on the stack
            if len(state.stx_stack) > 0:
                state.error = None
            else:
                raise SyntaxError(f"Parsing failed: empty result")
        else:
            raise SyntaxError(f"Parsing failed: {state.error}")
    
    # After handling special cases above, expect exactly one node on the stack
    if len(state.stx_stack) != 1:
        raise ValueError(f"Parsing resulted in {len(state.stx_stack)} nodes, expected 1")

    # Extract the single node from the stack
    result = state.stx_stack[0]
    
    return result


def token_fn() -> Parser:
    """
    Parse a token (number, identifier, or symbol).
    
    Returns:
        A parser that parses a token
    """
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        if state.is_eof():
            # End of input
            return state.make_error("Expected token, got end of input")
        
        char = state.current_char()
        
        # Check for special tokens first
        special_tokens = {
            "forall": "forall",
            "lambda": "lambda",
            "Π": "Π",
            "λ": "λ",
            "→": "→",
            "->": "->",
        }
        
        for token, token_type in special_tokens.items():
            if state.remaining_length() >= len(token) and state.text[state.pos:state.pos+len(token)] == token:
                # Found special token
                next_pos = state.pos + len(token)
                
                # Check if the next character would make this part of a longer identifier
                if next_pos < len(state.text) and is_id_rest(state.text[next_pos]):
                    # Not a special token, just part of an identifier
                    continue
                
                # Consume the token
                new_state = state.advance(len(token))
                
                # Create special token node
                token_node = SyntaxNode(kind="ident", value=token)
                new_state.stx_stack.append(token_node)
                return new_state
        
        # Check for arrow token
        if char == '-' and state.remaining_length() >= 2 and state.text[state.pos+1] == '>':
            # Found arrow token
            new_state = state.advance(2)  # Consume "->"
            token_node = SyntaxNode(kind="arrow", value="->")
            new_state.stx_stack.append(token_node)
            return new_state
        
        # Check for identifier
        if is_id_first(char):
            # Parse identifier
            start_pos = state.pos
            state.pos += 1  # Consume first character
            
            # Consume rest of identifier
            while not state.is_eof() and is_id_rest(state.current_char()):
                state.pos += 1
                
            # Create identifier token node
            ident_str = state.text[start_pos:state.pos]
            ident_node = SyntaxNode(kind="ident", value=ident_str)
            state.stx_stack.append(ident_node)
            return state
        
        # Check for number
        if char.isdigit():
            # Parse number
            start_pos = state.pos
            
            # Check for hex (0x) or binary (0b) prefix
            if char == '0' and state.pos + 1 < len(state.text):
                if state.text[state.pos + 1] in 'xX':  # Hex
                    state.pos += 2  # Skip '0x'
                    # Parse hex digits
                    has_hex_digit = False
                    while not state.is_eof():
                        ch = state.current_char()
                        if ch.isdigit() or 'a' <= ch.lower() <= 'f':
                            has_hex_digit = True
                            state.pos += 1
                        else:
                            break
                    
                    if not has_hex_digit:
                        return state.make_error("Expected hexadecimal digit after '0x'")
                    
                    # Create hex number token node
                    hex_str = state.text[start_pos:state.pos]
                    num_node = SyntaxNode(kind="num_lit", value=hex_str)
                    state.stx_stack.append(num_node)
                    return state
                
                elif state.text[state.pos + 1] in 'bB':  # Binary
                    state.pos += 2  # Skip '0b'
                    # Parse binary digits
                    has_binary_digit = False
                    while not state.is_eof():
                        ch = state.current_char()
                        if ch in '01':
                            has_binary_digit = True
                            state.pos += 1
                        else:
                            break
                    
                    if not has_binary_digit:
                        return state.make_error("Expected binary digit (0 or 1) after '0b'")
                    
                    # Create binary number token node
                    bin_str = state.text[start_pos:state.pos]
                    num_node = SyntaxNode(kind="num_lit", value=bin_str)
                    state.stx_stack.append(num_node)
                    return state
            
            # Regular decimal number or scientific notation
            state.pos += 1  # Consume first digit
            
            # Handle decimal point
            has_decimal = False
            has_scientific = False
            
            # Consume rest of number
            while not state.is_eof():
                char = state.current_char()
                if char.isdigit():
                    state.pos += 1
                elif char == '.' and not has_decimal:
                    has_decimal = True
                    state.pos += 1
                elif (char in 'eE') and not has_scientific:
                    # Scientific notation (e.g., 1e10, 2.5e-3)
                    has_scientific = True
                    state.pos += 1
                    
                    # Check for optional sign after 'e'
                    if not state.is_eof() and state.current_char() in '+-':
                        state.pos += 1
                    
                    # Must have at least one digit after 'e'
                    if state.is_eof() or not state.current_char().isdigit():
                        return state.make_error("Expected digit after 'e' in scientific notation")
                    
                    # Consume the remaining digits
                    while not state.is_eof() and state.current_char().isdigit():
                        state.pos += 1
                    
                    break  # Once we've processed scientific notation, we're done
                else:
                    break
                    
            # Create number token node
            num_str = state.text[start_pos:state.pos]
            num_node = SyntaxNode(kind="num_lit", value=num_str)
            state.stx_stack.append(num_node)
            return state
        
        # Check for symbol
        if not char.isspace():
            # Parse symbol (any non-whitespace character)
            symbol_node = SyntaxNode(kind="symbol", value=char)
            state.pos += 1  # Consume the symbol
            state.stx_stack.append(symbol_node)
            return state
        
        # If we get here, we couldn't parse a token
        return state.make_error(f"Unexpected character: {char}")
    
    return Parser(fn)


def whitespace() -> Parser:
    """Parse whitespace characters (spaces, tabs, newlines)."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        # Skip whitespace
        start_pos = state.pos
        while not state.is_eof() and state.current_char().isspace():
            state = state.advance(1)
            
        # Always succeed, even if we didn't consume any whitespace
        return state
    
    return Parser(fn)
