"""
Elaboration module for Pylean.

This module converts the Abstract Syntax Tree (AST) produced by the parser
(SyntaxNode objects) into the typed expressions used by the kernel
(kernel.expr.Expr objects). It also performs type checking during this process.
"""

from typing import Optional, Dict, Any, Tuple

# Parser imports
from pylean.parser.core import SyntaxNode
import pylean.parser.expr as parser_expr_kinds # To access node kinds like IDENT_EXPR

# Kernel imports
from pylean.kernel.expr import (
    Expr, ExprKind, VarExpr, ConstExpr, AppExpr, LambdaExpr, PiExpr, SortExpr, LetExpr, 
    Level, Name, BinderInfo,
    mk_var, mk_const, mk_app, mk_pi, mk_lambda, mk_let, mk_sort,
    instantiate, lift
)
from pylean.kernel.env import Environment
from pylean.kernel.typecheck import infer_type, check_type, is_type_convertible, Context, TypeError

# --- Elaboration Context ---

class ElaborationContext:
    """Context information needed during elaboration."""
    def __init__(self, env: Environment, local_ctx: Optional[Context] = None):
        self.env = env
        self.local_ctx = local_ctx if local_ctx is not None else Context()
        # Add other necessary context info, e.g., expected type

    def extend(self, name: Name, type: Expr) -> 'ElaborationContext':
        """Extend the local context with a new variable."""
        new_local_ctx = self.local_ctx.extend(name, type)
        return ElaborationContext(self.env, new_local_ctx)
    
    def find_local_var_index(self, name: Name) -> Optional[int]:
        """Find a variable's de Bruijn index by name in the local context."""
        for idx, ctx_name in enumerate(self.local_ctx.names):
            if ctx_name == name:
                return idx
        return None

# --- Elaboration Function ---

def elaborate(node: SyntaxNode, context: ElaborationContext) -> Expr:
    """
    Elaborates a SyntaxNode into a kernel Expr, performing type checking.

    Args:
        node: The SyntaxNode from the parser.
        context: The ElaborationContext containing environment and local bindings.

    Returns:
        The elaborated kernel Expr.

    Raises:
        TypeError: If type checking fails during elaboration or identifier not found.
    """
    kind = node.kind
    children = node.children

    if kind == parser_expr_kinds.IDENT_EXPR:
        if not children or not children[0].value:
            raise ValueError("Invalid IDENT_EXPR structure: Missing identifier value")
        ident_name_str = children[0].value
        ident_name = Name.from_string(ident_name_str)

        # 1. Check Local Context (for Var)
        local_var_idx = context.find_local_var_index(ident_name)
        if local_var_idx is not None:
            return mk_var(local_var_idx) # Use de Bruijn index

        # 2. Check Environment (for Const)
        decl = context.env.find_decl(ident_name)
        if decl is not None:
            # TODO: Handle universe levels properly.
            # For now, just create the constant without explicit levels.
            # We know the required parameters from decl.universe_params
            if decl.universe_params:
                print(f"Info: Constant '{ident_name}' expects universe parameters: {decl.universe_params}")
                # Here we would normally infer/provide the levels
                pass # Fall through to mk_const without levels for now
                
            return mk_const(ident_name) # Pass levels=[] or inferred levels later
        else:
            # 3. Error if not found
            raise TypeError(f"Unknown identifier: '{ident_name_str}'")

    elif kind == parser_expr_kinds.NUM_EXPR:
        # Convert number literal to kernel representation (e.g., Nat constant)
        # Example: Needs Nat type and zero/succ constructors in env
        if not children or not children[0].value:
             raise ValueError("Invalid NUM_EXPR structure: Missing number value")
        num_str = children[0].value
        try:
            num_val = int(num_str)
            if num_val < 0:
                raise TypeError("Negative literals not yet supported")
            # Assume Nat, zero, succ are defined constants
            nat_const = mk_const("Nat")
            zero_const = mk_const("Nat.zero")
            succ_const = mk_const("Nat.succ")
            
            current_expr = zero_const
            for _ in range(num_val):
                current_expr = mk_app(succ_const, current_expr)
            # We should probably typecheck this result, but deferring for now
            return current_expr
            
        except ValueError:
            raise TypeError(f"Invalid number literal: {num_str}")
        except KeyError as e:
             raise TypeError(f"Nat/zero/succ constants not found in environment: {e}")
             
    elif kind == parser_expr_kinds.PAREN_EXPR:
        # Elaborate the inner expression
        if len(children) == 3: # open_paren, expr, close_paren
            return elaborate(children[1], context)
        else:
            raise ValueError("Invalid PAREN_EXPR structure")
            
    elif kind == parser_expr_kinds.APP_EXPR:
        # Example: Simple non-dependent application
        if len(children) < 2:
             raise ValueError("Invalid APP_EXPR structure")
        
        # Elaborate function and first argument
        func_expr = elaborate(children[0], context)
        arg_expr = elaborate(children[1], context) # Assuming first arg is at index 1
        
        # Type check the application
        try:
            func_type = infer_type(context.env, context.local_ctx, func_expr)
            
            # Basic check: function type should be Pi
            if not isinstance(func_type, PiExpr):
                 raise TypeError(f"Function application requires a function type (PiExpr), got {func_type}")
            
            # Check if argument type matches domain type
            arg_type = infer_type(context.env, context.local_ctx, arg_expr)
            if not is_type_convertible(context.env, context.local_ctx, arg_type, func_type.type):
                raise TypeError(f"Argument type mismatch: expected {func_type.type}, got {arg_type}")
            
            # Create application expression
            app_expr = mk_app(func_expr, arg_expr)
            
            # Handle subsequent arguments if present (chaining applications)
            for i in range(2, len(children)):
                next_arg_expr = elaborate(children[i], context)
                
                # Check if the new expression can be applied to an argument
                current_app_type = infer_type(context.env, context.local_ctx, app_expr)
                if not isinstance(current_app_type, PiExpr):
                    raise TypeError(f"Function application requires a function type (PiExpr), got {current_app_type}")
                
                # Check if argument type matches domain type
                next_arg_type = infer_type(context.env, context.local_ctx, next_arg_expr)
                if not is_type_convertible(context.env, context.local_ctx, next_arg_type, current_app_type.type):
                    raise TypeError(f"Argument type mismatch for argument {i}: expected {current_app_type.type}, got {next_arg_type}")
                
                # Create the new application
                app_expr = mk_app(app_expr, next_arg_expr)
            
            return app_expr
            
        except TypeError as e:
            # Provide more informative error message
            raise TypeError(f"Type error in function application: {e}")
            
    elif kind in (
        parser_expr_kinds.ADD_EXPR, parser_expr_kinds.SUB_EXPR, 
        parser_expr_kinds.MUL_EXPR, parser_expr_kinds.DIV_EXPR,
        parser_expr_kinds.EQ_EXPR, parser_expr_kinds.LT_EXPR, 
        parser_expr_kinds.GT_EXPR, parser_expr_kinds.BINARY_OP
    ):
        if len(children) != 3: # lhs, op_token, rhs
            raise ValueError(f"Invalid binary operator structure for {kind}")
            
        lhs_node, op_node, rhs_node = children
        lhs_expr = elaborate(lhs_node, context)
        rhs_expr = elaborate(rhs_node, context)
        
        op_symbol = op_node.value
        if op_symbol is None:
            raise ValueError(f"Binary operator token has no value for {kind}")
            
        # Map operator symbol to kernel constant name
        # This is a simplification; real Lean might use notation tables
        op_map = {
            "+": Name.from_string("Nat.add"),
            "-": Name.from_string("Nat.sub"), # Assuming Nat subtraction
            "*": Name.from_string("Nat.mul"),
            "/": Name.from_string("Nat.div"), # Assuming Nat division
            "==": Name.from_string("Eq"),
            "<": Name.from_string("Lt"),
            ">": Name.from_string("Gt"),
            # Add mappings for other BINARY_OP symbols if the parser uses them
        }
        
        if op_symbol not in op_map:
            raise NotImplementedError(f"Elaboration for binary operator '{op_symbol}' not implemented.")
            
        op_const_name = op_map[op_symbol]
        
        try:
            # Check if the constant exists (basic check)
            if context.env.find_decl(op_const_name) is None:
                raise TypeError(f"Operator constant '{op_const_name}' not found in environment.")
                
            op_const = mk_const(op_const_name)
            
            # Construct the application: op lhs rhs
            app1 = mk_app(op_const, lhs_expr)
            app2 = mk_app(app1, rhs_expr)
            
            # Perform basic type checking (optional but recommended)
            # inferred_type = infer_type(context.env, context.local_ctx, app2)
            # print(f"Inferred type for {op_symbol}: {inferred_type}")
            
            return app2
            
        except TypeError as e:
            raise TypeError(f"Type error during binary operation '{op_symbol}': {e}")
            
    elif kind == parser_expr_kinds.UNARY_MINUS:
        # Example: Assuming Int.neg for negation
        if len(children) != 2: # minus_symbol, expr
             raise ValueError(f"Invalid unary operator structure for {kind}")
             
        op_node, expr_node = children
        inner_expr = elaborate(expr_node, context)
        op_symbol = op_node.value
        
        # Map operator symbol to kernel constant name
        op_const_name = Name.from_string("Int.neg") # Assuming integer negation
        
        try:
            if context.env.find_decl(op_const_name) is None:
                raise TypeError(f"Operator constant '{op_const_name}' not found in environment.")
                
            op_const = mk_const(op_const_name)
            app_expr = mk_app(op_const, inner_expr)
            
            # Optional type checking
            # inferred_type = infer_type(context.env, context.local_ctx, app_expr)
            
            return app_expr
        except TypeError as e:
             raise TypeError(f"Type error during unary operation '{op_symbol}': {e}")

    elif kind == parser_expr_kinds.UNARY_NOT:
        # Example: Assuming Bool.not for logical negation
        if len(children) != 2: # not_symbol, expr
             raise ValueError(f"Invalid unary operator structure for {kind}")
             
        op_node, expr_node = children
        inner_expr = elaborate(expr_node, context)
        op_symbol = op_node.value
        
        # Map operator symbol to kernel constant name
        op_const_name = Name.from_string("Bool.not") # Assuming boolean negation
        
        try:
            if context.env.find_decl(op_const_name) is None:
                raise TypeError(f"Operator constant '{op_const_name}' not found in environment.")
                
            op_const = mk_const(op_const_name)
            app_expr = mk_app(op_const, inner_expr)
            
            # Optional type checking
            # inferred_type = infer_type(context.env, context.local_ctx, app_expr)
            
            return app_expr
        except TypeError as e:
             raise TypeError(f"Type error during unary operation '{op_symbol}': {e}")

    elif kind == parser_expr_kinds.LAMBDA_EXPR:
        # Lambda expression: λ(x:Type), body
        if len(children) != 3:  # var_name, var_type, body
            raise ValueError(f"Invalid lambda expression structure: expected 3 children, got {len(children)}")
        
        var_node, type_node, body_node = children
        
        # Get variable name
        if var_node.kind != "ident" or var_node.value is None:
            raise ValueError(f"Expected identifier for lambda variable, got {var_node.kind}")
        var_name = Name.from_string(var_node.value)
        
        try:
            # Elaborate the variable type
            var_type = elaborate(type_node, context)
            
            # Extend the context with the new variable binding
            extended_context = context.extend(var_name, var_type)
            
            # Elaborate the body in the extended context
            body_expr = elaborate(body_node, extended_context)
            
            # Create the lambda expression
            lambda_expr = mk_lambda(var_name, var_type, body_expr)
            
            return lambda_expr
        except TypeError as e:
            raise TypeError(f"Type error in lambda expression for '{var_name}': {e}")
        except ValueError as e:
            raise ValueError(f"Value error in lambda expression for '{var_name}': {e}")
    
    elif kind == parser_expr_kinds.PI_EXPR:
        # Pi type (dependent function type): Π(x:Type), Type
        if len(children) != 3:  # var_name, var_type, body_type
            raise ValueError(f"Invalid pi type structure: expected 3 children, got {len(children)}")
        
        var_node, type_node, body_type_node = children
        
        # Get variable name
        if var_node.kind != "ident" or var_node.value is None:
            raise ValueError(f"Expected identifier for pi variable, got {var_node.kind}")
        var_name = Name.from_string(var_node.value)
        
        try:
            # Elaborate the variable type
            var_type = elaborate(type_node, context)
            
            # Extend the context with the new variable binding
            extended_context = context.extend(var_name, var_type)
            
            # Elaborate the body type in the extended context
            body_type = elaborate(body_type_node, extended_context)
            
            # Create the pi type expression
            pi_expr = mk_pi(var_name, var_type, body_type)
            
            return pi_expr
        except TypeError as e:
            raise TypeError(f"Type error in Pi type for '{var_name}': {e}")
        except ValueError as e:
            raise ValueError(f"Value error in Pi type for '{var_name}': {e}")

    # ... add cases for other expression kinds ...
    else:
        raise NotImplementedError(f"Elaboration for node kind '{kind}' not implemented yet")

def check_expr(node: SyntaxNode, expected_type: Expr, context: ElaborationContext) -> Expr:
    """
    Elaborate an expression and check that it has the expected type.
    
    Returns the elaborated expression if it type checks.
    """
    expr = elaborate(node, context)
    check_type(context.env, context.local_ctx, expr, expected_type)
    return expr

def infer_expr(node: SyntaxNode, context: ElaborationContext) -> Tuple[Expr, Expr]:
    """
    Elaborate an expression and infer its type.
    
    Returns a tuple of (elaborated expr, inferred type).
    """
    expr = elaborate(node, context)
    type_expr = infer_type(context.env, context.local_ctx, expr)
    return expr, type_expr 