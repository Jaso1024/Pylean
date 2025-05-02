"""
C backend for code generation.

This module implements a backend that generates C code from Pylean expressions.
"""

from typing import Dict, List, Optional, Set, Tuple, Union, Any, cast
import re

from pylean.kernel import (
    Expr, ExprKind, Environment, Declaration, DeclKind,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi, mk_match,
    MatchExpr, Alternative, Pattern, ExternExpr, ExternDecl, ConstExpr, LambdaExpr
)
from pylean.codegen.backend import Backend


class CBackend(Backend):
    """
    Backend for generating C code from Pylean expressions.
    """
    
    def __init__(self, env: Environment):
        """
        Initialize the C backend.
        
        Args:
            env: The environment containing declarations
        """
        super().__init__(env)
        self.temp_var_counter = 0
        self.indent_level = 0
        self.used_declarations: Set[str] = set()
        
        # Optimization data structures
        self.constant_values: Dict[str, str] = {}  # For constant folding
        self.expr_cache: Dict[str, str] = {}  # For common subexpression elimination
        self.function_sizes: Dict[str, int] = {}  # For function inlining decisions
        self.call_counts: Dict[str, int] = {}  # For dead code elimination
        self.optimization_level = 1  # Default optimization level
    
    def _gen_temp_var(self) -> str:
        """Generate a temporary variable name."""
        self.temp_var_counter += 1
        return f"temp_{self.temp_var_counter}"
    
    def _indent(self) -> str:
        """Return the current indentation string."""
        return "  " * self.indent_level
    
    def set_optimization_level(self, level: int) -> None:
        """
        Set the optimization level.
        
        Args:
            level: Optimization level (0-3)
        """
        self.optimization_level = level
    
    def _get_expr_hash(self, expr: Expr) -> str:
        """
        Generate a hash for an expression for caching.
        
        Args:
            expr: The expression to hash
            
        Returns:
            A string hash of the expression
        """
        if expr.kind == ExprKind.VAR:
            return f"var_{expr.idx}"
        elif expr.kind == ExprKind.SORT:
            return f"sort_{expr.level}"
        elif expr.kind == ExprKind.CONST:
            return f"const_{expr.name}"
        elif expr.kind == ExprKind.APP:
            return f"app_{self._get_expr_hash(expr.fn)}_{self._get_expr_hash(expr.arg)}"
        elif expr.kind == ExprKind.LAMBDA:
            return f"lambda_{expr.name}_{self._get_expr_hash(expr.body)}"
        elif expr.kind == ExprKind.PI:
            return f"pi_{expr.name}_{self._get_expr_hash(expr.body)}"
        else:
            return f"expr_{id(expr)}"
    
    def _should_inline(self, fn_name: str) -> bool:
        """
        Determine if a function should be inlined.
        
        Args:
            fn_name: Name of the function
            
        Returns:
            True if the function should be inlined, False otherwise
        """
        # Inline small functions or frequently called ones
        return (fn_name in self.function_sizes and 
                (self.function_sizes[fn_name] < 50 or 
                 self.call_counts.get(fn_name, 0) > 5))
    
    def _fold_constants(self, expr: Expr) -> Optional[str]:
        """
        Attempt to fold constants in an expression.
        
        Args:
            expr: The expression to fold
            
        Returns:
            The folded expression as C code, or None if folding failed
        """
        if self.optimization_level < 1:
            return None
            
        # Check for simple constant expressions
        if expr.kind == ExprKind.CONST:
            const_name = str(expr.name)
            if const_name in self.constant_values:
                return self.constant_values[const_name]
        
        # Check for simple arithmetic expressions
        elif expr.kind == ExprKind.APP and expr.fn.kind == ExprKind.APP:
            op_expr = expr.fn.fn
            if op_expr.kind == ExprKind.CONST:
                op_name = str(op_expr.name)
                
                # Try to evaluate simple arithmetic operations
                if op_name in ["add", "sub", "mul", "div"]:
                    left_arg = expr.fn.arg
                    right_arg = expr.arg
                    
                    # Recursively fold the arguments
                    left_val = self._fold_constants(left_arg)
                    right_val = self._fold_constants(right_arg)
                    
                    # If both arguments are constants, fold the operation
                    if (left_val and right_val and 
                        left_val.isdigit() and right_val.isdigit()):
                        left_num = int(left_val)
                        right_num = int(right_val)
                        
                        if op_name == "add":
                            return str(left_num + right_num)
                        elif op_name == "sub":
                            return str(left_num - right_num)
                        elif op_name == "mul":
                            return str(left_num * right_num)
                        elif op_name == "div" and right_num != 0:
                            return str(left_num // right_num)
        
        return None
    
    def _eliminate_common_subexpr(self, expr: Expr) -> Optional[str]:
        """
        Check if an expression has already been computed.
        
        Args:
            expr: The expression to check
            
        Returns:
            The variable containing the expression value, or None if not found
        """
        if self.optimization_level < 2:
            return None
            
        expr_hash = self._get_expr_hash(expr)
        return self.expr_cache.get(expr_hash)
    
    def compile_expr(self, expr: Expr, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Compile an expression to C code.
        
        Args:
            expr: The expression to compile
            context: Optional context for variable mapping
            
        Returns:
            The compiled C code
        """
        if context is None:
            context = {}
        
        # Try constant folding
        folded = self._fold_constants(expr)
        if folded:
            return folded
        
        # Try common subexpression elimination
        if self.optimization_level >= 2:
            cached = self._eliminate_common_subexpr(expr)
            if cached:
                return cached
        
        # Normal compilation based on expression type
        if expr.kind == ExprKind.VAR:
            # Look up variable in context if available
            var_idx = expr.idx
            if f"var_{var_idx}" in context:
                return context[f"var_{var_idx}"]
            # Otherwise use default naming
            return f"var_{var_idx}"
            
        elif expr.kind == ExprKind.SORT:
            # Sorts don't have a runtime representation in simplest form
            return "lean_box(0)"  # Lean uses box(0) for unit
            
        elif expr.kind == ExprKind.CONST:
            # Constant reference
            const_name = str(expr.name)
            self.used_declarations.add(const_name)
            
            # Track call counts for inlining and dead code elimination
            self.call_counts[const_name] = self.call_counts.get(const_name, 0) + 1
            
            # Check for inlining if optimization is enabled
            if self.optimization_level >= 3 and self._should_inline(const_name):
                # Get the declaration
                decl = self.env.find_decl(const_name)
                if decl and hasattr(decl, 'value') and decl.value is not None:
                    # Inline the function body
                    return f"/* inlined {const_name} */ " + self.compile_expr(decl.value, context)
            
            return f"lean_{const_name}()"
            
        elif expr.kind == ExprKind.APP:
            # Function application: f(x)
            fn_expr = expr.fn
            arg_expr = expr.arg
            
            # Handle multi-argument applications
            if fn_expr.kind == ExprKind.APP:
                # Collect all arguments
                args = [arg_expr]
                curr = fn_expr
                while curr.kind == ExprKind.APP:
                    args.insert(0, curr.arg)
                    curr = curr.fn
                
                if curr.kind == ExprKind.CONST:
                    const_name = str(curr.name)
                    self.used_declarations.add(const_name)
                    
                    # Track call counts
                    self.call_counts[const_name] = self.call_counts.get(const_name, 0) + 1
                    
                    # Check for inlining
                    if self.optimization_level >= 3 and self._should_inline(const_name):
                        decl = self.env.find_decl(const_name)
                        if decl and hasattr(decl, 'value') and decl.value is not None:
                            # Create a new context for the inline expansion
                            inline_context = context.copy()
                            
                            # Map arguments to variables in the function body
                            lambda_expr = decl.value
                            for i, arg in enumerate(args):
                                param_name = f"var_{i}"
                                arg_code = self.compile_expr(arg, context)
                                inline_context[param_name] = arg_code
                            
                            # Inline the function body
                            return f"/* inlined {const_name} */ " + self.compile_expr(lambda_expr.body, inline_context)
                    
                    # Generate normal function call
                    args_code = [self.compile_expr(arg, context) for arg in args]
                    return f"lean_{const_name}({', '.join(args_code)})"
            
            # Normal single argument function call
            fn_code = self.compile_expr(fn_expr, context)
            arg_code = self.compile_expr(arg_expr, context)
            
            # Handle function pointers vs. direct calls
            if fn_expr.kind == ExprKind.CONST:
                return f"{fn_code}({arg_code})"
            else:
                return f"(({fn_code})({arg_code}))"
            
        elif expr.kind == ExprKind.LAMBDA:
            # Lambda expressions become C function declarations
            var_name = f"var_{0}"  # The bound variable
            
            # Create a new context for the lambda body
            lambda_context = context.copy()
            lambda_context[var_name] = var_name
            
            body_code = self.compile_expr(expr.body, lambda_context)
            lambda_name = self._gen_temp_var()
            lambda_code = f"lean_object* {lambda_name}(lean_object* {var_name}) {{\n"
            lambda_code += f"{self._indent()}  return {body_code};\n"
            lambda_code += f"{self._indent()}}}"
            
            # Store function size for inlining decisions
            self.function_sizes[lambda_name] = len(body_code)
            
            return lambda_name
            
        elif expr.kind == ExprKind.PI:
            # Pi types become function types in C
            # In a simple implementation, we just treat all Pi types as function pointers
            return "lean_object* (*)(lean_object*)"
            
        elif expr.kind == ExprKind.MATCH:
            # Match expressions (pattern matching)
            match_expr = cast(MatchExpr, expr)
            
            # Generate scrutinee
            scrutinee_code = self.compile_expr(match_expr.scrutinee, context)
            
            # Generate a switch statement
            match_var = self._gen_temp_var()
            result_var = self._gen_temp_var()
            
            match_code = f"// Pattern matching\n{self._indent()}"
            match_code += f"lean_object* {match_var} = {scrutinee_code};\n{self._indent()}"
            match_code += f"lean_object* {result_var};\n{self._indent()}"
            match_code += f"switch (lean_get_tag({match_var})) {{\n"
            
            # Generate a case for each alternative
            for i, alt in enumerate(match_expr.alternatives):
                # Get the constructor tag
                constructor_info = self.env.get_constructor_info(alt.pattern.constructor)
                if constructor_info is None:
                    tag = hash(str(alt.pattern.constructor)) % 100000
                else:
                    tag = constructor_info['tag']
                
                match_code += f"{self._indent()}  case {tag}: {{\n"
                
                # Extract pattern variables
                alt_context = context.copy()
                for field_idx, field_name in enumerate(alt.pattern.fields):
                    field_var = self._gen_temp_var()
                    match_code += f"{self._indent()}    lean_object* {field_var} = "
                    match_code += f"lean_ctor_get({match_var}, {field_idx});\n"
                    alt_context[field_name] = field_var
                
                # Compile the alternative expression
                alt_code = self.compile_expr(alt.expr, alt_context)
                match_code += f"{self._indent()}    {result_var} = {alt_code};\n"
                match_code += f"{self._indent()}    break;\n"
                match_code += f"{self._indent()}  }}\n"
            
            # Default case
            match_code += f"{self._indent()}  default:\n"
            match_code += f"{self._indent()}    // Error: non-exhaustive pattern match\n"
            match_code += f"{self._indent()}    lean_panic_unreachable();\n"
            match_code += f"{self._indent()}}}\n"
            
            # Store result in the expression cache for CSE
            if self.optimization_level >= 2:
                expr_hash = self._get_expr_hash(expr)
                self.expr_cache[expr_hash] = result_var
            
            return result_var
            
        elif expr.kind == ExprKind.EXTERN:
            # External function reference
            extern_expr = cast(ExternExpr, expr)
            return extern_expr.c_name
            
        else:
            # Other expression types (let, meta, etc.)
            return f"/* Unsupported expression kind: {expr.kind} */ lean_box(0)"
    
    def compile_decl(self, decl: Declaration) -> str:
        """
        Compile a declaration to C code.
        
        Args:
            decl: The declaration to compile
            
        Returns:
            The compiled C code
        """
        if str(decl.name) in self.emitted_decls:
            return ""  # Already emitted
            
        self.emitted_decls.add(str(decl.name))
        name = str(decl.name)
        
        # Skip declarations that are never used (dead code elimination)
        if self.optimization_level >= 2 and name not in self.call_counts:
            return f"/* Eliminated unused declaration: {name} */"
            
        if decl.kind == DeclKind.DEF:
            # Generate code for a definition
            if hasattr(decl, 'value') and decl.value is not None:
                value_code = self.compile_expr(decl.value)
                
                # Store constant values for folding
                if self.optimization_level >= 1 and isinstance(value_code, str) and value_code.isdigit():
                    self.constant_values[name] = value_code
                
                # Store function size for inlining decisions
                self.function_sizes[name] = len(value_code)
                
                return f"lean_object* lean_{name}() {{\n{self._indent()}  return {value_code};\n{self._indent()}}}"
            else:
                return f"lean_object* lean_{name}() {{\n{self._indent()}  return lean_box(0);\n{self._indent()}}}"
            
        elif decl.kind == DeclKind.CONSTANT:
            # Generate code for a constant
            return f"extern lean_object* lean_{name}(void);"
            
        elif decl.kind == DeclKind.EXTERN:
            # External function declaration
            extern_decl = cast(ExternDecl, decl)
            param_types = ', '.join(['lean_object*'] * len(extern_decl.param_types))
            return f"extern lean_object* {extern_decl.c_name}({param_types});"
            
        elif decl.kind == DeclKind.INDUCTIVE:
            # Generate header for inductive type
            return f"// Inductive type: {name}"
            
        else:
            return f"/* Unsupported declaration kind: {decl.kind} for {decl.name} */"
    
    def generate_program(self, decls: List[Declaration], main_fn: Optional[str] = None) -> str:
        """
        Generate a complete C program from a list of declarations.
        
        Args:
            decls: The declarations to include in the program
            main_fn: Optional name of the main function
            
        Returns:
            The complete C program code
        """
        # Reset counters and caches
        self.expr_cache = {}
        self.constant_values = {}
        self.function_sizes = {}
        
        # Pre-analyze for optimization
        if self.optimization_level > 0:
            self._analyze_declarations(decls)
        
        # Generate headers
        program = "/**\n * Generated C code from Pylean\n */\n\n"
        program += "#include <lean/lean.h>\n\n"
        
        # Forward declarations for all functions
        for decl in decls:
            if self.should_generate_code(decl) and decl.kind == DeclKind.DEF:
                name = str(decl.name)
                program += f"lean_object* lean_{name}(void);\n"
        
        program += "\n"
        
        # Generate declarations
        for decl in decls:
            if self.should_generate_code(decl):
                code = self.compile_decl(decl)
                if code:
                    program += code + "\n\n"
        
        # Generate main function if requested
        if main_fn:
            program += "int main(int argc, char** argv) {\n"
            program += f"  lean_object* result = lean_{main_fn}();\n"
            program += "  /* Process result */\n"
            program += "  printf(\"Result: %p\\n\", result);\n"
            program += "  return 0;\n"
            program += "}\n"
        
        return program
    
    def _analyze_declarations(self, decls: List[Declaration]) -> None:
        """
        Analyze declarations for optimization.
        
        Args:
            decls: The declarations to analyze
        """
        # First pass: collect function sizes and constant values
        for decl in decls:
            if decl.kind == DeclKind.DEF and hasattr(decl, 'value') and decl.value is not None:
                name = str(decl.name)
                
                # Compute rough size of the function body
                size = self._estimate_expr_size(decl.value)
                self.function_sizes[name] = size
                
                # Check for constant definitions
                if decl.value.kind == ExprKind.CONST:
                    const_name = str(decl.value.name)
                    if const_name.isdigit():
                        self.constant_values[name] = const_name
        
        # Second pass: analyze function calls and dependencies
        for decl in decls:
            if hasattr(decl, 'value') and decl.value is not None:
                self._analyze_expr_calls(decl.value)
    
    def _estimate_expr_size(self, expr: Expr) -> int:
        """
        Estimate the 'size' of an expression for inlining decisions.
        
        Args:
            expr: The expression to analyze
            
        Returns:
            A size estimate
        """
        if expr.kind in [ExprKind.VAR, ExprKind.SORT, ExprKind.CONST]:
            return 1
        elif expr.kind == ExprKind.APP:
            return self._estimate_expr_size(expr.fn) + self._estimate_expr_size(expr.arg) + 1
        elif expr.kind in [ExprKind.LAMBDA, ExprKind.PI]:
            return self._estimate_expr_size(expr.body) + 2
        elif expr.kind == ExprKind.MATCH:
            match_expr = cast(MatchExpr, expr)
            size = self._estimate_expr_size(match_expr.scrutinee) + 2
            for alt in match_expr.alternatives:
                size += self._estimate_expr_size(alt.expr) + len(alt.pattern.fields) + 2
            return size
        else:
            return 5  # Default size for other expression types
    
    def _analyze_expr_calls(self, expr: Expr) -> None:
        """
        Analyze an expression to track function calls.
        
        Args:
            expr: The expression to analyze
        """
        if expr.kind == ExprKind.CONST:
            const_name = str(expr.name)
            self.call_counts[const_name] = self.call_counts.get(const_name, 0) + 1
        elif expr.kind == ExprKind.APP:
            self._analyze_expr_calls(expr.fn)
            self._analyze_expr_calls(expr.arg)
        elif expr.kind in [ExprKind.LAMBDA, ExprKind.PI]:
            self._analyze_expr_calls(expr.body)
        elif expr.kind == ExprKind.MATCH:
            match_expr = cast(MatchExpr, expr)
            self._analyze_expr_calls(match_expr.scrutinee)
            for alt in match_expr.alternatives:
                self._analyze_expr_calls(alt.expr) 