"""
Test module for the type checking system in the Pylean kernel.
"""

import unittest
from pylean.kernel import (
    Name, Level, Environment, Context, TypeError,
    Expr, VarExpr, SortExpr, ConstExpr, AppExpr, LambdaExpr, PiExpr, LetExpr, ExprKind,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi, mk_let,
    infer_type, check_type, ensure_type_is_valid, is_type_convertible,
    mk_std_env
)


class TestTypeChecking(unittest.TestCase):
    """Tests for the type checking system."""
    
    def setUp(self):
        """Set up for tests."""
        self.env = mk_std_env()
        self.ctx = Context()
    
    def test_infer_var_type(self):
        """Test inferring the type of a variable."""
        # Create a context with a variable
        var_type = mk_sort(0)  # Prop
        ctx = self.ctx.extend(Name.from_string("x"), var_type)
        
        # Infer the type of the variable
        var = mk_var(0)  # de Bruijn index 0
        type_expr = infer_type(self.env, ctx, var)
        
        self.assertEqual(type_expr.kind, var_type.kind)
        self.assertEqual(str(type_expr), str(var_type))
    
    def test_infer_var_type_error(self):
        """Test inferring the type of a non-existent variable."""
        var = mk_var(0)  # No variables in context
        with self.assertRaises(TypeError):
            infer_type(self.env, self.ctx, var)
    
    def test_infer_sort_type(self):
        """Test inferring the type of a sort."""
        # Sort 0 (Prop) has type Sort 1 (Type 0)
        sort0 = mk_sort(0)
        type_expr = infer_type(self.env, self.ctx, sort0)
        
        self.assertEqual(type_expr.kind, ExprKind.SORT)
        self.assertEqual(str(type_expr), "Type 1")
        
        # Sort 1 (Type 0) has type Sort 2 (Type 1)
        sort1 = mk_sort(1)
        type_expr = infer_type(self.env, self.ctx, sort1)
        
        self.assertEqual(type_expr.kind, ExprKind.SORT)
        self.assertEqual(str(type_expr), "Type 2")
    
    def test_infer_const_type(self):
        """Test inferring the type of a constant."""
        # Prop has type Type
        const = mk_const("Prop")
        type_expr = infer_type(self.env, self.ctx, const)
        
        self.assertEqual(type_expr.kind, ExprKind.SORT)
        self.assertEqual(str(type_expr), "Prop")  # Prop has type Prop in standard env
    
    def test_infer_const_type_error(self):
        """Test inferring the type of a non-existent constant."""
        const = mk_const("NonExistent")
        with self.assertRaises(TypeError):
            infer_type(self.env, self.ctx, const)
    
    def test_infer_app_type(self):
        """Test inferring the type of an application."""
        # Create a function type: Prop -> Prop
        fn_name = Name.from_string("f")
        dom_type = mk_sort(0)  # Prop
        cod_type = mk_sort(0)  # Prop
        fn_type = mk_pi(Name(), dom_type, cod_type)  # Prop -> Prop
        
        # Create a context with the function
        ctx = self.ctx.extend(fn_name, fn_type)
        
        # Create an application: f Prop
        fn = mk_var(0)  # de Bruijn index 0 (f)
        arg = mk_const("Prop")
        app = mk_app(fn, arg)
        
        # Infer the type of the application
        type_expr = infer_type(self.env, ctx, app)
        
        self.assertEqual(type_expr.kind, ExprKind.SORT)
        self.assertEqual(str(type_expr), "Prop")
    
    def test_infer_app_type_error(self):
        """Test inferring the type of an application with a non-function."""
        # Create a context with a non-function
        var_type = mk_sort(0)  # Prop
        ctx = self.ctx.extend(Name.from_string("x"), var_type)
        
        # Create an application with a non-function: x Prop
        fn = mk_var(0)  # de Bruijn index 0 (x: Prop)
        arg = mk_const("Prop")
        app = mk_app(fn, arg)
        
        # Should raise a type error
        with self.assertRaises(TypeError):
            infer_type(self.env, ctx, app)
    
    def test_infer_lambda_type(self):
        """Test inferring the type of a lambda expression."""
        # Create a lambda expression: λ(x: Prop), x
        var_name = Name.from_string("x")
        var_type = mk_sort(0)  # Prop
        body = mk_var(0)  # de Bruijn index 0 (x)
        lambda_expr = mk_lambda(var_name, var_type, body)
        
        # Infer the type of the lambda
        type_expr = infer_type(self.env, self.ctx, lambda_expr)
        
        self.assertEqual(type_expr.kind, ExprKind.PI)
        # Should be Prop -> Prop
        self.assertEqual(str(type_expr), "Prop → Prop")
    
    def test_infer_pi_type(self):
        """Test inferring the type of a pi expression."""
        # Create a pi expression: Π(x: Prop), Prop
        var_name = Name.from_string("x")
        var_type = mk_sort(0)  # Prop
        body = mk_sort(0)  # Prop
        pi_expr = mk_pi(var_name, var_type, body)
        
        # Infer the type of the pi
        type_expr = infer_type(self.env, self.ctx, pi_expr)
        
        self.assertEqual(type_expr.kind, ExprKind.SORT)
        # Should be Type 1 (max level + 1 in the current implementation)
        self.assertEqual(str(type_expr), "Type 1")
    
    def test_infer_let_type(self):
        """Test inferring the type of a let expression."""
        # Create a let expression: let x: Prop := Prop; x
        var_name = Name.from_string("x")
        var_type = mk_sort(0)  # Prop
        var_value = mk_const("Prop")
        body = mk_var(0)  # de Bruijn index 0 (x)
        let_expr = mk_let(var_name, var_type, var_value, body)
        
        # Infer the type of the let
        type_expr = infer_type(self.env, self.ctx, let_expr)
        
        self.assertEqual(type_expr.kind, ExprKind.SORT)
        self.assertEqual(str(type_expr), "Prop")
    
    def test_type_convertible(self):
        """Test checking if two types are convertible."""
        # Simple structural equality
        type1 = mk_sort(0)  # Prop
        type2 = mk_sort(0)  # Prop
        self.assertTrue(is_type_convertible(self.env, self.ctx, type1, type2))
        
        # Different sorts
        type1 = mk_sort(0)  # Prop
        type2 = mk_sort(1)  # Type 0
        self.assertFalse(is_type_convertible(self.env, self.ctx, type1, type2))
        
        # Function types
        type1 = mk_pi(Name(), mk_sort(0), mk_sort(0))  # Prop -> Prop
        type2 = mk_pi(Name(), mk_sort(0), mk_sort(0))  # Prop -> Prop
        self.assertTrue(is_type_convertible(self.env, self.ctx, type1, type2))
        
        # Different function types
        type1 = mk_pi(Name(), mk_sort(0), mk_sort(0))  # Prop -> Prop
        type2 = mk_pi(Name(), mk_sort(0), mk_sort(1))  # Prop -> Type 0
        self.assertFalse(is_type_convertible(self.env, self.ctx, type1, type2))
    
    def test_ensure_type_is_valid(self):
        """Test ensuring that a type expression is valid."""
        # Prop is a valid type
        type_expr = mk_sort(0)  # Prop
        ensure_type_is_valid(self.env, self.ctx, type_expr)  # Should not raise
        
        # Type is a valid type
        type_expr = mk_sort(1)  # Type 0
        ensure_type_is_valid(self.env, self.ctx, type_expr)  # Should not raise
        
        # Function type is a valid type
        type_expr = mk_pi(Name(), mk_sort(0), mk_sort(0))  # Prop -> Prop
        ensure_type_is_valid(self.env, self.ctx, type_expr)  # Should not raise
        
        # A value with non-sort type is not a valid type
        nat_zero = mk_const("Nat.zero")  # Nat.zero is a value, not a type
        ctx = self.ctx.extend(Name.from_string("x"), nat_zero)
        type_expr = mk_var(0)  # x:Nat.zero is not a type
        with self.assertRaises(TypeError):
            ensure_type_is_valid(self.env, ctx, type_expr)


if __name__ == "__main__":
    unittest.main() 