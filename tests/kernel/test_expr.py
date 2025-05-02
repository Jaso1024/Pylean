"""
Tests for the expression module in the kernel.

This module contains tests for the Lean4 expression data structures
and utility functions for manipulating expressions.
"""

import unittest
from pylean.kernel.expr import (
    ExprKind, Name, Level, BinderInfo,
    Expr, VarExpr, SortExpr, ConstExpr, AppExpr, LambdaExpr, PiExpr, LetExpr, MetaExpr, LocalExpr,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi, mk_let, mk_meta, mk_local,
    occurs_in, lift, instantiate
)


class TestName(unittest.TestCase):
    """Tests for the Name class."""
    
    def test_create_empty_name(self):
        """Test creating an empty/anonymous name."""
        name = Name()
        self.assertEqual(str(name), "")
        self.assertTrue(name.is_anonymous())
    
    def test_create_simple_name(self):
        """Test creating a simple, single-part name."""
        name = Name(("x",))
        self.assertEqual(str(name), "x")
        self.assertFalse(name.is_anonymous())
    
    def test_create_hierarchical_name(self):
        """Test creating a hierarchical, multi-part name."""
        name = Name(("lean", "kernel", "Expr"))
        self.assertEqual(str(name), "lean.kernel.Expr")
        self.assertFalse(name.is_anonymous())
    
    def test_from_string(self):
        """Test creating a name from a string with dot notation."""
        name = Name.from_string("lean.kernel.Expr")
        self.assertEqual(name.parts, ("lean", "kernel", "Expr"))
        self.assertEqual(str(name), "lean.kernel.Expr")
    
    def test_append(self):
        """Test appending a part to a name."""
        name1 = Name.from_string("lean.kernel")
        name2 = name1.append("Expr")
        self.assertEqual(str(name1), "lean.kernel")
        self.assertEqual(str(name2), "lean.kernel.Expr")


class TestExpressions(unittest.TestCase):
    """Tests for expression classes and utility functions."""
    
    def test_var_expr(self):
        """Test creating and manipulating variable expressions."""
        var = mk_var(1)
        self.assertEqual(var.kind, ExprKind.VAR)
        self.assertEqual(var.idx, 1)
        self.assertEqual(str(var), "#1")
    
    def test_sort_expr(self):
        """Test creating and manipulating sort expressions."""
        # Create Prop
        prop = mk_sort(0)
        self.assertEqual(prop.kind, ExprKind.SORT)
        self.assertEqual(str(prop), "Prop")
        
        # Create Type
        type1 = mk_sort(1)
        self.assertEqual(type1.kind, ExprKind.SORT)
        self.assertEqual(str(type1), "Type 1")
        
        # Create Type with named universe
        type_u = mk_sort("u")
        self.assertEqual(type_u.kind, ExprKind.SORT)
        self.assertEqual(str(type_u), "Type u")
    
    def test_const_expr(self):
        """Test creating and manipulating constant expressions."""
        # Create constant without universe levels
        nat = mk_const("Nat")
        self.assertEqual(nat.kind, ExprKind.CONST)
        self.assertEqual(str(nat), "Nat")
        
        # Create constant with universe levels
        list_u = mk_const("List", [Level(Name.from_string("u"))])
        self.assertEqual(list_u.kind, ExprKind.CONST)
        self.assertEqual(str(list_u), "List.{u}")
    
    def test_app_expr(self):
        """Test creating and manipulating application expressions."""
        # Create simple application: f x
        f = mk_const("f")
        x = mk_var(0)
        app = mk_app(f, x)
        self.assertEqual(app.kind, ExprKind.APP)
        self.assertEqual(str(app), "f #0")
        
        # Create nested application: g (f x)
        g = mk_const("g")
        app2 = mk_app(g, app)
        self.assertEqual(app2.kind, ExprKind.APP)
        self.assertEqual(str(app2), "g (f #0)")
    
    def test_lambda_expr(self):
        """Test creating and manipulating lambda expressions."""
        # Create simple lambda: λx, x
        x_name = "x"
        x_type = mk_sort(0)  # Prop
        x_body = mk_var(0)
        lambda_x = mk_lambda(x_name, x_type, x_body)
        self.assertEqual(lambda_x.kind, ExprKind.LAMBDA)
        self.assertEqual(str(lambda_x), "λ(x : Prop), #0")
        
        # Create lambda with implicit parameter: λ{x}, x
        binder_info = BinderInfo(is_implicit=True)
        lambda_x_implicit = mk_lambda(x_name, x_type, x_body, binder_info)
        self.assertEqual(str(lambda_x_implicit), "λ{x : Prop}, #0")
    
    def test_pi_expr(self):
        """Test creating and manipulating pi expressions."""
        # Create simple function type: Prop → Prop
        x_name = Name()  # Anonymous name
        x_type = mk_sort(0)  # Prop
        x_body = mk_sort(0)  # Prop
        arr_type = mk_pi(x_name, x_type, x_body)
        self.assertEqual(arr_type.kind, ExprKind.PI)
        self.assertEqual(str(arr_type), "Prop → Prop")
        
        # Create dependent function type: Π(x : Prop), Prop
        x_name = "x"
        x_type = mk_sort(0)  # Prop
        x_body = mk_app(mk_const("P"), mk_var(0))  # P x
        pi_type = mk_pi(x_name, x_type, x_body)
        self.assertEqual(pi_type.kind, ExprKind.PI)
        self.assertEqual(str(pi_type), "Π(x : Prop), P #0")
    
    def test_let_expr(self):
        """Test creating and manipulating let expressions."""
        # Create simple let: let x : Prop := true; x
        x_name = "x"
        x_type = mk_sort(0)  # Prop
        x_value = mk_const("true")
        x_body = mk_var(0)
        let_expr = mk_let(x_name, x_type, x_value, x_body)
        self.assertEqual(let_expr.kind, ExprKind.LET)
        self.assertEqual(str(let_expr), "let x : Prop := true; #0")
    
    def test_meta_expr(self):
        """Test creating and manipulating metavariable expressions."""
        # Create simple metavariable: ?m
        m_name = "m"
        meta = mk_meta(m_name)
        self.assertEqual(meta.kind, ExprKind.META)
        self.assertEqual(str(meta), "?m")
    
    def test_local_expr(self):
        """Test creating and manipulating local expressions."""
        # Create simple local: x
        x_name = "x"
        x_type = mk_sort(0)  # Prop
        local = mk_local(x_name, x_type)
        self.assertEqual(local.kind, ExprKind.LOCAL)
        self.assertEqual(str(local), "x")
    
    def test_occurs_in(self):
        """Test the occurs_in function."""
        # Test variable
        var = mk_var(0)
        self.assertTrue(occurs_in(0, var))
        self.assertFalse(occurs_in(1, var))
        
        # Test application
        app = mk_app(mk_var(1), mk_var(0))
        self.assertTrue(occurs_in(0, app))
        self.assertTrue(occurs_in(1, app))
        self.assertFalse(occurs_in(2, app))
        
        # Test lambda
        lambda_expr = mk_lambda("x", mk_sort(0), mk_var(0))
        self.assertTrue(occurs_in(0, lambda_expr.body))
        self.assertFalse(occurs_in(1, lambda_expr.body))
    
    def test_lift(self):
        """Test the lift function."""
        # Test lifting a variable
        var = mk_var(0)
        lifted_var = lift(var, 1, 0)
        self.assertEqual(str(lifted_var), "#1")
        
        # Test not lifting a variable below the threshold
        var2 = mk_var(0)
        not_lifted_var = lift(var2, 1, 1)
        self.assertEqual(str(not_lifted_var), "#0")
        
        # Test lifting in an application
        app = mk_app(mk_var(0), mk_var(1))
        lifted_app = lift(app, 2, 0)
        self.assertEqual(str(lifted_app), "#2 #3")
    
    def test_instantiate(self):
        """Test the instantiate function."""
        # Test substituting a variable
        var = mk_var(0)
        subst = mk_const("a")
        result = instantiate(var, subst)
        self.assertEqual(str(result), "a")
        
        # Test substituting in a lambda body
        body = mk_app(mk_var(1), mk_var(0))  # f x
        lambda_expr = mk_lambda("x", mk_sort(0), body)
        subst = mk_const("a")
        result = instantiate(lambda_expr.body, subst, 0)
        self.assertEqual(str(result), "#0 a")


if __name__ == "__main__":
    unittest.main() 