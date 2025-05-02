"""
Tests for the replace_expr function in tactic.py.

This module tests the behavior of the replace_expr function
that is used to substitute expressions in tactics like induction.
"""

import unittest
from pylean.kernel import (
    Expr, Name, ExprKind, 
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi, mk_let,
    Kernel, Context, Environment
)
from pylean.kernel.tactic import replace_expr
from pylean.kernel.reduce import is_def_eq


class TestReplaceExpr(unittest.TestCase):
    """Test case for the replace_expr function."""
    
    def setUp(self):
        """Set up a kernel and environment for testing."""
        self.kernel = Kernel()
        self.env = self.kernel.env
        
    def test_replace_var(self):
        """Test replacing a variable with another expression."""
        # Create expressions
        x = mk_var(0)  # De Bruijn index 0
        y = mk_var(1)  # De Bruijn index 1
        
        # Replace x with y
        result = replace_expr(x, x, y)
        
        # Check that x was replaced with y
        self.assertEqual(result.kind, ExprKind.VAR)
        self.assertEqual(result.idx, 1)
    
    def test_replace_const(self):
        """Test replacing a constant with another expression."""
        # Create expressions
        nat = mk_const("Nat")
        bool_expr = mk_const("Bool")
        
        # Replace Nat with Bool
        result = replace_expr(nat, nat, bool_expr)
        
        # Check that Nat was replaced with Bool
        self.assertEqual(result.kind, ExprKind.CONST)
        self.assertEqual(str(result.name), "Bool")
    
    def test_replace_in_app(self):
        """Test replacing in a function application."""
        # Create expressions
        f = mk_const("f")
        x = mk_const("x")
        y = mk_const("y")
        
        # Create f(x)
        f_x = mk_app(f, x)
        
        # Replace x with y in f(x) to get f(y)
        result = replace_expr(f_x, x, y)
        
        # Check that result is f(y)
        self.assertEqual(result.kind, ExprKind.APP)
        self.assertEqual(str(result.fn.name), "f")
        self.assertEqual(str(result.arg.name), "y")
    
    def test_replace_in_lambda(self):
        """Test replacing in a lambda expression."""
        # Create expressions
        nat = mk_const("Nat")
        bool_expr = mk_const("Bool")
        x = mk_var(0)  # Reference to the bound variable
        
        # Create lambda x: Nat, x
        lambda_expr = mk_lambda("x", nat, x)
        
        # Replace Nat with Bool to get lambda x: Bool, x
        result = replace_expr(lambda_expr, nat, bool_expr)
        
        # Check the result
        self.assertEqual(result.kind, ExprKind.LAMBDA)
        self.assertEqual(str(result.name), "x")
        self.assertEqual(str(result.type.name), "Bool")
        self.assertEqual(result.body.kind, ExprKind.VAR)
        self.assertEqual(result.body.idx, 0)
    
    def test_replace_in_pi(self):
        """Test replacing in a Pi expression."""
        # Create expressions
        nat = mk_const("Nat")
        bool_expr = mk_const("Bool")
        x = mk_var(0)  # Reference to the bound variable
        
        # Create Pi x: Nat, Nat
        pi_expr = mk_pi("x", nat, nat)
        
        # Replace all Nat with Bool to get Pi x: Bool, Bool
        result = replace_expr(pi_expr, nat, bool_expr)
        
        # Check the result
        self.assertEqual(result.kind, ExprKind.PI)
        self.assertEqual(str(result.name), "x")
        self.assertEqual(str(result.type.name), "Bool")
        self.assertEqual(str(result.body.name), "Bool")
    
    def test_dont_replace_bound_var(self):
        """Test that we don't replace a bound variable inside its binder."""
        # Create expressions
        nat = mk_const("Nat")
        x = mk_var(0)  # Reference to the bound variable
        y = mk_const("y")  # Some other expression
        
        # Create lambda x: Nat, x
        lambda_expr = mk_lambda("x", nat, x)
        
        # Try to replace x with y 
        # This should not affect the lambda body since x is bound
        result = replace_expr(lambda_expr, x, y)
        
        # Check that the lambda body is still x
        self.assertEqual(result.kind, ExprKind.LAMBDA)
        self.assertEqual(result.body.kind, ExprKind.VAR)
        self.assertEqual(result.body.idx, 0)
    
    def test_replace_with_def_eq(self):
        """Test replacement using definitional equality."""
        # Define two definitionally equal but not syntactically equal expressions
        # First, add some definitions to the environment
        nat = mk_const("Nat")
        
        # Create a definition for identity function
        self.kernel = self.kernel.add_constant("id", 
            mk_pi("A", mk_sort(0), 
                mk_pi("x", mk_var(0), mk_var(1))
            )
        )
        
        self.kernel = self.kernel.add_definition("id_nat", 
            mk_pi("x", nat, nat),
            mk_lambda("x", nat, mk_var(0))
        )
        
        # Create expressions
        id_nat_x = mk_app(mk_const("id_nat"), mk_const("n"))
        id_x = mk_const("n")  # Simplified
        
        # Create a test expression using id_nat_x
        test_expr = mk_app(mk_const("f"), id_nat_x)
        
        # Try to replace id_nat_x with id_x
        result = replace_expr(test_expr, id_nat_x, id_x, self.kernel.env)
        
        # This test might fail if the definitional equality checking is not working
        # but we expect the replacement to work based on structural equality
        self.assertEqual(result.kind, ExprKind.APP)
        self.assertEqual(str(result.fn.name), "f")
        self.assertEqual(result.arg.kind, ExprKind.CONST)
        self.assertEqual(str(result.arg.name), "n")
    
    def test_induction_hypothesis_case(self):
        """Test the specific case from induction tactic where the issue was observed."""
        # This is a simplified version of what happens in InductionTactic.apply
        
        # Create a simple induction target expression
        # P(n) where P is some predicate and n is the induction variable
        P = mk_const("P")
        n = mk_const("n")
        P_n = mk_app(P, n)
        
        # Create an expression for the induction argument
        # For example, in the inductive step we'd replace n with k
        k = mk_const("k")
        
        # Replace n with k in P(n) to get P(k)
        result = replace_expr(P_n, n, k, self.kernel.env)
        
        # Check that we get P(k)
        self.assertEqual(result.kind, ExprKind.APP)
        self.assertEqual(str(result.fn.name), "P")
        self.assertEqual(str(result.arg.name), "k")


if __name__ == "__main__":
    unittest.main() 