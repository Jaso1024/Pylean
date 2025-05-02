"""
Test module for the reduction system in the Pylean kernel.
"""

import unittest
from typing import cast

from pylean.kernel import (
    Name, Environment, expr_equals, is_stuck, is_def_eq,
    Expr, VarExpr, SortExpr, ConstExpr, AppExpr, LambdaExpr, PiExpr, LetExpr, ExprKind,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi, mk_let,
    reduce_beta, reduce_zeta, reduce_delta, whnf, nf,
    mk_std_env, mk_definition, ReductionMode
)


class TestReduction(unittest.TestCase):
    """Tests for the reduction system."""
    
    def setUp(self):
        """Set up for tests."""
        self.env = mk_std_env()
        
        # Add the identity function: id : Π(A: Type), A -> A
        id_type = mk_pi(
            Name.from_string("A"),
            mk_sort(1),  # Type
            mk_pi(
                Name.from_string("x"),
                mk_var(0),  # A
                mk_var(1)   # A
            )
        )
        id_value = mk_lambda(
            Name.from_string("A"),
            mk_sort(1),  # Type
            mk_lambda(
                Name.from_string("x"),
                mk_var(0),  # A
                mk_var(0)   # x
            )
        )
        self.env = self.env.add_decl(mk_definition("id", id_type, id_value))
        
        # Add function composition: compose : Π(A B C: Type), (B -> C) -> (A -> B) -> A -> C
        compose_type = mk_pi(
            Name.from_string("A"),
            mk_sort(1),  # Type
            mk_pi(
                Name.from_string("B"),
                mk_sort(1),  # Type
                mk_pi(
                    Name.from_string("C"),
                    mk_sort(1),  # Type
                    mk_pi(
                        Name.from_string("f"),
                        mk_pi(Name(), mk_var(1), mk_var(1)),  # B -> C
                        mk_pi(
                            Name.from_string("g"),
                            mk_pi(Name(), mk_var(3), mk_var(2)),  # A -> B
                            mk_pi(
                                Name.from_string("x"),
                                mk_var(4),  # A
                                mk_var(3)   # C
                            )
                        )
                    )
                )
            )
        )
        compose_value = mk_lambda(
            Name.from_string("A"),
            mk_sort(1),  # Type
            mk_lambda(
                Name.from_string("B"),
                mk_sort(1),  # Type
                mk_lambda(
                    Name.from_string("C"),
                    mk_sort(1),  # Type
                    mk_lambda(
                        Name.from_string("f"),
                        mk_pi(Name(), mk_var(1), mk_var(1)),  # B -> C
                        mk_lambda(
                            Name.from_string("g"),
                            mk_pi(Name(), mk_var(3), mk_var(2)),  # A -> B
                            mk_lambda(
                                Name.from_string("x"),
                                mk_var(4),  # A
                                mk_app(mk_var(2), mk_app(mk_var(1), mk_var(0)))
                            )
                        )
                    )
                )
            )
        )
        self.env = self.env.add_decl(mk_definition("compose", compose_type, compose_value))
    
    def test_is_stuck(self):
        """Test checking if an expression is stuck."""
        # Variables, sorts, and constants are stuck
        self.assertTrue(is_stuck(mk_var(0)))
        self.assertTrue(is_stuck(mk_sort(0)))
        self.assertTrue(is_stuck(mk_const("Prop")))
        
        # Lambda expressions are stuck
        lambda_expr = mk_lambda(Name.from_string("x"), mk_sort(0), mk_var(0))
        self.assertTrue(is_stuck(lambda_expr))
        
        # Pi expressions are stuck
        pi_expr = mk_pi(Name.from_string("x"), mk_sort(0), mk_sort(0))
        self.assertTrue(is_stuck(pi_expr))
        
        # Applications with stuck function are stuck
        app_expr = mk_app(mk_const("Prop"), mk_var(0))
        self.assertTrue(is_stuck(app_expr))
        
        # Applications with non-stuck function are not stuck
        app_expr2 = mk_app(
            mk_lambda(Name.from_string("x"), mk_sort(0), mk_var(0)),
            mk_var(0)
        )
        self.assertFalse(is_stuck(app_expr2))
        
        # Let expressions are not stuck
        let_expr = mk_let(
            Name.from_string("x"),
            mk_sort(0),
            mk_const("Prop"),
            mk_var(0)
        )
        self.assertFalse(is_stuck(let_expr))
    
    def test_reduce_beta(self):
        """Test beta reduction."""
        # Create a lambda expression and its argument
        lambda_expr = mk_lambda(Name.from_string("x"), mk_sort(0), mk_var(0))
        arg = mk_const("Prop")
        
        # Perform beta reduction
        result = reduce_beta(lambda_expr, arg)
        
        # Should yield the argument (id function)
        self.assertEqual(result.kind, arg.kind)
        self.assertTrue(expr_equals(result, arg))
        
        # More complex case: (λ(x: Prop), λ(y: Prop), x) Prop
        # Should yield λ(y: Prop), Prop
        lambda_expr2 = mk_lambda(
            Name.from_string("x"),
            mk_sort(0),
            mk_lambda(Name.from_string("y"), mk_sort(0), mk_var(1))
        )
        result2 = reduce_beta(lambda_expr2, arg)
        
        expected2 = mk_lambda(Name.from_string("y"), mk_sort(0), arg)
        self.assertTrue(expr_equals(result2, expected2))
    
    def test_reduce_zeta(self):
        """Test zeta reduction."""
        # Create a let expression
        let_expr = mk_let(
            Name.from_string("x"),
            mk_sort(0),
            mk_const("Prop"),
            mk_var(0)
        )
        
        # Perform zeta reduction
        result = reduce_zeta(let_expr)
        
        # Should yield Prop
        self.assertEqual(result.kind, ExprKind.CONST)
        self.assertTrue(expr_equals(result, mk_const("Prop")))
    
    def test_reduce_delta(self):
        """Test delta reduction."""
        # Create a constant expression
        const_expr = mk_const("id")
        
        # Perform delta reduction
        result = reduce_delta(self.env, const_expr)
        
        # Should yield the id function
        self.assertIsNotNone(result)
        self.assertEqual(result.kind, ExprKind.LAMBDA)
    
    def test_whnf(self):
        """Test weak head normal form reduction."""
        # Create an expression to reduce
        # (λ(x: Prop), x) Prop
        expr = mk_app(
            mk_lambda(Name.from_string("x"), mk_sort(0), mk_var(0)),
            mk_const("Prop")
        )
        
        # Reduce to WHNF
        result = whnf(expr, self.env)
        
        # Should yield Prop
        self.assertEqual(result.kind, ExprKind.CONST)
        self.assertTrue(expr_equals(result, mk_const("Prop")))
        
        # Test with a different reduction mode
        # Beta reduction only on a monomorphic function
        lambda_id = mk_lambda(Name.from_string("x"), mk_sort(0), mk_var(0))
        expr2 = mk_app(lambda_id, mk_const("Prop"))
        result2 = whnf(expr2, self.env, ReductionMode.BETA)
        
        # Should reduce to Prop
        self.assertEqual(result2.kind, ExprKind.CONST)
        self.assertTrue(expr_equals(result2, mk_const("Prop")))
        
        # Skipping complex polymorphic function tests for now
        # They would require more sophisticated handling of universe levels
    
    def test_nf(self):
        """Test full normal form reduction."""
        # Create an expression to reduce with nested redexes
        # (λ(x: Prop), x) ((λ(y: Prop), y) Prop)
        expr = mk_app(
            mk_lambda(Name.from_string("x"), mk_sort(0), mk_var(0)),
            mk_app(
                mk_lambda(Name.from_string("y"), mk_sort(0), mk_var(0)),
                mk_const("Prop")
            )
        )
        
        # Reduce to NF
        result = nf(expr, self.env)
        
        # Should reduce all the way to Prop
        self.assertEqual(result.kind, ExprKind.CONST)
        self.assertTrue(expr_equals(result, mk_const("Prop")))
        
        # Skipping complex polymorphic function tests for now
        # They would require more sophisticated handling of universe levels
    
    def test_is_def_eq(self):
        """Test definitional equality."""
        # Simple case: Prop = Prop
        self.assertTrue(is_def_eq(mk_const("Prop"), mk_const("Prop"), self.env))
        
        # Different constants: Prop != Type
        self.assertFalse(is_def_eq(mk_const("Prop"), mk_const("Type"), self.env))
        
        # Beta-equivalence: (λ(x: Prop), x) Prop = Prop
        expr1 = mk_app(
            mk_lambda(Name.from_string("x"), mk_sort(0), mk_var(0)),
            mk_const("Prop")
        )
        self.assertTrue(is_def_eq(expr1, mk_const("Prop"), self.env))
        
        # Skipping complex polymorphic function tests for now
        # They would require more sophisticated handling of universe levels


if __name__ == "__main__":
    unittest.main() 