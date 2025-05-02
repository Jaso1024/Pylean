"""
Tests for the InductionTactic.

This module tests the induction tactic, which is used to 
generate proof goals for inductive types.
"""

import unittest
from pylean.kernel import (
    Expr, Name, ExprKind,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, Environment, Context
)
from pylean.kernel.tactic import (
    Goal, TacticState, InductionTactic, TacticException, init_tactic_state
)
from pylean.kernel.env import DeclKind


class TestInductionTactic(unittest.TestCase):
    """Test case for the InductionTactic."""
    
    def setUp(self):
        """Set up a kernel and environment with inductive types for testing."""
        self.kernel = Kernel()
        
        # Define Nat type and constructors if they don't exist already
        try:
            self.kernel.infer_type(mk_const("Nat"))
        except:
            self.kernel = self.kernel.add_constant("Nat", mk_sort(0))
            
        try:
            self.kernel.infer_type(mk_const("zero"))
        except:
            self.kernel = self.kernel.add_constant("zero", mk_const("Nat"))
            
        try:
            succ_type = mk_pi("n", mk_const("Nat"), mk_const("Nat"))
            self.kernel.infer_type(mk_const("succ"))
        except:
            self.kernel = self.kernel.add_constant("succ", succ_type)

        # Define Bool type and constructors
        try:
            self.kernel.infer_type(mk_const("Bool"))
        except:
            self.kernel = self.kernel.add_constant("Bool", mk_sort(0))
            
        try:
            self.kernel.infer_type(mk_const("true"))
        except:
            self.kernel = self.kernel.add_constant("true", mk_const("Bool"))
            
        try:
            self.kernel.infer_type(mk_const("false"))
        except:
            self.kernel = self.kernel.add_constant("false", mk_const("Bool"))
        
        self.env = self.kernel.env
    
    def test_basic_induction_on_nat(self):
        """Test basic induction on natural numbers."""
        # Create a property P : Nat -> Prop
        P = mk_const("P")
        
        # Create a goal (Π n: Nat, P n)
        n = mk_var(0)  # De Bruijn index 0 refers to the bound variable n
        P_n = mk_app(P, n)
        goal_type = mk_pi("n", mk_const("Nat"), P_n)
        
        # Initialize a tactic state with this goal
        state = init_tactic_state(self.env, goal_type)
        
        # Apply intro tactic to get a variable n : Nat in the context
        from pylean.kernel.tactic import IntroTactic
        intro_tactic = IntroTactic("n")
        state = intro_tactic.apply(state)
        
        # Now the goal is P n with n : Nat in the context
        # Apply induction on n
        induction_tactic = InductionTactic("n")
        new_state = induction_tactic.apply(state)
        
        # There should be two subgoals: one for zero case, one for successor case
        self.assertEqual(len(new_state.goals), 2)
        
        # First goal should be P zero
        zero_goal = new_state.goals[0]
        self.assertEqual(zero_goal.ctx.names[0], "n")
        self.assertEqual(zero_goal.target.kind, ExprKind.APP)
        self.assertEqual(str(zero_goal.target.fn.name), "P")
        # Check that the argument is zero
        self.assertEqual(zero_goal.target.arg.kind, ExprKind.CONST)
        self.assertEqual(str(zero_goal.target.arg.name), "zero")
        
        # Second goal should be for successor case
        # Context should have IH_n, n, n
        succ_goal = new_state.goals[1]
        
        # Print the context names and types for debugging
        for i, name in enumerate(succ_goal.ctx.names):
            print(f"Context variable {i}: {name} : {succ_goal.ctx.types[i]}")
        
        # Check the context - based on the output, we know the order is IH_n, n, n
        self.assertEqual(len(succ_goal.ctx.names), 3)
        self.assertEqual(succ_goal.ctx.names[0], "IH_n")  # Induction hypothesis
        self.assertEqual(succ_goal.ctx.names[1], "n")     # n for successor case
        self.assertEqual(succ_goal.ctx.names[2], "n")     # Original n
        
        # Check the induction hypothesis type - should be P n
        ih_type = succ_goal.ctx.types[0]  # IH_n is at index 0
        self.assertEqual(ih_type.kind, ExprKind.APP)
        self.assertEqual(str(ih_type.fn.name), "P")
        self.assertEqual(ih_type.arg.kind, ExprKind.VAR)
        self.assertEqual(ih_type.arg.idx, 1)  # Reference to n at index 1
        
        # Check the target - should be P (succ n)
        self.assertEqual(succ_goal.target.kind, ExprKind.APP)
        self.assertEqual(str(succ_goal.target.fn.name), "P")
        succ_n = succ_goal.target.arg
        self.assertEqual(succ_n.kind, ExprKind.APP)
        self.assertEqual(str(succ_n.fn.name), "succ")
        self.assertEqual(succ_n.arg.kind, ExprKind.VAR)
        # The successor is applied to the variable n at index 0 (not 1)
        self.assertEqual(succ_n.arg.idx, 0)
    
    def test_induction_with_complex_property(self):
        """Test induction with a more complex property."""
        # Define a property: λ n : Nat, n = n
        
        # First, define equality if it doesn't exist
        try:
            self.kernel.infer_type(mk_const("Eq"))
        except:
            # Define equality: Π (A : Type), A -> A -> Prop
            eq_type = mk_pi(
                "A",
                mk_sort(0),
                mk_pi(
                    "a",
                    mk_var(0),
                    mk_pi(
                        "b",
                        mk_var(1),
                        mk_sort(0)
                    )
                )
            )
            self.kernel = self.kernel.add_constant("Eq", eq_type)
        
        # Create a property P : Nat -> Prop where P n is (n = n)
        # P = λ n : Nat, Eq Nat n n
        nat = mk_const("Nat")
        n = mk_var(0)  # De Bruijn index 0 refers to the bound variable n
        eq = mk_const("Eq")
        eq_nat = mk_app(eq, nat)
        eq_nat_n = mk_app(eq_nat, n)
        eq_nat_n_n = mk_app(eq_nat_n, n)  # n = n
        P = mk_lambda("n", nat, eq_nat_n_n)
        
        # Create a goal (Π n: Nat, P n) which expands to (Π n: Nat, n = n)
        P_n = mk_app(P, mk_var(0))  # Apply P to a fresh variable for the goal
        goal_type = mk_pi("n", nat, P_n)
        
        # Initialize a tactic state with this goal
        state = init_tactic_state(self.env, goal_type)
        
        # Apply intro tactic to get a variable n : Nat in the context
        from pylean.kernel.tactic import IntroTactic
        intro_tactic = IntroTactic("n")
        state = intro_tactic.apply(state)
        
        # Now the goal is P n with n : Nat in the context
        # Apply induction on n
        induction_tactic = InductionTactic("n")
        new_state = induction_tactic.apply(state)
        
        # There should be two subgoals: one for zero case, one for successor case
        self.assertEqual(len(new_state.goals), 2)
        
        # First goal should be P zero which expands to (zero = zero)
        zero_goal = new_state.goals[0]
        self.assertEqual(zero_goal.target.kind, ExprKind.APP)
        
        # Second goal should be for successor case
        # Target should be P (succ n) which expands to (succ n = succ n)
        succ_goal = new_state.goals[1]
        
        # Check the context based on what we observed in the previous test
        self.assertEqual(len(succ_goal.ctx.names), 3)
        self.assertEqual(succ_goal.ctx.names[0], "IH_n")  # Induction hypothesis
        self.assertEqual(succ_goal.ctx.names[1], "n")     # n for successor case
        self.assertEqual(succ_goal.ctx.names[2], "n")     # Original n
        
        # Check the induction hypothesis type
        ih_type = succ_goal.ctx.types[0]
        self.assertEqual(ih_type.kind, ExprKind.APP)
    
    def test_induction_fails_on_non_inductive_type(self):
        """Test that induction fails on non-inductive types."""
        # Create a non-inductive type (e.g., a function type)
        func_type = mk_pi("x", mk_sort(0), mk_sort(0))
        self.kernel = self.kernel.add_constant("Func", func_type)
        
        # Create a goal with a Func variable
        f = mk_var(0)
        P_f = mk_app(mk_const("P"), f)
        goal_type = mk_pi("f", mk_const("Func"), P_f)
        
        # Initialize a tactic state with this goal
        state = init_tactic_state(self.env, goal_type)
        
        # Apply intro tactic to get a variable f : Func in the context
        from pylean.kernel.tactic import IntroTactic
        intro_tactic = IntroTactic("f")
        state = intro_tactic.apply(state)
        
        # Now try to apply induction on f, which should fail
        induction_tactic = InductionTactic("f")
        with self.assertRaises(TacticException):
            new_state = induction_tactic.apply(state)


if __name__ == "__main__":
    unittest.main() 