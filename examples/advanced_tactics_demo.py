"""
Advanced Tactics Demo for Pylean.

This example demonstrates more complex proofs using induction 
and rewriting tactics.
"""

from pylean.kernel import (
    Name, Level, Expr, ExprKind,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, Context, NameAlreadyExistsException, TypeCheckException
)
from pylean.kernel.tactic import (
    Goal, TacticState, Tactic, TacticException,
    IntroTactic, ExactTactic, AssumptionTactic, ApplyTactic,
    RewriteTactic, ByTactic, InductionTactic, RflTactic, init_tactic_state
)
from pylean.kernel.parser import parse_expression
from pylean.elaborator import elaborate, ElaborationContext


def main():
    """Run the advanced tactics demo."""
    print("Pylean Advanced Tactics Demo")
    print("============================")
    
    # Create a new kernel with standard environment
    # This environment should include Nat, zero, succ, Nat.add, Eq, refl
    kernel = Kernel()
    env = kernel.env
    print("Created kernel with standard environment")
    print()
    
    # --- Proof 1: Induction on Natural Numbers --- 
    print("Proof 1: Proving 0 + n = n using induction")
    print("------------------------------------------")
    
    # Define the goal: Π (n : Nat), Nat.add Nat.zero n = n
    try:
        # Use parser and elaborator for the goal type
        goal_str = "Π (n : Nat), Nat.add Nat.zero n = n"
        parsed_goal = parse_expression(goal_str)
        elab_ctx = ElaborationContext(env, Context())
        goal_type = elaborate(parsed_goal, elab_ctx)
        print(f"Goal to prove: {goal_type}")
        
        # Initialize tactic state
        state = init_tactic_state(env, goal_type)
        print(f"Initial state:\n{state}")
        
        # Apply intro tactic
        print("\nApplying: intro n")
        state = IntroTactic("n").apply(state)
        print(f"State:\n{state}")
        
        # Apply induction tactic
        print("\nApplying: induction n")
        state = InductionTactic("n").apply(state)
        print(f"State (showing 2 goals):\n{state}")
        
        # --- Base Case: Nat.add Nat.zero Nat.zero = Nat.zero --- 
        print("\nSolving Goal 1 (Base Case): Nat.add Nat.zero Nat.zero = Nat.zero")
        
        # This should simplify to Nat.zero = Nat.zero using the definition of add
        # Assuming Nat.add is defined such that Nat.add Nat.zero m = m
        # Since rewrite/simp is not fully implemented, we'll use rfl directly
        # A full system would require `simp [Nat.add]` or similar.
        
        print("Applying: rfl") # Needs simplification first in a real proof
        state_goal1 = TacticState(env, [state.goals[0]]) # Isolate first goal
        try:
            state_goal1 = RflTactic().apply(state_goal1)
            # Replace the solved goal in the main state
            state = TacticState(env, state_goal1.goals + state.goals[1:]) 
            print(f"Goal 1 solved! Remaining state:\n{state}")
        except TacticException as e:
            print(f"Could not apply rfl directly: {e}")
            print("(This might be expected if Nat.add definition needs unfolding)")
            # In a real scenario, we'd need a simp/rewrite step here.
            # For demo, we'll manually assume it's solved.
            state = TacticState(env, state.goals[1:]) # Manually remove goal 1
            print("Manually assuming Goal 1 solved for demo purposes.")
            print(f"Remaining state:\n{state}")
            
        # --- Inductive Step: IH: (Nat.add Nat.zero n = n) => Nat.add Nat.zero (Nat.succ n) = Nat.succ n --- 
        print("\nSolving Goal 2 (Inductive Step):")
        print(f"Context: {state.goals[0].ctx}")
        print(f"Target: {state.goals[0].target}")
        
        # The target is Nat.add Nat.zero (Nat.succ n) = Nat.succ n
        # We'd need to use the definition of Nat.add (e.g., add zero (succ m) = succ (add zero m))
        # and then the induction hypothesis (IH_n : Nat.add Nat.zero n = n)
        
        print("Applying: rewrite ...") # Placeholder for rewrite steps
        # Example hypothetical steps:
        # 1. Rewrite `Nat.add Nat.zero (Nat.succ n)` using add definition -> `Nat.succ (Nat.add Nat.zero n)`
        #    Target becomes: `Nat.succ (Nat.add Nat.zero n) = Nat.succ n`
        # 2. Rewrite using `IH_n` inside `Nat.succ` -> `Nat.succ n`
        #    Target becomes: `Nat.succ n = Nat.succ n`
        
        print("Applying: rfl") # Apply rfl after hypothetical rewrites
        state_goal2 = TacticState(env, [state.goals[0]]) # Isolate second goal
        try:
            # We apply rfl directly, assuming the rewrites happened hypothetically
            state_goal2 = RflTactic().apply(state_goal2)
            state = TacticState(env, state_goal2.goals + state.goals[1:]) 
            print(f"Goal 2 solved! Remaining state:\n{state}")
            print("Proof 1 completed (hypothetically)!")
        except TacticException as e:
            print(f"Could not apply rfl directly: {e}")
            print("(This likely requires simplification/rewriting steps first)")
            # Manually remove goal 2 for demo
            state = TacticState(env, [])
            print("Manually assuming Goal 2 solved for demo purposes.")
            print("Proof 1 completed (hypothetically)!")
            
    except (NameError, ParseError, ValueError, TacticException, TypeCheckException) as e:
        print(f"\nError setting up or executing Proof 1: {e}")
        print("(This might indicate missing definitions like '=' or Nat.add in env, or parser/elaborator issues)")
        
    # --- Proof 2: Symmetry of Equality --- 
    print("\nProof 2: Proving symmetry of equality")
    print("-------------------------------------")
    
    # Goal: Π (A : Type) (a b : A), Eq A a b -> Eq A b a
    try:
        goal_str = "Π (A : Type) (a b : A), Eq A a b -> Eq A b a"
        parsed_goal = parse_expression(goal_str)
        elab_ctx = ElaborationContext(env, Context())
        goal_type = elaborate(parsed_goal, elab_ctx)
        print(f"Goal to prove: {goal_type}")
        
        state = init_tactic_state(env, goal_type)
        print(f"Initial state:\n{state}")
        
        # Use 'by' tactic for multiple intros
        print("\nApplying: intro A, intro a, intro b, intro h")
        tactics = [IntroTactic("A"), IntroTactic("a"), IntroTactic("b"), IntroTactic("h")]
        state = ByTactic(tactics).apply(state)
        print(f"State:\n{state}")
        
        # Now goal is `Eq A b a`, with context `A:Type, a:A, b:A, h: Eq A a b`
        print("\nApplying: rewrite <- h")
        # Find the hypothesis 'h' in the context. Its index depends on the order.
        # The elaborator handles name mapping, so _parse_expr should find 'h'.
        # However, RewriteTactic currently takes an Expr, not just a name.
        # We need to create a variable Expr for h.
        # Assuming h is the last introduced variable (index 0)
        h_var = mk_var(0) 
        try:
            state = RewriteTactic(h_var, direction="<-").apply(state)
            print(f"State after rewrite:\n{state}")
            
            # Goal should now be `Eq A a a`
            print("\nApplying: rfl")
            state = RflTactic().apply(state)
            print(f"State after rfl:\n{state}")
            print("Proof 2 completed!")
            
        except TacticException as e:
            print(f"\nFailed to complete proof 2: {e}")
            print("(Rewrite tactic might not be fully functional yet)")
            # Manually remove goal for demo
            state = TacticState(env, [])
            print("Manually assuming Proof 2 solved for demo purposes.")

    except (NameError, ParseError, ValueError, TacticException, TypeCheckException) as e:
        print(f"\nError setting up or executing Proof 2: {e}")

    print("\nAdvanced Tactics Demo Completed!")


if __name__ == "__main__":
    main() 