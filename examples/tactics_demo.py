#!/usr/bin/env python3
"""
Demo of tactics in Pylean.

This example shows how to use the tactics system to construct
proofs step-by-step rather than directly writing proof terms.
"""

from pylean.kernel import (
    Name, Level, Expr, ExprKind,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, NameAlreadyExistsException, TypeCheckException
)
from pylean.kernel.tactic import (
    Goal, TacticState, Tactic, TacticException,
    IntroTactic, ExactTactic, AssumptionTactic, ApplyTactic,
    RewriteTactic, ByTactic, init_tactic_state
)


def main():
    """Run the tactics demo."""
    print("PyLean Tactics Demo")
    print("=================")
    
    # Create a new kernel with standard environment
    kernel = Kernel()
    print("Created kernel with standard environment")
    print()
    
    # Step 1: Define base types and operations
    print("Step 1: Define base types and operations")
    print("------------------------------------")
    
    # Define Prop type (should already exist)
    try:
        kernel.infer_type(mk_sort(0))
        print("Prop already exists in the environment")
    except Exception:
        print("Adding Prop to the environment")
        # No need to add Prop, it's a built-in sort
    
    # Define implication
    impl_type = mk_pi(
        "a",
        mk_sort(0),
        mk_pi(
            "b",
            mk_sort(0),
            mk_sort(0)
        )
    )
    try:
        kernel.infer_type(mk_const("implies"))
        print("implies already exists")
    except Exception:
        kernel = kernel.add_constant("implies", impl_type)
        print("Added implies constant with type Prop -> Prop -> Prop")
    
    # Define equality if it doesn't exist
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
    try:
        kernel.infer_type(mk_const("Eq"))
        print("Eq already exists")
    except Exception:
        kernel = kernel.add_constant("Eq", eq_type)
        print("Added Eq constant with type Π (A : Type), A -> A -> Prop")
    
    # Define reflexivity axiom for equality
    refl_type = mk_pi(
        "A",
        mk_sort(0),
        mk_pi(
            "a",
            mk_var(0),
            mk_app(
                mk_app(
                    mk_app(
                        mk_const("Eq"),
                        mk_var(1)
                    ),
                    mk_var(0)
                ),
                mk_var(0)
            )
        )
    )
    try:
        kernel.infer_type(mk_const("refl"))
        print("refl already exists")
    except Exception:
        kernel = kernel.add_axiom("refl", refl_type)
        print("Added refl axiom with type Π (A : Type), Π (a : A), Eq A a a")
    
    # Define Nat type for examples
    nat_type = mk_sort(0)
    try:
        kernel.infer_type(mk_const("Nat"))
        print("Nat already exists")
    except Exception:
        kernel = kernel.add_constant("Nat", nat_type)
        print("Added Nat type")
    
    # Step 2: Simple identity proof using tactics
    print("\nStep 2: Simple identity proof using tactics")
    print("----------------------------------------")
    
    # Prove: Π (A : Prop), A -> A
    # This is the identity function for propositions
    identity_type = mk_pi(
        "A",
        mk_sort(0),
        mk_pi(
            "a",
            mk_var(0),
            mk_var(1)
        )
    )
    
    print(f"Goal to prove: {identity_type}")
    
    # Create an initial tactic state
    state = init_tactic_state(kernel.env, identity_type)
    print(f"Initial state:\n{state}")
    
    # Apply intro tactic to introduce A
    intro_A = IntroTactic("A")
    state = intro_A.apply(state)
    print(f"\nAfter intro A:\n{state}")
    
    # Apply intro tactic to introduce a
    intro_a = IntroTactic("a")
    state = intro_a.apply(state)
    print(f"\nAfter intro a:\n{state}")
    
    # Apply assumption tactic to finish the proof
    assumption = AssumptionTactic()
    state = assumption.apply(state)
    print(f"\nAfter assumption:\n{state}")
    
    print("Proof completed!")
    
    # Step 3: More complex proof with rewriting
    print("\nStep 3: More complex proof with rewriting")
    print("---------------------------------------")
    
    # Prove: Π (A : Type), Π (a b : A), Eq A a b -> Eq A b a
    # This is symmetry of equality
    symmetry_type = mk_pi(
        "A",
        mk_sort(0),
        mk_pi(
            "a",
            mk_var(0),
            mk_pi(
                "b",
                mk_var(1),
                mk_pi(
                    "h",
                    mk_app(
                        mk_app(
                            mk_app(
                                mk_const("Eq"),
                                mk_var(2)
                            ),
                            mk_var(1)
                        ),
                        mk_var(0)
                    ),
                    mk_app(
                        mk_app(
                            mk_app(
                                mk_const("Eq"),
                                mk_var(3)
                            ),
                            mk_var(1)
                        ),
                        mk_var(2)
                    )
                )
            )
        )
    )
    
    print(f"Goal to prove: {symmetry_type}")
    
    # Create an initial tactic state
    state = init_tactic_state(kernel.env, symmetry_type)
    print(f"Initial state:\n{state}")
    
    # Apply intro tactics to introduce variables
    intro_A = IntroTactic("A")
    intro_a = IntroTactic("a")
    intro_b = IntroTactic("b")
    intro_h = IntroTactic("h")
    
    # Apply the tactics in sequence using the by tactic
    by_tactic = ByTactic([intro_A, intro_a, intro_b, intro_h])
    state = by_tactic.apply(state)
    print(f"\nAfter introducing variables:\n{state}")
    
    # Rewrite using h
    # This would use the equality h : Eq A a b to rewrite the goal
    # Although this is just a demo and the rewrite may not actually work
    try:
        # Apply rewrite tactic (if we had implemented it fully)
        h_var = mk_var(0)  # Reference to h in the context
        rewrite = RewriteTactic(h_var, direction="<-")
        state = rewrite.apply(state)
        print(f"\nAfter rewrite using h:\n{state}")
        
        # Apply reflexivity to solve the goal
        refl_expr = mk_app(
            mk_app(
                mk_const("refl"),
                mk_const("A")
            ),
            mk_const("b")
        )
        exact = ExactTactic(refl_expr)
        state = exact.apply(state)
        print(f"\nAfter exact refl:\n{state}")
        
        print("Proof completed!")
    except TacticException as e:
        print(f"\nFailed to complete proof: {e}")
        print("This is expected as we're just demonstrating the tactics API")
        print("A complete implementation would require more sophisticated")
        print("rewriting and proof term construction.")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main() 