"""
Standard Library Demo for Pylean.

This example demonstrates how to use the Pylean standard library
for defining and working with proofs and data structures.
"""

from pylean.kernel import (
    Expr, ExprKind, Name, 
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, Context, Goal, TacticState
)
from pylean.kernel.tactic import (
    Tactic, IntroTactic, ExactTactic, AssumptionTactic, 
    ApplyTactic, RewriteTactic, RflTactic
)
from pylean.stdlib import init_stdlib


def demonstrate_logic(kernel: Kernel) -> None:
    """
    Demonstrate the logic module.
    
    Args:
        kernel: The kernel with the standard library loaded
    """
    print("\nLogic Module Demonstration")
    print("=========================")
    
    # Get the environment
    env = kernel.env
    
    # Show some basic definitions
    print("\nBasic logic definitions:")
    print(f"Prop type: {kernel.infer_type(mk_const('Prop'))}")
    print(f"True type: {kernel.infer_type(mk_const('True'))}")
    print(f"False type: {kernel.infer_type(mk_const('False'))}")
    print(f"Not type: {kernel.infer_type(mk_const('Not'))}")
    print(f"And type: {kernel.infer_type(mk_const('And'))}")
    print(f"Or type: {kernel.infer_type(mk_const('Or'))}")
    print(f"Eq type: {kernel.infer_type(mk_const('Eq'))}")
    
    # Prove a simple theorem: True
    print("\nProving: True")
    goal = Goal(ctx=Context(), target=mk_const("True"))
    state = TacticState(env=env, goals=[goal])
    
    # Apply the ExactTactic with True.intro
    exact_tactic = ExactTactic(mk_const("True.intro"))
    final_state = exact_tactic.apply(state)
    
    print(f"Result: {len(final_state.goals) == 0}")  # Should be true (no goals left)
    print(f"Proof: {final_state.proof}")
    
    # Prove: A → A (identity principle)
    print("\nProving: A → A (identity principle)")
    
    # Create the goal: ∀ (A : Prop), A → A
    a_to_a_goal = Goal(
        ctx=Context(),
        target=mk_pi(
            "A", mk_const("Prop"),
            mk_pi("HA", mk_var(0), mk_var(1))
        )
    )
    state = TacticState(env=env, goals=[a_to_a_goal])
    
    # Apply intro tactics to get A : Prop, HA : A in context
    intro_a = IntroTactic("A")
    state = intro_a.apply(state)
    intro_ha = IntroTactic("HA")
    state = intro_ha.apply(state)
    
    # Now we have A : Prop, HA : A ⊢ A as the goal
    # Use the assumption tactic
    assumption = AssumptionTactic()
    final_state = assumption.apply(state)
    
    print(f"Result: {len(final_state.goals) == 0}")  # Should be true (no goals left)
    print(f"Proof: {final_state.proof}")


def demonstrate_nat(kernel: Kernel) -> None:
    """
    Demonstrate the nat module.
    
    Args:
        kernel: The kernel with the standard library loaded
    """
    print("\nNatural Numbers Module Demonstration")
    print("===================================")
    
    # Get the environment
    env = kernel.env
    
    # Show some basic definitions
    print("\nBasic Nat definitions:")
    print(f"Nat type: {kernel.infer_type(mk_const('Nat'))}")
    print(f"zero type: {kernel.infer_type(mk_const('zero'))}")
    print(f"succ type: {kernel.infer_type(mk_const('succ'))}")
    print(f"add type: {kernel.infer_type(mk_const('add'))}")
    print(f"mul type: {kernel.infer_type(mk_const('mul'))}")
    
    # Show numerals
    print("\nNumerals:")
    print(f"one = {kernel.reduce(mk_const('one'))}")
    print(f"two = {kernel.reduce(mk_const('two'))}")
    print(f"three = {kernel.reduce(mk_const('three'))}")
    
    # Demonstrate function application
    print("\nComputing some expressions:")
    
    # double(1)
    double_one = mk_app(mk_const("double"), mk_const("one"))
    double_one_result = kernel.reduce(double_one)
    print(f"double(one) = {double_one_result}")
    
    # 2 + 3
    add_two_three = mk_app(
        mk_app(mk_const("add"), mk_const("two")),
        mk_const("three")
    )
    # Note: We can't evaluate this immediately since add is an axiom without an implementation
    print(f"two + three = {add_two_three}")
    
    # Prove: 0 + n = n
    print("\nProving: add_zero : ∀ n, 0 + n = n")
    
    # Create a goal instance for the theorem (it's already defined as an axiom)
    add_zero_type = kernel.infer_type(mk_const("add_zero"))
    add_zero_goal = Goal(ctx=Context(), target=add_zero_type)
    
    # Show the theorem type
    print(f"Theorem type: {add_zero_type}")


def demonstrate_list(kernel: Kernel) -> None:
    """
    Demonstrate the list module.
    
    Args:
        kernel: The kernel with the standard library loaded
    """
    print("\nList Module Demonstration")
    print("========================")
    
    # Get the environment
    env = kernel.env
    
    # Show some basic definitions
    print("\nBasic List definitions:")
    print(f"List type: {kernel.infer_type(mk_const('List'))}")
    print(f"nil type: {kernel.infer_type(mk_const('nil'))}")
    print(f"cons type: {kernel.infer_type(mk_const('cons'))}")
    print(f"append type: {kernel.infer_type(mk_const('append'))}")
    print(f"map type: {kernel.infer_type(mk_const('map'))}")
    print(f"fold type: {kernel.infer_type(mk_const('fold'))}")
    
    # Create a sample list: [1, 2, 3] of natural numbers
    print("\nCreating a sample list [1, 2, 3]:")
    
    # Define the type parameter (Nat)
    nat_type = mk_const("Nat")
    
    # Create empty list: nil Nat
    empty_list = mk_app(mk_const("nil"), nat_type)
    
    # Create [3]
    list_3 = mk_app(
        mk_app(mk_app(mk_const("cons"), nat_type), mk_const("three")),
        empty_list
    )
    
    # Create [2, 3]
    list_2_3 = mk_app(
        mk_app(mk_app(mk_const("cons"), nat_type), mk_const("two")),
        list_3
    )
    
    # Create [1, 2, 3]
    list_1_2_3 = mk_app(
        mk_app(mk_app(mk_const("cons"), nat_type), mk_const("one")),
        list_2_3
    )
    
    print(f"List [1, 2, 3] = {list_1_2_3}")
    
    # Demonstrate list operations
    print("\nList operations:")
    
    # Prove: append (nil A) l = l
    print("\nProving: append_nil : ∀ A (l : List A), append A (nil A) l = l")
    
    # Create a goal instance for the theorem (it's already defined as an axiom)
    append_nil_type = kernel.infer_type(mk_const("append_nil"))
    append_nil_goal = Goal(ctx=Context(), target=append_nil_type)
    
    # Show the theorem type
    print(f"Theorem type: {append_nil_type}")


def main():
    """Run the standard library demo."""
    print("Pylean Standard Library Demo")
    print("===========================")
    
    # Initialize a kernel with the standard library
    kernel = init_stdlib()
    
    # Print the number of declarations in the standard library
    print(f"\nStandard library loaded with {len(kernel.env.decls)} declarations")
    
    # Demonstrate different modules
    demonstrate_logic(kernel)
    demonstrate_nat(kernel)
    demonstrate_list(kernel)
    
    print("\nStandard Library Demo Completed!")


if __name__ == "__main__":
    main() 