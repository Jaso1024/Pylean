#!/usr/bin/env python3
"""
Demo of polymorphic sets in Pylean.

This example shows how to define and work with polymorphic sets
in the Pylean kernel, including basic set operations like union,
intersection, and membership testing.
"""

from pylean.kernel import (
    Name, Level, Expr, ExprKind,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, ReductionStrategy, ReductionMode, reduce
)


def main():
    """Run the set operations demo."""
    print("PyLean Polymorphic Sets Demo")
    print("===========================")
    
    # Create a new kernel with standard environment
    kernel = Kernel()
    print("Created kernel with standard environment")
    print()
    
    # Step A: Define the Bool type for predicate returns
    print("Step A: Define the Bool type")
    print("-------------------------")
    
    bool_type = mk_sort(0)  # Type of Bool is Prop
    kernel = kernel.add_constant("Bool", bool_type)
    print("Added Bool constant with type Prop")
    
    kernel = kernel.add_axiom("true", mk_const("Bool"))
    print("Added true constructor for Bool")
    
    kernel = kernel.add_axiom("false", mk_const("Bool"))
    print("Added false constructor for Bool")
    
    # Step B: Define the Set type
    print("\nStep B: Define the Set type")
    print("------------------------")
    
    # Define the Set type constructor: Set : Type -> Type
    # A Set α is essentially a predicate α -> Bool
    set_type = mk_pi("α", mk_sort(0), mk_sort(0))
    kernel = kernel.add_constant("Set", set_type)
    print("Added Set type constructor with type Type -> Type")
    
    # Define the actual Set implementation as a function type
    set_impl_type = mk_pi(
        "α", 
        mk_sort(0), 
        mk_pi(
            "s",
            mk_app(mk_const("Set"), mk_var(0)),
            mk_pi("x", mk_var(1), mk_const("Bool"))
        )
    )
    kernel = kernel.add_constant("set_predicate", set_impl_type)
    print("Added set_predicate function to represent sets as predicates")
    
    # Define empty set: empty : Π (α : Type), Set α
    empty_type = mk_pi("α", mk_sort(0), mk_app(mk_const("Set"), mk_var(0)))
    kernel = kernel.add_constant("empty", empty_type)
    print("Added empty set constructor")
    
    # Define empty set predicate: Π (α : Type), Π (x : α), set_predicate α (empty α) x = false
    empty_pred_type = mk_pi(
        "α",
        mk_sort(0),
        mk_pi(
            "x",
            mk_var(0),
            mk_app(
                mk_app(
                    mk_app(
                        mk_const("set_predicate"),
                        mk_var(1)
                    ),
                    mk_app(
                        mk_const("empty"),
                        mk_var(1)
                    )
                ),
                mk_var(0)
            )
        )
    )
    kernel = kernel.add_definition("empty_pred", mk_const("Bool"), mk_const("false"))
    print("Defined empty set predicate to always return false")
    
    # Define singleton: singleton : Π (α : Type), α -> Set α
    singleton_type = mk_pi(
        "α",
        mk_sort(0),
        mk_pi(
            "a",
            mk_var(0),
            mk_app(mk_const("Set"), mk_var(1))
        )
    )
    kernel = kernel.add_constant("singleton", singleton_type)
    print("Added singleton set constructor")
    
    # Define Eq type if it doesn't exist
    try:
        kernel.infer_type(mk_const("Eq"))
        print("Eq type already exists")
    except Exception:
        # Add Eq type: Π (A : Type), A → A → Prop
        eq_type = mk_pi(
            "A",
            mk_sort(0),
            mk_pi(
                "a",
                mk_var(0),
                mk_pi(
                    "b",
                    mk_var(1),
                    mk_sort(0)  # Prop
                )
            )
        )
        kernel = kernel.add_constant("Eq", eq_type)
        print("Added Eq type")
    
    # Step C: Define set operations
    print("\nStep C: Define set operations")
    print("--------------------------")
    
    # Define membership operation
    # member : Π (α : Type), α -> Set α -> Bool
    member_type = mk_pi(
        "α",
        mk_sort(0),
        mk_pi(
            "x",
            mk_var(0),
            mk_pi(
                "s",
                mk_app(mk_const("Set"), mk_var(1)),
                mk_const("Bool")
            )
        )
    )
    member_value = mk_lambda(
        "α",
        mk_sort(0),
        mk_lambda(
            "x",
            mk_var(0),
            mk_lambda(
                "s",
                mk_app(mk_const("Set"), mk_var(1)),
                mk_app(
                    mk_app(
                        mk_app(
                            mk_const("set_predicate"),
                            mk_var(2)
                        ),
                        mk_var(0)
                    ),
                    mk_var(1)
                )
            )
        )
    )
    kernel = kernel.add_definition("member", member_type, member_value)
    print("Defined member operation for testing set membership")
    
    # Define union operation
    # union : Π (α : Type), Set α -> Set α -> Set α
    union_type = mk_pi(
        "α",
        mk_sort(0),
        mk_pi(
            "s1",
            mk_app(mk_const("Set"), mk_var(0)),
            mk_pi(
                "s2",
                mk_app(mk_const("Set"), mk_var(1)),
                mk_app(mk_const("Set"), mk_var(2))
            )
        )
    )
    kernel = kernel.add_constant("union", union_type)
    print("Added union operation for sets")
    
    # Union predicate: x ∈ (s1 ∪ s2) iff x ∈ s1 or x ∈ s2
    union_pred_type = mk_pi(
        "α",
        mk_sort(0),
        mk_pi(
            "s1",
            mk_app(mk_const("Set"), mk_var(0)),
            mk_pi(
                "s2",
                mk_app(mk_const("Set"), mk_var(1)),
                mk_pi(
                    "x",
                    mk_var(2),
                    mk_app(
                        mk_app(
                            mk_app(
                                mk_const("set_predicate"),
                                mk_var(3)
                            ),
                            mk_app(
                                mk_app(
                                    mk_app(
                                        mk_const("union"),
                                        mk_var(3)
                                    ),
                                    mk_var(2)
                                ),
                                mk_var(1)
                            )
                        ),
                        mk_var(0)
                    )
                )
            )
        )
    )
    
    # Define intersection operation
    # intersection : Π (α : Type), Set α -> Set α -> Set α
    intersection_type = mk_pi(
        "α",
        mk_sort(0),
        mk_pi(
            "s1",
            mk_app(mk_const("Set"), mk_var(0)),
            mk_pi(
                "s2",
                mk_app(mk_const("Set"), mk_var(1)),
                mk_app(mk_const("Set"), mk_var(2))
            )
        )
    )
    kernel = kernel.add_constant("intersection", intersection_type)
    print("Added intersection operation for sets")
    
    # Define complement operation
    # complement : Π (α : Type), Set α -> Set α
    complement_type = mk_pi(
        "α",
        mk_sort(0),
        mk_pi(
            "s",
            mk_app(mk_const("Set"), mk_var(0)),
            mk_app(mk_const("Set"), mk_var(1))
        )
    )
    kernel = kernel.add_constant("complement", complement_type)
    print("Added complement operation for sets")
    
    # Step D: Define some example sets and operations
    print("\nStep D: Define example sets and operations")
    print("---------------------------------------")
    
    # Define Nat type for examples
    nat_type = mk_sort(0)
    kernel = kernel.add_constant("Nat", nat_type)
    print("Added Nat type for examples")
    
    kernel = kernel.add_axiom("zero", mk_const("Nat"))
    print("Added zero constructor for Nat")
    
    succ_type = mk_pi("n", mk_const("Nat"), mk_const("Nat"))
    kernel = kernel.add_constant("succ", succ_type)
    print("Added succ constructor for Nat")
    
    one_body = mk_app(mk_const("succ"), mk_const("zero"))
    kernel = kernel.add_definition("one", mk_const("Nat"), one_body)
    print("Defined one = succ zero")
    
    two_body = mk_app(mk_const("succ"), mk_const("one"))
    kernel = kernel.add_definition("two", mk_const("Nat"), two_body)
    print("Defined two = succ one")
    
    # Define example sets
    # evens = {0, 2, 4, ...}
    evens_type = mk_app(mk_const("Set"), mk_const("Nat"))
    kernel = kernel.add_constant("evens", evens_type)
    print("Added evens set of even natural numbers")
    
    # odds = {1, 3, 5, ...}
    odds_type = mk_app(mk_const("Set"), mk_const("Nat"))
    kernel = kernel.add_constant("odds", odds_type)
    print("Added odds set of odd natural numbers")
    
    # Step E: Test set operations
    print("\nStep E: Test set operations")
    print("-----------------------")
    
    # Test membership
    zero_in_evens = mk_app(
        mk_app(
            mk_app(
                mk_const("member"),
                mk_const("Nat")
            ),
            mk_const("zero")
        ),
        mk_const("evens")
    )
    print(f"zero ∈ evens: {kernel.infer_type(zero_in_evens)}")
    
    one_in_odds = mk_app(
        mk_app(
            mk_app(
                mk_const("member"),
                mk_const("Nat")
            ),
            mk_const("one")
        ),
        mk_const("odds")
    )
    print(f"one ∈ odds: {kernel.infer_type(one_in_odds)}")
    
    # Test union
    nat_set = mk_app(
        mk_app(
            mk_app(
                mk_const("union"),
                mk_const("Nat")
            ),
            mk_const("evens")
        ),
        mk_const("odds")
    )
    print(f"evens ∪ odds = Nat: {kernel.infer_type(nat_set)}")
    
    # Test set axioms (just type checking here, not proving)
    print("\nStep F: Set-theoretic axioms")
    print("-------------------------")
    
    # Define some common set-theoretic axioms
    print("For a set s, s ∪ s = s (idempotence of union)")
    print("For sets s1 and s2, s1 ∪ s2 = s2 ∪ s1 (commutativity of union)")
    print("For sets s1, s2, and s3, (s1 ∪ s2) ∪ s3 = s1 ∪ (s2 ∪ s3) (associativity of union)")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main() 