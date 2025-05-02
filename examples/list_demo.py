#!/usr/bin/env python3
"""
Demo of polymorphic lists in Pylean.

This example shows how to define and work with polymorphic lists
in the Pylean kernel.
"""

from pylean.kernel import (
    Name, Level, Expr, ExprKind,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, ReductionStrategy, ReductionMode, reduce
)


def main():
    """Run the list demo."""
    print("PyLean Polymorphic Lists Demo")
    print("============================")
    
    # Create a new kernel with standard environment
    kernel = Kernel()
    print("Created kernel with standard environment")
    print()
    
    # Step A: Define the list type
    print("Step A: Define the list type")
    print("-------------------------")
    
    # Define the list type constructor: List : Type -> Type
    list_type = mk_pi("α", mk_sort(0), mk_sort(0))
    kernel = kernel.add_constant("List", list_type)
    print("Added List type constructor with type Type -> Type")
    
    # Define nil constructor: nil : Π (α : Type), List α
    nil_type = mk_pi("α", mk_sort(0), mk_app(mk_const("List"), mk_var(0)))
    kernel = kernel.add_constant("nil", nil_type)
    print("Added nil constructor for List")
    
    # Define cons constructor: cons : Π (α : Type), α -> List α -> List α
    cons_type = mk_pi(
        "α",
        mk_sort(0),
        mk_pi(
            "head",
            mk_var(0),
            mk_pi(
                "tail",
                mk_app(mk_const("List"), mk_var(1)),
                mk_app(mk_const("List"), mk_var(2))
            )
        )
    )
    kernel = kernel.add_constant("cons", cons_type)
    print("Added cons constructor for List")
    
    # Step B: Define some basic types to work with
    print("\nStep B: Define some basic types to work with")
    print("----------------------------------------")
    
    # Define Nat type
    nat_type = mk_sort(0)
    kernel = kernel.add_constant("Nat", nat_type)
    print("Added Nat type")
    
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
    
    three_body = mk_app(mk_const("succ"), mk_const("two"))
    kernel = kernel.add_definition("three", mk_const("Nat"), three_body)
    print("Defined three = succ two")
    
    # Step C: Create and manipulate lists
    print("\nStep C: Create and manipulate lists")
    print("--------------------------------")
    
    # Create an empty list of Nat
    empty_nat_list = mk_app(mk_const("nil"), mk_const("Nat"))
    print("Created empty list of Nat")
    print(f"Type: {kernel.infer_type(empty_nat_list)}")
    
    # Create a list with one element: [1]
    list_one = mk_app(
        mk_app(
            mk_app(
                mk_const("cons"),
                mk_const("Nat")
            ),
            mk_const("one")
        ),
        empty_nat_list
    )
    print("Created list [1]")
    print(f"Type: {kernel.infer_type(list_one)}")
    
    # Create a list with multiple elements: [1, 2, 3]
    list_one_two_three = mk_app(
        mk_app(
            mk_app(
                mk_const("cons"),
                mk_const("Nat")
            ),
            mk_const("one")
        ),
        mk_app(
            mk_app(
                mk_app(
                    mk_const("cons"),
                    mk_const("Nat")
                ),
                mk_const("two")
            ),
            mk_app(
                mk_app(
                    mk_app(
                        mk_const("cons"),
                        mk_const("Nat")
                    ),
                    mk_const("three")
                ),
                empty_nat_list
            )
        )
    )
    print("Created list [1, 2, 3]")
    print(f"Type: {kernel.infer_type(list_one_two_three)}")
    
    # Step D: Define list operations
    print("\nStep D: Define list operations")
    print("---------------------------")
    
    # Define length operation
    # length : Π (α : Type), List α -> Nat
    length_type = mk_pi(
        "α",
        mk_sort(0),
        mk_pi(
            "xs",
            mk_app(mk_const("List"), mk_var(0)),
            mk_const("Nat")
        )
    )
    kernel = kernel.add_constant("length", length_type)
    print("Added length operation")
    
    # Define axioms for length
    # length_nil: Π (α : Type), length α (nil α) = zero
    length_nil_type = mk_pi(
        "α",
        mk_sort(0),
        mk_app(
            mk_app(
                mk_app(
                    mk_const("Eq"),
                    mk_const("Nat")
                ),
                mk_app(
                    mk_app(
                        mk_const("length"),
                        mk_var(0)
                    ),
                    mk_app(
                        mk_const("nil"),
                        mk_var(0)
                    )
                )
            ),
            mk_const("zero")
        )
    )
    
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
    
    try:
        kernel = kernel.add_axiom("length_nil", length_nil_type)
        print("Added length_nil axiom")
    except Exception as e:
        print(f"Error adding length_nil axiom: {e}")
    
    # Try to evaluate the length of our lists
    print("\nEvaluating list operations:")
    
    # Length of empty list
    empty_list_length = mk_app(
        mk_app(
            mk_const("length"),
            mk_const("Nat")
        ),
        empty_nat_list
    )
    print(f"Length of empty list: {kernel.infer_type(empty_list_length)}")
    
    # Length of list [1]
    list_one_length = mk_app(
        mk_app(
            mk_const("length"),
            mk_const("Nat")
        ),
        list_one
    )
    print(f"Length of list [1]: {kernel.infer_type(list_one_length)}")
    
    # Length of list [1, 2, 3]
    list_three_length = mk_app(
        mk_app(
            mk_const("length"),
            mk_const("Nat")
        ),
        list_one_two_three
    )
    print(f"Length of list [1, 2, 3]: {kernel.infer_type(list_three_length)}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main() 