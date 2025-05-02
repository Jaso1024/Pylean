#!/usr/bin/env python3
"""
Demo of the Pylean tactics REPL for interactive theorem proving.

This example shows how to use the interactive REPL for
applying tactics to prove theorems step-by-step.
"""

from pylean.kernel import (
    Name, Level, Expr, ExprKind,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, NameAlreadyExistsException, TypeCheckException
)
from pylean.kernel.tactic_repl import start_tactic_repl


def setup_kernel() -> Kernel:
    """Set up a kernel with basic types and operations."""
    kernel = Kernel()
    
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
    except Exception:
        kernel = kernel.add_axiom("refl", refl_type)
        print("Added refl axiom with type Π (A : Type), Π (a : A), Eq A a a")
    
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
    except Exception:
        kernel = kernel.add_constant("implies", impl_type)
        print("Added implies constant with type Prop -> Prop -> Prop")
    
    # Define and
    and_type = mk_pi(
        "a",
        mk_sort(0),
        mk_pi(
            "b",
            mk_sort(0),
            mk_sort(0)
        )
    )
    try:
        kernel.infer_type(mk_const("and"))
    except Exception:
        kernel = kernel.add_constant("and", and_type)
        print("Added and constant with type Prop -> Prop -> Prop")
    
    # Define Nat type for examples
    nat_type = mk_sort(0)
    try:
        kernel.infer_type(mk_const("Nat"))
    except Exception:
        kernel = kernel.add_constant("Nat", nat_type)
        print("Added Nat type")
    
    print("Kernel setup complete\n")
    return kernel


def identity_theorem() -> Expr:
    """
    Create the identity theorem: Π (A : Prop), A -> A.
    
    Returns:
        The type of the identity theorem
    """
    return mk_pi(
        "A",
        mk_sort(0),
        mk_pi(
            "a",
            mk_var(0),
            mk_var(1)
        )
    )


def modus_ponens_theorem() -> Expr:
    """
    Create the modus ponens theorem: Π (A B : Prop), (A -> B) -> A -> B.
    
    Returns:
        The type of the modus ponens theorem
    """
    return mk_pi(
        "A",
        mk_sort(0),
        mk_pi(
            "B",
            mk_sort(0),
            mk_pi(
                "H1",
                mk_app(
                    mk_app(
                        mk_const("implies"),
                        mk_var(1)
                    ),
                    mk_var(0)
                ),
                mk_pi(
                    "H2",
                    mk_var(2),
                    mk_var(1)
                )
            )
        )
    )


def main():
    """Run the tactics REPL demo."""
    print("PyLean Tactics REPL Demo")
    print("=======================")
    
    # Set up the kernel with basic types and operations
    kernel = setup_kernel()
    
    # Choose a theorem to prove
    print("Available theorems to prove:")
    print("1. Identity: Π (A : Prop), A -> A")
    print("2. Modus Ponens: Π (A B : Prop), (A -> B) -> A -> B")
    
    while True:
        choice = input("Enter theorem number (1-2) or 'q' to quit: ")
        if choice.lower() == 'q':
            print("Exiting demo")
            return
        
        try:
            choice_num = int(choice)
            if choice_num == 1:
                theorem = identity_theorem()
                name = "Identity"
                break
            elif choice_num == 2:
                theorem = modus_ponens_theorem()
                name = "Modus Ponens"
                break
            else:
                print("Invalid choice, please enter 1 or 2")
        except ValueError:
            print("Invalid input, please enter a number")
    
    print(f"\nSelected theorem: {name}")
    
    if choice_num == 1:
        print("\nProving: Π (A : Prop), A -> A")
        print("\nHint: This can be proven with the following tactics:")
        print("1. intro A       (introduces A : Prop)")
        print("2. intro a       (introduces a : A)")
        print("3. assumption    (completes the proof by using assumption a)")
    elif choice_num == 2:
        print("\nProving: Π (A B : Prop), (A -> B) -> A -> B")
        print("\nHint: This can be proven with the following tactics:")
        print("1. intro A       (introduces A : Prop)")
        print("2. intro B       (introduces B : Prop)")
        print("3. intro H1      (introduces H1 : A -> B)")
        print("4. intro H2      (introduces H2 : A)")
        print("5. apply H1      (applies the implication H1)")
        print("6. assumption    (completes the proof with H2)")
    
    print("\nStarting tactics REPL...\n")
    
    # Start the REPL for the selected theorem
    start_tactic_repl(kernel, theorem)


if __name__ == "__main__":
    main() 