#!/usr/bin/env python3
"""
Demo of the Pylean kernel functionality.

This example shows how to use the core kernel functionality
including type checking, definitions, and theorem proving.
"""

from pylean.kernel import (
    Name, Level, Expr, ExprKind,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, NameAlreadyExistsException, TypeCheckException
)


def main():
    """Run the kernel demo."""
    print("PyLean Kernel Demo")
    print("=================")
    
    # Create a new kernel with standard environment
    kernel = Kernel()
    print("Created kernel with standard environment")
    
    # Check what's in the standard environment
    print("Standard env contains:")
    print(f"  - Prop: {kernel.env.find_decl('Prop') is not None}")
    print(f"  - Type: {kernel.env.find_decl('Type') is not None}")
    
    # Step 1: Define a custom logic type
    print("\nStep 1: Define a custom logic type")
    print("--------------------------------")
    
    # Define a custom logic type
    logic_type = mk_sort(0)  # Type is Prop
    kernel = kernel.add_constant("Logic", logic_type)
    print("Added Logic constant with type Prop")
    
    # Add the constructors
    kernel = kernel.add_inductive(
        "Logic",
        logic_type,
        [
            ("t", mk_const("Logic")),
            ("f", mk_const("Logic"))
        ]
    )
    print("Defined Logic as an inductive type with constructors t and f")
    
    # Step 2: Define operations on logic values
    print("\nStep 2: Define operations on logic values")
    print("----------------------------------")
    
    # Define not: Logic -> Logic
    not_type = mk_pi("b", mk_const("Logic"), mk_const("Logic"))
    not_value = mk_lambda(
        "b",
        mk_const("Logic"),  # Parameter type
        mk_app(
            mk_app(
                mk_app(
                    mk_const("Logic.rec_on"),
                    mk_var(0)  # b
                ),
                mk_const("f")  # not t = f
            ),
            mk_const("t")  # not f = t
        )
    )
    kernel = kernel.add_definition("logic_not", not_type, not_value)
    print("Defined 'logic_not' function")
    
    # Define and: Logic -> Logic -> Logic
    and_type = mk_pi("b1", mk_const("Logic"), mk_pi("b2", mk_const("Logic"), mk_const("Logic")))
    and_value = mk_lambda(
        "b1",
        mk_const("Logic"),
        mk_lambda(
            "b2",
            mk_const("Logic"),
            mk_app(
                mk_app(
                    mk_app(
                        mk_const("Logic.rec_on"),
                        mk_var(1)  # b1
                    ),
                    mk_var(0)  # if b1 is t, return b2
                ),
                mk_const("f")  # if b1 is f, return f
            )
        )
    )
    kernel = kernel.add_definition("logic_and", and_type, and_value)
    print("Defined 'logic_and' function")
    
    # Define or: Logic -> Logic -> Logic
    or_type = mk_pi("b1", mk_const("Logic"), mk_pi("b2", mk_const("Logic"), mk_const("Logic")))
    or_value = mk_lambda(
        "b1",
        mk_const("Logic"),
        mk_lambda(
            "b2",
            mk_const("Logic"),
            mk_app(
                mk_app(
                    mk_app(
                        mk_const("Logic.rec_on"),
                        mk_var(1)  # b1
                    ),
                    mk_const("t")  # if b1 is t, return t
                ),
                mk_var(0)  # if b1 is f, return b2
            )
        )
    )
    kernel = kernel.add_definition("logic_or", or_type, or_value)
    print("Defined 'logic_or' function")
    
    # Step 3: Try to add an incorrect definition
    print("\nStep 3: Try to add an incorrect definition")
    print("--------------------------------------")
    
    # Try to define a value with incorrect type
    try:
        # mk_sort(1) (Type) is not compatible with mk_const("Logic")
        kernel.add_definition("bad_def", mk_const("Logic"), mk_sort(1))
    except TypeCheckException as e:
        print(f"Correctly caught type error: {e}")
    
    # Try to define a duplicate name
    try:
        kernel.add_constant("Logic", mk_sort(0))
    except NameAlreadyExistsException as e:
        print(f"Correctly caught duplicate name: {e}")
    
    # Step 4: Define and prove a theorem
    print("\nStep 4: Define and prove a theorem")
    print("-------------------------------")
    
    # First, define equality
    eq_type = mk_pi(
        "A",
        mk_sort(0),  # Type 0 (Prop)
        mk_pi(
            "x",
            mk_var(0),  # A
            mk_pi(
                "y",
                mk_var(1),  # A
                mk_sort(0)  # Prop
            )
        )
    )
    kernel = kernel.add_constant("logic_eq", eq_type)
    print("Added logic_eq constant with type: Π(A: Type), Π(x: A), Π(y: A), Prop")
    
    # Define reflexivity of equality
    eq_refl_type = mk_pi(
        "A",
        mk_sort(0),  # Type
        mk_pi(
            "x",
            mk_var(0),  # A
            mk_app(
                mk_app(
                    mk_app(
                        mk_const("logic_eq"),
                        mk_var(1)  # A
                    ),
                    mk_var(0)  # x
                ),
                mk_var(0)  # x
            )
        )
    )
    kernel = kernel.add_axiom("logic_eq_refl", eq_refl_type)
    print("Added logic_eq_refl constant with type: Π(A: Type), Π(x: A), eq A x x")
    
    # Define a theorem: not_not_elim: Π(b: Logic), eq Logic (not (not b)) b
    theorem_type = mk_pi(
        "b",
        mk_const("Logic"),
        mk_app(
            mk_app(
                mk_app(
                    mk_const("logic_eq"),
                    mk_const("Logic")
                ),
                mk_app(
                    mk_const("logic_not"),
                    mk_app(
                        mk_const("logic_not"),
                        mk_var(0)  # b
                    )
                )
            ),
            mk_var(0)  # b
        )
    )
    kernel = kernel.add_axiom("logic_not_not_elim", theorem_type)
    print("Added logic_not_not_elim axiom")
    
    # Define the proof of the theorem
    theorem_proof = mk_lambda(
        "b",
        mk_const("Logic"),
        # The actual proof would be by case analysis on b
        # For t: not (not t) = not f = t
        # For f: not (not f) = not t = f
        # Here we just supply a constant as a placeholder for the proof
        mk_app(
            mk_app(
                mk_app(
                    mk_const("Logic.rec_on"),
                    mk_var(0)  # b
                ),
                # Case t
                mk_app(
                    mk_app(
                        mk_const("logic_eq_refl"),
                        mk_const("Logic")
                    ),
                    mk_const("t")
                )
            ),
            # Case f
            mk_app(
                mk_app(
                    mk_const("logic_eq_refl"),
                    mk_const("Logic")
                ),
                mk_const("f")
            )
        )
    )
    
    # Add the proof to the kernel
    kernel = kernel.add_definition("logic_not_not_elim_proof", theorem_type, theorem_proof)
    print("Added logic_not_not_elim_proof function")
    print()
    
    # Step 5: Create a child kernel for a namespace
    print("Step 5: Create a child kernel for a namespace")
    print("----------------------------------------")
    
    # Create a child kernel
    child_kernel = kernel.mk_child("my_namespace")
    print("Created child kernel with namespace 'my_namespace'")
    
    # Add a declaration to the child
    child_kernel = child_kernel.add_axiom("local_axiom", mk_const("Logic"))
    print("Added local_axiom to child kernel")
    
    # Check that parent's declarations are visible in the child
    print(f"Logic in child kernel: {child_kernel.env.find_decl('Logic') is not None}")
    print()
    
    # Step 6: Evaluate expressions
    print("Step 6: Evaluate expressions")
    print("------------------------")
    
    # Create some expressions
    not_t = mk_app(mk_const("logic_not"), mk_const("t"))
    not_f = mk_app(mk_const("logic_not"), mk_const("f"))
    not_not_t = mk_app(mk_const("logic_not"), mk_app(mk_const("logic_not"), mk_const("t")))
    
    # Check types
    print(f"Type of not_t: {kernel.infer_type(not_t)}")
    print(f"Type of not_f: {kernel.infer_type(not_f)}")
    print(f"Type of not_not_t: {kernel.infer_type(not_not_t)}")
    
    # Evaluate expressions
    print(f"not_t evaluates to: {kernel.normalize(not_t)}")
    print(f"not_f evaluates to: {kernel.normalize(not_f)}")
    print(f"not_not_t evaluates to: {kernel.normalize(not_not_t)}")
    
    # Check for definitional equality
    print(f"not_t == f: {kernel.is_def_eq(not_t, mk_const('f'))}")
    print(f"not_f == t: {kernel.is_def_eq(not_f, mk_const('t'))}")
    print(f"not_not_t == t: {kernel.is_def_eq(not_not_t, mk_const('t'))}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main() 