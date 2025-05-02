"""
Logic module for the standard library.

This module provides basic logical types and operations,
such as True, False, And, Or, Implies, and Not.
"""

from pylean.kernel import (
    Expr, ExprKind, Environment, Declaration, DeclKind, Kernel, Context,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi
)


def init_logic(kernel: Kernel) -> Kernel:
    """
    Initialize the logic module.
    
    This adds all logic definitions to a kernel.
    
    Args:
        kernel: The kernel to extend
        
    Returns:
        The extended kernel
    """
    # Get the environment
    env = kernel.env
    
    # Define Prop as the sort of propositions
    kernel = kernel.add_constant("Prop", mk_sort(0))
    
    # Define True
    kernel = kernel.add_constant("True", mk_const("Prop"))
    
    # Define True.intro : True (proof of True)
    kernel = kernel.add_constant("True.intro", mk_const("True"))
    
    # Define False
    kernel = kernel.add_constant("False", mk_const("Prop"))
    
    # Define False.elim : ∀ (P : Prop), False → P
    false_elim_type = mk_pi(
        "P", mk_const("Prop"),
        mk_pi("H", mk_const("False"), mk_var(1))
    )
    kernel = kernel.add_constant("False.elim", false_elim_type)
    
    # Define Not
    # not P := P → False
    not_type = mk_pi("P", mk_const("Prop"), mk_const("Prop"))
    not_def = mk_lambda(
        "P", mk_const("Prop"),
        mk_pi("H", mk_var(0), mk_const("False"))
    )
    kernel = kernel.add_definition("Not", not_type, not_def)
    
    # Define And
    # and P Q := ∀ (C : Prop), (P → Q → C) → C
    and_type = mk_pi(
        "P", mk_const("Prop"),
        mk_pi("Q", mk_const("Prop"), mk_const("Prop"))
    )
    and_def = mk_lambda(
        "P", mk_const("Prop"),
        mk_lambda(
            "Q", mk_const("Prop"),
            mk_pi(
                "C", mk_const("Prop"),
                mk_pi(
                    "H", mk_pi("HP", mk_var(2), mk_pi("HQ", mk_var(2), mk_var(2))),
                    mk_var(1)
                )
            )
        )
    )
    kernel = kernel.add_definition("And", and_type, and_def)
    
    # Define And.intro : ∀ (P Q : Prop), P → Q → And P Q
    and_intro_type = mk_pi(
        "P", mk_const("Prop"),
        mk_pi(
            "Q", mk_const("Prop"),
            mk_pi(
                "HP", mk_var(1),
                mk_pi(
                    "HQ", mk_var(1),
                    mk_app(mk_app(mk_const("And"), mk_var(3)), mk_var(2))
                )
            )
        )
    )
    and_intro_def = mk_lambda(
        "P", mk_const("Prop"),
        mk_lambda(
            "Q", mk_const("Prop"),
            mk_lambda(
                "HP", mk_var(1),
                mk_lambda(
                    "HQ", mk_var(1),
                    mk_lambda(
                        "C", mk_const("Prop"),
                        mk_lambda(
                            "H", mk_pi("HP", mk_var(5), mk_pi("HQ", mk_var(4), mk_var(3))),
                            mk_app(mk_app(mk_var(0), mk_var(3)), mk_var(1))
                        )
                    )
                )
            )
        )
    )
    kernel = kernel.add_definition("And.intro", and_intro_type, and_intro_def)
    
    # Define And.left : ∀ (P Q : Prop), And P Q → P
    and_left_type = mk_pi(
        "P", mk_const("Prop"),
        mk_pi(
            "Q", mk_const("Prop"),
            mk_pi(
                "H", mk_app(mk_app(mk_const("And"), mk_var(1)), mk_var(0)),
                mk_var(2)
            )
        )
    )
    and_left_def = mk_lambda(
        "P", mk_const("Prop"),
        mk_lambda(
            "Q", mk_const("Prop"),
            mk_lambda(
                "H", mk_app(mk_app(mk_const("And"), mk_var(1)), mk_var(0)),
                mk_app(
                    mk_app(
                        mk_app(mk_var(0), mk_var(2)),
                        mk_lambda(
                            "HP", mk_var(3),
                            mk_lambda(
                                "HQ", mk_var(3),
                                mk_var(1)
                            )
                        )
                    ),
                    mk_lambda(
                        "HP", mk_var(3),
                        mk_lambda(
                            "HQ", mk_var(3),
                            mk_var(1)
                        )
                    )
                )
            )
        )
    )
    kernel = kernel.add_definition("And.left", and_left_type, and_left_def)
    
    # Define And.right : ∀ (P Q : Prop), And P Q → Q
    and_right_type = mk_pi(
        "P", mk_const("Prop"),
        mk_pi(
            "Q", mk_const("Prop"),
            mk_pi(
                "H", mk_app(mk_app(mk_const("And"), mk_var(1)), mk_var(0)),
                mk_var(1)
            )
        )
    )
    and_right_def = mk_lambda(
        "P", mk_const("Prop"),
        mk_lambda(
            "Q", mk_const("Prop"),
            mk_lambda(
                "H", mk_app(mk_app(mk_const("And"), mk_var(1)), mk_var(0)),
                mk_app(
                    mk_app(
                        mk_app(mk_var(0), mk_var(1)),
                        mk_lambda(
                            "HP", mk_var(3),
                            mk_lambda(
                                "HQ", mk_var(3),
                                mk_var(0)
                            )
                        )
                    ),
                    mk_lambda(
                        "HP", mk_var(3),
                        mk_lambda(
                            "HQ", mk_var(3),
                            mk_var(0)
                        )
                    )
                )
            )
        )
    )
    kernel = kernel.add_definition("And.right", and_right_type, and_right_def)
    
    # Define Or
    # or P Q := ∀ (C : Prop), (P → C) → (Q → C) → C
    or_type = mk_pi(
        "P", mk_const("Prop"),
        mk_pi("Q", mk_const("Prop"), mk_const("Prop"))
    )
    or_def = mk_lambda(
        "P", mk_const("Prop"),
        mk_lambda(
            "Q", mk_const("Prop"),
            mk_pi(
                "C", mk_const("Prop"),
                mk_pi(
                    "HP", mk_pi("P", mk_var(3), mk_var(1)),
                    mk_pi(
                        "HQ", mk_pi("Q", mk_var(3), mk_var(2)),
                        mk_var(2)
                    )
                )
            )
        )
    )
    kernel = kernel.add_definition("Or", or_type, or_def)
    
    # Define Or.inl : ∀ (P Q : Prop), P → Or P Q
    or_inl_type = mk_pi(
        "P", mk_const("Prop"),
        mk_pi(
            "Q", mk_const("Prop"),
            mk_pi(
                "HP", mk_var(1),
                mk_app(mk_app(mk_const("Or"), mk_var(2)), mk_var(1))
            )
        )
    )
    or_inl_def = mk_lambda(
        "P", mk_const("Prop"),
        mk_lambda(
            "Q", mk_const("Prop"),
            mk_lambda(
                "HP", mk_var(1),
                mk_lambda(
                    "C", mk_const("Prop"),
                    mk_lambda(
                        "HPC", mk_pi("P", mk_var(4), mk_var(1)),
                        mk_lambda(
                            "HQC", mk_pi("Q", mk_var(4), mk_var(2)),
                            mk_app(mk_var(2), mk_var(3))
                        )
                    )
                )
            )
        )
    )
    kernel = kernel.add_definition("Or.inl", or_inl_type, or_inl_def)
    
    # Define Or.inr : ∀ (P Q : Prop), Q → Or P Q
    or_inr_type = mk_pi(
        "P", mk_const("Prop"),
        mk_pi(
            "Q", mk_const("Prop"),
            mk_pi(
                "HQ", mk_var(0),
                mk_app(mk_app(mk_const("Or"), mk_var(2)), mk_var(1))
            )
        )
    )
    or_inr_def = mk_lambda(
        "P", mk_const("Prop"),
        mk_lambda(
            "Q", mk_const("Prop"),
            mk_lambda(
                "HQ", mk_var(0),
                mk_lambda(
                    "C", mk_const("Prop"),
                    mk_lambda(
                        "HPC", mk_pi("P", mk_var(4), mk_var(1)),
                        mk_lambda(
                            "HQC", mk_pi("Q", mk_var(4), mk_var(2)),
                            mk_app(mk_var(1), mk_var(3))
                        )
                    )
                )
            )
        )
    )
    kernel = kernel.add_definition("Or.inr", or_inr_type, or_inr_def)
    
    # Define Implies (→) as a built-in operator
    # Note: Pi types already represent implication for propositions
    
    # Define Eq (equality)
    # eq {A : Type} (a b : A) : Prop := ∀ (P : A → Prop), P a → P b
    eq_type = mk_pi(
        "A", mk_sort(1),
        mk_pi(
            "a", mk_var(0),
            mk_pi("b", mk_var(1), mk_const("Prop"))
        )
    )
    eq_def = mk_lambda(
        "A", mk_sort(1),
        mk_lambda(
            "a", mk_var(0),
            mk_lambda(
                "b", mk_var(1),
                mk_pi(
                    "P", mk_pi("x", mk_var(2), mk_const("Prop")),
                    mk_pi(
                        "Ha", mk_app(mk_var(0), mk_var(2)),
                        mk_app(mk_var(1), mk_var(2))
                    )
                )
            )
        )
    )
    kernel = kernel.add_definition("Eq", eq_type, eq_def)
    
    # Define Eq.refl : ∀ (A : Type) (a : A), Eq A a a
    eq_refl_type = mk_pi(
        "A", mk_sort(1),
        mk_pi(
            "a", mk_var(0),
            mk_app(mk_app(mk_app(mk_const("Eq"), mk_var(1)), mk_var(0)), mk_var(0))
        )
    )
    eq_refl_def = mk_lambda(
        "A", mk_sort(1),
        mk_lambda(
            "a", mk_var(0),
            mk_lambda(
                "P", mk_pi("x", mk_var(1), mk_const("Prop")),
                mk_lambda(
                    "Ha", mk_app(mk_var(0), mk_var(1)),
                    mk_var(0)
                )
            )
        )
    )
    kernel = kernel.add_definition("Eq.refl", eq_refl_type, eq_refl_def)
    
    # Define the substitution principle
    # subst : ∀ (A : Type) (P : A → Prop) (a b : A), Eq A a b → P a → P b
    subst_type = mk_pi(
        "A", mk_sort(1),
        mk_pi(
            "P", mk_pi("x", mk_var(0), mk_const("Prop")),
            mk_pi(
                "a", mk_var(1),
                mk_pi(
                    "b", mk_var(2),
                    mk_pi(
                        "H", mk_app(mk_app(mk_app(mk_const("Eq"), mk_var(3)), mk_var(1)), mk_var(0)),
                        mk_pi(
                            "Pa", mk_app(mk_var(3), mk_var(2)),
                            mk_app(mk_var(4), mk_var(2))
                        )
                    )
                )
            )
        )
    )
    subst_def = mk_lambda(
        "A", mk_sort(1),
        mk_lambda(
            "P", mk_pi("x", mk_var(0), mk_const("Prop")),
            mk_lambda(
                "a", mk_var(1),
                mk_lambda(
                    "b", mk_var(2),
                    mk_lambda(
                        "H", mk_app(mk_app(mk_app(mk_const("Eq"), mk_var(3)), mk_var(1)), mk_var(0)),
                        mk_lambda(
                            "Pa", mk_app(mk_var(3), mk_var(2)),
                            mk_app(mk_app(mk_var(1), mk_var(3)), mk_var(0))
                        )
                    )
                )
            )
        )
    )
    kernel = kernel.add_definition("Eq.subst", subst_type, subst_def)
    
    return kernel


# Export common names
__all__ = [
    'init_logic',
] 