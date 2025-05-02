"""
List module for the standard library.

This module provides definitions and operations for
lists, including construction, append, map, and fold.
"""

from pylean.kernel import (
    Expr, ExprKind, Environment, Declaration, DeclKind, Kernel, Context,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi
)


def init_list(kernel: Kernel) -> Kernel:
    """
    Initialize the list module.
    
    This adds all list definitions to a kernel.
    
    Args:
        kernel: The kernel to extend
        
    Returns:
        The extended kernel
    """
    # Get the environment
    env = kernel.env
    
    # Define List
    # List : Type → Type
    list_type = mk_pi("A", mk_sort(1), mk_sort(1))
    kernel = kernel.add_constant("List", list_type)
    
    # Define nil : ∀ A, List A
    nil_type = mk_pi("A", mk_sort(1), mk_app(mk_const("List"), mk_var(0)))
    kernel = kernel.add_constant("nil", nil_type)
    
    # Define cons : ∀ A, A → List A → List A
    cons_type = mk_pi(
        "A", mk_sort(1),
        mk_pi(
            "head", mk_var(0),
            mk_pi(
                "tail", mk_app(mk_const("List"), mk_var(1)),
                mk_app(mk_const("List"), mk_var(2))
            )
        )
    )
    kernel = kernel.add_constant("cons", cons_type)
    
    # Define append : ∀ A, List A → List A → List A
    append_type = mk_pi(
        "A", mk_sort(1),
        mk_pi(
            "l1", mk_app(mk_const("List"), mk_var(0)),
            mk_pi(
                "l2", mk_app(mk_const("List"), mk_var(1)),
                mk_app(mk_const("List"), mk_var(2))
            )
        )
    )
    kernel = kernel.add_constant("append", append_type)
    
    # Define append_nil : ∀ A (l : List A), append A (nil A) l = l
    append_nil_type = mk_pi(
        "A", mk_sort(1),
        mk_pi(
            "l", mk_app(mk_const("List"), mk_var(0)),
            mk_app(
                mk_app(
                    mk_app(
                        mk_const("Eq"),
                        mk_app(mk_const("List"), mk_var(1))
                    ),
                    mk_app(
                        mk_app(
                            mk_app(
                                mk_const("append"),
                                mk_var(1)
                            ),
                            mk_app(mk_const("nil"), mk_var(1))
                        ),
                        mk_var(0)
                    )
                ),
                mk_var(0)
            )
        )
    )
    kernel = kernel.add_constant("append_nil", append_nil_type)
    
    # Define append_cons : ∀ A (h : A) (t l : List A),
    #   append A (cons A h t) l = cons A h (append A t l)
    append_cons_type = mk_pi(
        "A", mk_sort(1),
        mk_pi(
            "h", mk_var(0),
            mk_pi(
                "t", mk_app(mk_const("List"), mk_var(1)),
                mk_pi(
                    "l", mk_app(mk_const("List"), mk_var(2)),
                    mk_app(
                        mk_app(
                            mk_app(
                                mk_const("Eq"),
                                mk_app(mk_const("List"), mk_var(3))
                            ),
                            mk_app(
                                mk_app(
                                    mk_app(
                                        mk_const("append"),
                                        mk_var(3)
                                    ),
                                    mk_app(
                                        mk_app(
                                            mk_app(
                                                mk_const("cons"),
                                                mk_var(3)
                                            ),
                                            mk_var(2)
                                        ),
                                        mk_var(1)
                                    )
                                ),
                                mk_var(0)
                            )
                        ),
                        mk_app(
                            mk_app(
                                mk_app(
                                    mk_const("cons"),
                                    mk_var(3)
                                ),
                                mk_var(2)
                            ),
                            mk_app(
                                mk_app(
                                    mk_app(
                                        mk_const("append"),
                                        mk_var(3)
                                    ),
                                    mk_var(1)
                                ),
                                mk_var(0)
                            )
                        )
                    )
                )
            )
        )
    )
    kernel = kernel.add_constant("append_cons", append_cons_type)
    
    # Define map : ∀ A B, (A → B) → List A → List B
    map_type = mk_pi(
        "A", mk_sort(1),
        mk_pi(
            "B", mk_sort(1),
            mk_pi(
                "f", mk_pi("a", mk_var(1), mk_var(1)),
                mk_pi(
                    "l", mk_app(mk_const("List"), mk_var(2)),
                    mk_app(mk_const("List"), mk_var(2))
                )
            )
        )
    )
    kernel = kernel.add_constant("map", map_type)
    
    # Define map_nil : ∀ A B (f : A → B), map A B f (nil A) = nil B
    map_nil_type = mk_pi(
        "A", mk_sort(1),
        mk_pi(
            "B", mk_sort(1),
            mk_pi(
                "f", mk_pi("a", mk_var(1), mk_var(1)),
                mk_app(
                    mk_app(
                        mk_app(
                            mk_const("Eq"),
                            mk_app(mk_const("List"), mk_var(1))
                        ),
                        mk_app(
                            mk_app(
                                mk_app(
                                    mk_app(
                                        mk_const("map"),
                                        mk_var(2)
                                    ),
                                    mk_var(1)
                                ),
                                mk_var(0)
                            ),
                            mk_app(mk_const("nil"), mk_var(2))
                        )
                    ),
                    mk_app(mk_const("nil"), mk_var(1))
                )
            )
        )
    )
    kernel = kernel.add_constant("map_nil", map_nil_type)
    
    # Define map_cons : ∀ A B (f : A → B) (h : A) (t : List A),
    #   map A B f (cons A h t) = cons B (f h) (map A B f t)
    map_cons_type = mk_pi(
        "A", mk_sort(1),
        mk_pi(
            "B", mk_sort(1),
            mk_pi(
                "f", mk_pi("a", mk_var(1), mk_var(1)),
                mk_pi(
                    "h", mk_var(2),
                    mk_pi(
                        "t", mk_app(mk_const("List"), mk_var(3)),
                        mk_app(
                            mk_app(
                                mk_app(
                                    mk_const("Eq"),
                                    mk_app(mk_const("List"), mk_var(3))
                                ),
                                mk_app(
                                    mk_app(
                                        mk_app(
                                            mk_app(
                                                mk_const("map"),
                                                mk_var(4)
                                            ),
                                            mk_var(3)
                                        ),
                                        mk_var(2)
                                    ),
                                    mk_app(
                                        mk_app(
                                            mk_app(
                                                mk_const("cons"),
                                                mk_var(4)
                                            ),
                                            mk_var(1)
                                        ),
                                        mk_var(0)
                                    )
                                )
                            ),
                            mk_app(
                                mk_app(
                                    mk_app(
                                        mk_const("cons"),
                                        mk_var(3)
                                    ),
                                    mk_app(mk_var(2), mk_var(1))
                                ),
                                mk_app(
                                    mk_app(
                                        mk_app(
                                            mk_app(
                                                mk_const("map"),
                                                mk_var(4)
                                            ),
                                            mk_var(3)
                                        ),
                                        mk_var(2)
                                    ),
                                    mk_var(0)
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    kernel = kernel.add_constant("map_cons", map_cons_type)
    
    # Define fold : ∀ A B, (A → B → B) → B → List A → B
    fold_type = mk_pi(
        "A", mk_sort(1),
        mk_pi(
            "B", mk_sort(1),
            mk_pi(
                "f", mk_pi("a", mk_var(1), mk_pi("b", mk_var(1), mk_var(1))),
                mk_pi(
                    "init", mk_var(1),
                    mk_pi(
                        "l", mk_app(mk_const("List"), mk_var(3)),
                        mk_var(2)
                    )
                )
            )
        )
    )
    kernel = kernel.add_constant("fold", fold_type)
    
    # Define fold_nil : ∀ A B (f : A → B → B) (init : B),
    #   fold A B f init (nil A) = init
    fold_nil_type = mk_pi(
        "A", mk_sort(1),
        mk_pi(
            "B", mk_sort(1),
            mk_pi(
                "f", mk_pi("a", mk_var(1), mk_pi("b", mk_var(1), mk_var(1))),
                mk_pi(
                    "init", mk_var(1),
                    mk_app(
                        mk_app(
                            mk_app(
                                mk_const("Eq"),
                                mk_var(1)
                            ),
                            mk_app(
                                mk_app(
                                    mk_app(
                                        mk_app(
                                            mk_app(
                                                mk_const("fold"),
                                                mk_var(3)
                                            ),
                                            mk_var(2)
                                        ),
                                        mk_var(1)
                                    ),
                                    mk_var(0)
                                ),
                                mk_app(mk_const("nil"), mk_var(3))
                            )
                        ),
                        mk_var(0)
                    )
                )
            )
        )
    )
    kernel = kernel.add_constant("fold_nil", fold_nil_type)
    
    # Define fold_cons : ∀ A B (f : A → B → B) (init : B) (h : A) (t : List A),
    #   fold A B f init (cons A h t) = f h (fold A B f init t)
    fold_cons_type = mk_pi(
        "A", mk_sort(1),
        mk_pi(
            "B", mk_sort(1),
            mk_pi(
                "f", mk_pi("a", mk_var(1), mk_pi("b", mk_var(1), mk_var(1))),
                mk_pi(
                    "init", mk_var(1),
                    mk_pi(
                        "h", mk_var(3),
                        mk_pi(
                            "t", mk_app(mk_const("List"), mk_var(4)),
                            mk_app(
                                mk_app(
                                    mk_app(
                                        mk_const("Eq"),
                                        mk_var(3)
                                    ),
                                    mk_app(
                                        mk_app(
                                            mk_app(
                                                mk_app(
                                                    mk_app(
                                                        mk_const("fold"),
                                                        mk_var(5)
                                                    ),
                                                    mk_var(4)
                                                ),
                                                mk_var(3)
                                            ),
                                            mk_var(2)
                                        ),
                                        mk_app(
                                            mk_app(
                                                mk_app(
                                                    mk_const("cons"),
                                                    mk_var(5)
                                                ),
                                                mk_var(1)
                                            ),
                                            mk_var(0)
                                        )
                                    )
                                ),
                                mk_app(
                                    mk_app(
                                        mk_var(3),
                                        mk_var(1)
                                    ),
                                    mk_app(
                                        mk_app(
                                            mk_app(
                                                mk_app(
                                                    mk_app(
                                                        mk_const("fold"),
                                                        mk_var(5)
                                                    ),
                                                    mk_var(4)
                                                ),
                                                mk_var(3)
                                            ),
                                            mk_var(2)
                                        ),
                                        mk_var(0)
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    kernel = kernel.add_constant("fold_cons", fold_cons_type)
    
    # Define List induction principle
    # list_ind : ∀ A (P : List A → Prop), P (nil A) →
    #            (∀ (h : A) (t : List A), P t → P (cons A h t)) →
    #            ∀ (l : List A), P l
    list_ind_type = mk_pi(
        "A", mk_sort(1),
        mk_pi(
            "P", mk_pi("l", mk_app(mk_const("List"), mk_var(0)), mk_const("Prop")),
            mk_pi(
                "H1", mk_app(mk_var(0), mk_app(mk_const("nil"), mk_var(1))),
                mk_pi(
                    "H2", mk_pi(
                        "h", mk_var(2),
                        mk_pi(
                            "t", mk_app(mk_const("List"), mk_var(3)),
                            mk_pi(
                                "Ht", mk_app(mk_var(3), mk_var(0)),
                                mk_app(
                                    mk_var(4),
                                    mk_app(
                                        mk_app(
                                            mk_app(
                                                mk_const("cons"),
                                                mk_var(4)
                                            ),
                                            mk_var(2)
                                        ),
                                        mk_var(1)
                                    )
                                )
                            )
                        )
                    ),
                    mk_pi(
                        "l", mk_app(mk_const("List"), mk_var(3)),
                        mk_app(mk_var(2), mk_var(0))
                    )
                )
            )
        )
    )
    kernel = kernel.add_constant("list_ind", list_ind_type)
    
    return kernel


# Export common names
__all__ = [
    'init_list',
] 