"""
Natural numbers module for the standard library.

This module provides definitions and operations for
natural numbers, including addition, multiplication, and proofs.
"""

from pylean.kernel import (
    Expr, ExprKind, Environment, Declaration, DeclKind, Kernel, Context,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi
)


def init_nat(kernel: Kernel) -> Kernel:
    """
    Initialize the natural numbers module.
    
    This adds all natural number definitions to a kernel.
    
    Args:
        kernel: The kernel to extend
        
    Returns:
        The extended kernel
    """
    # Get the environment
    env = kernel.env
    
    # Define Nat
    kernel = kernel.add_constant("Nat", mk_sort(0))
    
    # Define zero : Nat
    kernel = kernel.add_constant("zero", mk_const("Nat"))
    
    # Define succ : Nat → Nat
    succ_type = mk_pi("n", mk_const("Nat"), mk_const("Nat"))
    kernel = kernel.add_constant("succ", succ_type)
    
    # Define add : Nat → Nat → Nat
    add_type = mk_pi(
        "m", mk_const("Nat"),
        mk_pi("n", mk_const("Nat"), mk_const("Nat"))
    )
    kernel = kernel.add_constant("add", add_type)
    
    # Define add_zero : ∀ n, add zero n = n
    add_zero_type = mk_pi(
        "n", mk_const("Nat"),
        mk_app(
            mk_app(
                mk_app(
                    mk_const("Eq"),
                    mk_const("Nat")
                ),
                mk_app(mk_app(mk_const("add"), mk_const("zero")), mk_var(0))
            ),
            mk_var(0)
        )
    )
    kernel = kernel.add_constant("add_zero", add_zero_type)
    
    # Define add_succ : ∀ m n, add (succ m) n = succ (add m n)
    add_succ_type = mk_pi(
        "m", mk_const("Nat"),
        mk_pi(
            "n", mk_const("Nat"),
            mk_app(
                mk_app(
                    mk_app(
                        mk_const("Eq"),
                        mk_const("Nat")
                    ),
                    mk_app(
                        mk_app(
                            mk_const("add"),
                            mk_app(mk_const("succ"), mk_var(1))
                        ),
                        mk_var(0)
                    )
                ),
                mk_app(
                    mk_const("succ"),
                    mk_app(mk_app(mk_const("add"), mk_var(1)), mk_var(0))
                )
            )
        )
    )
    kernel = kernel.add_constant("add_succ", add_succ_type)
    
    # Define mul : Nat → Nat → Nat
    mul_type = mk_pi(
        "m", mk_const("Nat"),
        mk_pi("n", mk_const("Nat"), mk_const("Nat"))
    )
    kernel = kernel.add_constant("mul", mul_type)
    
    # Define mul_zero : ∀ n, mul zero n = zero
    mul_zero_type = mk_pi(
        "n", mk_const("Nat"),
        mk_app(
            mk_app(
                mk_app(
                    mk_const("Eq"),
                    mk_const("Nat")
                ),
                mk_app(mk_app(mk_const("mul"), mk_const("zero")), mk_var(0))
            ),
            mk_const("zero")
        )
    )
    kernel = kernel.add_constant("mul_zero", mul_zero_type)
    
    # Define mul_succ : ∀ m n, mul (succ m) n = add n (mul m n)
    mul_succ_type = mk_pi(
        "m", mk_const("Nat"),
        mk_pi(
            "n", mk_const("Nat"),
            mk_app(
                mk_app(
                    mk_app(
                        mk_const("Eq"),
                        mk_const("Nat")
                    ),
                    mk_app(
                        mk_app(
                            mk_const("mul"),
                            mk_app(mk_const("succ"), mk_var(1))
                        ),
                        mk_var(0)
                    )
                ),
                mk_app(
                    mk_app(
                        mk_const("add"),
                        mk_var(0)
                    ),
                    mk_app(mk_app(mk_const("mul"), mk_var(1)), mk_var(0))
                )
            )
        )
    )
    kernel = kernel.add_constant("mul_succ", mul_succ_type)
    
    # Define Nat induction principle
    # nat_ind : ∀ (P : Nat → Prop), P zero → (∀ n, P n → P (succ n)) → ∀ n, P n
    nat_ind_type = mk_pi(
        "P", mk_pi("n", mk_const("Nat"), mk_const("Prop")),
        mk_pi(
            "H1", mk_app(mk_var(0), mk_const("zero")),
            mk_pi(
                "H2", mk_pi(
                    "n", mk_const("Nat"),
                    mk_pi(
                        "Hn", mk_app(mk_var(2), mk_var(0)),
                        mk_app(mk_var(3), mk_app(mk_const("succ"), mk_var(1)))
                    )
                ),
                mk_pi(
                    "n", mk_const("Nat"),
                    mk_app(mk_var(3), mk_var(0))
                )
            )
        )
    )
    kernel = kernel.add_constant("nat_ind", nat_ind_type)
    
    # Define a simple function: double(n) = add n n
    double_type = mk_pi("n", mk_const("Nat"), mk_const("Nat"))
    n = mk_var(0)  # The bound variable 'n'
    add = mk_const("add")
    add_n = mk_app(add, n)
    double_body = mk_app(add_n, n)  # add n n
    double_def = mk_lambda("n", mk_const("Nat"), double_body)
    kernel = kernel.add_definition("double", double_type, double_def)
    
    # Define simple numerals
    # 1 = succ zero
    one_def = mk_app(mk_const("succ"), mk_const("zero"))
    kernel = kernel.add_definition("one", mk_const("Nat"), one_def)
    
    # 2 = succ one = succ (succ zero)
    two_def = mk_app(mk_const("succ"), mk_const("one"))
    kernel = kernel.add_definition("two", mk_const("Nat"), two_def)
    
    # 3 = succ two = succ (succ (succ zero))
    three_def = mk_app(mk_const("succ"), mk_const("two"))
    kernel = kernel.add_definition("three", mk_const("Nat"), three_def)
    
    # 4 = succ three = succ (succ (succ (succ zero)))
    four_def = mk_app(mk_const("succ"), mk_const("three"))
    kernel = kernel.add_definition("four", mk_const("Nat"), four_def)
    
    # 5 = succ four = succ (succ (succ (succ (succ zero))))
    five_def = mk_app(mk_const("succ"), mk_const("four"))
    kernel = kernel.add_definition("five", mk_const("Nat"), five_def)
    
    return kernel


# Export common names
__all__ = [
    'init_nat',
] 