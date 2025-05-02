#!/usr/bin/env python3
"""
Demo of the Pylean kernel reduction system.

This example shows how to use the environment, typechecking, 
and reduction system together to build and evaluate expressions.
"""

from pylean.kernel import (
    Name, Level, Expr, ExprKind,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi, mk_let,
    Kernel, ReductionStrategy, ReductionMode, reduce
)


def main():
    """Run the reduction demo."""
    print("PyLean Reduction Demo")
    print("====================")
    
    # Create a new kernel with standard environment
    kernel = Kernel()
    print("Created kernel with standard environment")
    print()
    
    # Define natural numbers
    print("Step A: Define natural numbers (Church encoding)")
    print("----------------------------------------------")
    
    # Define the natural number type: Nat
    kernel = kernel.add_constant("Nat", mk_sort(0))
    print("Added Nat constant with type Prop")
    
    # Define zero: λ(s: Nat -> Nat), λ(z: Nat), z
    zero_type = define_nat_type(kernel)
    zero_value = mk_lambda(
        "s",
        mk_pi(Name(), mk_const("Nat"), mk_const("Nat")),
        mk_lambda("z", mk_const("Nat"), mk_var(0))
    )
    kernel = kernel.add_definition("zero", zero_type, zero_value)
    print("Defined 'zero' as Church numeral")
    
    # Define successor: λ(n: Nat), λ(s: Nat -> Nat), λ(z: Nat), s (n s z)
    succ_type = mk_pi("n", zero_type, zero_type)
    succ_value = mk_lambda(
        "n",
        zero_type,
        mk_lambda(
            "s",
            mk_pi(Name(), mk_const("Nat"), mk_const("Nat")),
            mk_lambda(
                "z",
                mk_const("Nat"),
                mk_app(
                    mk_var(1),
                    mk_app(
                        mk_app(mk_var(2), mk_var(1)),
                        mk_var(0)
                    )
                )
            )
        )
    )
    kernel = kernel.add_definition("succ", succ_type, succ_value)
    print("Defined 'succ' as Church numeral successor")
    
    # Define one = succ zero
    one_expr = mk_app(mk_const("succ"), mk_const("zero"))
    kernel = kernel.add_definition("one", zero_type, one_expr)
    print("Defined 'one' as succ zero")
    
    # Define two = succ one
    two_expr = mk_app(mk_const("succ"), mk_const("one"))
    kernel = kernel.add_definition("two", zero_type, two_expr)
    print("Defined 'two' as succ one")
    
    # Define three = succ two
    three_expr = mk_app(mk_const("succ"), mk_const("two"))
    kernel = kernel.add_definition("three", zero_type, three_expr)
    print("Defined 'three' as succ two")
    
    # Define addition: λ(m: Nat), λ(n: Nat), λ(s: Nat -> Nat), λ(z: Nat), m s (n s z)
    add_type = mk_pi("m", zero_type, mk_pi("n", zero_type, zero_type))
    add_value = mk_lambda(
        "m",
        zero_type,
        mk_lambda(
            "n",
            zero_type,
            mk_lambda(
                "s",
                mk_pi(Name(), mk_const("Nat"), mk_const("Nat")),
                mk_lambda(
                    "z",
                    mk_const("Nat"),
                    mk_app(
                        mk_app(mk_var(3), mk_var(1)),
                        mk_app(
                            mk_app(mk_var(2), mk_var(1)),
                            mk_var(0)
                        )
                    )
                )
            )
        )
    )
    kernel = kernel.add_definition("add", add_type, add_value)
    print("Defined 'add' function for Church numerals")
    
    # Define multiplication: λ(m: Nat), λ(n: Nat), λ(s: Nat -> Nat), λ(z: Nat), m (n s) z
    mul_type = mk_pi("m", zero_type, mk_pi("n", zero_type, zero_type))
    mul_value = mk_lambda(
        "m",
        zero_type,
        mk_lambda(
            "n",
            zero_type,
            mk_lambda(
                "s",
                mk_pi(Name(), mk_const("Nat"), mk_const("Nat")),
                mk_lambda(
                    "z",
                    mk_const("Nat"),
                    mk_app(
                        mk_app(
                            mk_var(3),
                            mk_app(mk_var(2), mk_var(1))
                        ),
                        mk_var(0)
                    )
                )
            )
        )
    )
    kernel = kernel.add_definition("mul", mul_type, mul_value)
    print("Defined 'mul' function for Church numerals")
    print()
    
    # Step B: Reduction demonstrations
    print("Step B: Demonstration of reduction")
    print("-------------------------------")
    
    # Evaluate 1 + 1
    print("\nEvaluating 1 + 1:")
    one_plus_one = mk_app(mk_app(mk_const("add"), mk_const("one")), mk_const("one"))
    print(f"Expression: {one_plus_one}")
    
    # Type check the expression
    type_of_expr = kernel.infer_type(one_plus_one)
    print(f"Type: {type_of_expr}")
    
    # Reduce to WHNF (partial reduction)
    reduced_whnf = kernel.reduce(one_plus_one, ReductionStrategy.WHNF)
    print(f"WHNF: {reduced_whnf}")
    
    # Reduce to NF (full reduction)
    reduced_nf = kernel.normalize(one_plus_one)
    print(f"NF: {reduced_nf}")
    
    # Check if 1+1 = 2
    is_equal = kernel.is_def_eq(one_plus_one, mk_const("two"))
    print(f"1 + 1 = 2: {is_equal}")
    
    # Evaluate 2 × 3
    print("\nEvaluating 2 × 3:")
    two_times_three = mk_app(mk_app(mk_const("mul"), mk_const("two")), mk_const("three"))
    print(f"Expression: {two_times_three}")
    
    # Type check the expression
    type_of_expr = kernel.infer_type(two_times_three)
    print(f"Type: {type_of_expr}")
    
    # Check if 2 × 3 = 6 (we haven't defined 6, so we'll construct it)
    six_expr = mk_app(mk_const("succ"), 
              mk_app(mk_const("succ"), 
              mk_app(mk_const("succ"), 
              mk_app(mk_const("succ"), 
              mk_app(mk_const("succ"), mk_const("one"))))))
    
    # Definitionally equal?
    is_equal = kernel.is_def_eq(two_times_three, six_expr)
    print(f"2 × 3 = 6: {is_equal}")
    
    # Try a more complex expression: (2 + 3) × (1 + 1)
    print("\nEvaluating (2 + 3) × (1 + 1):")
    complex_expr = mk_app(
        mk_app(
            mk_const("mul"),
            mk_app(mk_app(mk_const("add"), mk_const("two")), mk_const("three"))
        ),
        mk_app(mk_app(mk_const("add"), mk_const("one")), mk_const("one"))
    )
    print(f"Expression: {complex_expr}")
    
    # Reduce with different strategies
    reduced_whnf = kernel.reduce(complex_expr, ReductionStrategy.WHNF)
    print(f"WHNF (partial reduction): expression too large to display")
    
    reduced_nf = kernel.normalize(complex_expr)
    print(f"NF (full reduction): expression too large to display")
    
    # Check if (2 + 3) × (1 + 1) = 10
    ten_expr = mk_app(mk_const("succ"), 
               mk_app(mk_const("succ"), 
               mk_app(mk_const("succ"), 
               mk_app(mk_const("succ"), 
               mk_app(mk_const("succ"), 
               mk_app(mk_const("succ"), 
               mk_app(mk_const("succ"), 
               mk_app(mk_const("succ"), 
               mk_app(mk_const("succ"), 
               mk_app(mk_const("succ"), mk_const("zero")))))))))))
    
    is_equal = kernel.is_def_eq(complex_expr, ten_expr)
    print(f"(2 + 3) × (1 + 1) = 10: {is_equal}")
    print()
    
    # Demonstration of reduction modes
    print("Step C: Demonstration of different reduction modes")
    print("-----------------------------------------------")
    
    # Create an expression with different types of redexes
    # let x := zero in (λy. y) (succ x)
    expr = mk_let(
        "x",
        zero_type,
        mk_const("zero"),
        mk_app(
            mk_lambda("y", zero_type, mk_var(0)),
            mk_app(mk_const("succ"), mk_var(0))
        )
    )
    print(f"Expression: {expr}")
    
    # Beta reduction only - won't unfold the let or reduce "succ zero"
    beta_only = reduce(expr, kernel.env, ReductionStrategy.WHNF, ReductionMode.BETA)
    print(f"Beta reduction only: {beta_only}")
    
    # Zeta reduction only - will unfold the let but won't reduce "succ zero" or the lambda
    zeta_only = reduce(expr, kernel.env, ReductionStrategy.WHNF, ReductionMode.ZETA)
    print(f"Zeta reduction only: {zeta_only}")
    
    # All reductions - will do everything
    all_reductions = kernel.normalize(expr)
    print(f"All reductions: {all_reductions}")
    print()
    
    print("Demo completed successfully!")


def define_nat_type(kernel):
    """Define the type for Church numerals."""
    # Nat = Π(s: Nat -> Nat), Π(z: Nat), Nat
    return mk_pi(
        "s",
        mk_pi(Name(), mk_const("Nat"), mk_const("Nat")),
        mk_pi("z", mk_const("Nat"), mk_const("Nat"))
    )


if __name__ == "__main__":
    main() 