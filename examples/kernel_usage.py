#!/usr/bin/env python3
"""
Example script demonstrating the usage of the Pylean kernel module.

This script creates various expressions and manipulates them using the
utility functions provided by the kernel module.
"""

from pylean.kernel import (
    ExprKind, Name, Level, BinderInfo,
    Expr, VarExpr, SortExpr, ConstExpr, AppExpr, LambdaExpr, PiExpr, LetExpr, MetaExpr, LocalExpr,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi, mk_let, mk_meta, mk_local,
    occurs_in, lift, instantiate
)


def main():
    """Main function demonstrating kernel module usage."""
    print("Pylean Kernel Module Example")
    print("===========================\n")
    
    # Create basic types
    print("Creating basic types:")
    prop = mk_sort(0)
    type1 = mk_sort(1)
    print(f"  Prop: {prop}")
    print(f"  Type 1: {type1}")
    print()
    
    # Create some constants
    print("Creating constants:")
    nat = mk_const("Nat")
    bool_type = mk_const("Bool")
    true = mk_const("true")
    add = mk_const("add")
    print(f"  nat: {nat}")
    print(f"  bool: {bool_type}")
    print(f"  true: {true}")
    print()
    
    # Create a simple function type: Nat → Nat
    print("Creating function types:")
    nat_fn_nat = mk_pi(Name(), nat, nat)
    print(f"  Nat → Nat: {nat_fn_nat}")
    
    # Create a dependent function type: Π(n : Nat), Vec Nat n
    vec = mk_const("Vec")
    n = mk_local("n", nat)
    vec_nat_n = mk_app(mk_app(vec, nat), mk_var(0))  # Vec Nat n
    vec_type = mk_pi("n", nat, vec_nat_n)
    print(f"  Π(n : Nat), Vec Nat n: {vec_type}")
    print()
    
    # Create an application: add 1 2
    print("Creating applications:")
    one = mk_const("1")
    two = mk_const("2")
    add_1_2 = mk_app(mk_app(add, one), two)
    print(f"  add 1 2: {add_1_2}")
    print()
    
    # Create a lambda expression: λ(x : Nat), add x 1
    print("Creating lambda expressions:")
    x_body = mk_app(mk_app(add, mk_var(0)), one)
    lambda_x = mk_lambda("x", nat, x_body)
    print(f"  λ(x : Nat), add x 1: {lambda_x}")
    
    # Create a lambda with implicit parameter: λ{x : Nat}, x
    implicit_binder = BinderInfo(is_implicit=True)
    lambda_implicit = mk_lambda("x", nat, mk_var(0), implicit_binder)
    print(f"  λ{{x : Nat}}, x: {lambda_implicit}")
    print()
    
    # Demonstrate expression utilities
    print("Demonstrating utility functions:")
    
    # Check if variable occurs in expression
    app_expr = mk_app(mk_var(1), mk_var(0))
    print(f"  Expression: {app_expr}")
    print(f"  Variable #0 occurs: {occurs_in(0, app_expr)}")
    print(f"  Variable #1 occurs: {occurs_in(1, app_expr)}")
    print(f"  Variable #2 occurs: {occurs_in(2, app_expr)}")
    print()
    
    # Lift variables in an expression
    var_expr = mk_var(0)
    lifted_expr = lift(var_expr, 2, 0)
    print(f"  Original: {var_expr}")
    print(f"  Lifted by 2: {lifted_expr}")
    print()
    
    # Instantiate a variable in an expression
    fn_body = mk_app(mk_var(1), mk_var(0))  # f x
    fn_value = mk_const("g")
    result = instantiate(fn_body, fn_value, 1)
    print(f"  Original: {fn_body}")
    print(f"  After replacing #1 with 'g': {result}")
    print()
    
    # More complex example: beta reduction of (λx.x+1) 5
    print("Beta reduction example:")
    # Create λx.x+1
    x_plus_1 = mk_app(mk_app(add, mk_var(0)), one)
    lambda_x_plus_1 = mk_lambda("x", nat, x_plus_1)
    # Create (λx.x+1) 5
    five = mk_const("5")
    app_lambda_5 = mk_app(lambda_x_plus_1, five)
    print(f"  (λx.x+1) 5: {app_lambda_5}")
    
    # Perform beta reduction by instantiating the lambda body with the argument
    reduced = instantiate(lambda_x_plus_1.body, five)
    print(f"  Reduced: {reduced}")
    print()


if __name__ == "__main__":
    main() 