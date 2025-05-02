"""
Parsing Lean4 Code with Pylean

This example demonstrates how to use the Pylean parser module to parse Lean4 code.
"""

from pylean.parser import parse_expr

def main():
    """Run the parsing examples and print the results."""
    # Simple arithmetic expression
    print("Parsing: 1 + 2 * 3")
    expr1 = parse_expr("1 + 2 * 3")
    print(f"Result: {expr1}")
    print()

    # Lambda expression
    print("Parsing: λx : Nat, x + 1")
    expr2 = parse_expr("λx : Nat, x + 1")
    print(f"Result: {expr2}")
    print()

    # Nested lambdas
    print("Parsing: λ(f : Nat → Nat) (x : Nat), f (f x)")
    expr3 = parse_expr("λ(f : Nat → Nat) (x : Nat), f (f x)")
    print(f"Result: {expr3}")
    print()

    # Function application
    print("Parsing: f x y z")
    expr4 = parse_expr("f x y z")
    print(f"Result: {expr4}")
    print()

    # Parenthesized expressions
    print("Parsing: f (g x) (h y z)")
    expr5 = parse_expr("f (g x) (h y z)")
    print(f"Result: {expr5}")
    print()

if __name__ == "__main__":
    main() 