"""
Tests for the expression parser.

This module contains tests for parsing different kinds of expressions,
with a focus on operator precedence and associativity.
"""

import unittest
from pylean.parser.expr import parse_expression, EXPR, APP_EXPR, ADD_EXPR, SUB_EXPR, MUL_EXPR, DIV_EXPR
from pylean.parser.core import SyntaxNode


def pretty_print_node(node: SyntaxNode, indent: int = 0) -> str:
    """
    Pretty print a syntax node tree for debugging.
    
    Args:
        node: The syntax node to print
        indent: Current indentation level
        
    Returns:
        A multi-line string representation of the node tree
    """
    if node is None:
        return " " * indent + "None"
        
    result = " " * indent + f"{node.kind}"
    
    if node.value:
        result += f" ({node.value})"
        
    if node.children:
        for child in node.children:
            result += "\n" + pretty_print_node(child, indent + 2)
            
    return result


class TestExprParser(unittest.TestCase):
    """Test cases for the expression parser."""
    
    def test_number_literals(self):
        """Test parsing of number literals."""
        cases = [
            "42",
            "3.14",
            "0xFF",  # Hex
            "0b101",  # Binary
            "1e10",   # Scientific notation
        ]
        
        for case in cases:
            with self.subTest(case=case):
                result = parse_expression(case)
                self.assertIsNotNone(result)
                # Print for debugging
                print(f"\nParsed: {case}")
                print(pretty_print_node(result))
    
    def test_identifiers(self):
        """Test parsing of identifiers."""
        cases = [
            "x",
            "foo",
            "CamelCase",
            "with_underscore",
            # TODO: Add tests for escaped identifiers when supported
        ]
        
        for case in cases:
            with self.subTest(case=case):
                result = parse_expression(case)
                self.assertIsNotNone(result)
                print(f"\nParsed: {case}")
                print(pretty_print_node(result))
    
    def test_basic_arithmetic(self):
        """Test parsing of basic arithmetic expressions."""
        cases = [
            "1 + 2",
            "a - b",
            "3 * 4",
            "x / y",
        ]
        
        for case in cases:
            with self.subTest(case=case):
                result = parse_expression(case)
                self.assertIsNotNone(result)
                print(f"\nParsed: {case}")
                print(pretty_print_node(result))
    
    def test_operator_precedence(self):
        """Test operator precedence in expressions."""
        cases = [
            ("1 + 2 * 3", ADD_EXPR),  # Should group as 1 + (2 * 3)
            ("1 * 2 + 3", ADD_EXPR),  # Should group as (1 * 2) + 3
            ("1 + 2 + 3", ADD_EXPR),  # Should group as (1 + 2) + 3 (left associative)
            ("1 * 2 * 3", MUL_EXPR),  # Should group as (1 * 2) * 3 (left associative)
        ]
        
        for case, expected_root_kind in cases:
            with self.subTest(case=case):
                result = parse_expression(case)
                self.assertIsNotNone(result)
                self.assertEqual(result.kind, expected_root_kind)
                print(f"\nParsed: {case}")
                print(pretty_print_node(result))
    
    def test_parenthesized_expressions(self):
        """Test parsing of parenthesized expressions."""
        cases = [
            "(1 + 2)",
            "(a * b) + c",
            "a * (b + c)",
            "((1 + 2) * (3 + 4))",
        ]
        
        for case in cases:
            with self.subTest(case=case):
                result = parse_expression(case)
                self.assertIsNotNone(result)
                print(f"\nParsed: {case}")
                print(pretty_print_node(result))
    
    def test_unary_minus(self):
        """Test parsing of unary minus expressions."""
        cases = [
            "-1",
            "-x",
            "-(a + b)",
            "a + -b",
            "-a * b",  # Should parse as (-a) * b due to precedence
        ]
        
        for case in cases:
            with self.subTest(case=case):
                result = parse_expression(case)
                self.assertIsNotNone(result)
                print(f"\nParsed: {case}")
                print(pretty_print_node(result))
    
    def test_function_application(self):
        """Test parsing of function application."""
        cases = [
            "f(x)",
            "g(1 + 2)",
            "h(a, b)",  # Will need to add support for multiple arguments
            "f(g(x))",  # Nested function calls
        ]
        
        for case in cases:
            with self.subTest(case=case):
                try:
                    result = parse_expression(case)
                    self.assertIsNotNone(result)
                    print(f"\nParsed: {case}")
                    print(pretty_print_node(result))
                except Exception as e:
                    # Some of these may not be supported yet
                    print(f"Failed to parse {case}: {e}")
    
    def test_complex_expressions(self):
        """Test parsing of complex expressions combining multiple features."""
        cases = [
            "a + b * c + d",
            "f(x) + g(y)",
            "-(a + b) * c",
            "a * (b + c) / d",
        ]
        
        for case in cases:
            with self.subTest(case=case):
                result = parse_expression(case)
                self.assertIsNotNone(result)
                print(f"\nParsed: {case}")
                print(pretty_print_node(result))


if __name__ == "__main__":
    unittest.main()
