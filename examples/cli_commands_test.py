#!/usr/bin/env python3
"""
Test script for Pylean CLI commands.
This script tests various commands by directly calling methods on the CLI object
rather than relying on interactive input.
"""

from pylean.cli import PyleanREPL
from pylean.kernel import (
    Name, Level, Expr, ExprKind,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, ReductionStrategy, ReductionMode
)
from pylean.kernel.tactic import init_tactic_state
from pylean.kernel.tactic_repl import TacticREPL
from pylean.elaborator import ElaborationContext

def run_test(title, fn):
    """Run a test with a title header."""
    print(f"\n{'=' * 50}")
    print(f"TEST: {title}")
    print(f"{'=' * 50}")
    fn()
    print("Test completed")

def test_parse_commands():
    """Test various parse commands."""
    repl = PyleanREPL()
    
    print("\n1. Parsing lambda expression:")
    repl.cmd_parse("λ x : Nat, x + 1")
    
    print("\n2. Parsing lambda expression with parentheses:")
    repl.cmd_parse("λ (x : Nat), x + 1")
    
    print("\n3. Attempt to parse Pi type (this will likely fail):")
    repl.cmd_parse("Π (A : Type), A -> A")
    
    print("\n4. Attempt to parse Pi type without symbol (this will likely fail):")
    repl.cmd_parse("forall (A : Type), A -> A")
    
    print("\n5. Attempt to parse arrow type (this will likely fail):")
    repl.cmd_parse("A -> A")

def test_reduce_commands():
    """Test various reduce commands."""
    repl = PyleanREPL()
    
    print("\n1. Reducing lambda application:")
    repl.cmd_reduce("(λ x : Nat, x) 0")
    
    print("\n2. Reducing simple expression:")
    repl.cmd_reduce("1 + 1")

def test_tactic_command():
    """Test tactic command with direct kernel creation."""
    # Create kernel and id_type directly since parsing Pi types seems problematic
    kernel = Kernel()
    
    # Create the identity function type: Π (A : Type), A -> A
    # First, create the inner function type: A -> A 
    inner_type = mk_pi("x", mk_var(0), mk_var(1))  # Π (x : A), A
    
    # Now create the outer Pi type: Π (A : Type), inner_type
    id_type = mk_pi("A", mk_sort(0), inner_type)  # Π (A : Type), Π (x : A), A
    
    print(f"\nCreated identity function type directly: {id_type}")
    
    # Create the tactic REPL object (initialized with goal_type)
    tactic_repl = TacticREPL(kernel, id_type)
    
    # Show initial state
    print("\nInitial tactic state:")
    tactic_repl.do_show("")
    
    # Test intro tactic
    print("\nRunning 'intro A' tactic:")
    tactic_repl.do_intro("A")
    
    print("\nRunning 'intro x' tactic:")
    tactic_repl.do_intro("x")
    
    print("\nRunning 'exact x' tactic:")
    tactic_repl.do_exact("x")

def main():
    """Run all CLI command tests."""
    print("Pylean CLI Command Tests")
    print("=======================")
    
    run_test("Parse Commands", test_parse_commands)
    run_test("Reduce Commands", test_reduce_commands)
    run_test("Tactic Command", test_tactic_command)
    
    print("\nAll tests completed")

if __name__ == "__main__":
    main() 