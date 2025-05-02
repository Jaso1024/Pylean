#!/usr/bin/env python3
"""
Demo of the Pylean command-line interface.

This example shows how to use the Pylean CLI to interact
with the theorem prover from the command line.
"""

import sys
import os
import subprocess

# Add the project root to the Python path if running from examples directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pylean.cli import PyleanREPL


def main():
    """Run the CLI demo."""
    print("PyLean CLI Demo")
    print("===============")
    
    # Demonstrate how to use the CLI programmatically
    print("\nMethod 1: Using the CLI programmatically")
    print("---------------------------------------")
    print("You can use the PyleanREPL class directly in Python code:")
    print("```python")
    print("from pylean.cli import PyleanREPL")
    print("repl = PyleanREPL()")
    print("repl.run()")
    print("```")
    
    # Demonstrate how to use the CLI from the command line
    print("\nMethod 2: Using the CLI from the command line")
    print("-------------------------------------------")
    print("You can run the CLI directly from the command line:")
    print("```bash")
    print("python -m pylean.cli")
    print("```")
    
    # Show examples of CLI commands
    print("\nExamples of CLI commands:")
    print("------------------------")
    print(":help             - Display help information")
    print(":kernel           - Show kernel information")
    print(":env              - List environment declarations")
    print(":tactic Π (A : Type), A -> A - Enter tactic mode to prove identity theorem")
    print(":type Nat         - Show the type of Nat")
    print(":reduce (λ x : Nat, x) 0 - Reduce a lambda application")
    print(":parse λ (x : Nat), x - Show the syntax tree of a lambda expression")
    
    # Demonstrate interactive usage
    print("\nInteractive Demo")
    print("---------------")
    print("Would you like to start an interactive CLI session? (y/n)")
    choice = input("> ")
    
    if choice.lower().startswith('y'):
        try:
            print("\nStarting interactive Pylean REPL...\n")
            repl = PyleanREPL()
            repl.run()
        except Exception as e:
            print(f"Error starting REPL: {e}")
    else:
        print("\nSkipping interactive demo.")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main() 