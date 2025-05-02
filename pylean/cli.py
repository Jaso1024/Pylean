#!/usr/bin/env python3
"""
Command-line interface for Pylean.

This module provides a command-line interface for interacting with
the Pylean theorem prover, including a REPL for interactive theorem proving.
"""

import argparse
import sys
import os
import subprocess
from typing import List, Optional, Dict, Any
import re

from pylean.kernel import (
    Expr, Name, Level, ExprKind,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, Environment, ReductionStrategy, ReductionMode
)
from pylean.kernel.tactic_repl import TacticREPL, start_tactic_repl
from pylean.parser.core import (
    ParserContext, ParserState, InputContext,
    SyntaxNode, Parser
)
from pylean.parser.expr import parse_expression
from pylean.elaborator import ElaborationContext, elaborate


class PyleanREPL:
    """
    Interactive REPL for Pylean that integrates parser, elaborator, and kernel.
    """
    
    WELCOME_MESSAGE = """
    ╔════════════════════════════════════════╗
    ║  Pylean Interactive Theorem Prover     ║
    ║  Version 0.1.0                         ║
    ║  Type :help for available commands     ║
    ║  Type :quit to exit                    ║
    ╚════════════════════════════════════════╝
    """
    
    PROMPT = "pylean> "
    
    def __init__(self):
        """Initialize the REPL with a kernel and standard environment."""
        self.kernel = Kernel()
        self.elab_context = ElaborationContext(self.kernel.env)
        self.history: List[str] = []
        self.commands: Dict[str, Any] = {
            ":help": self.cmd_help,
            ":quit": self.cmd_quit,
            ":exit": self.cmd_quit,
            ":kernel": self.cmd_kernel_info,
            ":env": self.cmd_env,
            ":tactic": self.cmd_tactic,
            ":reduce": self.cmd_reduce,
            ":type": self.cmd_type,
            ":parse": self.cmd_parse,
        }
    
    def run(self):
        """Start the REPL."""
        print(self.WELCOME_MESSAGE)
        
        while True:
            try:
                line = input(self.PROMPT)
                self.history.append(line)
                
                if not line.strip():
                    continue
                
                # Handle commands (starting with :)
                if line.strip().startswith(":"):
                    command = line.strip().split()[0]
                    args = line.strip()[len(command):].strip()
                    
                    if command in self.commands:
                        self.commands[command](args)
                    else:
                        print(f"Unknown command: {command}")
                        print("Type :help for available commands")
                else:
                    # Process as expression
                    self.process_expression(line)
            except KeyboardInterrupt:
                print("\nUse :quit to exit")
            except EOFError:
                print("\nExiting Pylean")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def process_expression(self, expr_str: str):
        """
        Process an expression string: parse -> elaborate -> type check.
        
        Args:
            expr_str: The expression string to process
        """
        try:
            # Parse the expression directly using parse_expression
            try:
                syntax_node = parse_expression(expr_str)
            except SyntaxError as e:
                print(f"Parse error: {e}")
                return
            except ValueError as e:
                print(f"Parse error: {e}")
                return
            
            # Elaborate the expression
            try:
                expr = elaborate(syntax_node, self.elab_context)
                expr_type = self.kernel.infer_type(expr)
                
                print(f"Expression: {expr}")
                print(f"Type: {expr_type}")
            except Exception as e:
                print(f"Elaboration/type checking error: {e}")
        except Exception as e:
            print(f"Error processing expression: {e}")
    
    def cmd_help(self, args: str):
        """Display help information."""
        print("Available commands:")
        print("  :help             - Display this help message")
        print("  :quit, :exit      - Exit the REPL")
        print("  :kernel           - Display kernel information")
        print("  :env              - Display environment information")
        print("  :tactic           - Enter tactic mode for theorem proving")
        print("  :reduce <expr>    - Reduce an expression")
        print("  :type <expr>      - Infer the type of an expression")
        print("  :parse <expr>     - Parse an expression and show the syntax tree")
        print("\nFor any other input, Pylean will try to parse, elaborate, and type check it.")
    
    def cmd_quit(self, args: str):
        """Exit the REPL."""
        print("Exiting Pylean")
        sys.exit(0)
    
    def cmd_kernel_info(self, args: str):
        """Display information about the kernel."""
        print("Kernel Information:")
        print(f"  Environment size: {len(self.kernel.env.declarations)} declarations")
        
        # TODO: Add more kernel information
    
    def cmd_env(self, args: str):
        """Display information about the environment."""
        print("Environment Declarations:")
        
        for name, decl in self.kernel.env.declarations.items():
            print(f"  {name}: {decl.type}")
        
        if args.strip():
            # Look up specific declaration
            name = args.strip()
            decl = self.kernel.env.find_decl(name)
            if decl:
                print(f"\nDeclaration for {name}:")
                print(f"  Type: {decl.type}")
                print(f"  Kind: {decl.__class__.__name__}")
                if hasattr(decl, "value"):
                    print(f"  Value: {decl.value}")
            else:
                print(f"\nDeclaration '{name}' not found")
    
    def cmd_tactic(self, args: str):
        """Enter tactic mode for interactive theorem proving."""
        if not args.strip():
            print("Usage: :tactic <theorem_type>")
            return
        
        try:
            # Parse the theorem type directly
            try:
                syntax_node = parse_expression(args)
            except SyntaxError as e:
                print(f"Parse error: {e}")
                return
            except ValueError as e:
                print(f"Parse error: {e}")
                return
            
            theorem_type = elaborate(syntax_node, self.elab_context)
            
            # Start tactic REPL
            print(f"\nStarting tactic mode for theorem: {theorem_type}")
            start_tactic_repl(self.kernel, theorem_type)
            
            print("\nExited tactic mode")
        except Exception as e:
            print(f"Error starting tactic mode: {e}")
    
    def cmd_reduce(self, args: str):
        """Reduce an expression."""
        if not args.strip():
            print("Usage: :reduce <expression>")
            return
        
        try:
            # Check if this is a lambda application like "(λ x : Nat, x) 0"
            lambda_application_match = re.match(r'\s*\(\s*λ\s*.*\)\s*(\S+)\s*$', args)
            
            if lambda_application_match:
                # This is a lambda application
                # Find the closing parenthesis for the lambda
                lambda_end_idx = args.rindex(')')
                lambda_expr_str = args[:lambda_end_idx+1].strip()
                arg_str = args[lambda_end_idx+1:].strip()
                
                # Parse and elaborate the lambda expression
                try:
                    lambda_node = parse_expression(lambda_expr_str)
                    lambda_expr = elaborate(lambda_node, self.elab_context)
                    
                    # Parse and elaborate the argument
                    arg_node = parse_expression(arg_str)
                    arg_expr = elaborate(arg_node, self.elab_context)
                    
                    # Create the application expression
                    app_expr = mk_app(lambda_expr, arg_expr)
                    
                    # Reduce the application expression
                    reduced_expr = self.kernel.reduce(app_expr, strategy=ReductionStrategy.NF)
                    
                    print(f"Original: {app_expr}")
                    print(f"Reduced:  {reduced_expr}")
                    return
                except Exception as e:
                    print(f"Error handling lambda application: {e}")
                    # Continue with normal processing as fallback
            
            # Normal processing for other expressions
            try:
                syntax_node = parse_expression(args)
            except SyntaxError as e:
                print(f"Parse error: {e}")
                return
            except ValueError as e:
                print(f"Parse error: {e}")
                return
            
            expr = elaborate(syntax_node, self.elab_context)
            
            # Reduce the expression
            reduced_expr = self.kernel.reduce(
                expr, 
                strategy=ReductionStrategy.NF  # Normal form
            )
            
            print(f"Original: {expr}")
            print(f"Reduced:  {reduced_expr}")
        except Exception as e:
            print(f"Error reducing expression: {e}")
    
    def cmd_type(self, args: str):
        """Infer the type of an expression."""
        if not args.strip():
            print("Usage: :type <expression>")
            return
        
        try:
            # Parse the expression directly
            try:
                syntax_node = parse_expression(args)
            except SyntaxError as e:
                print(f"Parse error: {e}")
                return
            except ValueError as e:
                print(f"Parse error: {e}")
                return
            
            expr = elaborate(syntax_node, self.elab_context)
            expr_type = self.kernel.infer_type(expr)
            
            print(f"Expression: {expr}")
            print(f"Type: {expr_type}")
        except Exception as e:
            print(f"Error inferring type: {e}")
    
    def cmd_parse(self, args: str):
        """Parse an expression and display the syntax tree."""
        if not args.strip():
            print("Usage: :parse <expression>")
            return
        
        try:
            # Parse the expression directly
            try:
                syntax_node = parse_expression(args)
            except SyntaxError as e:
                print(f"Parse error: {e}")
                return
            except ValueError as e:
                print(f"Parse error: {e}")
                return
            
            # Print syntax tree
            self._print_syntax_tree(syntax_node)
        except Exception as e:
            print(f"Error parsing expression: {e}")
    
    def _print_syntax_tree(self, node: SyntaxNode, indent: int = 0, is_last: bool = True):
        """Print a syntax tree with indentation."""
        prefix = "    " * indent
        if indent > 0:
            prefix = prefix[:-4] + ("└── " if is_last else "├── ")
        
        # Print node info
        value_str = f" = '{node.value}'" if node.value else ""
        print(f"{prefix}{node.kind}{value_str}")
        
        # Print children
        if node.children:
            for i, child in enumerate(node.children):
                is_last_child = (i == len(node.children) - 1)
                self._print_syntax_tree(child, indent + 1, is_last_child)


def run_demo(demo_name: str = None):
    """
    Run a demo to showcase Pylean features.
    
    Args:
        demo_name: Optional name of a specific demo to run
    """
    # Get the examples directory
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
    
    if not os.path.isdir(examples_dir):
        print(f"Error: Examples directory not found: {examples_dir}")
        return
    
    # List of available demos
    available_demos = {
        "pattern": "advanced_pattern_matching_demo.py",
        "tailcall": "tail_call_optimization_demo.py",
        "ffi": "ffi_demo.py",
        "typeclass": "typeclass_demo.py",
        "tactics": "tactics_demo.py",
        "all": None  # Special case to run all demos
    }
    
    if demo_name and demo_name not in available_demos:
        print(f"Error: Unknown demo '{demo_name}'")
        print("Available demos:")
        for name in available_demos:
            if name != "all":
                print(f"  {name}: {available_demos[name]}")
        return
    
    def run_demo_file(demo_file):
        demo_path = os.path.join(examples_dir, demo_file)
        if not os.path.isfile(demo_path):
            print(f"Warning: Demo file not found: {demo_path}")
            return False
            
        print(f"\n{'-' * 60}")
        print(f"Running demo: {demo_file}")
        print(f"{'-' * 60}\n")
        
        # Run the demo script
        result = subprocess.run([sys.executable, demo_path], capture_output=False)
        return result.returncode == 0
    
    if demo_name == "all" or demo_name is None:
        # Run all demos
        for name, demo_file in available_demos.items():
            if name != "all" and demo_file:
                run_demo_file(demo_file)
    else:
        # Run specific demo
        run_demo_file(available_demos[demo_name])


def main():
    """Main entry point for the Pylean CLI."""
    parser = argparse.ArgumentParser(description="Pylean Theorem Prover")
    
    parser.add_argument("--version", action="store_true", help="Print version information")
    parser.add_argument("--repl", action="store_true", help="Start REPL mode (default if no arguments)")
    parser.add_argument("--file", type=str, help="Execute commands from a file")
    parser.add_argument("--demo", type=str, nargs='?', const="all", 
                        help="Run a demonstration (options: pattern, tailcall, ffi, typeclass, tactics, all)")
    
    args = parser.parse_args()
    
    if args.version:
        print("Pylean Theorem Prover v1.0.0")
        return
    
    if args.demo:
        run_demo(args.demo)
        return
    
    if args.file:
        print(f"Executing commands from file: {args.file}")
        # TODO: Implement file execution
        print("File execution not yet implemented")
        return
    
    # Default: start REPL
    repl = PyleanREPL()
    repl.run()


if __name__ == "__main__":
    main() 