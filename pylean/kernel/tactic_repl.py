"""
Tactics REPL for interactive theorem proving.

This module provides a Read-Eval-Print Loop (REPL) for
interactive theorem proving using the tactics system.
"""

import cmd
import re
import sys
from typing import Dict, List, Optional, Tuple

# Parser and Elaborator imports
from pylean.parser import parse_expression, ParseError, SyntaxNode
from pylean.elaborator import elaborate, ElaborationContext

# Import from kernel modules directly to avoid circular imports
from pylean.kernel.expr import (
    Expr, mk_const, mk_var, mk_sort, mk_app, mk_lambda, mk_pi
)
from pylean.kernel.env import Environment
from pylean.kernel.kernel import Kernel
from pylean.kernel.typecheck import Context, TypeError as KernelTypeError
from pylean.kernel.tactic import (
    Goal, TacticState, Tactic, TacticException,
    IntroTactic, ExactTactic, AssumptionTactic, ApplyTactic, 
    RewriteTactic, ByTactic, InductionTactic, init_tactic_state,
    CasesTactic, RflTactic, ExfalsoTactic, ContradictionTactic, SimpTactic,
    DestructTactic
)


class TacticREPL(cmd.Cmd):
    """
    Interactive REPL for applying tactics to prove theorems.
    
    This class implements a command-line interface for
    interactive theorem proving using the tactics system.
    """
    
    intro = "Pylean Tactics REPL - Type 'help' for available commands."
    prompt = "tactic> "
    
    def __init__(self, kernel: Kernel, goal_type: Expr):
        """
        Initialize the tactics REPL.
        
        Args:
            kernel: The kernel instance with an environment
            goal_type: The type to prove
        """
        super().__init__()
        self.kernel = kernel
        self.initial_goal_type = goal_type
        self.state = init_tactic_state(kernel.env, goal_type)
        self.history: List[TacticState] = [self.state]
        # No longer need var_map with the real parser/elaborator
        # self.var_map: Dict[str, int] = {}
        # self._update_var_map()
    
    # _update_var_map is no longer needed as the elaborator handles context
    # def _update_var_map(self) -> None:
    #     ...
    
    def _parse_expr(self, expr_str: str) -> Expr:
        """
        Parse an expression string into a kernel expression using the Pylean parser
        and elaborator.
        
        Args:
            expr_str: The expression string to parse
            
        Returns:
            The parsed and elaborated kernel expression
            
        Raises:
            ValueError: If parsing or elaboration fails.
        """
        expr_str = expr_str.strip()
        if not expr_str:
            raise ValueError("Expression string cannot be empty")
            
        try:
            # 1. Parse the expression string
            parsed_node = parse_expression(expr_str)
            if not isinstance(parsed_node, SyntaxNode):
                # Handle cases where parsing might return None or other types
                # Although parse_expression should ideally raise ParseError
                raise ValueError(f"Parsing failed for: {expr_str}")

            # 2. Create Elaboration Context
            # Use the context from the current goal if available
            current_ctx = Context() # Default empty context
            if self.state.goals:
                current_ctx = self.state.goals[0].ctx
            
            elab_context = ElaborationContext(self.kernel.env, current_ctx)
            
            # 3. Elaborate the parsed node
            elaborated_expr = elaborate(parsed_node, elab_context)
            return elaborated_expr
            
        except ParseError as e:
            raise ValueError(f"Parsing error: {e}")
        except KernelTypeError as e: # Catch kernel type errors from elaborator
            raise ValueError(f"Type error during elaboration: {e}")
        except Exception as e:
            # Catch any other unexpected errors during parsing/elaboration
            raise ValueError(f"Failed to parse or elaborate expression '{expr_str}': {e}")
    
    def emptyline(self) -> bool:
        """Handle empty line input."""
        return False  # Don't repeat the last command
    
    def do_show(self, arg: str) -> None:
        """Show the current proof state."""
        print(self.state)
    
    def do_intro(self, arg: str) -> None:
        """
        Introduce a variable from a Pi type.
        Usage: intro [name]
        """
        try:
            name = arg.strip() if arg.strip() else None
            tactic = IntroTactic(name)
            self.state = tactic.apply(self.state)
            self.history.append(self.state)
            print(self.state)
        except TacticException as e:
            print(f"Error: {e}")
    
    def do_exact(self, arg: str) -> None:
        """
        Provide an exact proof for the current goal.
        Usage: exact <expr>
        """
        if not arg.strip():
            print("Error: An expression is required")
            return
        
        try:
            expr = self._parse_expr(arg)
            tactic = ExactTactic(expr)
            self.state = tactic.apply(self.state)
            self.history.append(self.state)
            print(self.state)
        except (TacticException, ValueError) as e:
            print(f"Error: {e}")
    
    def do_assumption(self, arg: str) -> None:
        """
        Look for a hypothesis that matches the current goal.
        Usage: assumption
        """
        try:
            tactic = AssumptionTactic()
            self.state = tactic.apply(self.state)
            self.history.append(self.state)
            print(self.state)
        except TacticException as e:
            print(f"Error: {e}")
    
    def do_apply(self, arg: str) -> None:
        """
        Apply backward reasoning with an implication.
        Usage: apply <expr>
        """
        if not arg.strip():
            print("Error: An expression is required")
            return
        
        try:
            expr = self._parse_expr(arg)
            tactic = ApplyTactic(expr)
            self.state = tactic.apply(self.state)
            self.history.append(self.state)
            print(self.state)
        except (TacticException, ValueError) as e:
            print(f"Error: {e}")
    
    def do_rewrite(self, arg: str) -> None:
        """
        Rewrite the goal using an equality.
        Usage: rewrite [->|<-] <expr>
        """
        args = arg.strip().split()
        if not args:
            print("Error: An expression is required")
            return
        
        try:
            direction = "->"  # Default direction
            if args[0] in ["->", "<-"]:
                direction = args[0]
                expr_str = " ".join(args[1:])
            else:
                expr_str = arg
            
            expr = self._parse_expr(expr_str)
            tactic = RewriteTactic(expr, direction)
            self.state = tactic.apply(self.state)
            self.history.append(self.state)
            print(self.state)
        except (TacticException, ValueError) as e:
            print(f"Error: {e}")
    
    def do_induction(self, arg: str) -> None:
        """
        Apply induction on a variable of an inductive type.
        Usage: induction <var_name>
        """
        if not arg.strip():
            print("Error: A variable name is required")
            return
        
        try:
            var_name = arg.strip()
            tactic = InductionTactic(var_name)
            self.state = tactic.apply(self.state)
            self.history.append(self.state)
            print(self.state)
        except TacticException as e:
            print(f"Error: {e}")
    
    def do_cases(self, arg: str) -> None:
        """
        Apply case analysis on a variable of an inductive type.
        Usage: cases <var_name>
        """
        if not arg.strip():
            print("Error: A variable name is required")
            return
        
        try:
            var_name = arg.strip()
            tactic = CasesTactic(var_name)
            self.state = tactic.apply(self.state)
            self.history.append(self.state)
            print(self.state)
        except TacticException as e:
            print(f"Error: {e}")
    
    def do_rfl(self, arg: str) -> None:
        """
        Prove a reflexive equality (a = a).
        Usage: rfl
        """
        try:
            tactic = RflTactic()
            self.state = tactic.apply(self.state)
            self.history.append(self.state)
            print(self.state)
        except TacticException as e:
            print(f"Error: {e}")
    
    def do_exfalso(self, arg: str) -> None:
        """
        Convert the current goal to False for proof by contradiction.
        Usage: exfalso
        """
        try:
            tactic = ExfalsoTactic()
            self.state = tactic.apply(self.state)
            self.history.append(self.state)
            print(self.state)
        except TacticException as e:
            print(f"Error: {e}")
    
    def do_contradiction(self, arg: str) -> None:
        """
        Search for a contradiction in the context to solve the goal.
        Usage: contradiction
        """
        try:
            tactic = ContradictionTactic()
            self.state = tactic.apply(self.state)
            self.history.append(self.state)
            print(self.state)
        except TacticException as e:
            print(f"Error: {e}")
    
    def do_simp(self, arg: str) -> None:
        """
        Simplify the goal using rewrite rules.
        Usage: simp [rule1 rule2 ...]
        """
        rules = []
        if arg.strip():
            rule_names = arg.strip().split()
            for name in rule_names:
                try:
                    rule = self._parse_expr(name)
                    rules.append(rule)
                except Exception as e:
                    print(f"Error parsing rule '{name}': {e}")
                    return
        
        try:
            tactic = SimpTactic(rules=rules)
            self.state = tactic.apply(self.state)
            self.history.append(self.state)
            print(self.state)
        except TacticException as e:
            print(f"Error: {e}")
    
    def do_destruct(self, arg: str) -> None:
        """
        Destruct a hypothesis or the goal (currently supports And).
        Usage: destruct <hyp_name> [as [name1 name2 ...]]
               destruct goal
        """
        args = arg.strip().split()
        if not args:
            print("Error: Target name (hypothesis or 'goal') required")
            return
        
        target_name = args[0]
        new_names = []
        if len(args) > 1:
            if args[1] == "as" and len(args) > 2:
                # Remove brackets if present, e.g., [hp hq]
                names_str = " ".join(args[2:])
                names_str = names_str.strip("[]")
                new_names = names_str.split()
            elif args[1] != "as":
                 print("Error: Expected 'as' after target name for renaming")
                 return
                 
        try:
            tactic = DestructTactic(target_name, new_names)
            self.state = tactic.apply(self.state)
            self.history.append(self.state)
            # self._update_var_map() # Not needed anymore
            print(self.state)
        except TacticException as e:
            print(f"Error: {e}")
    
    def do_undo(self, arg: str) -> None:
        """
        Undo the last tactic application.
        Usage: undo
        """
        if len(self.history) <= 1:
            print("Nothing to undo")
            return
        
        self.history.pop()  # Remove current state
        self.state = self.history[-1]  # Go back to previous state
        print(self.state)
    
    def do_reset(self, arg: str) -> None:
        """
        Reset the proof to the initial state.
        Usage: reset
        """
        self.state = init_tactic_state(self.kernel.env, self.initial_goal_type)
        self.history = [self.state]
        print(self.state)
    
    def do_vars(self, arg: str) -> None:
        """
        Show the variables in the current context with their de Bruijn indices.
        Usage: vars
        """
        if not self.var_map:
            print("No variables in context")
            return
        
        print("Variables in current context:")
        for name, idx in self.var_map.items():
            print(f"  {name}: {idx}")
    
    def do_quit(self, arg: str) -> bool:
        """
        Quit the REPL.
        Usage: quit
        """
        print("Exiting Tactics REPL")
        return True
    
    def do_exit(self, arg: str) -> bool:
        """
        Exit the REPL.
        Usage: exit
        """
        return self.do_quit(arg)
    
    def do_help(self, arg: str) -> None:
        """Show help for commands."""
        if not arg.strip():
            print("\nAvailable commands:")
            print("  show       - Show the current proof state")
            print("  intro      - Introduce a variable from a Pi type")
            print("  exact      - Provide an exact proof for the current goal")
            print("  assumption - Look for a hypothesis that matches the current goal")
            print("  apply      - Apply backward reasoning with an implication")
            print("  rewrite    - Rewrite the goal using an equality")
            print("  induction  - Apply induction on a variable of an inductive type")
            print("  cases      - Apply case analysis on a variable of an inductive type")
            print("  rfl        - Prove a reflexive equality (a = a)")
            print("  exfalso    - Convert the current goal to False for proof by contradiction")
            print("  contradiction - Search for a contradiction in the context to solve the goal")
            print("  simp       - Simplify the goal using rewrite rules")
            print("  destruct   - Destruct a hypothesis or the goal (currently supports And)")
            print("  undo       - Undo the last tactic application")
            print("  reset      - Reset the proof to the initial state")
            print("  vars       - Show variables in the current context")
            print("  quit/exit  - Exit the REPL")
            print("  help       - Show this help message or help for a specific command")
            print("\nFor help on a specific command, type 'help <command>'")
        else:
            super().do_help(arg)


def mk_tactic_repl(env: Environment, goal_type: Expr) -> TacticREPL:
    """
    Create a new tactic REPL.
    
    Args:
        env: The environment
        goal_type: The type to prove
        
    Returns:
        A new tactic REPL
    """
    # Create a kernel instance with the environment
    kernel = Kernel(env)
    
    # Create and return a new tactic REPL
    return TacticREPL(kernel, goal_type)


def start_tactic_repl(kernel: Kernel, goal_type: Expr) -> None:
    """
    Start a tactic REPL for interactive theorem proving.
    
    Args:
        kernel: The kernel instance
        goal_type: The type to prove
    """
    repl = TacticREPL(kernel, goal_type)
    repl.cmdloop()


if __name__ == "__main__":
    # Simple example usage if run directly
    kernel = Kernel()
    
    # Define the identity theorem for Prop: Π (A : Prop), A → A
    identity_type = mk_pi(
        "A",
        mk_sort(0),
        mk_pi(
            "a",
            mk_var(0),
            mk_var(1)
        )
    )
    
    start_tactic_repl(kernel, identity_type) 