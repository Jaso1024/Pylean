"""
Tactics module for the Pylean theorem prover.

This module implements a basic tactics system for constructing proofs
step-by-step, inspired by Lean's tactics language.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable, Set

from pylean.kernel.expr import (
    Expr, Name, ExprKind, SortExpr, ConstExpr, AppExpr, LambdaExpr, PiExpr,
    mk_sort, mk_const, mk_app, mk_lambda, mk_pi, mk_let, mk_var
)
from pylean.kernel.env import Environment, DeclKind
from pylean.kernel.typecheck import Context, infer_type, check_type
from pylean.kernel.reduce import reduce, is_def_eq, ReductionStrategy


@dataclass
class Goal:
    """
    Represents a proof goal.
    
    A goal consists of a context (local hypotheses) and
    a target type that we need to prove.
    """
    ctx: Context
    target: Expr
    
    def __str__(self) -> str:
        """String representation of the goal."""
        hyps = []
        # The context stores names and types separately
        for i in range(len(self.ctx.types)):
            name = self.ctx.names[i]
            type_expr = self.ctx.types[i]
            hyps.append(f"{name} : {type_expr}")
        
        hyps_str = "\n".join(hyps)
        if hyps_str:
            return f"{hyps_str}\n⊢ {self.target}"
        else:
            return f"⊢ {self.target}"


@dataclass
class TacticState:
    """
    Represents the state of a tactic proof.
    
    Contains the environment, the current list of goals,
    and a partial proof term.
    """
    env: Environment
    goals: List[Goal]
    proof: Optional[Expr] = None
    
    def __str__(self) -> str:
        """String representation of the tactic state."""
        if not self.goals:
            return "No goals remaining."
        
        result = []
        for i, goal in enumerate(self.goals):
            if i == 0:
                result.append(f"Current goal:\n{goal}")
            else:
                result.append(f"Goal {i + 1}:\n{goal}")
        
        return "\n\n".join(result)


class TacticException(Exception):
    """Exception raised for tactic-related errors."""
    pass


class Tactic:
    """
    Base class for tactics.
    
    A tactic transforms a tactic state into a new tactic state,
    typically by transforming the current goal.
    """
    
    def apply(self, state: TacticState) -> TacticState:
        """
        Apply the tactic to the given state.
        
        Args:
            state: The current tactic state
        
        Returns:
            A new tactic state after applying the tactic
        
        Raises:
            TacticException: If the tactic cannot be applied
        """
        raise NotImplementedError("Tactics must implement apply method")


class IntroTactic(Tactic):
    """
    Implements the 'intro' tactic that introduces a new hypothesis.
    
    For a goal of the form (Π (x : A), B), this tactic introduces a
    new variable into the context and changes the goal to B.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the intro tactic.
        
        Args:
            name: Optional name for the introduced variable
        """
        self.name = name
    
    def apply(self, state: TacticState) -> TacticState:
        """Apply the intro tactic to the current state."""
        if not state.goals:
            raise TacticException("No goals to apply intro tactic")
        
        current_goal = state.goals[0]
        target = current_goal.target
        
        # The target must be a pi type for intro to work
        if target.kind != ExprKind.PI:
            raise TacticException("Cannot apply intro tactic to non-function type")
        
        pi_expr = target
        var_name = self.name if self.name else str(pi_expr.name)
        var_type = pi_expr.type  # The domain type in a Pi expression
        
        # Create new context with the variable added
        new_ctx = current_goal.ctx.extend(var_name, var_type)
        
        # Create new goal with updated context and target
        new_goal = Goal(ctx=new_ctx, target=pi_expr.body)
        
        # Update the goals in the state
        new_goals = [new_goal] + state.goals[1:]
        
        return TacticState(env=state.env, goals=new_goals, proof=state.proof)


class ExactTactic(Tactic):
    """
    Implements the 'exact' tactic that provides a direct proof.
    
    Given an expression that matches the goal type, this tactic
    completes the current goal.
    """
    
    def __init__(self, expr: Expr):
        """
        Initialize the exact tactic.
        
        Args:
            expr: The expression that should exactly match the goal type
        """
        self.expr = expr
    
    def apply(self, state: TacticState) -> TacticState:
        """Apply the exact tactic to the current state."""
        if not state.goals:
            raise TacticException("No goals to apply exact tactic")
        
        current_goal = state.goals[0]
        target = current_goal.target
        ctx = current_goal.ctx
        
        try:
            # Check that the expression has the right type
            check_type(state.env, ctx, self.expr, target)
            
            # If successful, remove the current goal
            new_goals = state.goals[1:]
            
            return TacticState(env=state.env, goals=new_goals, proof=self.expr)
        except Exception as e:
            raise TacticException(f"Expression does not match goal type: {e}")


class AssumptionTactic(Tactic):
    """
    Implements the 'assumption' tactic that looks for a matching hypothesis.
    
    This tactic looks through the context for a hypothesis that matches
    the goal type.
    """
    
    def apply(self, state: TacticState) -> TacticState:
        """Apply the assumption tactic to the current state."""
        if not state.goals:
            raise TacticException("No goals to apply assumption tactic")
        
        current_goal = state.goals[0]
        target = current_goal.target
        ctx = current_goal.ctx
        
        # Search for a variable in the context with the target type
        for idx, i in enumerate(range(len(ctx.types))):
            name = ctx.names[i]
            type_expr = ctx.types[i]
            try:
                if is_def_eq(type_expr, target, state.env):
                    # Found a matching assumption
                    # Create a variable reference to it
                    var = mk_var(len(ctx.types) - idx - 1)
                    
                    # Remove the current goal
                    new_goals = state.goals[1:]
                    
                    return TacticState(env=state.env, goals=new_goals, proof=var)
            except Exception:
                continue
        
        raise TacticException("No matching assumption found for the goal")


class ApplyTactic(Tactic):
    """
    Implements the 'apply' tactic for backward reasoning.
    
    Given an expression of type (Π (x₁ : A₁), ... (Π (xₙ : Aₙ), B)),
    where B matches the goal, this tactic will generate new goals
    for A₁, ..., Aₙ.
    """
    
    def __init__(self, expr: Expr):
        """
        Initialize the apply tactic.
        
        Args:
            expr: The expression to apply
        """
        self.expr = expr
    
    def apply(self, state: TacticState) -> TacticState:
        """Apply the apply tactic to the current state."""
        if not state.goals:
            raise TacticException("No goals to apply apply tactic")
        
        current_goal = state.goals[0]
        target = current_goal.target
        ctx = current_goal.ctx
        
        try:
            # Infer the type of the expression to apply
            expr_type = infer_type(state.env, ctx, self.expr)
            
            # Collect all arguments we'll need to provide
            args_needed = []
            curr_type = expr_type
            
            while curr_type.kind == ExprKind.PI:
                args_needed.append((curr_type.name, curr_type.type))
                curr_type = curr_type.body
            
            # Check if the result type matches the goal
            if not is_def_eq(state.env, curr_type, target):
                raise TacticException(f"Result type {curr_type} does not match goal {target}")
            
            # Create new goals for the arguments in reverse order
            new_goals = []
            for var_name, var_type in args_needed:
                new_goals.append(Goal(ctx=ctx, target=var_type))
            
            # Add remaining goals
            new_goals.extend(state.goals[1:])
            
            return TacticState(env=state.env, goals=new_goals, proof=state.proof)
        except Exception as e:
            raise TacticException(f"Failed to apply expression: {e}")


class RewriteTactic(Tactic):
    """
    Implements the 'rewrite' tactic for equality substitution.
    
    Given an equality proof h : a = b, this tactic replaces
    occurrences of a with b in the goal.
    """
    
    def __init__(self, eq_proof: Expr, direction: str = "->"):
        """
        Initialize the rewrite tactic.
        
        Args:
            eq_proof: An expression that should have an equality type
            direction: Direction of rewriting, either "->" or "<-"
        """
        self.eq_proof = eq_proof
        self.direction = direction
    
    def apply(self, state: TacticState) -> TacticState:
        """Apply the rewrite tactic to the current state."""
        if not state.goals:
            raise TacticException("No goals to apply rewrite tactic")
        
        current_goal = state.goals[0]
        target = current_goal.target
        ctx = current_goal.ctx
        
        try:
            # Infer the type of the equality proof
            eq_proof_type = infer_type(state.env, ctx, self.eq_proof)
            
            # Reduce to WHNF
            eq_proof_type = reduce(eq_proof_type, state.env, ReductionStrategy.WHNF, ReductionMode.ALL)
            
            # Check that it's an equality
            if not is_const_with_name(eq_proof_type.fn.fn.fn, "Eq"):
                raise TacticException(f"Expected equality type, got {eq_proof_type}")
            
            # Extract the left and right sides of the equality
            eq_type = eq_proof_type
            left_arg = eq_type.fn.arg     # a
            right_arg = eq_type.arg       # b
            
            # Determine what to replace based on direction
            if self.direction == "->":
                # Replace left with right
                pattern = left_arg
                replacement = right_arg
            elif self.direction == "<-":
                # Replace right with left
                pattern = right_arg
                replacement = left_arg
            else:
                raise TacticException(f"Invalid rewrite direction: {self.direction}")
            
            # Create a new target by replacing occurrences
            new_target = replace_expr(target, pattern, replacement, state.env)
            
            # Create a new goal with the updated target
            new_goal = Goal(ctx=ctx, target=new_target)
            
            # Update the goals in the state
            new_goals = [new_goal] + state.goals[1:]
            
            return TacticState(env=state.env, goals=new_goals, proof=state.proof)
        except Exception as e:
            raise TacticException(f"Failed to apply rewrite: {e}")


class ByTactic(Tactic):
    """
    Implements 'by' tactic that applies a sequence of tactics.
    
    This is a convenience tactic that applies a list of tactics
    in sequence.
    """
    
    def __init__(self, tactics: List[Tactic]):
        """
        Initialize the by tactic.
        
        Args:
            tactics: A list of tactics to apply in sequence
        """
        self.tactics = tactics
    
    def apply(self, state: TacticState) -> TacticState:
        """Apply the tactics in sequence."""
        current_state = state
        
        for tactic in self.tactics:
            current_state = tactic.apply(current_state)
        
        return current_state


class InductionTactic(Tactic):
    """
    Implements the 'induction' tactic for inductive types.
    
    For a goal with a variable of inductive type, this tactic
    generates subgoals for each constructor of the inductive type.
    """
    
    def __init__(self, var_name: str):
        """
        Initialize the induction tactic.
        
        Args:
            var_name: The variable name to apply induction on
        """
        self.var_name = var_name
    
    def _find_constructors(self, env: Environment, type_name: str) -> List[Tuple[str, Expr]]:
        """
        Find all constructors for a given inductive type.
        
        Args:
            env: The environment with declarations
            type_name: The name of the inductive type
            
        Returns:
            A list of (name, type) tuples for each constructor
        """
        constructors = []
        
        for name, decl in env.declarations.items():
            # We identify constructors by looking for declarations that:
            # 1. Are constants
            # 2. Have a return type that is the inductive type
            if decl.kind in [DeclKind.CONSTANT, DeclKind.DEF]:
                type_expr = decl.type
                # Check if the return type is the target inductive type
                # For constructors, we need to inspect the return type in the Pi type
                curr_type = type_expr
                while curr_type.kind == ExprKind.PI:
                    curr_type = curr_type.body
                
                # Check if the final return type is our inductive type
                if (curr_type.kind == ExprKind.CONST and 
                    str(curr_type.name) == type_name):
                    constructors.append((str(decl.name), type_expr))
        
        return constructors
    
    def apply(self, state: TacticState) -> TacticState:
        """Apply the induction tactic to the current state."""
        if not state.goals:
            raise TacticException("No goals to apply induction tactic")
        
        current_goal = state.goals[0]
        ctx = current_goal.ctx
        
        # Find the variable in the context
        var_idx = None
        var_type = None
        for i in range(len(ctx.names)):
            if ctx.names[i] == self.var_name:
                var_idx = len(ctx.names) - i - 1
                var_type = ctx.types[i]
                break
        
        if var_idx is None:
            raise TacticException(f"Variable '{self.var_name}' not found in context")
        
        # Get the type name (removing universe parameters if any)
        if var_type.kind != ExprKind.CONST:
            raise TacticException(f"Variable '{self.var_name}' is not of an inductive type")
        
        type_name = str(var_type.name)
        
        # Find the inductive type declaration in the environment
        decl = state.env.find_decl(type_name)
        
        # Special case for Nat if it's not defined as an inductive type
        special_handling = False
        if (decl is None or decl.kind != DeclKind.INDUCTIVE) and type_name == "Nat":
            # Check if we have zero and succ constructors defined
            zero_decl = state.env.find_decl("zero")
            succ_decl = state.env.find_decl("succ")
            
            if zero_decl is not None and succ_decl is not None:
                special_handling = True
                # Manually construct constructors for Nat
                constructors = [
                    ("zero", zero_decl.type),
                    ("succ", succ_decl.type)
                ]
        
        if not special_handling:
            if decl is None or decl.kind != DeclKind.INDUCTIVE:
                raise TacticException(f"Type '{type_name}' is not an inductive type or not found")
            
            # Get the list of constructors for this inductive type
            constructors = self._find_constructors(state.env, type_name)
            
            if not constructors:
                raise TacticException(f"No constructors found for inductive type '{type_name}'")
        
        # Create a variable expression for the induction variable
        induc_var = mk_var(var_idx)
        
        # Generate a subgoal for each constructor
        new_goals = []
        
        for constr_name, constr_type in constructors:
            # Create a new context for this constructor case
            new_ctx = ctx.copy()
            
            # Add new hypotheses based on constructor arguments
            # For example, for the successor case in naturals,
            # we'd add 'n: Nat' and 'IH: P(n)' to the context
            
            # Parse the constructor type to add appropriate hypotheses
            curr_type = constr_type
            args = []
            while curr_type.kind == ExprKind.PI:
                arg_name = str(curr_type.name)
                arg_type = curr_type.type  # Changed from var_type to type
                args.append((arg_name, arg_type))
                curr_type = curr_type.body
            
            # For each argument, add it to the context
            for arg_idx, (arg_name, arg_type) in enumerate(args):
                new_ctx = new_ctx.extend(arg_name, arg_type)
                
                # If the argument type is the same as the induction variable type,
                # add an induction hypothesis
                if is_def_eq(arg_type, var_type, state.env):
                    # For the induction hypothesis, we need to replace the induction variable
                    # with the current argument in the goal
                    arg_var = mk_var(len(new_ctx.types) - 1)  # The variable just added
                    ih_type = replace_expr(current_goal.target, induc_var, arg_var, state.env)
                    
                    # The induction hypothesis name
                    ih_name = f"IH_{arg_name}"
                    
                    # Add the induction hypothesis
                    new_ctx = new_ctx.extend(ih_name, ih_type)
            
            # Build the constructor application to substitute for the induction variable
            constructor = mk_const(constr_name)
            
            # Apply the constructor to its arguments (in reverse order due to De Bruijn indices)
            constructor_args = []
            for arg_idx, (arg_name, _) in enumerate(reversed(args)):
                constructor_args.append(mk_var(arg_idx))
            
            constructor_app = constructor
            for arg in constructor_args:
                constructor_app = mk_app(constructor_app, arg)
            
            # Create a new goal for this constructor case with the modified target
            # where the induction variable is replaced with the constructor application
            new_target = replace_expr(current_goal.target, induc_var, constructor_app, state.env)
            
            # Create a description of this case for reporting
            case_desc = f"Case for constructor {constr_name}:"
            
            new_goal = Goal(ctx=new_ctx, target=new_target)
            new_goals.append(new_goal)
        
        # Add remaining goals
        new_goals.extend(state.goals[1:])
        
        return TacticState(env=state.env, goals=new_goals, proof=state.proof)


class CasesTactic(Tactic):
    """
    Implements the 'cases' tactic for case analysis on inductive types.
    
    For a goal with a variable of inductive type, this tactic generates
    subgoals for each constructor, but without induction hypotheses.
    This is similar to pattern matching but in the context of proofs.
    """
    
    def __init__(self, var_name: str):
        """
        Initialize the cases tactic.
        
        Args:
            var_name: The variable name to perform case analysis on
        """
        self.var_name = var_name
    
    def _find_constructors(self, env: Environment, type_name: str) -> List[Tuple[str, Expr]]:
        """
        Find all constructors for a given inductive type.
        
        Args:
            env: The environment with declarations
            type_name: The name of the inductive type
            
        Returns:
            A list of (name, type) tuples for each constructor
        """
        constructors = []
        
        for name, decl in env.declarations.items():
            # We identify constructors by looking for declarations that:
            # 1. Are constants
            # 2. Have a return type that is the inductive type
            if decl.kind in [DeclKind.CONSTANT, DeclKind.DEF]:
                type_expr = decl.type
                # Check if the return type is the target inductive type
                # For constructors, we need to inspect the return type in the Pi type
                curr_type = type_expr
                while curr_type.kind == ExprKind.PI:
                    curr_type = curr_type.body
                
                # Check if the final return type is our inductive type
                if (curr_type.kind == ExprKind.CONST and 
                    str(curr_type.name) == type_name):
                    constructors.append((str(decl.name), type_expr))
        
        return constructors
    
    def apply(self, state: TacticState) -> TacticState:
        """Apply the cases tactic to the current state."""
        if not state.goals:
            raise TacticException("No goals to apply cases tactic")
        
        current_goal = state.goals[0]
        ctx = current_goal.ctx
        
        # Find the variable in the context
        var_idx = None
        var_type = None
        for i in range(len(ctx.names)):
            if ctx.names[i] == self.var_name:
                var_idx = len(ctx.names) - i - 1
                var_type = ctx.types[i]
                break
        
        if var_idx is None:
            raise TacticException(f"Variable '{self.var_name}' not found in context")
        
        # Get the type name (removing universe parameters if any)
        if var_type.kind != ExprKind.CONST:
            raise TacticException(f"Variable '{self.var_name}' is not of an inductive type")
        
        type_name = str(var_type.name)
        
        # Find the inductive type declaration in the environment
        decl = state.env.find_decl(type_name)
        
        # Special case for Nat if it's not defined as an inductive type
        special_handling = False
        if (decl is None or decl.kind != DeclKind.INDUCTIVE) and type_name == "Nat":
            # Check if we have zero and succ constructors defined
            zero_decl = state.env.find_decl("zero")
            succ_decl = state.env.find_decl("succ")
            
            if zero_decl is not None and succ_decl is not None:
                special_handling = True
                # Manually construct constructors for Nat
                constructors = [
                    ("zero", zero_decl.type),
                    ("succ", succ_decl.type)
                ]
        
        if not special_handling:
            if decl is None or decl.kind != DeclKind.INDUCTIVE:
                raise TacticException(f"Type '{type_name}' is not an inductive type or not found")
            
            # Get the list of constructors for this inductive type
            constructors = self._find_constructors(state.env, type_name)
            
            if not constructors:
                raise TacticException(f"No constructors found for inductive type '{type_name}'")
        
        # Create a variable expression for the cases variable
        cases_var = mk_var(var_idx)
        
        # Generate a subgoal for each constructor
        new_goals = []
        
        for constr_name, constr_type in constructors:
            # Create a new context for this constructor case
            new_ctx = ctx.copy()
            
            # Parse the constructor type to add appropriate hypotheses for arguments
            curr_type = constr_type
            args = []
            while curr_type.kind == ExprKind.PI:
                arg_name = str(curr_type.name)
                arg_type = curr_type.type
                args.append((arg_name, arg_type))
                curr_type = curr_type.body
            
            # For each argument, add it to the context
            for arg_idx, (arg_name, arg_type) in enumerate(args):
                new_ctx = new_ctx.extend(arg_name, arg_type)
            
            # Build the constructor application to substitute for the case variable
            constructor = mk_const(constr_name)
            
            # Apply the constructor to its arguments (in reverse order due to De Bruijn indices)
            constructor_args = []
            for arg_idx, (arg_name, _) in enumerate(reversed(args)):
                constructor_args.append(mk_var(arg_idx))
            
            constructor_app = constructor
            for arg in constructor_args:
                constructor_app = mk_app(constructor_app, arg)
            
            # Create a new goal for this constructor case with the modified target
            # where the case variable is replaced with the constructor application
            new_target = replace_expr(current_goal.target, cases_var, constructor_app, state.env)
            
            new_goal = Goal(ctx=new_ctx, target=new_target)
            new_goals.append(new_goal)
        
        # Add remaining goals
        new_goals.extend(state.goals[1:])
        
        return TacticState(env=state.env, goals=new_goals, proof=state.proof)


class RflTactic(Tactic):
    """
    Implements the 'rfl' tactic for proving reflexive equality goals.
    
    This tactic solves goals of the form 'a = a' by applying the
    reflexivity property of equality.
    """
    
    def apply(self, state: TacticState) -> TacticState:
        """Apply the rfl tactic to the current state."""
        if not state.goals:
            raise TacticException("No goals to apply rfl tactic")
        
        current_goal = state.goals[0]
        target = current_goal.target
        ctx = current_goal.ctx
        
        # The target should be an equality
        # Check if the target is of form Eq A a b
        if (target.kind != ExprKind.APP or 
            target.fn.kind != ExprKind.APP or
            target.fn.fn.kind != ExprKind.APP or
            target.fn.fn.fn.kind != ExprKind.CONST or
            str(target.fn.fn.fn.name) != "Eq"):
            raise TacticException("Goal is not an equality (Eq A a b)")
        
        # Extract the type and the two terms being compared
        eq_app = target
        b = eq_app.arg                  # Second term (right side of equality)
        a_app = eq_app.fn
        a = a_app.arg                   # First term (left side of equality)
        type_app = a_app.fn
        A = type_app.arg                # Type of the terms
        
        # Check if the two sides are definitionally equal (a = b)
        if not is_def_eq(a, b, state.env):
            raise TacticException("Terms are not definitionally equal; rfl cannot be applied")
        
        # Construct a proof using the refl axiom
        # refl has type: forall A, forall a:A, Eq A a a
        # So we need to instantiate it with the correct type and term
        
        # Get the refl axiom
        refl_decl = state.env.find_decl("refl")
        if refl_decl is None:
            raise TacticException("Cannot find 'refl' axiom in the environment")
        
        # Instantiate refl with the type A and term a
        refl = mk_const("refl")
        refl_A = mk_app(refl, A)
        refl_A_a = mk_app(refl_A, a)
        
        # The proof is refl A a
        proof = refl_A_a
        
        # Create a new state with the current goal solved
        new_goals = state.goals[1:]
        
        return TacticState(env=state.env, goals=new_goals, proof=proof)


class ExfalsoTactic(Tactic):
    """
    Implements the 'exfalso' tactic for proof by contradiction.
    
    This tactic changes the current goal to 'False', allowing the proof
    to proceed by deriving a contradiction from the assumptions.
    Once 'False' is proven, any goal can be derived using ex falso quodlibet.
    """
    
    def apply(self, state: TacticState) -> TacticState:
        """Apply the exfalso tactic to the current state."""
        if not state.goals:
            raise TacticException("No goals to apply exfalso tactic")
        
        current_goal = state.goals[0]
        ctx = current_goal.ctx
        
        # Find the False type in the environment
        false_decl = state.env.find_decl("False")
        if false_decl is None:
            # If False is not defined, we check for "Bot" as an alternative
            false_decl = state.env.find_decl("Bot")
            if false_decl is None:
                raise TacticException("Cannot find 'False' or 'Bot' type in the environment")
        
        false_type = mk_const(str(false_decl.name))
        
        # Check that false_type is actually a type
        try:
            false_type_type = infer_type(state.env, ctx, false_type)
            if not (false_type_type.kind == ExprKind.SORT):
                raise TacticException("'False' is not a proper type")
        except Exception as e:
            raise TacticException(f"Cannot verify that 'False' is a type: {e}")
        
        # Create a new goal with False as the target
        new_goal = Goal(ctx=current_goal.ctx, target=false_type)
        
        # We'll need to use ex falso quodlibet: False → any proposition
        # This could be implemented as a constant/axiom, but for now we generate the proof directly
        
        # Create a new state with the modified goal
        new_goals = [new_goal] + state.goals[1:]
        
        # The proof needs to be updated later when False is proven
        # For now, we leave it as in the original state
        return TacticState(env=state.env, goals=new_goals, proof=state.proof)


class ContradictionTactic(Tactic):
    """
    Implements the 'contradiction' tactic for finding and using contradictions.
    
    This tactic automatically searches for contradictions in the context,
    such as having both p and not p, and uses them to prove any goal.
    """
    
    def apply(self, state: TacticState) -> TacticState:
        """Apply the contradiction tactic to the current state."""
        if not state.goals:
            raise TacticException("No goals to apply contradiction tactic")
        
        current_goal = state.goals[0]
        ctx = current_goal.ctx
        env = state.env
        
        # Collect all hypotheses
        hypotheses = []
        for i in range(len(ctx.types)):
            name = ctx.names[i]
            type_expr = ctx.types[i]
            idx = len(ctx.types) - i - 1
            var = mk_var(idx)
            hypotheses.append((name, type_expr, var))
        
        # Look for standard negation: not p, represented as p -> False
        for i, (name_i, type_i, var_i) in enumerate(hypotheses):
            # Check if type_i is a negation: p -> False
            if type_i.kind == ExprKind.PI:
                # Potential negation, extract p
                p_type = type_i.type
                body = type_i.body
                
                # Check if the body is False
                is_false_body = False
                if body.kind == ExprKind.CONST:
                    false_name = str(body.name)
                    if false_name in ["False", "Bot"]:
                        is_false_body = True
                
                if is_false_body:
                    # This is indeed a negation: p -> False
                    # Look for a hypothesis matching p
                    for j, (name_j, type_j, var_j) in enumerate(hypotheses):
                        if i != j and is_def_eq(type_j, p_type, env):
                            # Found contradiction: p and not p
                            # Create a proof: (not p) p
                            proof = mk_app(var_i, var_j)
                            
                            # This gives us False, now use ex falso quodlibet
                            # to derive any target type
                            
                            # Remove the current goal
                            new_goals = state.goals[1:]
                            
                            return TacticState(env=state.env, goals=new_goals, proof=proof)
        
        # Look for direct contradiction: False or Bot in the context
        for _, type_expr, var in hypotheses:
            if (type_expr.kind == ExprKind.CONST and 
                str(type_expr.name) in ["False", "Bot"]):
                # We already have False in the context, use it
                # Remove the current goal
                new_goals = state.goals[1:]
                
                return TacticState(env=state.env, goals=new_goals, proof=var)
        
        # Look for equality contradictions: a = b and a ≠ b
        for i, (name_i, type_i, var_i) in enumerate(hypotheses):
            # Check if type_i is an equality: Eq A a b
            if (type_i.kind == ExprKind.APP and 
                type_i.fn.kind == ExprKind.APP and
                type_i.fn.fn.kind == ExprKind.APP and
                type_i.fn.fn.fn.kind == ExprKind.CONST and
                str(type_i.fn.fn.fn.name) == "Eq"):
                
                # Extract the equality components
                eq_app = type_i
                b = eq_app.arg
                a_app = eq_app.fn
                a = a_app.arg
                type_app = a_app.fn
                A = type_app.arg
                
                # Look for a negated equality: a ≠ b, which is (a = b) -> False
                for j, (name_j, type_j, var_j) in enumerate(hypotheses):
                    if i != j and type_j.kind == ExprKind.PI:
                        neg_eq_type = type_j.type
                        neg_body = type_j.body
                        
                        # Check if the body is False
                        is_false_body = False
                        if neg_body.kind == ExprKind.CONST:
                            false_name = str(neg_body.name)
                            if false_name in ["False", "Bot"]:
                                is_false_body = True
                        
                        if is_false_body:
                            # Check if the negation argument is an equality that contradicts
                            if is_def_eq(neg_eq_type, type_i, env):
                                # Found contradiction: a = b and a ≠ b
                                # Create a proof: (a ≠ b) (a = b)
                                proof = mk_app(var_j, var_i)
                                
                                # Remove the current goal
                                new_goals = state.goals[1:]
                                
                                return TacticState(env=state.env, goals=new_goals, proof=proof)
        
        raise TacticException("No contradiction found in the context")


class SimpTactic(Tactic):
    """
    Implements the 'simp' tactic for simplifying expressions using rewrite rules.
    
    This tactic applies a series of rewrite rules to simplify the goal.
    Rules can be provided explicitly or automatically extracted from the context.
    """
    
    def __init__(self, rules: List[Expr] = None, use_context: bool = True, only: List[str] = None):
        """
        Initialize the simp tactic.
        
        Args:
            rules: Optional list of expressions to use as rewrite rules
            use_context: Whether to use equalities from the context as rules
            only: Optional list of names to restrict which context equalities to use
        """
        self.rules = rules or []
        self.use_context = use_context
        self.only = only
    
    def apply(self, state: TacticState) -> TacticState:
        """Apply the simp tactic to the current state."""
        if not state.goals:
            raise TacticException("No goals to apply simp tactic")
        
        current_goal = state.goals[0]
        ctx = current_goal.ctx
        target = current_goal.target
        env = state.env
        
        # Collect all rewrite rules
        rewrite_rules = []
        
        # Add explicitly provided rules
        for rule in self.rules:
            # Verify that the rule is an equality
            try:
                rule_type = infer_type(env, ctx, rule)
                if (rule_type.kind == ExprKind.APP and 
                    rule_type.fn.kind == ExprKind.APP and
                    rule_type.fn.fn.kind == ExprKind.APP and
                    rule_type.fn.fn.fn.kind == ExprKind.CONST and
                    str(rule_type.fn.fn.fn.name) == "Eq"):
                    rewrite_rules.append(rule)
                else:
                    raise TacticException(f"Rule is not an equality: {rule}")
            except Exception as e:
                raise TacticException(f"Invalid rewrite rule: {e}")
        
        # Add equalities from context if requested
        if self.use_context:
            for i in range(len(ctx.types)):
                name = ctx.names[i]
                type_expr = ctx.types[i]
                
                # Skip if 'only' is specified and this name is not in the list
                if self.only and name not in self.only:
                    continue
                
                # Check if this is an equality type
                if (type_expr.kind == ExprKind.APP and 
                    type_expr.fn.kind == ExprKind.APP and
                    type_expr.fn.fn.kind == ExprKind.APP and
                    type_expr.fn.fn.fn.kind == ExprKind.CONST and
                    str(type_expr.fn.fn.fn.name) == "Eq"):
                    
                    # Create a variable reference to this hypothesis
                    idx = len(ctx.types) - i - 1
                    var = mk_var(idx)
                    rewrite_rules.append(var)
        
        if not rewrite_rules:
            raise TacticException("No rewrite rules available for simplification")
        
        # Apply all rewrite rules repeatedly until no more changes
        simplified_target = target
        made_progress = True
        max_iterations = 100  # Prevent infinite loops
        iterations = 0
        
        while made_progress and iterations < max_iterations:
            made_progress = False
            iterations += 1
            
            for rule in rewrite_rules:
                # Try to apply each rule to the target
                # Use the existing rewrite tactic logic
                rewrite_tactic = RewriteTactic(rule, "->")
                try:
                    # Create a temporary state with the current simplified target
                    temp_goal = Goal(ctx=ctx, target=simplified_target)
                    temp_state = TacticState(env=env, goals=[temp_goal])
                    
                    # Apply the rewrite
                    result_state = rewrite_tactic.apply(temp_state)
                    
                    # Check if the target changed
                    if result_state.goals and not is_def_eq(simplified_target, result_state.goals[0].target, env):
                        simplified_target = result_state.goals[0].target
                        made_progress = True
                except TacticException:
                    # Rewrite failed, just continue to the next rule
                    continue
        
        if iterations >= max_iterations:
            raise TacticException("Simplification exceeded maximum iterations, possible rewrite loop")
        
        if is_def_eq(simplified_target, target, env):
            # No simplification occurred
            raise TacticException("No simplification possible with the given rules")
        
        # Create a new goal with the simplified target
        new_goal = Goal(ctx=ctx, target=simplified_target)
        
        # Create a new state with the modified goal
        new_goals = [new_goal] + state.goals[1:]
        
        return TacticState(env=env, goals=new_goals, proof=state.proof)


class DestructTactic(Tactic):
    """
    Implements the 'destruct' tactic for breaking down inductive types 
    or propositions in hypotheses or the goal.
    
    Currently handles:
    - Hypotheses of type `And p q` -> Adds hypotheses `p` and `q`.
    - Goal of type `And p q` -> Creates two subgoals `p` and `q`.
    """
    
    def __init__(self, target_name: str, new_names: Optional[List[str]] = None):
        """
        Initialize the destruct tactic.
        
        Args:
            target_name: The name of the hypothesis or 'goal' to destruct.
            new_names: Optional list of names for the new components.
        """
        self.target_name = target_name
        self.new_names = new_names or []

    def apply(self, state: TacticState) -> TacticState:
        """Apply the destruct tactic."""
        if not state.goals:
            raise TacticException("No goals to apply destruct tactic")
            
        current_goal = state.goals[0]
        ctx = current_goal.ctx
        env = state.env
        
        # Case 1: Destruct a hypothesis
        if self.target_name != 'goal':
            target_var = None
            target_type = None
            target_idx = -1
            
            # Find the hypothesis by name
            for i in range(len(ctx.names)):
                if ctx.names[i] == self.target_name:
                    target_idx = len(ctx.names) - i - 1 # De Bruijn index
                    target_type = ctx.types[i]
                    target_var = mk_var(target_idx)
                    break
            
            if target_var is None or target_type is None:
                raise TacticException(f"Hypothesis '{self.target_name}' not found in context")
            
            # --- Handle And --- 
            # Check if target_type is `And p q`
            # Assume And is represented as mk_app(mk_app(mk_const("And"), p), q)
            if (target_type.kind == ExprKind.APP and 
                target_type.fn.kind == ExprKind.APP and 
                target_type.fn.fn.kind == ExprKind.CONST and 
                str(target_type.fn.fn.name) == "And"):
                
                p = target_type.fn.arg
                q = target_type.arg
                
                # Need And.intro, And.left, And.right definitions
                and_left_decl = env.find_decl("And.left")
                and_right_decl = env.find_decl("And.right")
                if not and_left_decl or not and_right_decl:
                    raise TacticException("Definitions for And.left/And.right not found")
                
                # Create proofs for p and q: And.left h, And.right h
                # Note: These proofs need the correct type arguments for And.left/right
                # For simplicity, we assume they are constants for now.
                # A full implementation needs to instantiate them.
                proof_p = mk_app(mk_const("And.left"), target_var) 
                proof_q = mk_app(mk_const("And.right"), target_var)
                
                # Create new context: remove original hypothesis, add new ones
                # This is complex due to De Bruijn indices needing adjustment.
                # Simplified approach: Add new hyps first, then conceptually remove old one.
                new_ctx = ctx
                name_p = self.new_names[0] if len(self.new_names) > 0 else "hp"
                name_q = self.new_names[1] if len(self.new_names) > 1 else "hq"
                
                # Problem: The proofs depend on the original var index, which changes
                # if we remove the old hypothesis first. 
                # A better approach involves substitution or more complex context manipulation.
                
                # --- Simplified Placeholder Implementation --- 
                # Add p and q directly as new hypotheses without proper proof terms
                # This is NOT sound, just a placeholder structure.
                new_ctx = new_ctx.extend(name_p, p)
                new_ctx = new_ctx.extend(name_q, q)
                
                # TODO: Implement proper context update and proof term generation
                # For now, we just keep the original goal but with the modified context
                new_goal = Goal(ctx=new_ctx, target=current_goal.target)
                new_goals = [new_goal] + state.goals[1:]
                return TacticState(env=env, goals=new_goals, proof=state.proof)

            else:
                raise TacticException(f"Destruct not yet implemented for hypothesis type: {target_type}")

        # Case 2: Destruct the goal
        else:
            target = current_goal.target
            # --- Handle And --- 
            if (target.kind == ExprKind.APP and 
                target.fn.kind == ExprKind.APP and 
                target.fn.fn.kind == ExprKind.CONST and 
                str(target.fn.fn.name) == "And"):
                
                p = target.fn.arg
                q = target.arg
                
                # Create two new goals
                goal_p = Goal(ctx=ctx, target=p)
                goal_q = Goal(ctx=ctx, target=q)
                
                # Replace current goal with the two new subgoals
                new_goals = [goal_p, goal_q] + state.goals[1:]
                
                # TODO: Need to construct the proof term (e.g., And.intro) later
                return TacticState(env=env, goals=new_goals, proof=state.proof)
                
            else:
                raise TacticException(f"Destruct not yet implemented for goal type: {target}")


# Helper functions

def is_const_with_name(expr: Expr, name: str) -> bool:
    """Check if an expression is a constant with the given name."""
    return expr.kind == ExprKind.CONST and str(expr.name) == name


def replace_expr(expr: Expr, pattern: Expr, replacement: Expr, env: Optional[Environment] = None) -> Expr:
    """
    Replace all occurrences of pattern with replacement in expr.
    
    Args:
        expr: The expression to transform
        pattern: The pattern to search for
        replacement: The replacement for the pattern
        env: Optional environment for reduction and equality checking
        
    Returns:
        A new expression with replacements applied
    """
    # If the expression is exactly the pattern, replace it
    if env is not None:
        # Use definitional equality if environment is provided
        try:
            # First try definitional equality (more accurate)
            if is_def_eq(expr, pattern, env):
                return replacement
        except Exception:
            # If definitional equality check fails, don't give up yet
            pass
    
    # Fall back to structural equality for specific expression kinds
    # This is more limited but doesn't require an environment
    if expr.kind == pattern.kind:
        if expr.kind == ExprKind.VAR and expr.idx == pattern.idx:
            return replacement
        elif expr.kind == ExprKind.CONST and expr.name == pattern.name:
            return replacement
        # Add more structural equality checks for other expression kinds
        elif expr.kind == ExprKind.SORT and pattern.kind == ExprKind.SORT:
            if str(expr.level) == str(pattern.level):
                return replacementt
    
    # Otherwise, recursively process subexpressions
    if expr.kind == ExprKind.APP:
        new_fn = replace_expr(expr.fn, pattern, replacement, env)
        new_arg = replace_expr(expr.arg, pattern, replacement, env)
        return mk_app(new_fn, new_arg)
    
    elif expr.kind == ExprKind.LAMBDA:
        # For binding expressions, we need to be careful with De Bruijn indices
        # The pattern might need to be adjusted when going under a binder
        
        # First replace in the type (outside the binder)
        new_type = replace_expr(expr.type, pattern, replacement, env)
        
        # For the body, we need special handling:
        # 1. If pattern is a variable that will be captured by this binder,
        #    we should not replace it inside the body
        # 2. Otherwise, we need to adjust indices in both pattern and replacement
        
        # Check if pattern is a variable that would be captured
        if pattern.kind == ExprKind.VAR and pattern.idx == 0:
            # This variable would be captured by the current binder, so don't replace in body
            new_body = expr.body
        else:
            # Prepare pattern and replacement for use inside the body
            # This is a simplification - a real implementation would properly
            # adjust De Bruijn indices for pattern and replacement
            # For now, just recurse into the body
            new_body = replace_expr(expr.body, pattern, replacement, env)
        
        return mk_lambda(expr.name, new_type, new_body)
    
    elif expr.kind == ExprKind.PI:
        # Similar to lambda case
        new_type = replace_expr(expr.type, pattern, replacement, env)
        
        if pattern.kind == ExprKind.VAR and pattern.idx == 0:
            new_body = expr.body
        else:
            new_body = replace_expr(expr.body, pattern, replacement, env)
        
        return mk_pi(expr.name, new_type, new_body)
    
    elif expr.kind == ExprKind.LET:
        new_type = replace_expr(expr.type, pattern, replacement, env)
        new_value = replace_expr(expr.value, pattern, replacement, env)
        
        if pattern.kind == ExprKind.VAR and pattern.idx == 0:
            new_body = expr.body
        else:
            new_body = replace_expr(expr.body, pattern, replacement, env)
        
        return mk_let(expr.name, new_type, new_value, new_body)
    
    # Other expression types (VAR, SORT, CONST) don't have subexpressions to replace
    return expr


def init_tactic_state(env: Environment, goal_type: Expr) -> TacticState:
    """
    Initialize a tactic state for proving a goal.
    
    Args:
        env: The environment with available declarations
        goal_type: The type to prove
        
    Returns:
        A new tactic state with a single goal
    """
    goal = Goal(ctx=Context(), target=goal_type)
    return TacticState(env=env, goals=[goal]) 