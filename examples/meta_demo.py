"""
Metaprogramming Demo for Pylean.

This example demonstrates how to use the Pylean metaprogramming
system to manipulate expressions and create custom tactics.
"""

from pylean.kernel import (
    Expr, ExprKind, Name, 
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, Context, Declaration, DeclKind
)
from pylean.kernel import Tactic, TacticState, Goal
from pylean.meta import (
    MetaExpr, MetaM, run_meta,
    make_tactic, tactic_to_meta
)


def create_nat_double_meta() -> MetaM[Expr]:
    """
    Create a meta program that defines a double function for natural numbers.
    
    Returns:
        A meta program that creates the definition
    """
    # Get the environment
    env_meta = MetaM.get_env()
    
    def create_double(env):
        # Create expressions for Nat, zero, and succ
        nat_meta = MetaExpr.const("Nat")
        zero_meta = MetaExpr.const("zero")
        succ_meta = MetaExpr.const("succ")
        
        # Create expressions for the function definition
        n_meta = MetaExpr.var(0)  # Variable with De Bruijn index 0
        
        # Create the add function
        add_meta = MetaExpr.const("add")
        
        # Create the expression: add n n
        add_n_meta = MetaExpr.app(add_meta, n_meta)
        double_body_meta = MetaExpr.app(add_n_meta, n_meta)
        
        # Create the type: Nat -> Nat
        double_type_meta = MetaExpr.pi("n", nat_meta, nat_meta)
        
        # Create the definition: λn:Nat. add n n
        double_def_meta = MetaExpr.lambda_expr("n", nat_meta, double_body_meta)
        
        return double_type_meta.bind(
            lambda double_type: double_def_meta.bind(
                lambda double_def: MetaM.pure((double_type, double_def))
            )
        )
    
    # Bind the environment to the create_double function
    return env_meta.bind(create_double)


class AutoRewriteTactic(Tactic):
    """
    A tactic that rewrites expressions using the first matching rewrite rule.
    
    This is implemented using the metaprogramming system to search for
    rewrite rules and apply them to expressions.
    """
    
    def __init__(self, lemmas: list = None):
        """
        Initialize the tactic.
        
        Args:
            lemmas: Optional list of lemma names to use for rewriting
        """
        super().__init__()
        self.lemmas = lemmas or []
    
    def apply(self, state: TacticState) -> TacticState:
        """
        Apply the tactic to a state.
        
        Args:
            state: The current tactic state
            
        Returns:
            The new tactic state
        """
        if not state.goals:
            return state  # No goals to process
        
        # Get the first goal
        goal = state.goals[0]
        target = goal.target
        
        # Define a meta function to rewrite the target
        def rewrite_meta():
            # Get the environment
            env_meta = MetaM.get_env()
            
            def find_and_apply_rewrite(env):
                # For each lemma, try to use it for rewriting
                for lemma_name in self.lemmas:
                    # Get the lemma expression
                    lemma_meta = MetaExpr.const(lemma_name)
                    
                    # Infer its type
                    lemma_type_meta = MetaM.infer_type(lemma_meta)
                    
                    # TODO: Check if the lemma is an equality and can be applied
                    # For now, just return the target as is
                    return MetaM.pure(target)
            
            return env_meta.bind(find_and_apply_rewrite)
        
        # Convert the meta function to a tactic
        meta_tactic = make_tactic(lambda args: rewrite_meta())
        
        # Apply the meta tactic
        new_state = meta_tactic().apply(state)
        
        return new_state


def main():
    """Run the metaprogramming demo."""
    print("Pylean Metaprogramming Demo")
    print("===========================")
    
    # Create a kernel with a standard environment
    kernel = Kernel()
    env = kernel.env
    
    # Add basic definitions to the environment
    print("\nDefining basic constants...")
    
    # Define Natural numbers (Nat)
    kernel = kernel.add_constant("Nat", mk_sort(0))
    
    # Define constructors for Nat
    kernel = kernel.add_constant("zero", mk_const("Nat"))
    
    # Define succ : Nat -> Nat
    succ_type = mk_pi("n", mk_const("Nat"), mk_const("Nat"))
    kernel = kernel.add_constant("succ", succ_type)
    
    # Define add : Nat -> Nat -> Nat
    add_type = mk_pi(
        "m", mk_const("Nat"),
        mk_pi("n", mk_const("Nat"), mk_const("Nat"))
    )
    kernel = kernel.add_constant("add", add_type)
    
    # Get the updated environment
    env = kernel.env
    
    print("\nDemonstrating metaprogramming...")
    
    # Use meta to define a double function
    double_meta = create_nat_double_meta()
    
    # Run the meta program
    double_type, double_def = run_meta(double_meta, env)
    
    print("\nGenerated expressions:")
    print(f"Type: {double_type}")
    print(f"Definition: {double_def}")
    
    # Add the definition to the environment
    kernel = kernel.add_definition("double", double_type, double_def)
    env = kernel.env
    
    # Demonstrate meta tactics
    print("\nDemonstrating meta tactics...")
    
    # Create a hypothetical goal
    ctx = Context()
    target = mk_app(mk_const("double"), mk_const("zero"))
    goal = Goal(ctx, target)
    
    # Create an auto-rewrite tactic with some lemmas
    auto_rewrite = AutoRewriteTactic(lemmas=["add_zero"])
    
    # Apply the tactic
    tactic_state = TacticState(env=env, goals=[goal])
    new_state = auto_rewrite.apply(tactic_state)
    
    print("\nBefore auto-rewrite:")
    print(f"Target: {goal.target}")
    
    print("\nAfter auto-rewrite:")
    if new_state.goals:
        print(f"Target: {new_state.goals[0].target}")
    else:
        print("Goal solved!")
    
    print("\nMetaprogramming Demo Completed!")


if __name__ == "__main__":
    main() 