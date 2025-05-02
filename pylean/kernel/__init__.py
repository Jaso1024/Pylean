"""
Main module for the Pylean theorem prover kernel.

This module exports the main classes and functions from the kernel.
"""

from pylean.kernel.expr import (
    # Types and classes
    ExprKind, Name, Level, Expr, VarExpr, SortExpr, ConstExpr, AppExpr,
    BinderInfo, LambdaExpr, PiExpr, LetExpr, MetaExpr, LocalExpr,
    Pattern, Alternative, MatchExpr, ExternExpr,
    
    # Factory functions
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi, mk_let, mk_meta,
    mk_local, mk_match, mk_pattern, mk_alternative, mk_extern,
    
    # Expression manipulation functions
    occurs_in, lift, instantiate
)

from pylean.kernel.env import (
    # Types and classes
    DeclKind, Declaration, AxiomDecl, DefinitionDecl, TheoremDecl,
    OpaqueDecl, ConstantDecl, InductiveDecl, ConstructorDecl, ExternDecl,
    Environment,
    
    # Factory functions
    mk_axiom, mk_definition, mk_theorem, mk_opaque, mk_constant,
    mk_constructor, mk_inductive, mk_extern_decl,
    
    # Environment creation
    mk_std_env
)

from pylean.kernel.typecheck import (
    # Type checking function
    check_type, infer_type, is_type_convertible,
    Context, TypeError, ensure_type_is_valid
)

from pylean.kernel.reduce import (
    # Reduction functions
    reduce_beta, reduce_delta, reduce_zeta,
    whnf, nf, is_stuck, reduce, is_def_eq, expr_equals,
    ReductionStrategy, ReductionMode
)

from pylean.kernel.tactic import (
    # Tactic types and classes
    TacticException, Goal, TacticState, Tactic,
    IntroTactic, ExactTactic, AssumptionTactic, ApplyTactic, RewriteTactic,
    InductionTactic, CasesTactic, RflTactic, ExfalsoTactic, ContradictionTactic,
    SimpTactic, DestructTactic, ByTactic,
    
    # Helper functions
    init_tactic_state, replace_expr, is_const_with_name
)

from pylean.kernel.tactic_repl import (
    # Tactic REPL
    TacticREPL
)

from pylean.kernel.typeclass import (
    # Type class types and classes
    TypeClass, TypeClassInstance, TypeClassEnvironment, TypeClassElaborator,
    
    # Type class functions
    mk_type_class, mk_type_class_instance
)

from pylean.kernel.eliminator import (
    # Eliminator generation
    generate_recursor, generate_induction_principle, generate_eliminators
)

from pylean.kernel.pattern_opt import (
    # Pattern matching optimization
    check_exhaustiveness, optimize_match, generate_decision_tree,
    PatternOptimizer, DecisionTree
)

from pylean.kernel.kernel import (
    # Kernel class
    Kernel, KernelException, TypeCheckException, NameAlreadyExistsException
)

__all__ = [
    # From expr.py
    'ExprKind', 'Name', 'Level', 'Expr', 'VarExpr', 'SortExpr', 'ConstExpr', 'AppExpr',
    'BinderInfo', 'LambdaExpr', 'PiExpr', 'LetExpr', 'MetaExpr', 'LocalExpr',
    'Pattern', 'Alternative', 'MatchExpr', 'ExternExpr',
    'mk_var', 'mk_sort', 'mk_const', 'mk_app', 'mk_lambda', 'mk_pi', 'mk_let', 'mk_meta',
    'mk_local', 'mk_match', 'mk_pattern', 'mk_alternative', 'mk_extern',
    'occurs_in', 'lift', 'instantiate',
    
    # From env.py
    'DeclKind', 'Declaration', 'AxiomDecl', 'DefinitionDecl', 'TheoremDecl',
    'OpaqueDecl', 'ConstantDecl', 'InductiveDecl', 'ConstructorDecl', 'ExternDecl',
    'Environment',
    'mk_axiom', 'mk_definition', 'mk_theorem', 'mk_opaque', 'mk_constant',
    'mk_constructor', 'mk_inductive', 'mk_extern_decl',
    'mk_std_env',
    
    # From typecheck.py
    'check_type', 'infer_type', 'is_type_convertible',
    'Context', 'TypeError', 'ensure_type_is_valid',
    
    # From reduce.py
    'reduce_beta', 'reduce_delta', 'reduce_zeta',
    'whnf', 'nf', 'is_stuck', 'reduce', 'is_def_eq', 'expr_equals',
    'ReductionStrategy', 'ReductionMode',
    
    # From tactic.py
    'TacticException', 'Goal', 'TacticState', 'Tactic',
    'IntroTactic', 'ExactTactic', 'AssumptionTactic', 'ApplyTactic', 'RewriteTactic',
    'InductionTactic', 'CasesTactic', 'RflTactic', 'ExfalsoTactic', 'ContradictionTactic',
    'SimpTactic', 'DestructTactic', 'ByTactic',
    'init_tactic_state', 'replace_expr', 'is_const_with_name',
    
    # From tactic_repl.py
    'TacticREPL',
    
    # From typeclass.py
    'TypeClass', 'TypeClassInstance', 'TypeClassEnvironment', 'TypeClassElaborator',
    'mk_type_class', 'mk_type_class_instance',
    
    # From eliminator.py
    'generate_recursor', 'generate_induction_principle', 'generate_eliminators',
    
    # From pattern_opt.py
    'check_exhaustiveness', 'optimize_match', 'generate_decision_tree',
    'PatternOptimizer', 'DecisionTree',
    
    # From kernel.py
    'Kernel', 'KernelException', 'TypeCheckException', 'NameAlreadyExistsException'
] 