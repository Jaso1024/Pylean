"""
Pylean metaprogramming module.

This module implements metaprogramming facilities for extending
Pylean's capabilities at compile time, similar to Lean's meta system.
"""

from pylean.meta.expr import MetaExpr, MetaM, run_meta
from pylean.meta.tactics import make_tactic, tactic_to_meta

__all__ = [
    'MetaExpr',
    'MetaM',
    'run_meta',
    'make_tactic',
    'tactic_to_meta',
] 