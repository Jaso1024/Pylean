"""
Pylean: A Python implementation of the Lean4 theorem prover.

This package provides a Python implementation of the Lean4 theorem prover,
making it available as a Python module through pip.
"""

__version__ = "0.1.0"

# Import core modules for easy access
from pylean.kernel import (
    Expr, Name, Level, ExprKind,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi
)

# Optional CLI import - not required for library usage
try:
    from pylean.cli import PyleanREPL
except ImportError:
    # CLI dependencies might be missing in minimal installation
    pass

# Define function to start the REPL
def start_repl():
    """Start the Pylean interactive REPL."""
    from pylean.cli import PyleanREPL
    repl = PyleanREPL()
    repl.run() 