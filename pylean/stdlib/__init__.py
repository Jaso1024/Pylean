"""
Pylean standard library.

This module provides a standard library of definitions, theorems,
and tactics for Pylean, similar to Lean's standard library.
"""

from pylean.stdlib.init import init_stdlib
from pylean.stdlib.logic import *
from pylean.stdlib.nat import *
from pylean.stdlib.list import *

__all__ = [
    'init_stdlib',
] 