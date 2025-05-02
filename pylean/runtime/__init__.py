"""
Pylean runtime support module.

This module provides runtime support for compiled Pylean code,
including memory management and runtime functions.
"""

from pylean.runtime.memory import (
    Box, Object, alloc_object, free_object, 
    inc_ref, dec_ref, get_rc
)

__all__ = [
    'Box',
    'Object',
    'alloc_object',
    'free_object',
    'inc_ref',
    'dec_ref',
    'get_rc'
] 