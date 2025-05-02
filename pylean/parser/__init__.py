"""
Pylean parser module.

This module implements a parser for the Lean4 programming language syntax,
including support for expressions, commands, tactics, and other Lean4-specific
syntax constructs.
"""

from pylean.parser.core import (
    Parser, ParserContext, ParserState, SyntaxNode, SourceInfo,
    ParseError, InputContext, and_then, or_else, many, many1, optional
)

from pylean.parser.expr import (
    parse_expr, parse_expression
)

__all__ = [
    # Core parser types
    'Parser', 'ParserContext', 'ParserState', 'SyntaxNode', 'SourceInfo',
    'ParseError', 'InputContext',
    
    # Core parser combinators
    'and_then', 'or_else', 'many', 'many1', 'optional',
    
    # Expression parsing
    'parse_expr', 'parse_expression',
] 