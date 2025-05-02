from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
import functools

from pylean.parser.core import Parser, ParserState, SyntaxNode, mk_token_node, node, token_fn, peek_token_node

KIND_ANY = "<any>"

@dataclass
class TokenMap:
    """A map from token names to parsers with priorities."""
    tokens: Dict[str, List[Tuple[Parser, int]]] = field(default_factory=dict)

    def insert(self, token: str, parser: Parser, priority: int) -> None:
        """Insert a parser for a token with given priority."""
        if token not in self.tokens:
            self.tokens[token] = []
        self.tokens[token].append((parser, priority))
    
    def find(self, token: str) -> List[Tuple[Parser, int]]:
        """Find all parsers for a given token."""
        return self.tokens.get(token, [])


@dataclass
class PrattParsingTables:
    """Tables for Pratt parsing."""
    # For leading/prefix/atomic parsers
    leading_table: TokenMap = field(default_factory=TokenMap)
    leading_parsers: List[Tuple[Parser, int]] = field(default_factory=list)
    
    # For trailing/infix/postfix parsers
    trailing_table: TokenMap = field(default_factory=TokenMap)
    trailing_parsers: List[Tuple[Parser, int]] = field(default_factory=list)


def indexed(token_map: TokenMap, state: ParserState) -> Tuple[ParserState, List[Tuple[Parser, int]]]:
    """Find parsers that match the current token."""
    # Peek at the next token node without modifying state
    token = peek_token_node(state.context, state)

    # If no token was peeked (EOF or error during peek)
    if token is None:
        # Return the original state unmodified, no parsers found
        return state, []

    # We have a token, proceed with lookup
    kind = token.kind
    value = token.value
    token_pos = token.info.pos if hasattr(token, 'info') and token.info is not None else state.pos

    parsers = []
    # Check for specific token value (symbols) in the map
    if kind == "symbol" and value in token_map.tokens:
        parsers = token_map.tokens[value]
    # Check for token kind in the map (e.g., "ident")
    elif kind in token_map.tokens:
        parsers = token_map.tokens[kind]
    # Check if the generic 'any' key exists (for fallback/general parsers)
    elif KIND_ANY in token_map.tokens:
         parsers = token_map.tokens[KIND_ANY]

    # IMPORTANT FIX: Even if peek_token_node sets an error in state,
    # we want to ignore it as long as we found a token and matched parsers.
    # This function should only peek ahead, not modify the state's error.
    clean_state = dataclasses.replace(state, error=None)
    
    # Return the original state (with any error cleared) and the found parsers
    return clean_state, parsers


def longest_match(left: Optional[Any], parsers: List[Tuple[Parser, int]], state: ParserState) -> ParserState:
    """Run all parsers and choose the one that consumes the most input or has highest priority.
    
    The selection criteria (in order) are:
    1. The parser that advances the furthest in the input
    2. If multiple parsers advance to the same position, the one with the highest priority
    """
    if not parsers:
        return state.make_error("No parsers available")
    
    # If there's only one parser, just run it
    if len(parsers) == 1:
        parser, _ = parsers[0]
        if left is not None:
            # For trailing parsers, push left node first, then run parser
            state.stx_stack.append(left)
            return parser(state.context, state)
        else:
            # For leading parsers, just run parser
            return parser(state.context, state)
    
    # Try all parsers and keep track of the best result
    best_state = None
    best_score = (-1, -1)  # (position, priority)
    
    for parser, priority in parsers:
        # Save the initial state
        initial_size = len(state.stx_stack)
        initial_pos = state.pos
        
        # Clone the state for this parser
        curr_state = state.clone()
        
        if left is not None:
            # For trailing parsers, push left node first
            curr_state.stx_stack.append(left)
        
        # Run the parser
        new_state = parser(state.context, curr_state)
        
        # If the parser failed, skip it
        if new_state.has_error():
            continue
        
        # Calculate how much input was consumed
        score = (new_state.pos, priority)
        
        # If this is the best parser so far, keep its result
        if best_state is None or score > best_score:
            best_state = new_state
            best_score = score
    
    # If no parser succeeded, return the original state
    if best_state is None:
        return state.make_error("No parser succeeded")
    
    return best_state


def leading_parser(kind: str, tables: PrattParsingTables, state: ParserState) -> ParserState:
    """Parse a leading (prefix/atomic) expression.
    
    This function handles parsing the first part of an expression - the "nud" (null denotation)
    parsers in traditional Pratt parsing terminology. These include:
    - Literals (numbers, strings, etc.)
    - Identifiers
    - Prefix operators (+x, -x, !x)
    - Parenthesized expressions
    
    Args:
        kind: The syntax node kind to create if needed
        tables: The Pratt parsing tables with leading parsers
        state: Current parser state
        
    Returns:
        Updated parser state with a parsed expression on the stack
    """
    initial_size = len(state.stx_stack)
    initial_pos = state.pos
    initial_prec = state.lhs_prec
    
    # Reset lhs_prec for leading expression
    state.lhs_prec = 0
    
    # Get parsers that match the next token
    state, parsers = indexed(tables.leading_table, state)
    
    if state.has_error():
        return state
    
    # Combine token-specific parsers with general leading parsers
    all_parsers = tables.leading_parsers + parsers
    if not all_parsers:
        # If there are no applicable parsers, report an error
        return state.make_error(f"Unexpected token at this position, expected {kind}")
    
    # Try each parser individually - prioritize token-specific parsers first
    success = False
    for parser, priority in all_parsers:
        parser_name = getattr(parser, '__name__', str(parser))
        # Create a clone for this parser attempt
        parser_state = state.clone()
        
        result_state = parser(state.context, parser_state)
        
        if not result_state.has_error():
            success = True
            state = result_state
            break
    
    # If no parser succeeded, return an error
    if not success:
        return state.make_error(f"No parser succeeded for token at position {state.pos}")
    
    # Create a node if multiple items were pushed to the stack
    if len(state.stx_stack) > initial_size + 1:
        # This shouldn't normally happen with a proper implementation
        # as each parser should push exactly one node
        nodes = state.stx_stack[initial_size:]
        
        # Pop nodes from stack
        state.stx_stack = state.stx_stack[:initial_size]
        
        # Create a new node and push it to the stack
        new_node = SyntaxNode(kind=kind, children=nodes)
        state.stx_stack.append(new_node)
    
    return state


def trailing_loop_step(tables: PrattParsingTables, left: Any, state: ParserState) -> ParserState:
    """Apply a single trailing parser.
    
    This function is used to parse a trailing (postfix/infix) operator or construct.
    These include:
    - Binary operators (x + y, x * y)
    - Postfix operators (x++, x--)
    - Call expressions (f(x))
    - Subscript expressions (a[i])
    - Member access (x.y)
    
    It tries to find a parser that can consume the next token and combines it
    with the left-hand value.
    
    Args:
        tables: The Pratt parsing tables with trailing parsers
        left: The left-hand node to operate on
        state: Current parser state
        
    Returns:
        Updated parser state after applying a trailing parser
    """
    # Get parsers that apply to the current token
    state, parsers = indexed(tables.trailing_table, state)
    
    if state.has_error():
        return state
    
    # Combine token-specific parsers with general trailing parsers
    all_parsers = parsers + tables.trailing_parsers
    if not all_parsers:
        return state  # No available trailing parser
    
    # Use all parsers - don't filter by precedence at this level
    filtered_parsers = all_parsers
    
    if not filtered_parsers:
        return state  # No parsers available
    
    # Get the parser and precedence
    parser, prec = filtered_parsers[0]
    
    # Apply the parser directly to the state
    # Pass the left node as the context parameter - this is the key change!
    result = parser(left, state)
    
    # Update state lhs_prec with the precedence of the chosen parser
    if not result.has_error():
        result.lhs_prec = prec
        return result
    else:
        return result


def trailing_loop(tables: PrattParsingTables, state: ParserState) -> ParserState:
    """Apply trailing parsers repeatedly as long as possible.
    
    This function is responsible for handling the "led" (left denotation) parsers
    in traditional Pratt parsing terminology. It repeatedly tries to apply
    trailing (postfix/infix) parsers as long as they're applicable.
    
    Args:
        tables: The Pratt parsing tables with trailing parsers
        state: Current parser state with a leftmost expression on the stack
        
    Returns:
        Updated parser state after all trailing parsers have been applied
    """
    loop_count = 0
    while True:
        loop_count += 1
        
        if len(state.stx_stack) == 0:
            # Nothing on the stack to operate on
            break
            
        # Save initial state for potential error recovery
        initial_size = len(state.stx_stack)
        initial_pos = state.pos
        
        # Save the left node before peeking at tokens
        left = state.stx_stack[-1]
        
        # Check if any trailing parsers are applicable
        _, parsers = indexed(tables.trailing_table, state)
        
        if state.has_error():
            # Discard token parse errors and break the trailing loop
            state = state.restore(initial_size, initial_pos)
            break
        
        # If no applicable trailing parsers, we're done
        if not parsers and not tables.trailing_parsers:
            break
        
        # Remove left node from stack - trailing parsers will handle it
        # The parser will be called with the left node as its context parameter
        left_node = state.stx_stack.pop()
        
        # Apply a trailing parser - note that we pass the left_node as the first parser parameter
        state = trailing_loop_step(tables, left_node, state)
        
        if state.has_error():
            # Handle error based on whether we made progress
            if state.pos == initial_pos:
                # Non-consuming error - restore state and put left back
                state = state.restore(initial_size - 1, initial_pos)
                state.stx_stack.append(left)
                break
            else:
                # Consuming error - just fail
                return state
    
    return state


def pratt_parser(kind: str, tables: PrattParsingTables, state: ParserState) -> ParserState:
    """Run a Pratt parser with the given tables and initial state.
    
    This implements the core of Pratt parsing:
    1. Parse a leading (prefix/atomic) expression
    2. Repeatedly apply trailing (infix/postfix) parsers as long as possible
    
    Args:
        kind: The syntax node kind to create for the expression
        tables: The Pratt parsing tables with leading and trailing parsers
        state: Initial parser state
        
    Returns:
        Updated parser state with a parsed expression on the stack
    """
    # First phase: Parse a leading expression
    state = leading_parser(kind, tables, state)
    
    if state.has_error():
        return state
    
    # Second phase: Parse trailing (infix/postfix) expressions/operators
    return trailing_loop(tables, state)
