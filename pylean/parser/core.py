# pylean/parser/core.py

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Tuple
from typing import Callable, TypeVar
import dataclasses
import copy # For deep copying states if needed

# --- Abstract Syntax Tree (AST) Node ---

@dataclass
class SourceInfo:
    """Basic source location information."""
    line: int
    column: int
    pos: int # Character offset from the start of the file
    end_pos: int

# A simplified representation of Lean's `Syntax` object
@dataclass
class SyntaxNode:
    kind: str # e.g., 'atom', 'ident', 'node'
    info: Optional[SourceInfo] = None
    # For atoms/identifiers
    value: Optional[str] = None
    # For compound nodes
    children: List['SyntaxNode'] = field(default_factory=list)
    # Lean-specific: Name for identifiers
    name_val: Optional[str] = None # Simplified Name representation
    # Store start position and end position
    span: Optional[Tuple[int, int]] = None

# --- Parsing Error ---

@dataclass
class ParseError:
    pos: int
    unexpected: str = ""
    expected: List[str] = field(default_factory=list)
    message: str = "" # More general message

    def __str__(self):
        if self.message:
            return f"Error at pos {self.pos}: {self.message}"
        parts = []
        if self.unexpected:
            parts.append(f"unexpected '{self.unexpected}'")
        if self.expected:
            exp_str = " or ".join(f"'{e}'" for e in self.expected)
            parts.append(f"expected {exp_str}")
        return f"Error at pos {self.pos}: {'; '.join(parts)}"

# --- Parser Context and State ---

@dataclass
class FileMap:
    """Maps character positions to line/column numbers."""
    source: str
    lines: List[int] = field(init=False) # Start position of each line

    def __post_init__(self):
        self.lines = [0] + [i + 1 for i, char in enumerate(self.source) if char == '\n']

    def get_line_col(self, pos: int) -> Tuple[int, int]:
        line = 0
        while line + 1 < len(self.lines) and self.lines[line + 1] <= pos:
            line += 1
        col = pos - self.lines[line]
        return line + 1, col + 1 # 1-based indexing for lines/cols

@dataclass
class InputContext:
    """Information about the input source."""
    input: str
    file_name: str
    file_map: FileMap = field(init=False)

    def __post_init__(self):
        self.file_map = FileMap(self.input)

@dataclass
class ParserContext:
    """Immutable context for the parser."""
    input_ctx: InputContext
    # We'll add environment info later if needed
    # curr_namespace: str = ""
    # open_decls: List[Any] = field(default_factory=list)
    prec: int = 0 # Current precedence level

@dataclass
class ParserState:
    """Mutable state during parsing."""
    context: ParserContext
    pos: int = 0
    stx_stack: List[SyntaxNode] = field(default_factory=list)
    error: Optional[ParseError] = None
    # For Pratt parsing
    lhs_prec: int = 0
    # Token cache for reuse
    token_cache: 'TokenCache' = field(default_factory=lambda: TokenCache())

    def has_error(self) -> bool:
        return self.error is not None

    def set_error(self, error: ParseError):
        # Keep the first error encountered at the furthest position
        if self.error is None or error.pos > self.error.pos:
            self.error = error
        elif error.pos == self.error.pos:
             # Merge expected lists if errors are at the same position
             self.error.expected.extend(e for e in error.expected if e not in self.error.expected)
             if not self.error.unexpected and error.unexpected:
                 self.error.unexpected = error.unexpected
             # Prioritize more specific 'unexpected' token if one exists
             if not self.error.message and error.message:
                 self.error.message = error.message

    def is_eof(self) -> bool:
        return self.pos >= len(self.context.input_ctx.input)

    def current_char(self) -> Optional[str]:
        if self.is_eof():
            return None
        return self.context.input_ctx.input[self.pos]

    def next_char(self) -> Optional[str]:
         if self.pos + 1 >= len(self.context.input_ctx.input):
             return None
         return self.context.input_ctx.input[self.pos + 1]
             
    def make_error(self, message: str, expected: Optional[List[str]] = None) -> 'ParserState':
        """Create an error at the current position."""
        if expected is None:
            expected = []
        self.set_error(ParseError(pos=self.pos, message=message, expected=expected))
        return self
    
    def set_pos(self, new_pos: int) -> 'ParserState':
        """Set the position to a new value."""
        self.pos = new_pos
        return self
    
    def restore(self, stack_size: int, position: int) -> 'ParserState':
        """Restore the state to a previous parser position (rewind)."""
        # Truncate the syntax stack
        while len(self.stx_stack) > stack_size:
            self.stx_stack.pop()
        
        # Reset position
        self.pos = position
        return self
        
    def clone(self) -> 'ParserState':
        """Create a copy of the current parser state."""
        new_state = ParserState(
            context=self.context,
            pos=self.pos,
            # Create a new list with the same elements
            stx_stack=[node for node in self.stx_stack],
            error=self.error,
            lhs_prec=self.lhs_prec
        )
        # Also clone the token cache
        if hasattr(self, 'token_cache'):
            new_state.token_cache = TokenCache(
                start_pos=self.token_cache.start_pos,
                stop_pos=self.token_cache.stop_pos,
                token=self.token_cache.token
            )
        return new_state

# --- Parser Function Type ---
ParserFn = Callable[['ParserContext', 'ParserState'], 'ParserState']

# --- Parser Class ---

@dataclass
class Parser:
    """Wraps a parser function."""
    fn: ParserFn
    # We can add info fields later if needed for optimization/debugging

    def __call__(self, context: 'ParserContext', state: 'ParserState') -> 'ParserState':
        """Executes the parser."""
        return self.fn(context, state)

    def __rrshift__(self, other): # Implement >> (for sequencing, less common in Python style)
        if isinstance(other, Parser):
            return and_then(other, self)
        return NotImplemented

    def __or__(self, other): # Implement | (for alternatives)
         if isinstance(other, Parser):
             return or_else(self, other)
         return NotImplemented

# --- Basic Combinators ---

def and_then(p1: Parser, p2: Parser) -> Parser:
    """Runs p1 followed by p2. Fails if p1 fails."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        state1 = p1(context, state)
        if state1.has_error():
            return state1
        return p2(context, state1)
    return Parser(fn)

def or_else(p1: Parser, p2: Parser) -> Parser:
    """Tries p1. If it fails *without consuming input*, tries p2."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        initial_pos = state.pos
        initial_stack_size = len(state.stx_stack)
        # Use deepcopy if state contains mutable objects that p1 might change even on failure
        # state_copy_for_p1 = copy.deepcopy(state)
        state1 = p1(context, state) # Or use state_copy_for_p1

        if not state1.has_error():
            return state1 # p1 succeeded

        # p1 failed, check if it consumed input
        if state1.pos != initial_pos:
            # p1 failed *after* consuming input. Commit to this branch and return the error.
            return state1
        else:
            # p1 failed *without* consuming input. Try p2.
            # Restore state for p2 (stack is implicitly restored as it wasn't changed, pos is same)
            # Keep error from p1 for potential merging
            state_for_p2 = dataclasses.replace(state, error=state1.error, stx_stack=state.stx_stack[:initial_stack_size]) # Ensure stack reset
            state2 = p2(context, state_for_p2)

            # Error merging: If p2 also failed at the *same initial position*
            if state2.has_error() and state1.error and state2.error.pos == initial_pos:
                 # Merge expected lists
                 merged_error = dataclasses.replace(state1.error)
                 merged_error.expected.extend(e for e in state2.error.expected if e not in merged_error.expected)
                 # Prioritize more specific 'unexpected' token if one exists
                 if not merged_error.unexpected and state2.error.unexpected:
                     merged_error.unexpected = state2.error.unexpected
                 # Prioritize more specific message if one exists
                 if not merged_error.message and state2.error.message:
                     merged_error.message = state2.error.message

                 final_state = dataclasses.replace(state, error=merged_error) # Final state based on original 'state'
                 final_state.stx_stack = state.stx_stack[:initial_stack_size] # Ensure stack is reset
                 return final_state
            else:
                 # Return state2 whether it succeeded or failed (at a different position)
                 return state2

    return Parser(fn)

def node(kind: str, p: Parser) -> Parser:
    """Runs p, then wraps the resulting nodes in a new node of the given kind."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        initial_stack_size = len(state.stx_stack)
        start_pos = state.pos
        state1 = p(context, state)
        if state1.has_error():
            # Restore stack if p failed
            state1.stx_stack = state.stx_stack[:initial_stack_size]
            return state1

        end_pos = state1.pos
        # Collect nodes pushed by p
        children = state1.stx_stack[initial_stack_size:]
        # Pop children
        state1.stx_stack = state1.stx_stack[:initial_stack_size]

        # Create source info (simplified)
        start_line, start_col = context.input_ctx.file_map.get_line_col(start_pos)
        end_line, end_col = context.input_ctx.file_map.get_line_col(end_pos) # Approx end col
        info = SourceInfo(line=start_line, column=start_col, pos=start_pos, end_pos=end_pos)

        # Create the new node
        new_node = SyntaxNode(kind=kind, info=info, children=children)
        state1.stx_stack.append(new_node)
        return state1
    return Parser(fn)

def atomic(p: Parser) -> Parser:
    """Runs p. If p fails after consuming input, resets pos before failing."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        initial_pos = state.pos
        # We need to potentially restore the stack as well if p modifies it before failing
        initial_stack_size = len(state.stx_stack)
        state1 = p(context, state) # Run p on the current state
        if state1.has_error() and state1.pos != initial_pos:
            # Failed after consuming input, reset position *and* stack, but keep the error
            final_state = dataclasses.replace(state1, pos=initial_pos)
            final_state.stx_stack = state.stx_stack[:initial_stack_size] # Restore stack
            return final_state
        else:
            # Succeeded, or failed without consuming input
            return state1
    return Parser(fn)

def optional(p: Parser) -> Parser:
    """Tries p. Succeeds always, pushing a 'null' node if p failed without consumption,
       or if p succeeded, or if p failed *with* consumption (Lean behavior)."""
    null_kind = "syntax_null" # Or some other indicator
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        initial_pos = state.pos
        initial_stack_size = len(state.stx_stack)
        state1 = p(context, state)

        if state1.has_error() and state1.pos == initial_pos:
            # p failed without consuming input. Succeed, push null node.
            final_state = dataclasses.replace(state, error=None) # Clear error
            final_state.stx_stack = state.stx_stack[:initial_stack_size] # Restore stack
            start_line, start_col = context.input_ctx.file_map.get_line_col(initial_pos)
            info = SourceInfo(line=start_line, column=start_col, pos=initial_pos, end_pos=initial_pos)
            final_state.stx_stack.append(SyntaxNode(kind=null_kind, info=info, children=[]))
            return final_state
        elif state1.has_error():
            # p failed *after* consuming input. Lean clears the error and pushes a null node
            # spanning the range where the error occurred.
            final_state = dataclasses.replace(state1, error=None) # Clear error!
            final_state.stx_stack = state.stx_stack[:initial_stack_size] # Restore stack to before p
            start_line, start_col = context.input_ctx.file_map.get_line_col(initial_pos)
            end_line, end_col = context.input_ctx.file_map.get_line_col(state1.pos) # End where p failed
            info = SourceInfo(line=start_line, column=start_col, pos=initial_pos, end_pos=state1.pos)
            final_state.stx_stack.append(SyntaxNode(kind=null_kind, info=info, children=[]))
            return final_state
        else:
            # p succeeded. Wrap its result(s) in a null node.
            children = state1.stx_stack[initial_stack_size:]
            final_state = state1 # Keep state from successful parse (pos, etc.)
            final_state.stx_stack = state.stx_stack[:initial_stack_size] # Pop children produced by p
            start_line, start_col = context.input_ctx.file_map.get_line_col(initial_pos)
            end_line, end_col = context.input_ctx.file_map.get_line_col(state1.pos) # End where p succeeded
            info = SourceInfo(line=start_line, column=start_col, pos=initial_pos, end_pos=state1.pos)
            final_state.stx_stack.append(SyntaxNode(kind=null_kind, info=info, children=children))
            return final_state

    return Parser(fn)

def lookahead(p: Parser) -> Parser:
    """Runs p, but resets state afterwards. Fails if p fails."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        initial_stack_size = len(state.stx_stack)
        state1 = p(context, state) # Run p on the current state
        # Regardless of success/failure, create the final state based on the *original* state
        final_state = dataclasses.replace(state, error=None) # Start with original state, clear error
        final_state.stx_stack = state.stx_stack[:initial_stack_size] # Ensure stack is original

        if state1.has_error():
            # p failed, propagate the error but keep original state
            final_state.error = state1.error
            return final_state
        else:
            # p succeeded, return original state with no error
            return final_state
    return Parser(fn)


def not_followed_by(p: Parser, description: str) -> Parser:
    """Succeeds if p fails, fails if p succeeds. Always resets state."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        initial_stack_size = len(state.stx_stack)
        state1 = p(context, state) # Try p

        # Always restore original state (pos, stack)
        final_state = dataclasses.replace(state, error=None)
        final_state.stx_stack = state.stx_stack[:initial_stack_size]

        if state1.has_error():
            # p failed, so notFollowedBy succeeds. Return original state.
            return final_state
        else:
            # p succeeded, so notFollowedBy fails. Return original state with error.
            final_state.set_error(ParseError(pos=state.pos, message=f"Unexpected {description}"))
            return final_state
    return Parser(fn)

def many(p: Parser) -> Parser:
    """Runs p zero or more times, collecting results in a node."""
    kind = "many"
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        initial_stack_size = len(state.stx_stack)
        start_pos = state.pos
        current_state = state
        iteration_states = [] # Keep track of states *after* each successful p

        while True:
            loop_start_pos = current_state.pos
            loop_start_stack_size = len(current_state.stx_stack)
            next_state = p(context, current_state)

            if next_state.has_error():
                # If p failed *without consuming input*, stop gracefully.
                if next_state.pos == loop_start_pos:
                    # Clear the error from the non-consuming failure.
                    current_state = dataclasses.replace(current_state, error=None)
                    break # Exit loop, current_state is the state before the failed attempt
                else:
                    # p failed *after consuming input*. This is an error in 'many'.
                    # We should probably return this error state.
                    # TODO: Consider if Lean bundles this error into the 'many' node.
                    # For now, propagate the error.
                    return next_state # Propagate error that occurred *during* many
            elif next_state.pos == loop_start_pos:
                 # p succeeded but didn't consume input (e.g., optional(epsilon)). Stop loop.
                 current_state = next_state # Keep the result of the non-consuming success
                 break # Stop loop
            else:
                 # p succeeded and consumed input, continue loop
                 iteration_states.append(next_state) # Store state *after* success
                 current_state = next_state

        # Loop finished. current_state is the state *after* the last successful or non-consuming parse.
        end_pos = current_state.pos
        # The children are the *differences* in the stack between the start and the final state.
        children = current_state.stx_stack[initial_stack_size:]

        # Reset the stack to its initial size before adding the 'many' node.
        current_state.stx_stack = current_state.stx_stack[:initial_stack_size]

        start_line, start_col = context.input_ctx.file_map.get_line_col(start_pos)
        end_line, end_col = context.input_ctx.file_map.get_line_col(end_pos)
        info = SourceInfo(line=start_line, column=start_col, pos=start_pos, end_pos=end_pos)

        # Add the 'many' node containing all children collected.
        current_state.stx_stack.append(SyntaxNode(kind=kind, info=info, children=children))

        # Ensure any error from the final state (e.g., a non-consuming failure error that was cleared) is gone.
        final_state = dataclasses.replace(current_state, error=None) # Ensure no lingering error from break condition

        return final_state

    return Parser(fn)


def many1(p: Parser, message: str = "Expected at least one item") -> Parser:
    """Runs p one or more times (equivalent to p >> many(p) conceptually)."""
    kind = "many1"
    def fn(context: ParserContext, state: ParserState) -> ParserState:
         initial_stack_size = len(state.stx_stack)
         start_pos = state.pos

         # --- Parse first item (p) ---
         state1 = p(context, state)
         if state1.has_error():
             # Failed on the very first item. Replace error with specific many1 error.
             final_error = ParseError(pos=start_pos, message=message, expected=state1.error.expected if state1.error else [])
             # Reset state completely to before p was tried
             final_state = dataclasses.replace(state, error=final_error)
             final_state.stx_stack = state.stx_stack[:initial_stack_size]
             return final_state
         elif state1.pos == start_pos:
              # p succeeded but didn't consume input on the first required try. Error.
              final_error = ParseError(pos=start_pos, message=f"{message} (parser did not consume input)")
              final_state = dataclasses.replace(state, error=final_error)
              final_state.stx_stack = state.stx_stack[:initial_stack_size]
              return final_state

         # --- Parse remaining items (many(p)) ---
         current_state = state1 # Start loop from state *after* first success
         while True:
            loop_start_pos = current_state.pos
            loop_start_stack_size = len(current_state.stx_stack)
            next_state = p(context, current_state)

            if next_state.has_error():
                if next_state.pos == loop_start_pos:
                    current_state = dataclasses.replace(current_state, error=None) # Clear error
                    break # Stop gracefully
                else:
                    return next_state # Propagate error from within loop
            elif next_state.pos == loop_start_pos:
                 current_state = next_state # Keep result
                 break # Stop loop
            else:
                 current_state = next_state # Continue loop

         # --- Wrap results ---
         end_pos = current_state.pos
         # Children are everything pushed after the initial stack state
         children = current_state.stx_stack[initial_stack_size:]
         current_state.stx_stack = current_state.stx_stack[:initial_stack_size] # Pop children

         start_line, start_col = context.input_ctx.file_map.get_line_col(start_pos)
         end_line, end_col = context.input_ctx.file_map.get_line_col(end_pos)
         info = SourceInfo(line=start_line, column=start_col, pos=start_pos, end_pos=end_pos)

         current_state.stx_stack.append(SyntaxNode(kind=kind, info=info, children=children))
         final_state = dataclasses.replace(current_state, error=None) # Clear any loop exit error

         return final_state

    return Parser(fn)

def fail(message: str, expected: Optional[List[str]] = None) -> Parser:
    """Parser that always fails."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        state.set_error(ParseError(pos=state.pos, message=message, expected=expected or []))
        return state
    # Set a distinct function name for easier debugging if needed
    fn.__name__ = f"fail_{message[:10].replace(' ','_')}"
    return Parser(fn)

def check_prec(prec: int) -> Parser:
    """Succeeds if context.prec <= prec."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        if context.prec <= prec:
            return state
        else:
            state.set_error(ParseError(pos=state.pos, expected=[f"token at precedence <= {prec}"]))
            return state
    return Parser(fn)

def check_lhs_prec(prec: int) -> Parser:
    """Succeeds if state.lhs_prec >= prec."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        if state.lhs_prec >= prec:
            return state
        else:
            # This error often implies missing parentheses
            state.set_error(ParseError(pos=state.pos, message=f"Unexpected token at this precedence level (lhs_prec {state.lhs_prec} < {prec}); consider parenthesizing the term"))
            return state
    return Parser(fn)

def set_lhs_prec(prec: int) -> Parser:
    """Sets state.lhs_prec = prec."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        if state.has_error(): return state # Don't change state if already failed
        # Create a new state object to maintain functional style if needed,
        # or modify in place if performance is critical and state is mutable class
        new_state = dataclasses.replace(state, lhs_prec=prec)
        return new_state
    return Parser(fn)

def leading_node(kind: str, prec: int, p: Parser) -> Parser:
    """Common pattern: check_prec(prec) >> node(kind, p) >> set_lhs_prec(prec)."""
    return and_then(check_prec(prec), and_then(node(kind, p), set_lhs_prec(prec)))

# Helper for epsilon parser (succeeds without consuming input)
def epsilon() -> Parser:
    """A parser that succeeds without consuming input or changing the stack."""
    return Parser(lambda ctx, state: state)

# Placeholder for adapting context
T = TypeVar('T')
def adapt_context(adapt_fn: Callable[[ParserContext], ParserContext], p: Parser) -> Parser:
     """Runs parser p with a modified context."""
     def fn(context: ParserContext, state: ParserState) -> ParserState:
         new_context = adapt_fn(context)
         # Pass the *original* context to the state's constructor/replace
         # The state holds a reference to the context it was created with.
         # We only modify the context *passed into* p.
         new_state = dataclasses.replace(state, context=new_context)
         # Run p with the modified context via the modified state
         result_state = p(new_context, new_state)
         # Restore the original context in the final state, keep other changes
         return dataclasses.replace(result_state, context=context)
     return Parser(fn)

def with_prec(prec: int, p: Parser) -> Parser:
     """Runs parser p with context.prec set to prec."""
     return adapt_context(lambda ctx: dataclasses.replace(ctx, prec=prec), p)

def trailing_node(kind: str, prec: int, lhs_prec: int, p: Parser) -> Parser:
     """Check prec, check lhsPrec, run p, combine with node on stack, set new lhsPrec."""
     # Simplified trailing_node_aux from before:
     def trailing_node_aux(kind: str, p: Parser) -> Parser:
        def fn(context: ParserContext, state: ParserState) -> ParserState:
            if not state.stx_stack:
                 state.set_error(ParseError(pos=state.pos, message=f"Syntax stack empty for trailing node '{kind}'"))
                 return state

            initial_stack_size = len(state.stx_stack) # Should be >= 1
            left_operand = state.stx_stack[-1] # Peek, don't pop yet
            start_pos = left_operand.info.pos if left_operand.info else state.pos # Approx start

            state1 = p(context, state)
            if state1.has_error():
                # Restore stack (only nodes added by p, keep original left operand)
                state1.stx_stack = state.stx_stack[:initial_stack_size]
                return state1

            end_pos = state1.pos
            # Collect nodes pushed by p
            children_from_p = state1.stx_stack[initial_stack_size:]
            state1.stx_stack = state1.stx_stack[:initial_stack_size - 1] # Pop left and p's children conceptually

            all_children = [left_operand] + children_from_p
            final_end_pos = max(child.info.end_pos for child in all_children if child.info) if all_children else end_pos
            start_line, start_col = context.input_ctx.file_map.get_line_col(start_pos)
            end_line, end_col = context.input_ctx.file_map.get_line_col(final_end_pos)
            info = SourceInfo(line=start_line, column=start_col, pos=start_pos, end_pos=final_end_pos)

            new_node = SyntaxNode(kind=kind, info=info, children=all_children)
            state1.stx_stack.append(new_node)
            return state1
        return Parser(fn)

     # Combine checks and aux node creation
     p_checked = and_then(check_prec(prec),
                   and_then(check_lhs_prec(lhs_prec),
                            trailing_node_aux(kind, p)))
     return and_then(p_checked, set_lhs_prec(prec))

# --- Character Level Parsers ---

def satisfy(predicate: Callable[[str], bool], description: str = "character") -> Parser:
    """Consumes and returns the next character if it satisfies the predicate."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        if state.is_eof():
            state.set_error(ParseError(pos=state.pos, expected=[description], message="end of input"))
            return state

        char = state.context.input_ctx.input[state.pos]
        if predicate(char):
            # Create an atom node for the character
            start_pos = state.pos
            end_pos = start_pos + 1
            start_line, start_col = context.input_ctx.file_map.get_line_col(start_pos)
            end_line, end_col = context.input_ctx.file_map.get_line_col(end_pos)
            info = SourceInfo(line=start_line, column=start_col, pos=start_pos, end_pos=end_pos)
            atom_node = SyntaxNode(kind="atom", info=info, children=[], value=char)

            next_state = dataclasses.replace(state, pos=end_pos, error=None)
            next_state.stx_stack.append(atom_node)
            return next_state
        else:
            state.set_error(ParseError(pos=state.pos, expected=[description], unexpected=char))
            return state
    return Parser(fn)

def ch(char_to_match: str) -> Parser:
    """Parses a specific character."""
    return satisfy(lambda c: c == char_to_match, description=f"'{char_to_match}'")

def string(s_to_match: str) -> Parser:
    """Parses a specific string."""
    if not s_to_match:
        return epsilon() # Parsing an empty string always succeeds

    def fn(context: ParserContext, state: ParserState) -> ParserState:
        start_pos = state.pos
        if state.pos + len(s_to_match) > len(state.context.input_ctx.input):
            state.set_error(ParseError(pos=state.pos, expected=[f"'{s_to_match}'"], message="end of input"))
            return state

        matched_str = state.context.input_ctx.input[state.pos : state.pos + len(s_to_match)]
        if matched_str == s_to_match:
             end_pos = state.pos + len(s_to_match)
             start_line, start_col = context.input_ctx.file_map.get_line_col(start_pos)
             end_line, end_col = context.input_ctx.file_map.get_line_col(end_pos)
             info = SourceInfo(line=start_line, column=start_col, pos=start_pos, end_pos=end_pos)
             # Often, string matches act as delimiters and don't create nodes,
             # but let's create an atom for consistency for now.
             atom_node = SyntaxNode(kind="atom", info=info, children=[], value=s_to_match)

             next_state = dataclasses.replace(state, pos=end_pos, error=None)
             next_state.stx_stack.append(atom_node)
             return next_state
        else:
            state.set_error(ParseError(pos=state.pos, expected=[f"'{s_to_match}'"], unexpected=matched_str[:1]))
            return state
    return Parser(fn)

def take_while(predicate: Callable[[str], bool]) -> Parser:
    """Consumes zero or more characters satisfying the predicate."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        start_pos = state.pos
        current_pos = start_pos
        input_str = state.context.input_ctx.input
        while current_pos < len(input_str) and predicate(input_str[current_pos]):
            current_pos += 1

        # Only create a node if input was consumed
        if current_pos > start_pos:
            matched_str = input_str[start_pos:current_pos]
            start_line, start_col = context.input_ctx.file_map.get_line_col(start_pos)
            end_line, end_col = context.input_ctx.file_map.get_line_col(current_pos)
            info = SourceInfo(line=start_line, column=start_col, pos=start_pos, end_pos=current_pos)
            atom_node = SyntaxNode(kind="atom", info=info, children=[], value=matched_str)

            next_state = dataclasses.replace(state, pos=current_pos, error=None)
            next_state.stx_stack.append(atom_node)
            return next_state
        else:
             # Succeeded without consuming input
             return dataclasses.replace(state, error=None)

    return Parser(fn)

def take_while1(predicate: Callable[[str], bool], error_msg: str) -> Parser:
    """Consumes one or more characters satisfying the predicate."""
    p = take_while(predicate)
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        start_pos = state.pos
        state1 = p(context, state)
        if state1.has_error(): # Should not happen with take_while
            return state1
        if state1.pos == start_pos:
             # Did not consume any input
             state.set_error(ParseError(pos=start_pos, message=error_msg))
             return state
        else:
             # Consumed at least one character
             return state1
    return Parser(fn)


# --- Whitespace and Comments ---

def _finish_block_comment(nesting: int) -> Parser:
    """Auxiliary parser for nested block comments."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        start_pos = state.pos
        input_str = state.context.input_ctx.input
        current_pos = start_pos
        current_nesting = nesting

        while current_pos < len(input_str):
            if current_pos + 2 <= len(input_str):
                sub = input_str[current_pos : current_pos + 2]
                if sub == "-/":
                    current_nesting -= 1
                    current_pos += 2
                    if current_nesting == 0:
                        # Successfully closed the comment block
                        # Don't create a node, just update position
                        return dataclasses.replace(state, pos=current_pos, error=None)
                elif sub == "/-":
                    current_nesting += 1
                    current_pos += 2
                else:
                    current_pos += 1 # Advance by one character
            else:
                 current_pos += 1 # Advance by one character

        # Reached EOF without closing comment
        state.set_error(ParseError(pos=start_pos, message="unterminated block comment"))
        return state

    # Create a parser that doesn't create a node, just consumes
    p = Parser(fn)
    def wrapper_fn(context: ParserContext, state: ParserState) -> ParserState:
        initial_stack_size = len(state.stx_stack)
        state1 = p(context, state)
        # Restore stack even on error to discard partial work if needed
        state1.stx_stack = state.stx_stack[:initial_stack_size]
        # Propagate the error if the comment was unterminated
        return state1
    return Parser(wrapper_fn)


def _block_comment() -> Parser:
    """Parses '/-' followed by a nested block comment."""
    # Parses '/-', then starts the recursive finish_block_comment
    # Doesn't produce a node
    p = and_then(string("/-"), _finish_block_comment(nesting=1))
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        initial_stack_size = len(state.stx_stack)
        state1 = p(context, state)
        if not state1.has_error(): # Only restore stack on success
            state1.stx_stack = state.stx_stack[:initial_stack_size]
        # Propagate error if '/-' failed or comment was unterminated
        return state1
    return Parser(fn)


def _line_comment() -> Parser:
    """Parses '--' or '/--' followed by the rest of the line."""
    # Doesn't produce a node
    p_dash = and_then(string("--"), take_while(lambda c: c != '\n'))
    p_slash = and_then(string("/--"), take_while(lambda c: c != '\n')) # Doc comment
    p = or_else(p_dash, p_slash)

    def fn(context: ParserContext, state: ParserState) -> ParserState:
         # We run 'p' but discard its result (the atom nodes it creates)
         initial_stack_size = len(state.stx_stack)
         state1 = p(context, state)
         # Error could occur if or_else fails (input doesn't start with -- or /--)
         if not state1.has_error():
             # Success, restore stack to discard nodes created by p
             state1.stx_stack = state.stx_stack[:initial_stack_size]
         return state1 # Propagate error or return success state
    return Parser(fn)

def _doc_block_comment() -> Parser:
     """Parses '/-!' followed by a nested block comment (Doc comment)."""
     # Doesn't produce a node
     p = and_then(string("/-!"), _finish_block_comment(nesting=1))
     def fn(context: ParserContext, state: ParserState) -> ParserState:
         initial_stack_size = len(state.stx_stack)
         state1 = p(context, state)
         if not state1.has_error(): # Only restore stack on success
            state1.stx_stack = state.stx_stack[:initial_stack_size]
         # Propagate error if '/-!' failed or comment was unterminated
         return state1
     return Parser(fn)


def _whitespace_char() -> Parser:
     """Consumes one whitespace character (space or newline)."""
     # Doesn't produce a node
     # TODO: Lean checks for invalid chars like tabs here.
     p = satisfy(lambda c: c.isspace(), "whitespace")
     def fn(context: ParserContext, state: ParserState) -> ParserState:
         initial_stack_size = len(state.stx_stack)
         state1 = p(context, state)
         if not state1.has_error():
             state1.stx_stack = state.stx_stack[:initial_stack_size]
         return state1
     return Parser(fn)


def whitespace() -> Parser:
    """Consumes whitespace and comments. Does not produce a node."""
    # Parses zero or more occurrences of (ws | line_comment | block_comment | doc_block_comment)
    # Order matters: check for block comment starts before line comment starts
    comment_parsers = or_else(_block_comment(), or_else(_doc_block_comment(), _line_comment()))
    single_ws_or_comment = or_else(_whitespace_char(), comment_parsers)

    p = many(single_ws_or_comment)

    def fn(context: ParserContext, state: ParserState) -> ParserState:
        # Run `many`, but discard the resulting 'many' node and its children.
        initial_stack_size = len(state.stx_stack)
        state1 = p(context, state)
        # `many` itself doesn't produce errors unless a child parser fails *after consuming input*,
        # which shouldn't happen here as our base parsers either succeed, fail without consuming,
        # or propagate errors like unterminated comments.
        # Restore the stack regardless.
        state1.stx_stack = state.stx_stack[:initial_stack_size]
        # Clear any error from `many` itself (though unlikely) and return the state.
        final_state = dataclasses.replace(state1, error=None)
        return final_state
    return Parser(fn)


# --- Token Parsers ---

def mk_token_node(kind: str, start_pos: int) -> Callable[[ParserContext, ParserState], ParserState]:
    """
    Helper function to be used with and_then after a raw token is parsed.
    Does not consume trailing whitespace and creates a SyntaxNode atom.
    Assumes the state's current `pos` marks the end of the raw token.
    """
    def post_process(context: ParserContext, state: ParserState) -> ParserState:
        if state.has_error():
            return state # Error occurred during raw token parsing

        token_end_pos = state.pos
        token_val = context.input_ctx.input[start_pos:token_end_pos]

        # Create source info
        # For tokens, leading trivia is usually handled by the preceding whitespace call,
        # so we assume empty leading trivia here relative to the token start.
        start_line, start_col = context.input_ctx.file_map.get_line_col(start_pos)
        # End pos for info is the end of the raw token *before* whitespace
        info = SourceInfo(line=start_line, column=start_col, pos=start_pos, end_pos=token_end_pos)

        # Create the atom node
        atom_node = SyntaxNode(kind=kind, info=info, children=[], value=token_val)

        # Update state: Use token_end_pos, clear error (might not be necessary), push node
        # Restore original stack before pushing
        final_state = dataclasses.replace(state, pos=token_end_pos, error=None)
        final_state.stx_stack.append(atom_node)
        return final_state

    return post_process


def symbol_core(sym: str) -> Parser:
    """Parses the raw string `sym`."""
    # This just parses the string, doesn't handle whitespace or node creation yet.
    return string(sym)


def symbol(sym: str) -> Parser:
    """Parses the symbol `sym`, consumes trailing whitespace, and creates a token node."""
    if not sym:
         raise ValueError("Cannot parse empty symbol")

    raw_parser = symbol_core(sym)

    def fn(context: ParserContext, state: ParserState) -> ParserState:
        start_pos = state.pos
        # 1. Try parsing the raw symbol
        state1 = raw_parser(context, state)
        if state1.has_error():
             return state1 # Failed to match the symbol string

        # 2. Check if it's followed by an invalid character (heuristic from Lean)
        # Avoid matching "foo" as a prefix of "foobar" if "foobar" could be an ident.
        # TODO: Refine this check based on Lean's `identFn` logic when we implement it.
        # For now, just check if the next char is alphanumeric.
        next_pos = state1.pos
        input_str = context.input_ctx.input
        if next_pos < len(input_str):
            next_char = input_str[next_pos]
            # A simple check: if the symbol could be a prefix of an identifier, fail.
            # This is not perfect but a starting point.
            if sym.isalnum() and next_char.isalnum():
                 # Create error on the original state
                 error_state = dataclasses.replace(state, error=ParseError(pos=start_pos, message=f"Symbol '{sym}' is prefix of identifier"))
                 return error_state

        # 3. If successful and check passes, consume whitespace and make node
        return mk_token_node("atom", start_pos)(context, state1)

    return Parser(fn)

def is_symbol_char(char: str) -> bool:
    """Checks if a character can be part of a symbol."""
    # Add other valid symbol characters as needed
    return char in "+-*/=<>!?:." or not char.isalnum() and not char.isspace() and char not in "()[]{}"

def token_fn(expected: Optional[List[str]] = None):
    """Parses the next token based on the current character."""
    if expected is None:
        expected = []
    
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        # Check if we're at EOF
        if state.is_eof():
            return state.make_error(f"Unexpected end of input{', expected ' + ', '.join(expected) if expected else ''}")
        
        # Check if we have a cached token at this position
        if state.token_cache is not None and state.token_cache.start_pos == state.pos:
            state.stx_stack.append(state.token_cache.token)
            return state.set_pos(state.token_cache.stop_pos)
        
        start_pos = state.pos
        curr = state.current_char()
        
        # Original state for restoring if needed
        original_state = state.clone()
        
        # Attempt to parse different token types
        if curr.isdigit():
            # Number literal
            state1 = number_literal()(context, state)
            # Cache successful token
            if not state1.has_error() and len(state1.stx_stack) > 0:
                token = state1.stx_stack[-1]
                state1.token_cache = TokenCache(start_pos, state1.pos, token)
            return state1
            
        elif is_id_first(curr) or is_id_escape_start(curr):
            # Identifier
            state1 = ident()(context, state)
            # Cache successful token
            if not state1.has_error() and len(state1.stx_stack) > 0:
                token = state1.stx_stack[-1]
                state1.token_cache = TokenCache(start_pos, state1.pos, token)
            return state1
            
        # Handle parentheses as individual tokens
        elif curr in "()[]{}":
            # Create token node for the parenthesis or bracket
            state1 = state.clone().set_pos(state.pos + 1)
            # Create source info
            start_line, start_col = context.input_ctx.file_map.get_line_col(start_pos)
            info = SourceInfo(line=start_line, column=start_col, pos=start_pos, end_pos=state1.pos)
            # Create the token node
            token = SyntaxNode(kind="symbol", value=curr, info=info)
            state1.stx_stack.append(token)
            state1.token_cache = TokenCache(start_pos, state1.pos, token)
            return state1
            
        # TODO: Add other token types (strings, chars, etc.)
        # For now, assume it's a symbol
        state1 = take_while1(
            lambda c: not c.isalnum() and not c.isspace() and c not in "()[]{}'\"`\"",
            "symbol"
        )(context, state)
        
        if state1.has_error():
            return state1
            
        # Create a token node
        state1 = mk_token_node("symbol", start_pos)(context, state1)
        
        # Cache the result
        if not state1.has_error() and len(state1.stx_stack) > 0:
            token = state1.stx_stack[-1]
            state1.token_cache = TokenCache(start_pos, state1.pos, token)
            
        return state1
    
    return Parser(fn)


def peek_token_node(context: ParserContext, state: ParserState) -> Optional[SyntaxNode]:
    """
    Peeks at the next token *without* modifying the input state stack or position.
    Handles whitespace and uses/updates the token cache on the passed state.
    Returns the peeked SyntaxNode or None if EOF or error during peek.
    """
    print(f"  peek_token_node: starting with state pos={state.pos}") # ADDED DEBUG
    
    # Consume leading whitespace
    state_after_ws = whitespace()(context, state)
    print(f"  peek_token_node: after whitespace pos={state_after_ws.pos}, error={state_after_ws.error}") # ADDED DEBUG
    
    # Check if we're at EOF after whitespace
    if state_after_ws.is_eof():
        print(f"  peek_token_node: EOF after whitespace") # ADDED DEBUG
        return None
    
    # Even if whitespace() reported an error, continue since it should never actually fail
    # Just clear any potential error from whitespace()
    state_after_ws = dataclasses.replace(state_after_ws, error=None)
    pos_after_ws = state_after_ws.pos

    # Check cache first - ensure token_cache exists before accessing it
    if state_after_ws.token_cache is not None and state_after_ws.token_cache.start_pos == pos_after_ws:
        print(f"  peek_token_node: cache hit for token at pos {pos_after_ws}") # ADDED DEBUG
        return state_after_ws.token_cache.token

    # Cache miss: Use token_fn to parse the token
    print(f"  peek_token_node: cache miss, calling token_fn() at pos {pos_after_ws}") # ADDED DEBUG
    
    # Create a temporary state *copy* to run token_fn without affecting original state's stack/pos
    temp_state = dataclasses.replace(state_after_ws, stx_stack=[])

    # Run token_fn on the temp state
    state1 = token_fn()(context, temp_state)
    print(f"  peek_token_node: token_fn returned state with pos={state1.pos}, error={state1.error}, stack_size={len(state1.stx_stack)}") # ADDED DEBUG

    # Check if token_fn succeeded and produced a token
    if not state1.has_error() and len(state1.stx_stack) > 0:
        parsed_token = state1.stx_stack[-1]
        print(f"  peek_token_node: found token kind={parsed_token.kind}, value={parsed_token.value}") # ADDED DEBUG

        # Let token_fn handle cache updating and return the parsed token
        return parsed_token
    else:
        # Error or no token found by token_fn
        print(f"  peek_token_node: token_fn failed or found no token") # ADDED DEBUG
        return None

@dataclass
class TokenCache:
    """Cache for token parsing results."""
    start_pos: int = -1  # Invalid position by default
    stop_pos: int = -1   # Invalid position by default
    token: Optional[SyntaxNode] = None


# --- Combinator Refinements (using whitespace) ---

def and_then_ws(p1: Parser, p2: Parser) -> Parser:
    """Sequences p1, whitespace, then p2."""
    return and_then(p1, and_then(whitespace(), p2))

# Override the >> operator
Parser.__rshift__ = and_then_ws

# Example usage: symbol("let") >> ident() >> symbol(":=") ...


# --- Precedence and Pratt Parsing Helpers ---

# --- Character Predicates (Simplified) ---

def is_digit(c: str) -> bool:
    return '0' <= c <= '9'

def is_hex_digit(c: str) -> bool:
    return '0' <= c <= '9' or 'a' <= c <= 'f' or 'A' <= c <= 'F'

def is_id_first(c: str) -> bool:
    # Very simplified: letter or underscore
    return c.isalpha() or c == '_'

def is_id_rest(c: str) -> bool:
    # Very simplified: letter, digit, or underscore
    return c.isalnum() or c == '_'

def is_id_escape_start(c: str) -> bool:
    # Placeholder - Lean uses specific unicode chars
    return c == '`' # Using backtick as a simple escape indicator

def is_id_escape_end(c: str) -> bool:
    return c == '`'

# --- Token Parsers ---

def mk_token_node(kind: str, start_pos: int) -> Callable[[ParserContext, ParserState], ParserState]:
    """
    Helper function to be used with and_then after a raw token is parsed.
    Does not consume trailing whitespace and creates a SyntaxNode atom.
    Assumes the state's current `pos` marks the end of the raw token.
    """
    def post_process(context: ParserContext, state: ParserState) -> ParserState:
        if state.has_error():
            return state # Error occurred during raw token parsing

        token_end_pos = state.pos
        token_val = context.input_ctx.input[start_pos:token_end_pos]

        # Create source info
        # For tokens, leading trivia is usually handled by the preceding whitespace call,
        # so we assume empty leading trivia here relative to the token start.
        start_line, start_col = context.input_ctx.file_map.get_line_col(start_pos)
        # End pos for info is the end of the raw token *before* whitespace
        info = SourceInfo(line=start_line, column=start_col, pos=start_pos, end_pos=token_end_pos)

        # Create the atom node
        atom_node = SyntaxNode(kind=kind, info=info, children=[], value=token_val)

        # Update state: Use token_end_pos, clear error (might not be necessary), push node
        # Restore original stack before pushing
        final_state = dataclasses.replace(state, pos=token_end_pos, error=None)
        final_state.stx_stack.append(atom_node)
        return final_state

    return post_process


# --- Number Literal Parser ---

def _take_digits(predicate: Callable[[str], bool], expected_msg: str, need_digit: bool) -> Parser:
    """Consumes digits based on predicate, allowing '_' separators."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        start_pos = state.pos
        current_pos = start_pos
        input_str = state.context.input_ctx.input
        consumed_digit = False

        while current_pos < len(input_str):
            char = input_str[current_pos]
            if predicate(char):
                consumed_digit = True
                current_pos += 1
            elif char == '_' and current_pos + 1 < len(input_str) and predicate(input_str[current_pos + 1]):
                # Allow underscore only if followed by a valid digit
                current_pos += 1 # Skip underscore
            else:
                break # Not a valid digit or separator

        if need_digit and not consumed_digit:
            state.set_error(ParseError(pos=start_pos, message=f"Expected {expected_msg}"))
            return state
        elif current_pos == start_pos and need_digit:
             state.set_error(ParseError(pos=start_pos, message=f"Expected {expected_msg}"))
             return state
        else:
            # Succeed, potentially consuming nothing if need_digit is false
            # Doesn't create a node, just updates position
            return dataclasses.replace(state, pos=current_pos, error=None)

    return Parser(fn)

def _parse_decimal_mantissa() -> Parser:
    """Parses optional '.' followed by digits."""
    p_dot = ch('.')
    p_digits = _take_digits(is_digit, "decimal digits after '.'", need_digit=True)

    # Modified: Make sure to parse the decimal point followed by at least one digit
    # This ensures we properly handle decimals like 3.14
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        # First try to parse the decimal point
        state1 = p_dot(context, state)
        if state1.has_error():
            return state1
            
        # Then try to parse the digits after the decimal point
        state2 = p_digits(context, state1)
        if state2.has_error():
            # If there are no digits after the decimal point, it might not be a decimal number
            # (like in 'x.y' which could be a field access), so propagate the error
            return state2
            
        return state2
        
    return Parser(fn)

def _parse_exponent() -> Parser:
    """Parses 'e'/'E', optional sign, and digits."""
    p_e = or_else(ch('e'), ch('E'))
    p_sign = optional(or_else(ch('+'), ch('-')))
    p_digits = _take_digits(is_digit, "exponent digits", need_digit=True)
    
    # Improved exponent parser with better error handling
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        # First, parse the 'e' or 'E'
        state1 = p_e(context, state)
        if state1.has_error():
            return state1
            
        # Then, try to parse an optional sign
        state2 = p_sign(context, state1)
        # p_sign is optional so it won't fail
        
        # Finally, parse the exponent digits (must have at least one)
        state3 = p_digits(context, state2)
        if state3.has_error():
            return state3
            
        return state3
        
    return Parser(fn)

def _decimal_number(start_pos: int) -> Parser:
    """Parses decimal part, optional scientific part, makes node."""
    p_digits = _take_digits(is_digit, "decimal number", need_digit=True) # Need at least one digit
    p_mantissa = optional(_parse_decimal_mantissa())
    p_exponent = optional(_parse_exponent())

    # Sequence: initial digits, optional mantissa, optional exponent
    # We need to determine if it's scientific based on whether mantissa or exponent parsed.
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        # Parse initial digits (always required)
        state1 = p_digits(context, state)
        if state1.has_error():
            return state1
            
        pos_after_digits = state1.pos

        # Try to parse decimal mantissa (optional)
        state2 = p_mantissa(context, state1)
        # Since p_mantissa is wrapped with optional(), it should never error
        pos_after_mantissa = state2.pos

        # Try to parse exponent (optional)
        state3 = p_exponent(context, state2)
        # Since p_exponent is wrapped with optional(), it should never error
        pos_after_exponent = state3.pos

        # Check if it's a scientific notation or decimal
        is_scientific = pos_after_mantissa > pos_after_digits or pos_after_exponent > pos_after_mantissa
        kind = "num_lit"  # Use num_lit for both regular and scientific (for simplicity)
        
        # Make node using the final state's position
        final_raw_state = state3
        return mk_token_node(kind, start_pos)(context, final_raw_state)

    return Parser(fn)

def _bin_number(start_pos: int) -> Parser:
    p = _take_digits(lambda c: c == '0' or c == '1', "binary digits", need_digit=True)
    return and_then(p, lambda ctx, s: mk_token_node("num_lit", start_pos)(ctx, s))

def _oct_number(start_pos: int) -> Parser:
    p = _take_digits(lambda c: '0' <= c <= '7', "octal digits", need_digit=True)
    return and_then(p, lambda ctx, s: mk_token_node("num_lit", start_pos)(ctx, s))

def _hex_number(start_pos: int) -> Parser:
    p = _take_digits(is_hex_digit, "hexadecimal digits", need_digit=True)
    return and_then(p, lambda ctx, s: mk_token_node("num_lit", start_pos)(ctx, s))

def number_literal() -> Parser:
    """Parses any number literal (dec, bin, oct, hex, scientific)."""
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        start_pos = state.pos
        if state.is_eof():
            state.set_error(ParseError(pos=start_pos, expected=["number literal"], message="end of input"))
            return state

        input_str = context.input_ctx.input
        char = input_str[start_pos]

        # First handle the special case of 0-prefixed literals
        if char == '0' and start_pos + 1 < len(input_str):
            next_char = input_str[start_pos + 1]
            # Check for binary, octal, or hex
            if next_char in 'bBoOxX':
                # Consume '0' and prefix char
                prefix_state = dataclasses.replace(state, pos=start_pos + 2) # Skip '0' and prefix
                if next_char in 'bB':  # Binary (0b...)
                    return _bin_number(start_pos)(context, prefix_state)
                elif next_char in 'oO':  # Octal (0o...)
                    return _oct_number(start_pos)(context, prefix_state)
                elif next_char in 'xX':  # Hex (0x...)
                    return _hex_number(start_pos)(context, prefix_state)
            # Decimal starting with 0 (e.g. 0.5 or 0123)
            else:
                # Just scan for a sequence of digits, optionally followed by '.' and/or 'e'/'E'
                pos = start_pos
                length = len(input_str)
                
                # Consume all digits
                while pos < length and is_digit(input_str[pos]):
                    pos += 1
                
                # Check for decimal point
                if pos < length and input_str[pos] == '.':
                    pos += 1
                    # Consume digits after decimal point
                    while pos < length and is_digit(input_str[pos]):
                        pos += 1
                
                # Check for scientific notation
                if pos < length and input_str[pos] in 'eE':
                    pos += 1
                    # Optional sign
                    if pos < length and input_str[pos] in '+-':
                        pos += 1
                    # Consume exponent digits
                    while pos < length and is_digit(input_str[pos]):
                        pos += 1
                
                # If we made any progress at all
                if pos > start_pos:
                    final_state = dataclasses.replace(state, pos=pos)
                    num_value = input_str[start_pos:pos]
                    token = SyntaxNode(kind="num_lit", value=num_value)
                    final_state.stx_stack.append(token)
                    return final_state
        
        # Handle any decimal number
        elif is_digit(char):
            # Directly parse the full number sequence without relying on sub-parsers
            pos = start_pos
            length = len(input_str)
            
            # Consume all digits
            while pos < length and is_digit(input_str[pos]):
                pos += 1
            
            # Check for decimal point
            if pos < length and input_str[pos] == '.':
                pos += 1
                # Consume digits after decimal point
                while pos < length and is_digit(input_str[pos]):
                    pos += 1
            
            # Check for scientific notation
            if pos < length and input_str[pos] in 'eE':
                pos += 1
                # Optional sign
                if pos < length and input_str[pos] in '+-':
                    pos += 1
                # Consume exponent digits
                while pos < length and is_digit(input_str[pos]):
                    pos += 1
            
            # If we made any progress at all
            if pos > start_pos:
                final_state = dataclasses.replace(state, pos=pos)
                num_value = input_str[start_pos:pos]
                token = SyntaxNode(kind="num_lit", value=num_value)
                final_state.stx_stack.append(token)
                return final_state
        
        # Not a number
        state.set_error(ParseError(pos=start_pos, expected=["number literal"], unexpected=char))
        return state
        
    return Parser(fn)


# --- Identifier Parser ---

def _parse_escaped_ident_part() -> Parser:
    """Parses `...` part of identifier."""
    # Consumes start escape, takes until end escape, consumes end escape
    p_start = satisfy(is_id_escape_start, "identifier escape start")
    p_content = take_while(lambda c: not is_id_escape_end(c))
    p_end = satisfy(is_id_escape_end, "identifier escape end")

    # Combine, but we only care about the content's value for the node
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        state1 = p_start(context, state)
        if state1.has_error(): return state1
        start_content_pos = state1.pos

        state2 = p_content(context, state1)
        if state2.has_error(): return state2 # Should not happen with take_while
        content_val = context.input_ctx.input[start_content_pos:state2.pos]

        state3 = p_end(context, state2)
        if state3.has_error(): return state3

        # Success. We don't create a sub-node here, just return the value implicitly via pos.
        # The main ident parser will extract the full raw string.
        return state3 # Final position is after end escape

    return Parser(fn)

def _parse_standard_ident_part() -> Parser:
    """Parses a standard identifier part (letters, numbers, _)."""
    # Consumes first char, then rest
    p_first = satisfy(is_id_first, "identifier start")
    p_rest = take_while(is_id_rest)

    # We need to ensure the first char is consumed, then optionally the rest.
    def fn(context: ParserContext, state: ParserState) -> ParserState:
        state1 = p_first(context, state)
        if state1.has_error(): return state1
        state2 = p_rest(context, state1)
        # take_while always succeeds, so just return state2
        return state2
    return Parser(fn)


def ident() -> Parser:
    """Parses an identifier (potentially multipart with ., potentially escaped)."""
    def core_ident_parser(context: ParserContext, state: ParserState) -> ParserState:
        start_pos = state.pos
        state1 = _parse_standard_ident_part()(context, state)
        if state1.has_error():
            return state1

        # Check for multipart identifier (like foo.bar)
        while not state1.has_error() and not state1.is_eof() and state1.current_char() == '.':
            # Temporarily parse '.' and the next part
            temp_state = state1.set_pos(state1.pos + 1) # Skip '.'
            temp_state = _parse_standard_ident_part()(context, temp_state)
            
            if temp_state.has_error():
                # If the part after '.' fails, it's not a multipart ident, break the loop
                break
            else:
                # Successfully parsed another part, update state1
                state1 = temp_state

        # Now, create the final node using mk_token_node (which handles position correctly)
        # No need to consume whitespace here, mk_token_node handles it.
        # state1 currently points to the end of the raw identifier string.
        return mk_token_node("ident", start_pos)(context, state1)

    # Return the parser for the core logic without any trailing whitespace consumption
    return Parser(core_ident_parser)


# --- Token Parsing ---
