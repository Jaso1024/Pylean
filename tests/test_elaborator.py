import pytest

# Pylean imports
from pylean.parser.core import SyntaxNode
from pylean.parser.expr import (
    IDENT_EXPR, NUM_EXPR, PAREN_EXPR, APP_EXPR, ADD_EXPR, EQ_EXPR, UNARY_NOT
)
from pylean.elaborator import ElaborationContext, elaborate
from pylean.kernel.typecheck import TypeError
from pylean.kernel.expr import (
    Expr, ConstExpr, VarExpr, AppExpr, Name, mk_const, mk_var, mk_app, mk_sort, mk_pi
)
from pylean.kernel.env import mk_std_env, Environment, mk_constant, mk_definition, mk_axiom
from pylean.kernel.typecheck import Context

# --- Fixtures ---

@pytest.fixture(scope="module")
def std_env() -> Environment:
    """Provides a standard environment with Nat, Bool, etc."""
    return mk_std_env()

@pytest.fixture
def elab_context(std_env: Environment) -> ElaborationContext:
    """Provides a basic elaboration context with the standard environment."""
    return ElaborationContext(env=std_env)

# Helper to create identifier SyntaxNodes
def mk_ident_node(name: str) -> SyntaxNode:
    # Simulates the structure the parser creates for IDENT_EXPR
    ident_token = SyntaxNode(kind="ident", value=name)
    return SyntaxNode(kind=IDENT_EXPR, children=[ident_token])

# Helper to create number SyntaxNodes
def mk_num_node(value: str) -> SyntaxNode:
    num_token = SyntaxNode(kind="num_lit", value=value)
    return SyntaxNode(kind=NUM_EXPR, children=[num_token])

# Helper to create operator symbol SyntaxNodes
def mk_op_node(op: str) -> SyntaxNode:
    return SyntaxNode(kind="symbol", value=op)

# --- Test Cases ---

def test_elaborate_identifier_const(elab_context: ElaborationContext):
    """Test elaborating a known constant identifier."""
    node = mk_ident_node("Nat")
    result = elaborate(node, elab_context)
    assert isinstance(result, ConstExpr)
    assert result.name == Name.from_string("Nat")

def test_elaborate_identifier_var(elab_context: ElaborationContext):
    """Test elaborating an identifier bound in the local context."""
    # Add 'x : Nat' to local context
    nat_type = mk_const("Nat")
    local_ctx = Context().extend(Name.from_string("x"), nat_type)
    context = ElaborationContext(elab_context.env, local_ctx)
    
    node = mk_ident_node("x")
    result = elaborate(node, context)
    assert isinstance(result, VarExpr)
    assert result.idx == 0 # Innermost bound variable

    # Test with nested context
    bool_type = mk_const("Bool")
    local_ctx_nested = local_ctx.extend(Name.from_string("y"), bool_type)
    context_nested = ElaborationContext(elab_context.env, local_ctx_nested)
    
    node_x = mk_ident_node("x")
    node_y = mk_ident_node("y")
    result_x = elaborate(node_x, context_nested)
    result_y = elaborate(node_y, context_nested)
    
    assert isinstance(result_x, VarExpr) and result_x.idx == 1 # x is now the second binder
    assert isinstance(result_y, VarExpr) and result_y.idx == 0 # y is the innermost

def test_elaborate_identifier_not_found(elab_context: ElaborationContext):
    """Test elaborating an unknown identifier."""
    node = mk_ident_node("unknown_var")
    with pytest.raises(TypeError, match="Unknown identifier"):
        elaborate(node, elab_context)

def test_elaborate_number(elab_context: ElaborationContext):
    """Test elaborating number literals."""
    zero_node = mk_num_node("0")
    one_node = mk_num_node("1")
    three_node = mk_num_node("3")

    zero_expr = elaborate(zero_node, elab_context)
    one_expr = elaborate(one_node, elab_context)
    three_expr = elaborate(three_node, elab_context)

    nat_zero = mk_const("Nat.zero")
    nat_succ = mk_const("Nat.succ")

    assert zero_expr == nat_zero
    assert one_expr == mk_app(nat_succ, nat_zero)
    assert three_expr == mk_app(nat_succ, mk_app(nat_succ, mk_app(nat_succ, nat_zero)))

def test_elaborate_paren(elab_context: ElaborationContext):
    """Test elaborating parenthesized expressions."""
    inner_node = mk_ident_node("Nat")
    paren_node = SyntaxNode(kind=PAREN_EXPR, children=[
        mk_op_node("("), inner_node, mk_op_node(")")
    ])
    result = elaborate(paren_node, elab_context)
    expected = elaborate(inner_node, elab_context)
    assert result == expected

def test_elaborate_application(elab_context: ElaborationContext):
    """Test elaborating function application."""
    # Nat.succ 0
    func_node = mk_ident_node("Nat.succ")
    arg_node = mk_num_node("0")
    app_node = SyntaxNode(kind=APP_EXPR, children=[func_node, arg_node]) # Simplified structure for testing

    result = elaborate(app_node, elab_context)
    
    expected_func = mk_const("Nat.succ")
    expected_arg = mk_const("Nat.zero")
    expected_app = mk_app(expected_func, expected_arg)
    
    assert result == expected_app

    # Add f : Nat -> Nat
    nat_type = mk_const("Nat")
    f_type = mk_pi("n", nat_type, nat_type)
    env = elab_context.env.add_decl(mk_constant("f", f_type))
    context_with_f = ElaborationContext(env, elab_context.local_ctx)

    # f 1
    f_node = mk_ident_node("f")
    one_node = mk_num_node("1")
    app_node_f = SyntaxNode(kind=APP_EXPR, children=[f_node, one_node])

    result_f = elaborate(app_node_f, context_with_f)
    expected_f = mk_const("f")
    expected_one = mk_app(mk_const("Nat.succ"), mk_const("Nat.zero"))
    expected_app_f = mk_app(expected_f, expected_one)

    assert result_f == expected_app_f

def test_elaborate_binary_op(elab_context: ElaborationContext):
    """Test elaborating binary operations."""
    # 1 + 2
    one_node = mk_num_node("1")
    two_node = mk_num_node("2")
    plus_node = mk_op_node("+")
    add_stx_node = SyntaxNode(kind=ADD_EXPR, children=[one_node, plus_node, two_node])

    result = elaborate(add_stx_node, elab_context)

    nat_add = mk_const("Nat.add")
    nat_succ = mk_const("Nat.succ")
    nat_zero = mk_const("Nat.zero")
    one_expr = mk_app(nat_succ, nat_zero)
    two_expr = mk_app(nat_succ, one_expr)
    expected = mk_app(mk_app(nat_add, one_expr), two_expr)

    assert result == expected

    # true == false (assuming Bool constants exist)
    true_node = mk_ident_node("true")
    false_node = mk_ident_node("false")
    eq_node = mk_op_node("==")
    eq_stx_node = SyntaxNode(kind=EQ_EXPR, children=[true_node, eq_node, false_node])
    
    # Add Bool constants if not in mk_std_env yet
    bool_type = mk_const("Bool") # Assumes Bool type exists
    env = elab_context.env
    if not env.find_decl("true"):
        env = env.add_decl(mk_constant("true", bool_type))
    if not env.find_decl("false"):
        env = env.add_decl(mk_constant("false", bool_type))
    context_with_bool = ElaborationContext(env, elab_context.local_ctx)


    result_eq = elaborate(eq_stx_node, context_with_bool)
    
    eq_const = mk_const("Eq") # Assumes Eq exists
    true_const = mk_const("true")
    false_const = mk_const("false")
    expected_eq = mk_app(mk_app(eq_const, true_const), false_const) # Eq true false

    assert result_eq == expected_eq

def test_elaborate_unary_op(elab_context: ElaborationContext):
    """Test elaborating unary operations."""
    # Bool.not true
    true_node = mk_ident_node("true")
    not_node = mk_op_node("not") # Assuming parser uses 'not' symbol value
    not_stx_node = SyntaxNode(kind=UNARY_NOT, children=[not_node, true_node])

    # Add Bool constants if not present
    bool_type = mk_const("Bool")
    env = elab_context.env
    if not env.find_decl("true"):
        env = env.add_decl(mk_constant("true", bool_type))
    if not env.find_decl("Bool.not"): # Assuming Bool.not : Bool -> Bool
        not_type = mk_pi("_", bool_type, bool_type)
        env = env.add_decl(mk_constant("Bool.not", not_type))
    context_with_bool = ElaborationContext(env, elab_context.local_ctx)

    result = elaborate(not_stx_node, context_with_bool)
    
    expected_not = mk_const("Bool.not")
    expected_true = mk_const("true")
    expected = mk_app(expected_not, expected_true)

    assert result == expected

# TODO: Add tests for elaboration error cases (e.g., type mismatches in App)
# TODO: Add tests involving more complex local contexts 