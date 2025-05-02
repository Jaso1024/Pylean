"""
Test module for the environment in the Pylean kernel.
"""

import unittest
from pylean.kernel import (
    Name, Level, Environment, DeclKind,
    AxiomDecl, DefinitionDecl, TheoremDecl, OpaqueDecl, ConstantDecl, InductiveDecl, ConstructorDecl,
    mk_axiom, mk_definition, mk_theorem, mk_opaque, mk_constant, mk_inductive, mk_constructor,
    mk_std_env,
    mk_sort, mk_const, mk_lambda, mk_pi, mk_app
)


class TestEnvironment(unittest.TestCase):
    """Tests for the Environment class."""
    
    def test_empty_environment(self):
        """Test creating an empty environment."""
        env = Environment()
        self.assertIsNotNone(env)
        self.assertEqual(len(env.declarations), 0)
    
    def test_standard_environment(self):
        """Test creating a standard environment."""
        env = mk_std_env()
        self.assertIsNotNone(env)
        self.assertTrue(env.is_constant("Prop"))
        self.assertTrue(env.is_constant("Type"))
    
    def test_add_decl(self):
        """Test adding a declaration to the environment."""
        env = Environment()
        
        # Create a simple axiom
        name = Name.from_string("test.axiom")
        type_expr = mk_sort(0)  # Prop
        axiom = mk_axiom(name, type_expr)
        
        # Add to environment
        new_env = env.add_decl(axiom)
        
        # Check that original environment is unchanged
        self.assertEqual(len(env.declarations), 0)
        
        # Check that new environment has the declaration
        self.assertEqual(len(new_env.declarations), 1)
        self.assertTrue(new_env.is_constant("test.axiom"))
        
        # Check that the declaration is retrievable
        decl = new_env.find_decl("test.axiom")
        self.assertIsNotNone(decl)
        self.assertEqual(decl.kind, DeclKind.AX)
        self.assertEqual(str(decl.name), "test.axiom")
    
    def test_find_decl(self):
        """Test finding a declaration in the environment."""
        env = Environment()
        
        # Create and add a declaration
        name = Name.from_string("test.constant")
        type_expr = mk_sort(0)  # Prop
        const = mk_constant(name, type_expr)
        
        env = env.add_decl(const)
        
        # Find the declaration by name string
        decl1 = env.find_decl("test.constant")
        self.assertIsNotNone(decl1)
        self.assertEqual(str(decl1.name), "test.constant")
        
        # Find the declaration by Name object
        decl2 = env.find_decl(Name.from_string("test.constant"))
        self.assertIsNotNone(decl2)
        self.assertEqual(str(decl2.name), "test.constant")
        
        # Try to find a non-existent declaration
        decl3 = env.find_decl("nonexistent")
        self.assertIsNone(decl3)
    
    def test_is_constant(self):
        """Test checking if a name refers to a constant in the environment."""
        env = Environment()
        
        # Add different kinds of declarations
        env = env.add_decl(mk_constant("test.constant", mk_sort(0)))
        env = env.add_decl(mk_axiom("test.axiom", mk_sort(0)))
        env = env.add_decl(mk_definition("test.definition", mk_sort(0), mk_sort(0)))
        
        # Check constants
        self.assertTrue(env.is_constant("test.constant"))
        self.assertTrue(env.is_constant("test.axiom"))
        self.assertTrue(env.is_constant("test.definition"))
        
        # Check non-existent
        self.assertFalse(env.is_constant("nonexistent"))
    
    def test_is_inductive(self):
        """Test checking if a name refers to an inductive type in the environment."""
        env = Environment()
        
        # Create a simple inductive type
        name = Name.from_string("test.bool")
        type_expr = mk_sort(0)  # Prop
        ctors = [
            mk_constructor("test.bool.true", mk_const("test.bool"), "test.bool"),
            mk_constructor("test.bool.false", mk_const("test.bool"), "test.bool")
        ]
        ind = mk_inductive(name, type_expr, ctors)
        
        # Add to environment
        env = env.add_decl(ind)
        
        # Check inductive
        self.assertTrue(env.is_inductive("test.bool"))
        self.assertFalse(env.is_inductive("test.bool.true"))  # Constructor, not inductive
        self.assertFalse(env.is_inductive("nonexistent"))
    
    def test_get_type(self):
        """Test getting the type of a declaration by name."""
        env = Environment()
        
        # Add a declaration
        name = Name.from_string("test.constant")
        type_expr = mk_sort(0)  # Prop
        const = mk_constant(name, type_expr)
        
        env = env.add_decl(const)
        
        # Get the type
        type_got = env.get_type("test.constant")
        self.assertIsNotNone(type_got)
        self.assertEqual(type_got.kind, type_expr.kind)
        
        # Try non-existent
        self.assertIsNone(env.get_type("nonexistent"))
    
    def test_get_value(self):
        """Test getting the value of a definition by name."""
        env = Environment()
        
        # Add a definition
        name = Name.from_string("test.definition")
        type_expr = mk_sort(0)  # Prop
        value = mk_const("Prop")
        defn = mk_definition(name, type_expr, value)
        
        env = env.add_decl(defn)
        
        # Get the value
        value_got = env.get_value("test.definition")
        self.assertIsNotNone(value_got)
        self.assertEqual(value_got.kind, value.kind)
        
        # Try non-existent
        self.assertIsNone(env.get_value("nonexistent"))
        
        # Try a constant (has no value)
        env = env.add_decl(mk_constant("test.constant", mk_sort(0)))
        self.assertIsNone(env.get_value("test.constant"))
    
    def test_get_constructors(self):
        """Test getting the constructors of an inductive type by name."""
        env = Environment()
        
        # Create a simple inductive type
        name = Name.from_string("test.bool")
        type_expr = mk_sort(0)  # Prop
        ctors = [
            mk_constructor("test.bool.true", mk_const("test.bool"), "test.bool"),
            mk_constructor("test.bool.false", mk_const("test.bool"), "test.bool")
        ]
        ind = mk_inductive(name, type_expr, ctors)
        
        # Add to environment
        env = env.add_decl(ind)
        
        # Get the constructors
        ctors_got = env.get_constructors("test.bool")
        self.assertEqual(len(ctors_got), 2)
        self.assertEqual(str(ctors_got[0].name), "test.bool.true")
        self.assertEqual(str(ctors_got[1].name), "test.bool.false")
        
        # Try non-existent
        self.assertEqual(env.get_constructors("nonexistent"), ())
        
        # Try a non-inductive declaration
        env = env.add_decl(mk_constant("test.constant", mk_sort(0)))
        self.assertEqual(env.get_constructors("test.constant"), ())
    
    def test_create_child(self):
        """Test creating a child environment."""
        parent = Environment()
        parent = parent.add_decl(mk_constant("parent.constant", mk_sort(0)))
        
        child = parent.create_child()
        child = child.add_decl(mk_constant("child.constant", mk_sort(0)))
        
        # Child can see parent's declarations
        self.assertIsNotNone(child.find_decl("parent.constant"))
        
        # Parent cannot see child's declarations
        self.assertIsNone(parent.find_decl("child.constant"))
        
        # Child's declarations are in child only
        self.assertEqual(len(child.declarations), 1)
        
        # Child has a reference to parent
        self.assertIs(child.parent_env, parent)


if __name__ == "__main__":
    unittest.main() 