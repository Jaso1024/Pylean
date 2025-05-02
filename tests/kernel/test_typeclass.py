"""
Tests for the type class system.
"""

import unittest
from pylean.kernel.expr import (
    Name, mk_sort, mk_const, mk_pi, mk_app, mk_var
)
from pylean.kernel.env import (
    Environment, mk_definition
)
from pylean.kernel.typecheck import (
    Context, infer_type
)
from pylean.kernel.typeclass import (
    TypeClass, TypeClassInstance, TypeClassEnvironment, TypeClassElaborator,
    mk_type_class, mk_type_class_instance
)


class TestTypeClass(unittest.TestCase):
    """Tests for the type class system."""
    
    def setUp(self):
        """Set up an environment with some type classes for testing."""
        self.env = Environment()
        
        # Create a type universe
        type_0 = mk_sort(0)  # Type 0
        
        # Define Show type class
        # class Show (α : Type) where
        #   show : α → String
        show_name = Name.from_string("Show")
        alpha_name = Name.from_string("α")
        
        # Parameter for Show: α (a type)
        param_names = [alpha_name]
        
        # Field: show : α → String
        string_type = mk_const("String")
        show_field_type = mk_pi(
            Name.from_string("x"),
            mk_var(0),  # α
            string_type
        )
        fields = {Name.from_string("show"): show_field_type}
        
        # Create Show type class
        self.show_class = mk_type_class(show_name, param_names, fields)
        
        # Create a TypeClassEnvironment
        self.tc_env = TypeClassEnvironment(self.env)
        self.tc_env = self.tc_env.add_class(self.show_class)
        
        # Define Nat type and String type as constants
        nat_type = mk_const("Nat")
        
        # Create Show instance for Nat
        # instance Show Nat where
        #   show n = "natural number"
        nat_show_name = Name.from_string("natShow")
        
        # For the test, we don't need the actual implementation, just the declaration
        nat_show_type = mk_app(mk_const("Show"), mk_const("Nat"))
        self.env = self.env.add_decl(mk_definition(nat_show_name, nat_show_type, mk_const("dummy_impl")))
        
        # The instance parameters: [Nat]
        params = [mk_const("Nat")]
        
        # The field implementations - just use dummy values for testing
        field_values = {Name.from_string("show"): mk_const("natShowImpl")}
        
        # Create the instance
        self.nat_show_instance = mk_type_class_instance("Show", nat_show_name, params, field_values)
        
        # Add the instance to the environment
        self.tc_env = self.tc_env.add_instance(self.nat_show_instance)
        
        # Create a TypeClassElaborator
        self.elaborator = TypeClassElaborator(self.tc_env)
    
    def test_type_class_creation(self):
        """Test creating a type class."""
        self.assertEqual(str(self.show_class.name), "Show")
        self.assertEqual(len(self.show_class.param_names), 1)
        self.assertEqual(str(self.show_class.param_names[0]), "α")
        self.assertEqual(len(self.show_class.fields), 1)
        self.assertTrue(Name.from_string("show") in self.show_class.fields)
    
    def test_instance_creation(self):
        """Test creating a type class instance."""
        self.assertEqual(str(self.nat_show_instance.class_name), "Show")
        self.assertEqual(str(self.nat_show_instance.instance_name), "natShow")
        self.assertEqual(len(self.nat_show_instance.params), 1)
        self.assertEqual(self.nat_show_instance.params[0].kind.name, "CONST")
        self.assertEqual(str(self.nat_show_instance.params[0].name), "Nat")
    
    def test_instance_lookup(self):
        """Test looking up an instance by class and parameters."""
        # Look up the Nat instance of Show
        instance = self.tc_env.find_instance(
            Name.from_string("Show"),
            [mk_const("Nat")],
            Context()
        )
        
        self.assertIsNotNone(instance)
        self.assertEqual(str(instance.instance_name), "natShow")
    
    def test_missing_instance_lookup(self):
        """Test looking up a non-existent instance."""
        # Look up a Show instance for a non-existent type
        instance = self.tc_env.find_instance(
            Name.from_string("Show"),
            [mk_const("NonExistent")],
            Context()
        )
        
        self.assertIsNone(instance)
    
    def test_instance_synthesis(self):
        """Test synthesizing an instance expression."""
        # Synthesize a Show instance for Nat
        instance_expr = self.tc_env.synthesize_instance(
            Name.from_string("Show"),
            [mk_const("Nat")],
            Context()
        )
        
        self.assertIsNotNone(instance_expr)
        self.assertEqual(instance_expr.kind.name, "APP")
        self.assertEqual(instance_expr.fn.kind.name, "CONST")
        self.assertEqual(str(instance_expr.fn.name), "natShow")
        self.assertEqual(str(instance_expr.arg.name), "Nat")
    
    def test_multiple_instances(self):
        """Test adding and finding multiple instances for a type class."""
        # Add another instance: Show Bool
        bool_show_name = Name.from_string("boolShow")
        bool_show_type = mk_app(mk_const("Show"), mk_const("Bool"))
        self.env = self.env.add_decl(mk_definition(bool_show_name, bool_show_type, mk_const("dummy_impl")))
        
        # The instance parameters: [Bool]
        params = [mk_const("Bool")]
        
        # The field implementations
        field_values = {Name.from_string("show"): mk_const("boolShowImpl")}
        
        # Create and add the instance
        bool_show_instance = mk_type_class_instance("Show", bool_show_name, params, field_values)
        self.tc_env = self.tc_env.add_instance(bool_show_instance)
        
        # Look up the Bool instance
        instance = self.tc_env.find_instance(
            Name.from_string("Show"),
            [mk_const("Bool")],
            Context()
        )
        
        self.assertIsNotNone(instance)
        self.assertEqual(str(instance.instance_name), "boolShow")
        
        # Look up the Nat instance to make sure it's still there
        instance = self.tc_env.find_instance(
            Name.from_string("Show"),
            [mk_const("Nat")],
            Context()
        )
        
        self.assertIsNotNone(instance)
        self.assertEqual(str(instance.instance_name), "natShow")


class TestTypeClassElaborator(unittest.TestCase):
    """Tests for the type class elaborator."""
    
    def setUp(self):
        """Set up an environment for testing the elaborator."""
        # This would be similar to TestTypeClass.setUp, but with more
        # complex type classes and instances for testing elaboration
        pass
    
    def test_elaborate_with_implicit_instances(self):
        """Test elaborating an expression with implicit instances."""
        # TODO: Implement this test
        pass


if __name__ == '__main__':
    unittest.main() 