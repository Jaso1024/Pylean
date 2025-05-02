"""
Test module for the Kernel class in the Pylean kernel.
"""

import unittest
from unittest.mock import patch, Mock

from pylean.kernel import (
    Name, Level, Expr, SortExpr, ConstExpr, AppExpr, LambdaExpr, PiExpr,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, KernelException, TypeCheckException, NameAlreadyExistsException, ExprKind,
    Environment
)


class TestKernel(unittest.TestCase):
    """Tests for the Kernel class."""
    
    def setUp(self):
        """Set up for tests."""
        self.kernel = Kernel()
    
    def test_create_kernel(self):
        """Test creating a kernel."""
        kernel = Kernel()
        self.assertIsNotNone(kernel)
        self.assertIsNotNone(kernel.env)
    
    def test_add_constant(self):
        """Test adding a constant to the kernel."""
        name = "test.constant"
        type_expr = mk_sort(0)  # Prop
        
        kernel1 = Kernel()
        kernel2 = kernel1.add_constant(name, type_expr)
        
        # Check that new kernel has the constant
        self.assertIsNotNone(kernel2.env.find_decl(name))
        self.assertEqual(str(kernel2.env.get_type(name)), str(type_expr))
        
        # Check that original kernel is unchanged
        self.assertIsNone(kernel1.env.find_decl(name))
    
    def test_add_axiom(self):
        """Test adding an axiom to the kernel."""
        name = "test.axiom"
        type_expr = mk_sort(0)  # Prop
        
        kernel1 = Kernel()
        kernel2 = kernel1.add_axiom(name, type_expr)
        
        # Check that new kernel has the axiom
        self.assertIsNotNone(kernel2.env.find_decl(name))
        self.assertEqual(str(kernel2.env.get_type(name)), str(type_expr))
        
        # Check that original kernel is unchanged
        self.assertIsNone(kernel1.env.find_decl(name))
    
    def test_add_axiom_duplicate(self):
        """Test adding a duplicate axiom to the kernel."""
        name = "test.axiom"
        type_expr = mk_sort(0)  # Prop
        
        kernel = Kernel().add_axiom(name, type_expr)
        
        # Adding duplicate should raise an exception
        with self.assertRaises(NameAlreadyExistsException):
            kernel.add_axiom(name, type_expr)
    
    def test_add_definition(self):
        """Test adding a definition to the kernel."""
        name = "test.definition"
        type_expr = mk_pi(Name.from_string("x"), mk_sort(0), mk_sort(0))  # Prop -> Prop
        value = mk_lambda(Name.from_string("x"), mk_sort(0), mk_var(0))  # λ(x: Prop), x
        
        kernel1 = Kernel()
        kernel2 = kernel1.add_definition(name, type_expr, value)
        
        # Check that new kernel has the definition
        self.assertIsNotNone(kernel2.env.find_decl(name))
        self.assertEqual(str(kernel2.env.get_type(name)), str(type_expr))
        self.assertEqual(str(kernel2.env.get_value(name)), str(value))
        
        # Check that original kernel is unchanged
        self.assertIsNone(kernel1.env.find_decl(name))
    
    def test_add_definition_type_error(self):
        """Test adding a definition with a type error to the kernel."""
        name = "test.definition"
        type_expr = mk_pi(Name.from_string("x"), mk_sort(0), mk_sort(0))  # Prop -> Prop
        value = mk_var(0)  # This will cause a type error
        
        kernel = Kernel()
        
        # Adding definition with type error should raise an exception
        with self.assertRaises(TypeCheckException):
            kernel.add_definition(name, type_expr, value)
    
    def test_add_theorem(self):
        """Test adding a theorem to the kernel."""
        name = "test.theorem"
        type_expr = mk_sort(0)  # Prop
        proof = mk_const("proof")  # This would normally cause a type error, but we'll mock it
        
        # Mock type checker to avoid the type error
        with patch('pylean.kernel.kernel.check_type'):
            kernel1 = Kernel()
            kernel2 = kernel1.add_theorem(name, type_expr, proof)
            
            # Check that new kernel has the theorem
            self.assertIsNotNone(kernel2.env.find_decl(name))
            self.assertEqual(str(kernel2.env.get_type(name)), str(type_expr))
    
    def test_add_inductive(self):
        """Test adding an inductive type to the kernel."""
        # Create a simple inductive type with a unique name to avoid collisions
        name = "test.new_bool_type"
        type_expr = mk_sort(0)  # Prop
        
        # Create a fresh kernel with empty environment
        fresh_kernel = Kernel(env=Environment())
        
        # First add the type as a constant to the environment 
        # This is necessary because constructors reference the type
        kernel_with_type = fresh_kernel.add_constant(name, type_expr)
        
        # Now the constructors can reference the type
        constructors = [
            ("test.new_bool_type.true", mk_const("test.new_bool_type")),
            ("test.new_bool_type.false", mk_const("test.new_bool_type"))
        ]
        
        # Add the inductive type
        kernel = kernel_with_type.add_inductive(name, type_expr, constructors)
        
        # Check that the inductive type was added
        self.assertTrue(kernel.env.is_inductive(name))
        
        # Check that the constructors were added
        for ctor_name, _ in constructors:
            self.assertIsNotNone(kernel.env.find_decl(ctor_name))
    
    def test_infer_type(self):
        """Test inferring the type of an expression."""
        # Infer the type of Prop
        const = mk_const("Prop")
        type_expr = self.kernel.infer_type(const)
        
        self.assertEqual(type_expr.kind, ExprKind.SORT)
        self.assertEqual(str(type_expr), str(mk_sort(0)))
        
        # Infer the type of Type (universe level 0)
        const = mk_const("Type")
        type_expr = self.kernel.infer_type(const)
        
        self.assertEqual(type_expr.kind, ExprKind.SORT)
        self.assertEqual(str(type_expr), str(mk_sort(1)))
    
    def test_check_type(self):
        """Test checking the type of an expression."""
        # Check that Prop has type Type
        expr = mk_const("Prop")
        expected_type = mk_sort(0)
        
        # This should not raise an exception
        self.kernel.check_type(expr, expected_type)
        
        # Check with an incorrect type
        incorrect_type = mk_pi(Name(), mk_sort(0), mk_sort(0))  # Prop -> Prop
        
        # This should raise an exception
        with self.assertRaises(TypeCheckException):
            self.kernel.check_type(expr, incorrect_type)
    
    def test_create_child_kernel(self):
        """Test creating a child kernel."""
        parent = Kernel()
        child = parent.create_child_kernel()
        
        # Check that child has a new environment with parent
        self.assertIsNotNone(child.env)
        self.assertIsNotNone(child.env.parent_env)
        self.assertIs(child.env.parent_env, parent.env)
    
    def test_bool_example(self):
        """Test a complete example with Boolean algebra."""
        # Start with a fresh kernel with a completely empty environment
        # rather than the standard environment which already has Bool
        kernel = Kernel(env=Environment())
        
        # Add the bool type with a different name
        kernel = kernel.add_constant("CustomBool", mk_sort(0))
        
        # Add true and false
        kernel = kernel.add_constant("CustomBool.true", mk_const("CustomBool"))
        kernel = kernel.add_constant("CustomBool.false", mk_const("CustomBool"))
        
        # Add not: Bool -> Bool
        not_type = mk_pi(
            Name.from_string("b"),
            mk_const("CustomBool"),
            mk_const("CustomBool")
        )
        kernel = kernel.add_constant("CustomBool.not", not_type)
        
        # Add and: Bool -> Bool -> Bool
        and_type = mk_pi(
            Name.from_string("a"),
            mk_const("CustomBool"),
            mk_pi(
                Name.from_string("b"),
                mk_const("CustomBool"),
                mk_const("CustomBool")
            )
        )
        kernel = kernel.add_constant("CustomBool.and", and_type)
        
        # Add or: Bool -> Bool -> Bool
        or_type = mk_pi(
            Name.from_string("a"),
            mk_const("CustomBool"),
            mk_pi(
                Name.from_string("b"),
                mk_const("CustomBool"),
                mk_const("CustomBool")
            )
        )
        kernel = kernel.add_constant("CustomBool.or", or_type)
        
        # Check that all the declarations were added
        self.assertIsNotNone(kernel.env.find_decl("CustomBool"))
        self.assertIsNotNone(kernel.env.find_decl("CustomBool.true"))
        self.assertIsNotNone(kernel.env.find_decl("CustomBool.false"))
        self.assertIsNotNone(kernel.env.find_decl("CustomBool.not"))
        self.assertIsNotNone(kernel.env.find_decl("CustomBool.and"))
        self.assertIsNotNone(kernel.env.find_decl("CustomBool.or"))


if __name__ == "__main__":
    unittest.main() 