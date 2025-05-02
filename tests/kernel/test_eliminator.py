"""
Tests for the eliminator generation system.
"""

import unittest
from pylean.kernel.expr import (
    Name, mk_sort, mk_const, mk_pi, mk_app, mk_var
)
from pylean.kernel.env import (
    Environment, mk_inductive, mk_constructor
)
from pylean.kernel.eliminator import (
    generate_recursor, generate_induction_principle, generate_eliminators
)


class TestEliminator(unittest.TestCase):
    """Tests for the eliminator generation system."""
    
    def setUp(self):
        """Set up an environment with some inductive types for testing."""
        self.env = Environment()
        
        # Add Type 0 (Prop) constant
        prop_name = Name.from_string("Prop")
        prop_type = mk_sort(1)  # Type 1
        self.env = self.env.add_decl(mk_inductive(prop_name, prop_type, []))
        
        # Define Nat inductive type
        nat_name = Name.from_string("Nat")
        nat_type = mk_sort(0)  # Type 0
        
        # Define zero constructor
        zero_name = Name.from_string("zero")
        zero_type = mk_const(nat_name)
        zero_constr = mk_constructor(zero_name, zero_type, nat_name)
        
        # Define succ constructor
        succ_name = Name.from_string("succ")
        succ_type = mk_pi(
            Name.from_string("n"),
            mk_const(nat_name),
            mk_const(nat_name)
        )
        succ_constr = mk_constructor(succ_name, succ_type, nat_name)
        
        # Add Nat to environment
        nat_decl = mk_inductive(nat_name, nat_type, [zero_constr, succ_constr])
        self.env = self.env.add_decl(nat_decl)
        
        # Define List inductive type
        list_name = Name.from_string("List")
        
        # Define type parameter
        type_param_name = Name.from_string("T")
        type_param_type = mk_sort(0)  # Type 0
        
        # List type is Type -> Type
        list_type = mk_pi(
            type_param_name,
            type_param_type,
            mk_sort(0)  # Type 0
        )
        
        # Define nil constructor
        nil_name = Name.from_string("nil")
        # nil : Π (T : Type), List T
        nil_type = mk_pi(
            type_param_name,
            type_param_type,
            mk_app(mk_const(list_name), mk_var(0))
        )
        nil_constr = mk_constructor(nil_name, nil_type, list_name)
        
        # Define cons constructor
        cons_name = Name.from_string("cons")
        # cons : Π (T : Type), T -> List T -> List T
        cons_type = mk_pi(
            type_param_name,
            type_param_type,
            mk_pi(
                Name.from_string("head"),
                mk_var(0),  # T
                mk_pi(
                    Name.from_string("tail"),
                    mk_app(mk_const(list_name), mk_var(1)),  # List T
                    mk_app(mk_const(list_name), mk_var(2))  # List T
                )
            )
        )
        cons_constr = mk_constructor(cons_name, cons_type, list_name)
        
        # Add List to environment
        list_decl = mk_inductive(list_name, list_type, [nil_constr, cons_constr])
        self.env = self.env.add_decl(list_decl)
    
    def test_nat_recursor_generation(self):
        """Test generation of the Nat recursor."""
        # Get Nat declaration
        nat_decl = self.env.find_decl("Nat")
        self.assertIsNotNone(nat_decl)
        
        # Generate recursor
        nat_rec = generate_recursor(self.env, nat_decl)
        
        # Check recursor name
        self.assertEqual(str(nat_rec.name), "Nat.rec")
        
        # Add recursor to environment
        self.env = self.env.add_decl(nat_rec)
        
        # Verify recursor is in environment
        nat_rec_decl = self.env.find_decl("Nat.rec")
        self.assertIsNotNone(nat_rec_decl)
    
    def test_nat_induction_principle_generation(self):
        """Test generation of the Nat induction principle."""
        # Get Nat declaration
        nat_decl = self.env.find_decl("Nat")
        self.assertIsNotNone(nat_decl)
        
        # Generate induction principle
        nat_ind = generate_induction_principle(self.env, nat_decl)
        
        # Check induction principle name
        self.assertEqual(str(nat_ind.name), "Nat.ind")
        
        # Add induction principle to environment
        self.env = self.env.add_decl(nat_ind)
        
        # Verify induction principle is in environment
        nat_ind_decl = self.env.find_decl("Nat.ind")
        self.assertIsNotNone(nat_ind_decl)
    
    def test_list_eliminators_generation(self):
        """Test generation of List eliminators."""
        # Get List declaration
        list_decl = self.env.find_decl("List")
        self.assertIsNotNone(list_decl)
        
        # Generate eliminators
        self.env = generate_eliminators(self.env, list_decl)
        
        # Verify recursor is in environment
        list_rec_decl = self.env.find_decl("List.rec")
        self.assertIsNotNone(list_rec_decl)
        
        # Verify induction principle is in environment
        list_ind_decl = self.env.find_decl("List.ind")
        self.assertIsNotNone(list_ind_decl)
    
    def test_automatic_eliminator_generation(self):
        """Test automatic generation of eliminators when adding inductive types."""
        # Define a new inductive type
        bool_name = Name.from_string("Bool")
        bool_type = mk_sort(0)  # Type 0
        
        # Define true constructor
        true_name = Name.from_string("true")
        true_type = mk_const(bool_name)
        true_constr = mk_constructor(true_name, true_type, bool_name)
        
        # Define false constructor
        false_name = Name.from_string("false")
        false_type = mk_const(bool_name)
        false_constr = mk_constructor(false_name, false_type, bool_name)
        
        # Create Bool inductive declaration
        bool_decl = mk_inductive(bool_name, bool_type, [true_constr, false_constr])
        
        # Add Bool to environment - this should automatically generate eliminators
        self.env = self.env.add_decl(bool_decl)
        
        # Verify eliminators were automatically generated
        bool_rec_decl = self.env.find_decl("Bool.rec")
        self.assertIsNotNone(bool_rec_decl)
        
        bool_ind_decl = self.env.find_decl("Bool.ind")
        self.assertIsNotNone(bool_ind_decl)


if __name__ == '__main__':
    unittest.main() 