#!/usr/bin/env python3
"""
Pi Type Demo for Pylean.

This script demonstrates how to directly work with Pi types (dependent function types)
using the kernel API, without relying on the parser or tactic system.
"""

from pylean.kernel import (
    Name, Level, Expr, ExprKind,
    mk_var, mk_sort, mk_const, mk_app, mk_lambda, mk_pi,
    Kernel, ReductionStrategy, ReductionMode, reduce
)

def main():
    """Run the Pi type demo."""
    print("Pylean Pi Type Demo")
    print("==================")
    
    # Create a kernel with standard environment
    print("\nCreating kernel...")
    kernel = Kernel()
    
    # Add the Nat type to the environment if it doesn't exist
    print("\nSetting up environment:")
    try:
        if kernel.env.find_decl("Nat") is None:
            kernel = kernel.add_constant("Nat", mk_sort(0))
            print("   Added Nat type")
            
            # Add zero constructor
            kernel = kernel.add_axiom("zero", mk_const("Nat"))
            print("   Added zero constructor")
            
            # Add successor constructor
            succ_type = mk_pi("n", mk_const("Nat"), mk_const("Nat"))
            kernel = kernel.add_constant("succ", succ_type)
            print("   Added successor constructor")
    except Exception as e:
        print(f"   Error setting up environment: {e}")
    
    # 1. Define the identity function type with correct universe level
    print("\n1. Creating the identity function type:")
    # Use Type 1 instead of Prop to ensure it can contain Nat which is a Type 0
    univ_type = mk_sort(1)  # Type 1, which can contain Type 0 types like Nat
    # First create the inner function type (A → A)
    inner_type = mk_pi("x", mk_var(0), mk_var(1))  # Π (x : A), A
    # Now create the complete type: Π (A : Type 1), inner_type
    id_type = mk_pi("A", univ_type, inner_type)
    print(f"   Identity function type: {id_type}")
    
    # 2. Create a term for the identity function
    print("\n2. Creating the identity function term:")
    # Inner lambda: λ (x : A), x
    inner_lambda = mk_lambda("x", mk_var(0), mk_var(0))  # λ (x : A), x
    # Outer lambda: λ (A : Type 1), inner_lambda
    id_term = mk_lambda("A", univ_type, inner_lambda)
    print(f"   Identity function term: {id_term}")
    
    # 3. Type check the identity function
    print("\n3. Type checking the identity function:")
    try:
        id_type_check = kernel.infer_type(id_term)
        print(f"   Inferred type: {id_type_check}")
        print(f"   Expected type: {id_type}")
        # Check if types are identical (string comparison as fallback)
        print(f"   Types match: {str(id_type_check) == str(id_type)}")
    except Exception as e:
        print(f"   Type check error: {e}")
    
    # 4. Apply the identity function to a type and a value
    print("\n4. Applying the identity function:")
    try:
        # First, apply to the type 'Nat'
        id_nat = mk_app(id_term, mk_const("Nat"))
        print(f"   Identity function for Nat: {id_nat}")
        nat_type = kernel.infer_type(id_nat)
        print(f"   Type of id_nat: {nat_type}")
        
        # Then, apply to a value '0'
        id_nat_zero = mk_app(id_nat, mk_const("zero"))
        print(f"   Identity function applied to zero: {id_nat_zero}")
        zero_type = kernel.infer_type(id_nat_zero)
        print(f"   Type of id_nat_zero: {zero_type}")
    except Exception as e:
        print(f"   Application error: {e}")
    
    # 5. Reduce the application
    print("\n5. Reducing the application:")
    try:
        if 'id_nat_zero' in locals():
            reduced_term = kernel.reduce(id_nat_zero, strategy=ReductionStrategy.NF)
            print(f"   Original: {id_nat_zero}")
            print(f"   Reduced:  {reduced_term}")
        else:
            print("   Skipping reduction as application failed")
    except Exception as e:
        print(f"   Reduction error: {e}")
    
    # 6. Demo the failure of parsing Pi types
    print("\n6. Note on syntax limitations:")
    print("   The current parser doesn't directly support Pi types with the Π symbol.")
    print("   Instead, use the kernel API (mk_pi) to create Pi types programmatically.")
    print("   For interactive theorem proving, consider using lambda expressions (λ) which are supported.")
    
    print("\nDemo completed.")

if __name__ == "__main__":
    main() 