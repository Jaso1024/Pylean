#!/usr/bin/env python3
"""
Demonstration of the type class and eliminator systems in Pylean.

This script shows how to:
1. Define inductive types and automatically generate their eliminators
2. Define type classes and instances
3. Use type class resolution to infer instances
"""

from pylean.kernel import (
    # Environment
    Environment, mk_std_env,
    
    # Expressions
    Name, Level, mk_sort, mk_const, mk_var, mk_app, mk_pi, mk_lambda,
    
    # Declarations
    mk_inductive, mk_constructor, mk_definition,
    
    # Type checking
    infer_type, check_type, Context,
    
    # Eliminator system
    generate_recursor, generate_induction_principle, generate_eliminators,
    
    # Type class system
    TypeClass, TypeClassInstance, TypeClassEnvironment, TypeClassElaborator,
    mk_type_class, mk_type_class_instance
)


def demo_inductive_types_and_eliminators():
    """
    Demonstrate the inductive type and eliminator system.
    """
    print("=== Inductive Types and Eliminators Demo ===")
    env = mk_std_env()
    
    # Define a simple inductive type: Bool
    print("\nDefining Bool type...")
    bool_name = Name.from_string("Bool")
    bool_type = mk_sort(0)  # Type 0
    
    # Define constructors: true and false
    true_name = Name.from_string("true")
    true_type = mk_const(bool_name)
    true_constr = mk_constructor(true_name, true_type, bool_name)
    
    false_name = Name.from_string("false")
    false_type = mk_const(bool_name)
    false_constr = mk_constructor(false_name, false_type, bool_name)
    
    # Create Bool inductive declaration
    bool_decl = mk_inductive(bool_name, bool_type, [true_constr, false_constr])
    
    # Add Bool to environment - this will also generate eliminators
    env = env.add_decl(bool_decl)
    
    # Verify the recursor and induction principle were generated
    recursor = env.find_decl("Bool.rec")
    induction = env.find_decl("Bool.ind")
    
    if recursor:
        print(f"Recursor generated: {recursor.name}")
        print(f"Recursor type: {recursor.type}")
    else:
        print("Recursor was not generated!")
        
    if induction:
        print(f"Induction principle generated: {induction.name}")
        print(f"Induction principle type: {induction.type}")
    else:
        print("Induction principle was not generated!")
    
    # Define a more complex inductive type: List
    print("\nDefining List type...")
    list_name = Name.from_string("List")
    
    # List takes a type parameter
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
    nil_type = mk_pi(
        type_param_name,
        type_param_type,
        mk_app(mk_const(list_name), mk_var(0))
    )
    nil_constr = mk_constructor(nil_name, nil_type, list_name)
    
    # Define cons constructor
    cons_name = Name.from_string("cons")
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
    
    # Create List inductive declaration
    list_decl = mk_inductive(list_name, list_type, [nil_constr, cons_constr])
    
    # Add List to environment
    env = env.add_decl(list_decl)
    
    # Verify the recursor and induction principle were generated
    list_recursor = env.find_decl("List.rec")
    list_induction = env.find_decl("List.ind")
    
    if list_recursor:
        print(f"List recursor generated: {list_recursor.name}")
    else:
        print("List recursor was not generated!")
        
    if list_induction:
        print(f"List induction principle generated: {list_induction.name}")
    else:
        print("List induction principle was not generated!")
    
    return env


def demo_typeclasses(env):
    """
    Demonstrate the type class system.
    """
    print("\n=== Type Class System Demo ===")
    
    # Create a type class environment
    tc_env = TypeClassEnvironment(env)
    
    # Define a Show type class
    print("\nDefining Show type class...")
    show_name = Name.from_string("Show")
    alpha_name = Name.from_string("α")
    
    # Parameter for Show: α (a type)
    param_names = [alpha_name]
    
    # Define String type for simplicity
    string_type = mk_const(Name.from_string("String"))
    
    # Field: show : α → String
    show_field_type = mk_pi(
        Name.from_string("x"),
        mk_var(0),  # α
        string_type
    )
    fields = {Name.from_string("show"): show_field_type}
    
    # Create Show type class
    show_class = mk_type_class(show_name, param_names, fields)
    tc_env = tc_env.add_class(show_class)
    
    print(f"Type class created: {show_class.name}")
    print(f"Parameters: {[str(p) for p in show_class.param_names]}")
    print(f"Fields: {[str(f) for f in show_class.fields]}")
    
    # Create Show instance for Bool
    print("\nCreating Show instance for Bool...")
    bool_show_name = Name.from_string("boolShow")
    
    # Create a dummy type for the instance
    bool_show_type = mk_app(mk_const(show_name), mk_const(Name.from_string("Bool")))
    
    # Add the instance declaration to the environment
    env = env.add_decl(mk_definition(bool_show_name, bool_show_type, mk_const("dummy_impl")))
    tc_env.env = env  # Update the env reference
    
    # The instance parameters: [Bool]
    params = [mk_const(Name.from_string("Bool"))]
    
    # The field implementations - just use dummy values for testing
    field_values = {Name.from_string("show"): mk_const(Name.from_string("boolShowImpl"))}
    
    # Create the instance
    bool_show_instance = mk_type_class_instance(show_name, bool_show_name, params, field_values)
    tc_env = tc_env.add_instance(bool_show_instance)
    
    print(f"Instance created: {bool_show_instance.instance_name}")
    print(f"For class: {bool_show_instance.class_name}")
    print(f"With parameters: {[str(p.name) if hasattr(p, 'name') else str(p) for p in bool_show_instance.params]}")
    
    # Create a Show instance for List (only if the elements can be shown)
    print("\nCreating Show instance for List...")
    list_show_name = Name.from_string("listShow")
    
    # The instance type:
    # Π (T : Type), Show T → Show (List T)
    type_param = Name.from_string("T")
    type_param_type = mk_sort(0)
    
    # Show T
    show_t = mk_app(mk_const(show_name), mk_var(0))
    
    # Show (List T)
    list_t = mk_app(mk_const(Name.from_string("List")), mk_var(0))
    show_list_t = mk_app(mk_const(show_name), list_t)
    
    # Π (T : Type), Show T → Show (List T)
    list_show_type = mk_pi(
        type_param,
        type_param_type,
        mk_pi(
            Name.from_string("showT"),
            show_t,
            show_list_t
        )
    )
    
    # Add the instance declaration to the environment
    env = env.add_decl(mk_definition(list_show_name, list_show_type, mk_const("dummy_impl")))
    tc_env.env = env  # Update the env reference
    
    # The instance has parameters: [T, Show T]
    # But for simplicity in this demo, we'll just use concrete types
    nat_type = mk_const(Name.from_string("Nat"))
    list_nat = mk_app(mk_const(Name.from_string("List")), nat_type)
    
    # Create a context for testing
    ctx = Context()
    
    # Find a Show instance for Bool
    print("\nLooking up Show instance for Bool...")
    bool_instance = tc_env.find_instance(
        show_name,
        [mk_const(Name.from_string("Bool"))],
        ctx
    )
    
    if bool_instance:
        print(f"Found instance: {bool_instance.instance_name}")
    else:
        print("No instance found for Bool!")
    
    return tc_env


if __name__ == "__main__":
    env = demo_inductive_types_and_eliminators()
    tc_env = demo_typeclasses(env)
    
    print("\nDemo completed successfully!") 