"""
Memory management for the Pylean runtime.

This module implements basic memory management for
the Pylean runtime, using a simple reference counting
system similar to Lean.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ObjectKind(Enum):
    """
    Enumeration of the different kinds of runtime objects.
    """
    SCALAR = 0    # Simple scalar values (integers, etc.)
    CONSTRUCTOR = 1    # Constructor objects
    CLOSURE = 2    # Function closures
    ARRAY = 3    # Array objects
    STRING = 4    # String objects
    THUNK = 5    # Thunks (delayed computations)


class Object:
    """
    Base class for all runtime objects in Pylean.
    
    Each object has a reference count and a kind.
    """
    
    def __init__(self, kind: ObjectKind):
        """
        Initialize a runtime object.
        
        Args:
            kind: The kind of the object
        """
        self.kind = kind
        self.ref_count = 1  # All objects start with a reference count of 1
    
    def __del__(self):
        """Release any resources owned by this object."""
        pass


class Box(Object):
    """
    A box containing a simple scalar value.
    """
    
    def __init__(self, value: Any):
        """
        Initialize a box object.
        
        Args:
            value: The value to box
        """
        super().__init__(ObjectKind.SCALAR)
        self.value = value


class Constructor(Object):
    """
    A constructor object with a tag and fields.
    """
    
    def __init__(self, tag: int, fields: List[Object]):
        """
        Initialize a constructor object.
        
        Args:
            tag: The constructor tag
            fields: The constructor fields
        """
        super().__init__(ObjectKind.CONSTRUCTOR)
        self.tag = tag
        self.fields = fields


class Closure(Object):
    """
    A function closure with captured variables.
    """
    
    def __init__(self, fn_ptr, captured: List[Object]):
        """
        Initialize a closure object.
        
        Args:
            fn_ptr: The function pointer
            captured: The captured variables
        """
        super().__init__(ObjectKind.CLOSURE)
        self.fn_ptr = fn_ptr
        self.captured = captured


class Array(Object):
    """
    An array of objects.
    """
    
    def __init__(self, elements: List[Object]):
        """
        Initialize an array object.
        
        Args:
            elements: The array elements
        """
        super().__init__(ObjectKind.ARRAY)
        self.elements = elements
        self.size = len(elements)


class String(Object):
    """
    A string object.
    """
    
    def __init__(self, value: str):
        """
        Initialize a string object.
        
        Args:
            value: The string value
        """
        super().__init__(ObjectKind.STRING)
        self.value = value
        self.size = len(value)


# Global object registry for debugging/tracking
_object_registry: Dict[int, Object] = {}


def alloc_object(kind: ObjectKind, *args, **kwargs) -> Object:
    """
    Allocate a new object of the given kind.
    
    Args:
        kind: The kind of object to allocate
        *args: Additional positional arguments for the object constructor
        **kwargs: Additional keyword arguments for the object constructor
        
    Returns:
        The newly allocated object
    """
    obj = None
    
    if kind == ObjectKind.SCALAR:
        obj = Box(*args, **kwargs)
    elif kind == ObjectKind.CONSTRUCTOR:
        obj = Constructor(*args, **kwargs)
    elif kind == ObjectKind.CLOSURE:
        obj = Closure(*args, **kwargs)
    elif kind == ObjectKind.ARRAY:
        obj = Array(*args, **kwargs)
    elif kind == ObjectKind.STRING:
        obj = String(*args, **kwargs)
    else:
        raise ValueError(f"Unknown object kind: {kind}")
    
    _object_registry[id(obj)] = obj
    return obj


def free_object(obj: Object) -> None:
    """
    Free an object and release its resources.
    
    Args:
        obj: The object to free
    """
    # Remove from the registry
    _object_registry.pop(id(obj), None)
    
    # Call destructor to release resources
    obj.__del__()


def inc_ref(obj: Object) -> Object:
    """
    Increment the reference count of an object.
    
    Args:
        obj: The object to increment the reference count of
        
    Returns:
        The same object
    """
    obj.ref_count += 1
    return obj


def dec_ref(obj: Object) -> None:
    """
    Decrement the reference count of an object.
    If the reference count reaches 0, free the object.
    
    Args:
        obj: The object to decrement the reference count of
    """
    obj.ref_count -= 1
    if obj.ref_count <= 0:
        # First, recursively decrement references to contained objects
        if obj.kind == ObjectKind.CONSTRUCTOR:
            for field in obj.fields:
                dec_ref(field)
        elif obj.kind == ObjectKind.CLOSURE:
            for captured in obj.captured:
                dec_ref(captured)
        elif obj.kind == ObjectKind.ARRAY:
            for element in obj.elements:
                dec_ref(element)
        
        # Then free the object itself
        free_object(obj)


def get_rc(obj: Object) -> int:
    """
    Get the reference count of an object.
    
    Args:
        obj: The object to get the reference count of
        
    Returns:
        The reference count
    """
    return obj.ref_count 