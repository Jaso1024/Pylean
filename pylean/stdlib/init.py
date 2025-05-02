"""
Standard library initialization.

This module provides functions for initializing the standard library
and loading it into a Pylean environment.
"""

from pylean.kernel import Environment, Kernel
from pylean.stdlib import logic, nat, list


def init_stdlib(kernel: Kernel = None) -> Kernel:
    """
    Initialize the standard library.
    
    This adds all standard library definitions to a kernel,
    creating a new kernel if none is provided.
    
    Args:
        kernel: Optional existing kernel to extend
        
    Returns:
        A kernel with the standard library loaded
    """
    # Create a new kernel if none is provided
    if kernel is None:
        kernel = Kernel()
    
    # Initialize each library module
    kernel = logic.init_logic(kernel)
    kernel = nat.init_nat(kernel)
    kernel = list.init_list(kernel)
    
    return kernel 