"""
Module system implementation for Pylean.

This module implements the core classes and functions for
the Pylean module system, allowing for imports and exports.
"""

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

from pylean.kernel import (
    Name, Expr, Declaration, Environment,
    DeclKind, mk_const
)


@dataclass
class Import:
    """
    Represents an import statement in a module.
    
    Attributes:
        module_name: The name of the imported module
        imports: List of specific entities to import, or None for all
        alias: Optional module alias for the import
        is_explicit: Whether this is an explicit (open) import
    """
    module_name: str
    imports: Optional[List[str]] = None
    alias: Optional[str] = None
    is_explicit: bool = False


@dataclass
class Namespace:
    """
    Represents a namespace in a module.
    
    Attributes:
        name: The namespace name
        declarations: Declarations in this namespace
        subnamespaces: Child namespaces
    """
    name: str
    declarations: Dict[str, Declaration] = field(default_factory=dict)
    subnamespaces: Dict[str, 'Namespace'] = field(default_factory=dict)
    
    def get_full_name(self, parent_path: str = "") -> str:
        """
        Get the fully qualified name of this namespace.
        
        Args:
            parent_path: The parent namespace path
            
        Returns:
            The fully qualified name
        """
        if parent_path:
            return f"{parent_path}.{self.name}" if self.name else parent_path
        return self.name
    
    def add_declaration(self, name: str, decl: Declaration) -> None:
        """
        Add a declaration to this namespace.
        
        Args:
            name: The declaration name (without namespace)
            decl: The declaration to add
        """
        self.declarations[name] = decl
    
    def get_declaration(self, name: str) -> Optional[Declaration]:
        """
        Get a declaration from this namespace.
        
        Args:
            name: The declaration name (without namespace)
            
        Returns:
            The declaration, or None if not found
        """
        return self.declarations.get(name)
    
    def get_all_declarations(self) -> Dict[str, Declaration]:
        """
        Get all declarations in this namespace.
        
        Returns:
            A dictionary of declarations
        """
        return self.declarations.copy()
    
    def get_subnamespace(self, name: str, create_if_missing: bool = False) -> Optional['Namespace']:
        """
        Get a subnamespace by name.
        
        Args:
            name: The namespace name
            create_if_missing: Whether to create the namespace if it doesn't exist
            
        Returns:
            The namespace, or None if not found and not created
        """
        if name in self.subnamespaces:
            return self.subnamespaces[name]
        
        if create_if_missing:
            namespace = Namespace(name)
            self.subnamespaces[name] = namespace
            return namespace
        
        return None
    
    def get_namespace_by_path(self, path: List[str], create_if_missing: bool = False) -> Optional['Namespace']:
        """
        Get a namespace by path.
        
        Args:
            path: The namespace path components
            create_if_missing: Whether to create namespaces if they don't exist
            
        Returns:
            The namespace, or None if not found and not created
        """
        if not path:
            return self
        
        first, rest = path[0], path[1:]
        namespace = self.get_subnamespace(first, create_if_missing)
        
        if namespace is None:
            return None
        
        return namespace.get_namespace_by_path(rest, create_if_missing)
    
    def find_declaration(self, name_parts: List[str]) -> Optional[Declaration]:
        """
        Find a declaration by its name parts.
        
        Args:
            name_parts: The declaration name parts
            
        Returns:
            The declaration, or None if not found
        """
        if len(name_parts) == 1:
            return self.get_declaration(name_parts[0])
        
        namespace_name, rest = name_parts[0], name_parts[1:]
        namespace = self.get_subnamespace(namespace_name)
        
        if namespace is None:
            return None
        
        return namespace.find_declaration(rest)


@dataclass
class ModuleImport:
    """
    Represents an imported module with additional data.
    
    Attributes:
        name: The module name
        imported_names: Names imported from this module
        env_extension: An extension to the environment from this module
        alias: Optional module alias
    """
    name: str
    imported_names: List[str] = field(default_factory=list)
    env_extension: Dict[str, Declaration] = field(default_factory=dict)
    alias: Optional[str] = None


@dataclass
class ModuleData:
    """
    Contains all the data for a Pylean module.
    
    Attributes:
        root_namespace: The root namespace
        imports: The imports used by this module
    """
    root_namespace: Namespace = field(default_factory=lambda: Namespace(""))
    imports: List[Import] = field(default_factory=list)


class Module:
    """
    Represents a Pylean module.
    
    A module contains declarations organized in namespaces
    and imports from other modules.
    """
    
    def __init__(self, name: str, data: Optional[ModuleData] = None):
        """
        Initialize a module.
        
        Args:
            name: The module name
            data: Optional module data, or None for an empty module
        """
        self.name = name
        self.data = data or ModuleData()
        self.imported_modules: Dict[str, ModuleImport] = {}
    
    def add_declaration(self, decl: Declaration, namespace: str = "") -> None:
        """
        Add a declaration to the module.
        
        Args:
            decl: The declaration to add
            namespace: Optional namespace path (dot-separated)
        """
        # Split the namespace path
        path = namespace.split(".") if namespace else []
        
        # Get or create the namespace
        target_namespace = self.data.root_namespace.get_namespace_by_path(path, True)
        
        # Extract the declaration name (without namespace)
        decl_name = str(decl.name)
        if "." in decl_name:
            decl_name = decl_name.split(".")[-1]
        
        # Add the declaration to the namespace
        target_namespace.add_declaration(decl_name, decl)
    
    def add_import(self, import_stmt: Import) -> None:
        """
        Add an import statement to the module.
        
        Args:
            import_stmt: The import statement to add
        """
        self.data.imports.append(import_stmt)
    
    def get_declaration(self, name: str) -> Optional[Declaration]:
        """
        Get a declaration from the module.
        
        Args:
            name: The fully qualified name of the declaration
            
        Returns:
            The declaration, or None if not found
        """
        # Split the name into namespace parts and declaration name
        parts = name.split(".")
        
        return self.data.root_namespace.find_declaration(parts)
    
    def get_declarations_in_namespace(self, namespace: str = "") -> Dict[str, Declaration]:
        """
        Get all declarations in a namespace.
        
        Args:
            namespace: The namespace path (dot-separated)
            
        Returns:
            A dictionary of declarations
        """
        # Split the namespace path
        path = namespace.split(".") if namespace else []
        
        # Find the namespace
        target_namespace = self.data.root_namespace.get_namespace_by_path(path)
        if target_namespace is None:
            return {}
        
        return target_namespace.get_all_declarations()
    
    def get_all_declarations(self) -> Dict[str, Declaration]:
        """
        Get all declarations in the module.
        
        Returns:
            A dictionary of fully qualified names to declarations
        """
        result = {}
        
        def collect_declarations(namespace: Namespace, prefix: str = ""):
            for name, decl in namespace.declarations.items():
                full_name = f"{prefix}.{name}" if prefix else name
                result[full_name] = decl
            
            for name, ns in namespace.subnamespaces.items():
                new_prefix = f"{prefix}.{name}" if prefix else name
                collect_declarations(ns, new_prefix)
        
        collect_declarations(self.data.root_namespace)
        return result
    
    def get_imports(self) -> List[Import]:
        """
        Get all imports in the module.
        
        Returns:
            A list of import statements
        """
        return self.data.imports


def find_module(module_name: str, search_paths: List[str] = None) -> Optional[str]:
    """
    Find a module file by name.
    
    Args:
        module_name: The module name
        search_paths: Optional list of search paths, or None for default
        
    Returns:
        The path to the module file, or None if not found
    """
    if search_paths is None:
        # Default search paths
        search_paths = [".", os.path.expanduser("~/.pylean/modules")]
    
    # Replace dots with path separators
    rel_path = module_name.replace(".", os.path.sep)
    
    # Look for module files in search paths
    for path in search_paths:
        # Try .plm (Pylean Module) file
        module_path = os.path.join(path, f"{rel_path}.plm")
        if os.path.exists(module_path):
            return module_path
        
        # Try .py file (source file)
        module_path = os.path.join(path, f"{rel_path}.py")
        if os.path.exists(module_path):
            return module_path
    
    return None


def load_module(module_path: str) -> Module:
    """
    Load a module from a file.
    
    Args:
        module_path: The path to the module file
        
    Returns:
        The loaded module
    """
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Module file not found: {module_path}")
    
    # Get the module name from the file path
    basename = os.path.basename(module_path)
    module_name = os.path.splitext(basename)[0]
    
    # Check if it's a binary module file (.plm)
    if module_path.endswith(".plm"):
        with open(module_path, "rb") as f:
            data = pickle.load(f)
        return Module(module_name, data)
    
    # Otherwise, it's a source file that needs to be compiled
    # This would involve parsing and processing the file
    # For now, just return an empty module
    return Module(module_name)


def save_module(module: Module, output_path: str) -> None:
    """
    Save a module to a file.
    
    Args:
        module: The module to save
        output_path: The path to the output file
    """
    # Make sure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the module data to a binary file
    with open(output_path, "wb") as f:
        pickle.dump(module.data, f)


def parse_namespace_name(full_name: str) -> Tuple[List[str], str]:
    """
    Parse a fully qualified name into namespace path and local name.
    
    Args:
        full_name: The fully qualified name
        
    Returns:
        A tuple of (namespace_path, local_name)
    """
    parts = full_name.split(".")
    if len(parts) <= 1:
        return [], full_name
    
    return parts[:-1], parts[-1]


def make_fully_qualified_name(namespace: str, name: str) -> str:
    """
    Create a fully qualified name from a namespace and local name.
    
    Args:
        namespace: The namespace path (dot-separated)
        name: The local name
        
    Returns:
        The fully qualified name
    """
    if not namespace:
        return name
    
    return f"{namespace}.{name}" 