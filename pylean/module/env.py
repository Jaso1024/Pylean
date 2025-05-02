"""
Environment extensions for the module system.

This module implements functions for extending environments
with imported modules.
"""

from typing import Dict, List, Optional, Set, Tuple, Union

from pylean.kernel import Environment, Declaration, Name
from pylean.module.module import (
    Module, ModuleImport, find_module, load_module,
    make_fully_qualified_name, parse_namespace_name
)


class EnvExtension:
    """
    Represents an extension to an environment from imported modules.
    
    This class keeps track of imported declarations and their sources.
    """
    
    def __init__(self):
        """Initialize an empty environment extension."""
        self.decls: Dict[str, Declaration] = {}
        self.sources: Dict[str, str] = {}  # Maps decl names to module names
        self.imported_modules: Dict[str, ModuleImport] = {}
        self.open_namespaces: List[str] = []  # List of open namespaces
    
    def add_open_namespace(self, namespace: str) -> None:
        """
        Add an open namespace to the extension.
        
        Args:
            namespace: The namespace to open
        """
        if namespace not in self.open_namespaces:
            self.open_namespaces.append(namespace)
    
    def resolve_name(self, name: str) -> Optional[str]:
        """
        Resolve a name to its fully qualified form.
        
        Args:
            name: The name to resolve
            
        Returns:
            The fully qualified name, or None if not found
        """
        # Check if it's already a fully qualified name
        if name in self.decls:
            return name
        
        # Try open namespaces
        for namespace in self.open_namespaces:
            full_name = make_fully_qualified_name(namespace, name)
            if full_name in self.decls:
                return full_name
        
        # Try module aliases
        for module_name, module_import in self.imported_modules.items():
            if module_import.alias and name.startswith(f"{module_import.alias}."):
                # Replace the alias with the actual module name
                local_name = name[len(module_import.alias) + 1:]
                full_name = make_fully_qualified_name(module_name, local_name)
                if full_name in self.decls:
                    return full_name
        
        return None
    
    def add_module(self, module: Module, open_all: bool = False) -> None:
        """
        Add a module's declarations to the extension.
        
        Args:
            module: The module to add
            open_all: Whether to open all namespaces from this module
        """
        module_import = ModuleImport(module.name)
        
        # Add declarations from the module
        all_decls = module.get_all_declarations()
        for name, decl in all_decls.items():
            # Skip declarations that are already imported, unless this is a newer version
            if name in self.decls:
                # TODO: implement version checking logic
                continue
            
            # Add the declaration to our extension
            self.decls[name] = decl
            self.sources[name] = module.name
            module_import.imported_names.append(name)
        
        # If open_all is True, add the root namespace to open namespaces
        if open_all:
            self.add_open_namespace(module.name)
        
        # Store the module import
        self.imported_modules[module.name] = module_import
    
    def get_declaration(self, name: str) -> Optional[Declaration]:
        """
        Get a declaration from the extension.
        
        Args:
            name: The name of the declaration
            
        Returns:
            The declaration, or None if not found
        """
        full_name = self.resolve_name(name)
        if full_name:
            return self.decls.get(full_name)
        return None
    
    def get_source_module(self, name: str) -> Optional[str]:
        """
        Get the source module for a declaration.
        
        Args:
            name: The name of the declaration
            
        Returns:
            The source module name, or None if not found
        """
        full_name = self.resolve_name(name)
        if full_name:
            return self.sources.get(full_name)
        return None


def get_imported_modules(env: Environment) -> List[str]:
    """
    Get a list of imported modules from an environment.
    
    Args:
        env: The environment to check
        
    Returns:
        A list of imported module names
    """
    # This information would be stored in environment attributes
    # For now, return an empty list
    return []


def import_module(env: Environment, module_name: str,
                 search_paths: Optional[List[str]] = None,
                 alias: Optional[str] = None,
                 open_all: bool = False) -> Tuple[Environment, Module]:
    """
    Import a module into an environment.
    
    Args:
        env: The environment to extend
        module_name: The name of the module to import
        search_paths: Optional list of search paths
        alias: Optional alias for the module
        open_all: Whether to open all namespaces from this module
        
    Returns:
        A tuple of (extended environment, imported module)
        
    Raises:
        FileNotFoundError: If the module cannot be found
    """
    # Find the module file
    module_path = find_module(module_name, search_paths)
    if module_path is None:
        raise FileNotFoundError(f"Module not found: {module_name}")
    
    # Load the module
    module = load_module(module_path)
    
    # If an alias is provided, set it in the module
    if alias and module_name != alias:
        for module_import in module.imported_modules.values():
            if module_import.name == module_name:
                module_import.alias = alias
    
    # Create an extension from the module
    ext = EnvExtension()
    ext.add_module(module, open_all)
    
    # Recursively import dependencies
    for import_stmt in module.get_imports():
        # Skip modules that are already imported
        if import_stmt.module_name in get_imported_modules(env):
            continue
        
        # Import the dependency with the specified alias and open settings
        env, _ = import_module(
            env, import_stmt.module_name, search_paths,
            import_stmt.alias, import_stmt.is_explicit
        )
    
    # Extend the environment with the module's declarations
    new_env = env
    for name, decl in ext.decls.items():
        # Skip declarations that are already in the environment
        if name in env.decls:
            continue
        
        # Add the declaration to the environment
        new_env = new_env.add_decl(decl)
    
    # Return the extended environment and the imported module
    return new_env, module


def open_namespace(env: Environment, namespace: str) -> Environment:
    """
    Open a namespace in an environment.
    
    This allows access to declarations in the namespace without qualifying them.
    
    Args:
        env: The environment to extend
        namespace: The namespace to open
        
    Returns:
        The extended environment
    """
    # TODO: Implement namespace opening in the environment
    # For now, return the environment unchanged
    return env 