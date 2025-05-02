# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2023-07-20

### Added
- Core kernel module with expression types (variables, constants, applications, lambdas, etc.)
- Expression utility functions (lift, instantiate, occurs_in)
- Comprehensive test suite for the kernel module
- Example scripts demonstrating the kernel module usage
- Updated documentation with usage examples
- Proper Python package structure (setup.py, pyproject.toml, requirements.txt)
- GitHub Actions CI configuration
- Lambda expression parsing for both Unicode (λ) and ASCII ("lambda") syntax
- Support for basic binary operations with proper operator precedence
- Enhanced parser architecture with better error handling and recovery
- Comprehensive tests for expression parsing
- Updated documentation and project status
- Advanced pattern matching optimizations
  - Exhaustiveness checking to prevent runtime errors
  - Redundancy detection and elimination
  - Decision tree generation for efficient pattern matching
  - Support for nested patterns and pattern guards
- C backend optimizations
  - Constant folding for arithmetic expressions
  - Common subexpression elimination
  - Function inlining for small and frequently called functions
  - Dead code elimination for unused declarations
- Foreign Function Interface (FFI) for calling C functions from Pylean
- Tail call optimization for recursive functions in the LLVM backend
- Command-line interface with interactive REPL
- Demo mode for showcasing features
- PyPI package publishing support
- Extended documentation with installation and usage instructions

### Changed
- Cleaned up parser implementation by removing debug print statements
- Improved error handling in the parser
- Moved tests to a dedicated tests directory for better organization
- Improved the parser implementation to handle more complex expressions
- Updated the project status from Alpha to Beta
- Enhanced technical documentation with recent progress
- Updated technical documentation to reflect new features
- Improved the pattern matching implementation with optimization support
- Finalized module structure for package distribution

### Fixed
- Error handling in the parser's longest_match function
- Error handling in the parser's leading_parser function
- Missing __init__.py files in package directories
- Module imports for proper package installation
- CLI functionality for a better user experience

## [0.2.0] - 2023-05-15

### Added
- Type class system with instance resolution
- Tactic-based theorem proving capabilities
- Inductive type eliminators for proofs by induction
- Meta-programming support
- Basic standard library (logic, nat, list)

### Changed
- Improved environment implementation with better caching
- Enhanced type checking with more informative error messages

## [0.1.0] - 2023-01-10

### Added
- Core kernel with expression system and type checking
- Environment implementation for managing declarations
- Basic parser for Lean-like syntax
- De Bruijn indices for variable binding
- Support for Pi types, lambda expressions, and sort universes
- Type universe handling
- Basic REPL for interactive exploration
- Initial parser implementation with Pratt parsing for expressions
- Basic parsing for literals, identifiers, and operators
- Support for operator precedence and associativity
- Tests for basic expression parsing 