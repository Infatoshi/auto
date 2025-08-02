# CRUSH Command Reference

## Build Commands
- `python setup.py develop` - Install cutlass_dsl_template in development mode
- Use existing CUDA toolkit commands for compiling .cu files

## Test Commands
- `pytest` - Run all tests
- `pytest path/to/test_file.py::test_function` - Run a specific test

## Lint/Format Commands
- `black .` - Format Python code
- No isort found, using default import order

## Code Style Guidelines

### Imports
- Standard library imports first
- Third-party imports grouped together
- Local imports last
- Each group separated by blank line

### Formatting
- Use Black for consistent formatting
- Max line length: 88 characters
- Follow PEP 8 guidelines

### Naming Conventions
- Use snake_case for variables and functions
- Use PascalCase for classes
- Use uppercase for constants
- Prefix private members with underscore

### Types
- Use type hints for function signatures
- Use Optional[T] for nullable types
- Prefer specific types over Any

### Error Handling
- Use specific exception types
- Include meaningful error messages
- Handle exceptions at appropriate levels
- Use context managers for resource cleanup

## Project Notes
- CUDA kernels in .cu files
- Python interface code in cutlass_dsl_template/
- Configuration in configs/ directory