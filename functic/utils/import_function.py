import importlib
import re
import typing

import functic


def import_function(path: str) -> typing.Type[functic.FuncticBaseModel]:
    """
    Import and validate a FuncticBaseModel class from a module path.

    Args:
        path: Module path in format "module.path:class_name"
              Example: "functic.functions.examples.get_coordinates:GetCoordinates"

    Returns:
        The imported class that inherits from FuncticBaseModel

    Raises:
        ValueError: If path format is invalid or class doesn't inherit from
                   FuncticBaseModel
        ImportError: If module cannot be imported
        AttributeError: If class doesn't exist in module
    """
    # Validate path format - must contain exactly one ":"
    if ":" not in path:
        raise ValueError(
            f"Invalid path format: '{path}'. "
            "Expected format: 'module.path:class_name'"
        )

    if path.count(":") != 1:
        raise ValueError(
            f"Invalid path format: '{path}'. "
            "Path must contain exactly one ':' separator"
        )

    module_name, class_name = path.rsplit(":", 1)

    # Validate module name format (should be valid Python module path)
    if not module_name:
        raise ValueError("Module name cannot be empty")

    # Check if module name contains valid characters for Python modules
    module_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*$")
    if not module_pattern.match(module_name):
        raise ValueError(
            f"Invalid module name: '{module_name}'. "
            "Module name must be a valid Python module path"
        )

    # Validate class name format (should be valid Python identifier)
    if not class_name:
        raise ValueError("Class name cannot be empty")

    if not class_name.isidentifier():
        raise ValueError(
            f"Invalid class name: '{class_name}'. "
            "Class name must be a valid Python identifier"
        )

    try:
        # Import the module
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Cannot import module '{module_name}': {e}")

    try:
        # Get the class from the module
        cls = getattr(module, class_name)
    except AttributeError:
        raise AttributeError(
            f"Class '{class_name}' not found in module '{module_name}'"
        )

    # Check if it's a class (type)
    if not isinstance(cls, type):
        raise ValueError(f"'{class_name}' is not a class in module '{module_name}'")

    # Check if the class inherits from FuncticBaseModel
    if not issubclass(cls, functic.FuncticBaseModel):
        raise ValueError(
            f"Class '{class_name}' does not inherit from functic.FuncticBaseModel"
        )

    return cls
