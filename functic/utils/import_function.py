import importlib
import inspect
import re
import types
import typing

import functic


def import_module_with_target(path: str) -> tuple[types.ModuleType, str]:
    """
    Import module and extract target name from path format 'module.path:target_name'.
    Validates path format and module/target name syntax.
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

    module_name, target_name = path.rsplit(":", 1)

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
    if not target_name:
        raise ValueError("Class name cannot be empty")

    if not target_name.isidentifier():
        raise ValueError(
            f"Invalid target name: '{target_name}'. "
            "Target name must be a valid Python identifier"
        )

    try:
        # Import the module
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Cannot import module '{module_name}': {e}")

    return (module, target_name)


def import_functic_model_type(path: str) -> typing.Type[functic.FuncticBaseModel]:
    """
    Import and validate a FuncticBaseModel class from module path.
    Path format: 'module.path:class_name'
    Returns the class that inherits from FuncticBaseModel.
    """
    module, class_name = import_module_with_target(path)

    try:
        # Get the class from the module
        cls = getattr(module, class_name)
    except AttributeError:
        raise AttributeError(
            f"Class '{class_name}' not found in module '{module.__name__}'"
        )

    # Check if it's a class (type)
    if not isinstance(cls, type):
        raise ValueError(f"'{class_name}' is not a class in module '{module.__name__}'")

    # Check if the class inherits from FuncticBaseModel
    if not issubclass(cls, functic.FuncticBaseModel):
        raise ValueError(
            f"Class '{class_name}' does not inherit from functic.FuncticBaseModel"
        )

    return cls


def import_functic_function(
    path: str,
) -> typing.Union[
    typing.Callable[..., typing.Any],
    typing.Callable[..., typing.Coroutine[typing.Any, typing.Any, typing.Any]],
]:
    """
    Import and validate a function from module path.
    Function must have FuncticBaseModel as first parameter with type annotation.
    Path format: 'module.path:function_name'
    """
    module, function_name = import_module_with_target(path)

    try:
        function = getattr(module, function_name)
    except AttributeError:
        raise AttributeError(
            f"Function '{function_name}' not found in module '{module.__name__}'"
        )

    # Check if it's callable
    if not callable(function):
        raise ValueError(
            f"'{function_name}' is not callable in module '{module.__name__}'"
        )

    # Get function signature
    try:
        sig = inspect.signature(function)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot inspect signature of function '{function_name}': {e}")

    params = list(sig.parameters.values())

    # Check first parameter (request)
    first_param = params[0]
    if first_param.kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise ValueError(
            f"First parameter '{first_param.name}' must be positional in function "
            f"'{function_name}'"
        )

    # Check that first parameter is annotated as functic.FuncticBaseModel or subclass
    if first_param.annotation == inspect.Parameter.empty:
        raise ValueError(
            f"First parameter '{first_param.name}' must have type annotation "
            f"(functic.FuncticBaseModel or subclass) in function '{function_name}'"
        )

    # Handle string annotations and forward references
    annotation = first_param.annotation
    if isinstance(annotation, str):
        # Try to resolve the string annotation in the module's namespace
        try:
            annotation = eval(annotation, module.__dict__)
        except (NameError, AttributeError):
            # If we can't resolve it, we'll do a string comparison
            if annotation not in ["functic.FuncticBaseModel", "FuncticBaseModel"]:
                raise ValueError(
                    f"First parameter '{first_param.name}' must be annotated as "
                    f"functic.FuncticBaseModel or a subclass, got '{annotation}' "
                    f"in function '{function_name}'"
                )
            # If it's a string that matches, we'll trust it for now
            return function

    # Check if annotation is a type and is a subclass of FuncticBaseModel
    if isinstance(annotation, type) and issubclass(
        annotation, functic.FuncticBaseModel
    ):
        return function

    # If we get here, the annotation is not valid
    raise ValueError(
        f"First parameter '{first_param.name}' must be annotated as "
        f"functic.FuncticBaseModel or a subclass, got '{annotation}' "
        f"in function '{function_name}'"
    )
