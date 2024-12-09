import importlib
import importlib.util
import inspect
import typing
from typing import Any, Callable, Coroutine, ParamSpec, TypeVar

# Type variables for return types
R = TypeVar("R")
R_co = TypeVar("R_co", covariant=True)
P = ParamSpec("P")  # Type variable for function parameters
FunctionType = Callable[P, R]  # Regular function type
CoroutineType = Callable[P, Coroutine[Any, Any, R]]  # Coroutine function type


def import_function(
    path_spec: typing.Text,
    notation: typing.Literal["dot", "colon", ".", ":"] | typing.Text | None = None,
) -> FunctionType | CoroutineType:
    notation = ":" if notation == "colon" else notation
    notation = (
        ":" if notation is None and ":" in path_spec else notation
    )  # Higher priority than dot
    notation = "." if notation == "dot" else notation
    notation = "." if notation is None and "." in path_spec else notation

    # Check if notation is valid
    if notation is None:
        raise ValueError("Invalid notation: must be one of `dot`, `colon`, `.` or `:`.")
    # Check if notation is in path spec
    if notation not in path_spec:
        raise ValueError(
            "Path spec must contain a dot, e.g. `path.to.module.function`."
        )

    # Split path spec into model path and function name
    model_path, function_name = path_spec.rsplit(notation, 1)

    # Import with notation
    module = importlib.import_module(model_path)
    function = getattr(module, function_name)

    # Check if function is callable
    if not callable(function) and not inspect.iscoroutinefunction(function):
        raise ValueError(
            f"Function `{function_name}` in module `{model_path}` is not callable."
        )

    return function


if __name__ == "__main__":
    print(import_function("textwrap.dedent"))
    print(import_function("textwrap:dedent"))
    print(import_function("asyncio:gather"))
