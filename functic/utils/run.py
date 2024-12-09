import asyncio
import inspect
import typing
from typing import Any, Callable, Coroutine, ParamSpec, TypeVar

R = TypeVar("R")
P = ParamSpec("P")

FunctionType = Callable[P, R]  # Regular function type
CoroutineType = Callable[P, Coroutine[Any, Any, R]]  # Coroutine function type


async def run_func(
    func: FunctionType | CoroutineType[P, R], *args: P.args, **kwargs: P.kwargs
) -> R:
    if inspect.iscoroutinefunction(func):
        func = typing.cast(CoroutineType[P, R], func)
        return await func(*args, **kwargs)
    else:
        func = typing.cast(FunctionType[P, R], func)
        return await asyncio.to_thread(func, *args, **kwargs)


if __name__ == "__main__":

    async def test_func(name: str):
        return f"Hello, {name}!"

    res = asyncio.run(run_func(test_func, "John"))
    print(res)


def sync_run_func(
    func: FunctionType | CoroutineType[P, R], *args: P.args, **kwargs: P.kwargs
) -> R:
    if inspect.iscoroutinefunction(func):
        func = typing.cast(CoroutineType[P, R], func)
        return asyncio.run(func(*args, **kwargs))
    else:
        func = typing.cast(FunctionType[P, R], func)
        return func(*args, **kwargs)
