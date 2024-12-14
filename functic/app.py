import contextlib
import importlib
import inspect
import re
import typing

import fastapi
import pydantic
import pydantic_settings
from loguru import logger
from openai.types.shared.function_definition import FunctionDefinition

import functic
from functic.types.pagination import Pagination


@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI) -> typing.AsyncIterator[None]:
    import pymongo

    # Setup application
    logger.debug("Setting up application")
    app_settings = AppSettings()
    app.state.settings = app_settings
    app.state.logger = logger
    app.state.db = pymongo.MongoClient(
        app_settings.FUNCTIC_DATABASE_CONNECTION_STRING.get_secret_value(),
    )

    # Load functic functions
    logger.debug("Loading functic functions")
    functic_functions: typing.Dict[
        typing.Text, typing.Type[functic.FuncticBaseModel]
    ] = {}
    for module_name in app_settings.FUNCTIC_FUNCTIONS:
        logger.debug(f"Reading module: '{module_name}'")
        _mod = importlib.import_module(module_name)

        for cls_name, _cls in inspect.getmembers(_mod, inspect.isclass):
            if (
                _cls.__module__ == _mod.__name__  # The class is defined in the module
                and issubclass(
                    _cls, functic.FuncticBaseModel
                )  # The class is a subclass of FuncticBaseModel
            ):  # Filter out non-functic classes
                logger.debug(f"Validating functic class: '{cls_name}'")

                # Validate the function config
                _cls.functic_config.raise_if_invalid()

                _func_name = _cls.functic_config.name

                # Check for duplicate function names
                if _func_name in functic_functions:
                    logger.warning(
                        "There are multiple functions with the same name: "
                        + f"{_func_name}, overwriting the first one."
                        + "You might want to rename one of them to "
                        + "avoid this issue."
                    )

                functic_functions[_func_name] = _cls
                logger.info(f"Added function: '{_func_name}'")

    app.state.functic_functions = functic_functions

    yield


def create_app() -> fastapi.FastAPI:
    logger.debug("Creating application")
    app = fastapi.FastAPI(lifespan=lifespan)

    # Add routes
    @app.get("/functions")
    def api_list_functions(request: fastapi.Request) -> Pagination[FunctionDefinition]:
        functic_models: typing.List[typing.Type[functic.FuncticBaseModel]] = list(
            app.state.functic_functions.values()
        )
        return Pagination(data=[m.function_definition for m in functic_models])

    @app.get("/functions/{function_name}")
    def api_retrieve_function(
        request: fastapi.Request, function_name: str
    ) -> FunctionDefinition:
        functic_functions: typing.Dict[
            typing.Text, typing.Type[functic.FuncticBaseModel]
        ] = app.state.functic_functions
        if function_name not in functic_functions:
            raise fastapi.HTTPException(status_code=404, detail="Function not found")
        functic_model = functic_functions[function_name]
        return functic_model.function_definition

    return app


class AppSettings(pydantic_settings.BaseSettings):
    FUNCTIC_DATABASE_CONNECTION_STRING: pydantic.SecretStr = pydantic.Field(
        default=pydantic.SecretStr("mongodb://localhost:27017/"),
        description="The connection string to the Functic database",
    )
    FUNCTIC_DATABASE_NAME: str = pydantic.Field(
        default="functic",
        description="The name of the Functic database",
    )
    FUNCTIC_FUNCTIONS_REPOSITORY_TABLE_NAME: str = pydantic.Field(
        default="functions",
        description="The name of the Functic functions repository table",
    )
    FUNCTIC_FUNCTIONS: typing.List[typing.Text] = pydantic.Field(
        default_factory=lambda: [
            "functic.functions.azure.get_weather_forecast_daily",
            "functic.functions.azure.get_weather_forecast_hourly",
            "functic.functions.google.get_maps_geocode",
        ],
        description="The list of Functic functions",
    )

    @pydantic.field_validator("FUNCTIC_FUNCTIONS", mode="before")
    def split_functic_functions(cls, value):
        if isinstance(value, typing.Text):
            output: typing.List[typing.Text] = []
            for s in re.split(r"[;,]", value):
                s = s.strip(" '\"").strip()
                if s:
                    output.append(s)
            return output
        return value


app = create_app()
