import contextlib
import importlib
import inspect
import re
import typing

import fastapi
import pydantic
import pydantic_settings
from loguru import logger
from openai.types.beta.threads.required_action_function_tool_call import (
    Function,
    RequiredActionFunctionToolCall,
)
from openai.types.beta.threads.run import RequiredActionSubmitToolOutputs
from openai.types.shared.function_definition import FunctionDefinition

import functic
from functic.types.pagination import Pagination
from functic.types.tool_output import ToolOutput

type FuncticFunctions = typing.Dict[typing.Text, typing.Type[functic.FuncticBaseModel]]


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

    def depends_functic_functions() -> FuncticFunctions:
        return app.state.functic_functions

    # Add routes
    @app.get("/functions")
    async def api_list_functions(
        request: fastapi.Request,
        functic_functions: FuncticFunctions = fastapi.Depends(
            depends_functic_functions
        ),
    ) -> Pagination[FunctionDefinition]:
        return Pagination(
            data=[m.function_definition for m in list(functic_functions.values())]
        )

    @app.get("/functions/{function_name}")
    async def api_retrieve_function(
        request: fastapi.Request,
        function_name: typing.Text = fastapi.Path(...),
        functic_functions: FuncticFunctions = fastapi.Depends(
            depends_functic_functions
        ),
    ) -> FunctionDefinition:
        if function_name not in functic_functions:
            raise fastapi.HTTPException(status_code=404, detail="Function not found")

        functic_model = functic_functions[function_name]
        return functic_model.function_definition

    @app.post("/functions/invoke")
    async def api_invoke_function(
        function_invoke_request: Function = fastapi.Body(...),
        functic_functions: FuncticFunctions = fastapi.Depends(
            depends_functic_functions
        ),
    ) -> FunctionInvokeResponse:
        if function_invoke_request.name not in functic_functions:
            raise fastapi.HTTPException(status_code=404, detail="Function not found")

        functic_model = functic_functions[function_invoke_request.name]
        functic_obj = functic_model.from_args_str(function_invoke_request.arguments)
        await functic_obj.execute()
        return FunctionInvokeResponse(result=functic_obj.content_parsed)

    @app.post("/assistant/tool_call")
    async def api_assistant_tool_call(
        request: fastapi.Request,
        required_action_function_tool_call: RequiredActionFunctionToolCall,
        functic_functions: FuncticFunctions = fastapi.Depends(
            depends_functic_functions
        ),
    ) -> ToolOutput:
        if required_action_function_tool_call.function.name not in functic_functions:
            raise fastapi.HTTPException(status_code=404, detail="Function not found")

        # Create the Functic model
        functic_model = functic_functions[
            required_action_function_tool_call.function.name
        ]
        functic_obj = functic_model.from_args_str(
            required_action_function_tool_call.function.arguments
        )
        functic_obj.set_tool_call_id(required_action_function_tool_call.id)

        # Execute the function
        await functic_obj.execute()

        # Return the tool output
        return functic_obj.tool_output

    @app.post("/assistant/tool_calls")
    async def api_assistant_tool_calls(
        request: fastapi.Request,
        required_action_submit_tool_outputs: RequiredActionSubmitToolOutputs,
        functic_functions: FuncticFunctions = fastapi.Depends(
            depends_functic_functions
        ),
    ) -> AssistantToolCallsResponse:
        for tool_call in required_action_submit_tool_outputs.tool_calls:
            if tool_call.function.name not in functic_functions:
                raise fastapi.HTTPException(
                    status_code=404,
                    detail=f"Function '{tool_call.function.name}' not found",
                )

        # Execute the functions
        tool_outputs: typing.List[ToolOutput] = []
        for tool_call in required_action_submit_tool_outputs.tool_calls:
            functic_model = functic_functions[tool_call.function.name]
            functic_obj = functic_model.from_args_str(tool_call.function.arguments)
            functic_obj.set_tool_call_id(tool_call.id)

            await functic_obj.execute()

            tool_outputs.append(functic_obj.tool_output)

        return AssistantToolCallsResponse(tool_outputs=tool_outputs)

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


class FunctionInvokeResponse(pydantic.BaseModel):
    result: typing.Any


class AssistantToolCallsResponse(pydantic.BaseModel):
    tool_outputs: typing.List[ToolOutput]


app = create_app()
