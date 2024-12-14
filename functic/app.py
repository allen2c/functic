import contextlib
import importlib
import inspect
import re
import textwrap
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
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
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
    app = fastapi.FastAPI(
        title="Functic API",
        summary="API service for executing OpenAI function tools with ease",
        description=textwrap.dedent(
            """
            Functic API provides endpoints to manage and execute OpenAI function tools in a standardized way.

            Key features:
            - List and retrieve available function definitions
            - Execute functions via direct invocation
            - Handle OpenAI Chat Completion tool calls
            - Support for OpenAI Assistant tool calls
            - Built-in support for weather forecasts and geocoding functions

            The API integrates with OpenAI's function calling capabilities and provides a consistent interface
            for executing functions across different services like Azure Maps and Google Maps.
            """  # noqa: E501
        ).strip(),
        version=functic.__version__,
        lifespan=lifespan,
    )

    def depends_functic_functions(request: fastapi.Request) -> FuncticFunctions:
        return app.state.functic_functions

    # Add routes
    @app.get(
        "/functions",
        summary="List Available Functions",
        description="Retrieves a paginated list of all available function definitions that can be used as OpenAI function tools. Each function includes its name, description, and parameter schema.",  # noqa: E501
    )
    async def api_list_functions(
        request: fastapi.Request,
        functic_functions: FuncticFunctions = fastapi.Depends(
            depends_functic_functions
        ),
    ) -> Pagination[FunctionDefinition]:
        return Pagination(
            data=[m.function_definition for m in list(functic_functions.values())]
        )

    @app.get(
        "/functions/{function_name}",
        summary="Retrieve Function Definition",
        description="Retrieves the complete function definition for a specific function by its name. Returns detailed information including the function's name, description, parameters schema, and required fields.",  # noqa: E501
    )
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

    @app.post(
        "/functions/invoke",
        summary="Invoke Function",
        description="Executes a specific function with the provided arguments. The function must be registered in the system. Returns the function's execution result in a standardized format.",  # noqa: E501
    )
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

    @app.post(
        "/chat/tool_call",
        summary="Handle Chat-Based Tool Calls",
        description="Processes tool call requests initiated via chat interfaces. This endpoint validates the requested function, executes it with the provided arguments, and returns the result formatted for chat interactions.",  # noqa: E501
    )
    async def api_chat_tool_call(
        request: fastapi.Request,
        chat_completion_message_tool_call: ChatCompletionMessageToolCall,
        functic_functions: FuncticFunctions = fastapi.Depends(
            depends_functic_functions
        ),
    ) -> ChatCompletionToolMessageParam:
        if chat_completion_message_tool_call.function.name not in functic_functions:
            raise fastapi.HTTPException(status_code=404, detail="Function not found")

        # Create the Functic model
        functic_model = functic_functions[
            chat_completion_message_tool_call.function.name
        ]
        functic_obj = functic_model.from_args_str(
            chat_completion_message_tool_call.function.arguments
        )
        functic_obj.set_tool_call_id(chat_completion_message_tool_call.id)

        # Execute the function
        await functic_obj.execute()

        # Return the tool message
        return ChatCompletionToolMessageParam(
            {
                "role": "tool",
                "content": functic_obj.content_parsed,
                "tool_call_id": chat_completion_message_tool_call.id,
            }
        )

    @app.post(
        "/assistant/tool_call",
        summary="Execute Assistant-Initiated Tool Call",
        description="Handles tool call requests initiated by assistant actions. This endpoint ensures the requested function exists, executes it with the provided arguments, and returns the tool's output.",  # noqa: E501
    )
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

    @app.post(
        "/assistant/tool_calls",
        summary="Batch Execute Multiple Assistant-Initiated Tool Calls",
        description="Processes a batch of tool call requests initiated by assistant actions. This endpoint validates each requested function, executes them concurrently with the provided arguments, and returns a consolidated response containing all tool outputs.",  # noqa: E501
    )
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
