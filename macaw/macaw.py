import json
from typing import List, Optional, Type
from pydantic import BaseModel, Field  # , Type
from typing import Any, Dict
import openai
import anthropic
from mistralai.client import MistralClient

from openai.types.chat import ChatCompletion
from mistralai.models.chat_completion import ChatMessage

from anthropic.types import Message
from pydantic_core import PydanticUndefined
from print_color import print


class BaseTool(BaseModel):
    name: str = Field(description="Name of the tool.")
    description: str = Field(description="Description of the tool.")
    args_schema: Optional[Type[BaseModel]] = Field(
        description="Schema for the tool arguments.", default=None
    )
    return_direct: Optional[bool] = Field(
        description="Whether to return direct the tool response as final answer to a message?",
        default=False,
    )

    def _run(self, *args, **kwargs):
        """Placeholder for the run method."""
        raise NotImplementedError("Subclasses should implement this method.")

    class Config:
        validate_assignment = True


class LLM:
    """Base LLM class which allows you use make other types of LLMs ontop of it."""

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        """INIT LLM CLASS WITH BASIC PARAMS AND OTHERS INCLUDED IN THE KWARGS"""
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs
        self.LLMClient = None

    def set_client(self):
        """SET OR INIT THE LLM CLIENT (IF OPENAI THEN USING BASE URL AND API KEY)"""
        pass

    def get_kwargs(self):
        return self.kwargs

    def ava_llm_completions(
        self, messages: List, tools: List, is_function_based: bool = False, **extras
    ) -> Any:
        """GET COMPLETIONS FROM THE LLM CLIENT"""
        pass



class OpenaiLLM(LLM, openai.OpenAI):
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        self.kwargs = kwargs
        if base_url:
            self.LLMClient: openai = openai.OpenAI(base_url=base_url, api_key=api_key)
        else:
            self.LLMClient: openai = openai.OpenAI(
                # base_url=base_url,
                api_key=api_key
            )

    def get_kwargs(self):
        return self.kwargs

    def ava_llm_completions(
        self,
        messages: List,
        tools: List,
        is_function_based: bool = False,
        streaming: bool = False,
        logging: bool = False,
    ) -> ChatCompletion:
        self.kwargs["messages"] = messages
        self.kwargs["stream"] = streaming
        if tools != []:
            if not is_function_based:
                print("is function based in avallm", is_function_based)
                print("tools in ava_llm", tools)
                self.kwargs["tools"] = tools
                print("Self kwargs when function false", self.kwargs)
            else:
                self.kwargs["functions"] = tools
                print("Self kwargs when function true", self.kwargs)
       
        chat_completion = self.LLMClient.chat.completions.create(**self.kwargs)
        return chat_completion
    
    # def ava_chain_llm_completions(
    #     self,
    #     messages: List,
    #     tools: List,
    #     is_function_based: bool = False,
    #     streaming: bool = False,
    #     logging: bool = False,
    # ) -> ChatCompletion:
    #     self.kwargs["messages"] = messages
    #     self.kwargs["stream"] = streaming
    #     if tools != []:
    #         if not is_function_based:
    #             print("is function based in avallm", is_function_based)
    #             # print("tools in ava_llm", tools)
    #             # self.kwargs["tool_choice"] = None
    #             # tools[0]["type"]=None
    #             # self.kwargs["tools"] = tools
    #             # print("Self kwargs when function false", self.kwargs)
    #         else:
    #             self.kwargs["functions"] = tools
    #             print("Self kwargs when function true", self.kwargs)
    #     # print("Ava llm kwargs: ", self.kwargs)
    #     # print("Messages : ", messages)
    #     # Assuming 'client' defined somewhere before using it
    #     # model="accounts/fireworks/models/fw-function-call-34b-v0",
    #     # model="accounts/fireworks/models/firefunction-v1",
    #     # if not streaming:
    #     chat_completion = self.LLMClient.chat.completions.create(**self.kwargs)
    #     return chat_completion
    
   

    def ava_llm_streaming_completions(
        self,
        messages: List,
        tools: List,
        is_function_based: bool = False,
        streaming: bool = False,
        logging: bool = False,
    ) -> ChatCompletion:
        self.kwargs["messages"] = messages
        self.kwargs["stream"] = streaming
        if tools != []:
            if not is_function_based:
                self.kwargs["tools"] = tools
            else:
                self.kwargs["functions"] = tools
        # print("Ava llm kwargs: ", self.kwargs)
        # print("Messages : ", messages)
        # Assuming 'client' defined somewhere before using it
        # model="accounts/fireworks/models/fw-function-call-34b-v0",
        # model="accounts/fireworks/models/firefunction-v1",

        async def stream():
            try:
                completion = self.LLMClient.chat.completions.create(**self.kwargs)
                for line in completion:
                    if logging:
                        print("Openai LLM streaming line: ", line, line, color="green")
                    chunk = None
                    if line.choices[0]:
                        chunk = line.choices[0].delta.content
                    if chunk:
                        yield chunk.encode("utf-8") + b"\n"
            except Exception as e:
                # Log the error and continue streaming or stop gracefully
                print(f"An error occurred in streaming: {str(e)}")
                raise ValueError(
                    "An error occurred while making request to streaming endpoint: ", e
                )
                # You can decide to break the loop or log and continue based on your requirements
                # break


class MistralAILLM(LLM, openai.OpenAI):
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        self.kwargs = kwargs
        if base_url:
            self.LLMClient: MistralClient = MistralClient(
                api_key=api_key, max_retries=3, timeout=30
            )
        else:
            # ofr now in both cases we create simple client for mistral ai
            self.LLMClient: MistralClient = MistralClient(
                api_key=api_key, max_retries=3, timeout=30
            )

    def get_kwargs(self):
        return self.kwargs

    def ava_llm_completions(
        self,
        messages: List,
        tools: List,
        is_function_based: bool = False,
        streaming: bool = False,
        logging: bool = False,
    ) -> ChatMessage:
        self.kwargs["messages"] = messages
        # self.kwargs["stream"] = streaming
        # i think mistral ai doesnt accepts streaming param
        if tools != []:
            # self.kwargs["tool_choice"] = "auto"
            if not is_function_based:
                self.kwargs["tool"] = tools
            else:
                self.kwargs["functions"] = tools
        # print("Ava llm kwargs: ", self.kwargs)
        # print("Messages : ", messages)
        # Assuming 'client' defined somewhere before using it
        # model="accounts/fireworks/models/fw-function-call-34b-v0",
        # model="accounts/fireworks/models/firefunction-v1",
        # if not streaming:
        chat_completion = self.LLMClient.chat(**self.kwargs)
        # chat_completion = self.LLMClient.chat.completions.create(**self.kwargs)
        return chat_completion
    

    def ava_llm_streaming_completions(
        self,
        messages: List,
        tools: List,
        is_function_based: bool = False,
        streaming: bool = False,
        logging: bool = False,
    ) -> ChatCompletion:
        self.kwargs["messages"] = messages
        self.kwargs["stream"] = streaming
        if tools != []:
            if not is_function_based:
                self.kwargs["tools"] = tools
            else:
                self.kwargs["functions"] = tools
        # print("Ava llm kwargs: ", self.kwargs)
        # print("Messages : ", messages)
        # Assuming 'client' defined somewhere before using it
        # model="accounts/fireworks/models/fw-function-call-34b-v0",
        # model="accounts/fireworks/models/firefunction-v1",

        async def stream():
            try:
                completion = self.LLMClient.chat.completions.create(**self.kwargs)
                for line in completion:
                    if logging:
                        print("Openai LLM streaming line: ", line, line, color="green")
                    chunk = None
                    if line.choices[0]:
                        chunk = line.choices[0].delta.content
                    if chunk:
                        yield chunk.encode("utf-8") + b"\n"
            except Exception as e:
                # Log the error and continue streaming or stop gracefully
                print(f"An error occurred in streaming: {str(e)}")
                raise ValueError(
                    "An error occurred while making request to streaming endpoint: ", e
                )
                # You can decide to break the loop or log and continue based on your requirements
                # break


class ClaudeLLM(LLM):
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        self.kwargs = kwargs
        if base_url:
            self.LLMClient = anthropic.Anthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key=api_key,
            )
        else:
            self.LLMClient = anthropic.Anthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key=api_key,
            )

    def get_kwargs(self):
        return self.kwargs

    def ava_llm_completions(
        self, messages: List, tools: List, is_function_based: bool = False, **extras
    ) -> Message:
        self.kwargs.update(extras)
        self.kwargs["messages"] = messages
        if tools != []:
            if not is_function_based:
                self.kwargs["tools"] = tools
            else:
                self.kwargs["functions"] = tools
        # print("Ava llm kwargs: ", self.kwargs)
        # print("Messages : ", messages)
        # Assuming you have 'client' defined somewhere before using it
        # model="accounts/fireworks/models/fw-function-call-34b-v0",
        # model="accounts/fireworks/models/firefunction-v1",
        chat_completion = self.LLMClient.messages.create(**self.kwargs)
        return chat_completion


class CallbackHandler:
    def on_agent_run(self, input_msg: str):
        pass

    def on_tool_call(self, tool_name: str, tool_params: Dict):
        pass

    def on_general_response(self, response: str):
        pass


def map_type_to_json(type_info):
    type_mappings = {
        int: "number",
        float: "number",
        str: "string",
        bool: "boolean",
        # Add more type mappings as needed i guess
    }
    return type_mappings.get(type_info, str(type_info))


def convert_tool_to_json(
    tool: BaseTool,
) -> Dict[str, Any]:
    """
    Convert a tool object into a JSON representation.

    Args:
        tool (BaseTool): The tool object.

    Returns:
        Dict[str, Any]: JSON representation of the tool.
    """
    json_representation = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }

    # Add parameters from args_schema to the JSON representation
    required_args = []
    print("tool in convert tool",type(tool) )
    if tool.args_schema:
        print("args schema", tool.args_schema)
        for field_name, field_info in tool.args_schema.__annotations__.items():
            field_description = getattr(
                tool.args_schema.model_fields[field_name],
                "description",
                "No description available",
            )

            field_properties = {
                "type": map_type_to_json(field_info),
                "description": field_description,
            }
            if tool.args_schema.model_fields[field_name].repr:
                required_args.append(str(field_name))
            # Include default value if available
            # print("checking :", tool.args_schema.model_fields[field_name])
            # Include default value if available
            default_value = tool.args_schema.model_fields[field_name].default
            if default_value is not None and default_value is not PydanticUndefined:
                field_properties["default"] = default_value

            # Include enum values if available
            enums_present = tool.args_schema.model_fields[field_name].json_schema_extra

            # enum_values = getattr(tool.args_schema.model_fields[field_name].json_schema_extra, 'enumerate', None)
            if enums_present:
                enum_values = enums_present.get("enumerate", None)
                # print("Enum values: ", enum_values)
                if enum_values is not None:
                    field_properties["enum"] = enum_values

            json_representation["function"]["parameters"]["properties"][field_name] = (
                field_properties
            )
            json_representation["function"]["parameters"]["required"] = required_args

    # print("converted a tool: ", required_args)
    return json_representation


def convert_functions_to_json(
    tool: BaseTool,
) -> Dict[str, Any]:
    """
    Convert a tool object into a JSON representation.

    Args:
        tool (BaseTool): The tool object.

    Returns:
        Dict[str, Any]: JSON representation of the tool.
    """
    json_representation = {
        "name": tool.name,
        "description": tool.description,
        "parameters": {"type": "object", "properties": {}, "required": []},
    }

    # Add parameters from args_schema to the JSON representation
    required_args = []

    if tool.args_schema:
        for field_name, field_info in tool.args_schema.__annotations__.items():
            field_description = getattr(
                tool.args_schema.model_fields[field_name],
                "description",
                "No description available",
            )

            field_properties = {
                "type": map_type_to_json(field_info),
                "description": field_description,
            }
            if tool.args_schema.model_fields[field_name].repr:
                required_args.append(str(field_name))
            # Include default value if available
            # print("checking :", tool.args_schema.model_fields[field_name])
            # Include default value if available
            default_value = tool.args_schema.model_fields[field_name].default
            if default_value is not None and default_value is not PydanticUndefined:
                field_properties["default"] = default_value

            # Include enum values if available
            enums_present = tool.args_schema.model_fields[field_name].json_schema_extra

            # enum_values = getattr(tool.args_schema.model_fields[field_name].json_schema_extra, 'enumerate', None)
            if enums_present:
                enum_values = enums_present.get("enumerate", None)
                # print("Enum values: ", enum_values)
                if enum_values is not None:
                    field_properties["enum"] = enum_values

            json_representation["parameters"]["properties"][field_name] = (
                field_properties
            )
            json_representation["parameters"]["required"] = required_args

    # print("converted a tool: ", required_args)
    return json_representation


def convert_tools_to_json(
    tools: List[BaseTool], is_function_based: bool = False
) -> List[Dict[str, Any]]:
    """
    Convert a list of tool objects into a list of JSON representations.

    Args:
        tools (List[BaseTool]): List of tool objects.

    Returns:
        List[Dict[str, Any]]: List of JSON representations of the tools.
    """
    json_representations = []
    if not is_function_based:
        for tool in tools:
            json_representation = convert_tool_to_json(tool)
            json_representations.append(json_representation)
    else:
        for tool in tools:
            json_representation = convert_functions_to_json(tool)
            json_representations.append(json_representation)

    return json_representations


def extract_function_info(tool_call=None, is_function_based: bool = False):
    # print("extracting from: ", api_response)
    if tool_call:
        print("Extracting info for tool: ", tool_call)
        if not is_function_based:
            # Assuming there is only one tool_call for simplicity
            # tool_call =
            if tool_call.function:
                function_info = tool_call.function
                name = function_info.name
                params = function_info.arguments
                id = tool_call.id
                # if params == "{}":
                #     params = None
                return name, params, id
        else:
            # function_info = tool_call.function
            name = tool_call.name
            params = tool_call.arguments
            id = None
            return name, params, id

    return None, None, None


def find_and_execute_tool(
    tool_name: str,
    tools_list: List[BaseTool],
    tool_params: Optional[Dict[str, str]] = None,
):
    for tool in tools_list:
        if tool.name == tool_name:
            # tool.return_direct
            if tool_params:
                args_model = tool.args_schema(**tool_params)
                return tool._run(**args_model.dict()), tool.return_direct
            else:
                return tool._run(), tool.return_direct
    return None
