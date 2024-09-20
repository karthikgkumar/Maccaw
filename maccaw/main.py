from pydantic_core import PydanticUndefined
from datetime import datetime
import json
from typing import Dict, List, Optional, Callable

from maccaw import (
    BaseTool,
    ClaudeLLM,
    OpenaiLLM,
    MistralAILLM,
    LLM,
    convert_tools_to_json,
    extract_function_info,
    find_and_execute_tool,
    CallbackHandler,
)
import time
from print_color import print
from traceback import print_exc
from openai.types.chat import ChatCompletion
from mistralai.models.chat_completion import ChatMessage as MistralChatMessage


import requests
from io import BytesIO
import os

from maccaw import map_type_to_json


def get_current_timestamp():
    # Get the current timestamp with day of week
    current_timestamp = datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")

    return current_timestamp
class MaccawAgent:
    def __init__(
        self,
        sys_prompt: str,
        maccaw_llm: LLM,
        tools_list: Optional[List[BaseTool]] = [],
        pickup_mes_count: int = 4,
        logging: bool = False,
        use_system_prompt_as_context: bool = False,
        is_function_based: bool = False,
        max_agent_iterations: int = 4,
        throw_error_on_iteration_exceed: bool = False,
        callback_handler: CallbackHandler = None,
        agent_name_identifier: str = "agent",
        deeper_logs: bool = False,
        streaming: bool = False,
        include_message_timestap: bool = True,
        pikcup_mes_count_in_sys_history: int = 3,
        tts_streaming: bool = False,
    ):
        self.include_message_timestap = include_message_timestap
        self.use_system_prompt_as_context = use_system_prompt_as_context
        self.is_function_based = is_function_based
        self.sys_prompt_original = sys_prompt
        self.sys_prompt: str = sys_prompt
        self.messages: List = []
        self.tools_list: List = tools_list
        self.ava_llm: LLM = maccaw_llm
        self.streaming: bool = streaming
        self.pickup_mes_count: int = pickup_mes_count
        self.pikcup_mes_count_in_sys_history = pikcup_mes_count_in_sys_history
        self.isOpenaiLLM = isinstance(self.ava_llm, OpenaiLLM)
        if use_system_prompt_as_context:
            self.pickup_mes_count = pikcup_mes_count_in_sys_history
            self.system_prompt_contexts_history = """"""
            self.generate_system_prompt_with_context(
                self.system_prompt_contexts_history
            )
        self.agent_name_identifier = agent_name_identifier
        self.current_user_msg: str = None

        self.appendToMessages(role="system", content=sys_prompt)

        self.converted_tools_list: List[Dict] = convert_tools_to_json(
            tools=tools_list, is_function_based=self.is_function_based
        )
        self.logging: bool = logging
        self.deeper_logs = deeper_logs
        self.callback_handler: CallbackHandler = callback_handler
        self.throw_error_on_iteration_exceed: bool = throw_error_on_iteration_exceed
        self.max_agent_iterations: int = max_agent_iterations
        self.current_agent_iteration: int = 0
        print(f"{agent_name_identifier} TOOlS:")
        if self.logging and self.deeper_logs:
            for json_representation in self.converted_tools_list:
                print(json.dumps(json_representation, indent=2))

    def run(self, msg: str = None, actual_mes: str = None):
        """Runs the agent via main executor"""
        if msg:
            try:
                print(
                    f"\nRunning {self.agent_name_identifier} ... with input: '{msg}'",
                    "\n",
                    color="purple",
                )
                if not actual_mes:
                    self.appendToMessages(role="user", content=msg)
                else:
                    self.appendToMessages(role="user", content=actual_mes)

                if self.use_system_prompt_as_context:
                    if not self.include_message_timestap:
                        self.system_prompt_contexts_history += f"\nUSER: {msg}"
                    else:
                        self.system_prompt_contexts_history += (
                            f"\nUSER(SystemNote:{get_current_timestamp()}): {msg}"
                        )

                self.current_user_msg = msg
                # if not self.use_system_prompt_as_context:
                self.messages = self.trim_list(
                    input_list=self.messages, count=self.pickup_mes_count
                )
                return self.maccaw_main_executor(
                    messages=self.messages,
                    tools_list=self.tools_list,
                    ava_llm=self.ava_llm,
                )
            except ValueError as e:
                if self.logging:
                    print_exc()
                raise ValueError(f"Error in agent run : ", e)
        else:
            raise ValueError(
                f"Input to the agent to run was None or blank, please check it!!"
            )

    def generate_system_prompt_with_context(self, context_string=None):
        if context_string:
            self.sys_prompt = f"""{self.sys_prompt_original}\n
            Please take into account the preceding and ongoing dialogues between you and the user for context, and utilize this information to inform your subsequent responses."
            Below are the prior and current converstations between you and the user (with message SystemNote). Use it as context and information in further conversations with the User:
            {context_string}
            """
        else:
            self.sys_prompt = self.sys_prompt_original


    def appendToMessages(self, role: str, content: str):
        # if not to_add_system_context_history:
        print("appending sys : Currently LLM is openai? ", self.isOpenaiLLM)
        if role == "user" and self.include_message_timestap:
            content = f"(SystemNote:{get_current_timestamp()}) " + content

        if self.isOpenaiLLM:
            if self.ava_llm.kwargs.get("model", None) != "gpt-4o":
                self.messages.append({"role": role, "content": content})
            else:
                self.messages.append(
                    {"role": role, "content": [{"type": "text", "text": content}]}
                )

        elif isinstance(self.ava_llm, MistralAILLM):
            # if not role == "system":
            self.messages.append(MistralChatMessage(role=role, content=content))
        elif isinstance(self.ava_llm, ClaudeLLM):
            if not role == "system":
                self.messages.append({"role": role, "content": content})

       

    def generateSysMessageForLLM(self, content: str):
        # if not to_add_system_context_history:
        print("Currently LLM is openai? ", self.isOpenaiLLM)
        if self.isOpenaiLLM:
            return {"role": "system", "content": content}
        elif isinstance(self.ava_llm, MistralAILLM):
            # if not role == "system":
            return MistralChatMessage(role="system", content=content)

    def refreshSysMessage(self, content: str = None):
        self.messages[0] = self.generateSysMessageForLLM(content=self.sys_prompt)

    def updateSysMessage(self, content: str = None):
        if not content:
            self.messages[0] = self.generateSysMessageForLLM(content=self.sys_prompt)
        else:
            self.sys_prompt = content
            self.sys_prompt_original = content
            self.messages[0] = self.generateSysMessageForLLM(content=content)



    def getSystemMessage(
        self,
    ):
        return self.sys_prompt

    def clearMessageHistory(self, del_systen_chat_history: bool = False):
        if self.logging:
            print(f"Deleting chat history..🗑️ for {self.agent_name_identifier}")
        self.messages.clear()
        if self.use_system_prompt_as_context and del_systen_chat_history:
            self.system_prompt_contexts_history = ""
            self.generate_system_prompt_with_context(context_string=None)
        self.appendToMessages(role="system", content=self.sys_prompt)
        if self.logging and self.deeper_logs:
            print(
                "Agent chat history cleared 🧹: ", self.messages, "\n", color="magenta"
            )

    def trim_list(self, input_list: list, count: int):
        """
        Trim the list to the last 'count' items if the length is greater than 'count'.

        Parameters:
        - input_list (list): The input list to be trimmed.
        - count (int): The desired count of items in the final list.

        Returns:
        - list: Trimmed list.
        """
        print("trimmnig list")
        if len(input_list) < count:
            return input_list
        else:
            if not self.use_system_prompt_as_context:
                input_list = input_list[-count:]
                input_list[0] = self.generateSysMessageForLLM(content=self.sys_prompt)
            else:
                input_list = input_list[-count:]
                input_list[0] = self.generateSysMessageForLLM(content=self.sys_prompt)
            return input_list

    def maccaw_main_executor(
        self, messages: List[dict], tools_list: List[BaseTool], ava_llm: LLM
    ):
        """
        tools_list: is actual list of tools with tools as class ex: [mytool(),mytool2()]

        converted_tools_list: is the converted list of to json with schema

        messages: the list of user-assistant messages pairs
        """

        if self.current_agent_iteration <= self.max_agent_iterations:
            self.current_agent_iteration += 1

            if self.logging:
                print("Messages List: ", self.messages, "\n", color="purple")
            if self.callback_handler and hasattr(self.callback_handler, "on_agent_run"):
                self.callback_handler.on_agent_run(input_msg="agent started running!")
            llm_resp = None
            if self.messages and self.ava_llm:
                if self.isOpenaiLLM or isinstance(self.ava_llm, MistralAILLM):
                    return self.handle_openai_llm_completions(
                        messages=messages, tools_list=tools_list, ava_llm=ava_llm
                    )
                elif isinstance(ava_llm, ClaudeLLM):
                    return self.handle_claude_llm_completions(
                        messages=messages, tools_list=tools_list, ava_llm=ava_llm
                    )
            else:
                self.current_agent_iteration = 0

                raise ValueError(
                    f"Error: Check the passed message: {messages}, tools: {tools_list}, and ava llm: {ava_llm}"
                )
        else:
            self.current_agent_iteration = 0
            if not self.throw_error_on_iteration_exceed:
                return (
                    "Sorry! I wasn't able to complete you query after several tries!!."
                )
            raise ValueError(
                f"{self.agent_name_identifier} wasnt' able to come to the conclusion and was exceeding the max agent iteration count of {self.max_agent_iterations}"
            )

    def handle_claude_llm_completions(
        self, messages: List[dict], tools_list: List[BaseTool], ava_llm: ClaudeLLM
    ):
        llm_resp = ava_llm.ava_llm_completions(
            self.messages,
            self.converted_tools_list,
            is_function_based=self.is_function_based,
            system=self.sys_prompt,
        )
        if self.logging:
            print("OPENAI LLM RESP:", llm_resp, "\n", color="green")
        agent_response = llm_resp.content[0].text
        agent_mes = llm_resp.content[0].text
        self.appendToMessages(role="assistant", content=agent_mes)
        if self.use_system_prompt_as_context:
            if not self.include_message_timestap:
                self.system_prompt_contexts_history += f"\nYOU:{agent_mes}"
            else:
                self.system_prompt_contexts_history += (
                    f"\nYOU(SystemNote:{get_current_timestamp()}):{agent_mes}"
                )
            self.generate_system_prompt_with_context(
                context_string=self.system_prompt_contexts_history
            )
            self.updateSysMessage()
        if self.logging:
            print(
                f"{self.agent_name_identifier.capitalize()} message: ",
                agent_mes,
                color="yellow",
            )
            print()
        if self.callback_handler and hasattr(
            self.callback_handler, "on_general_response"
        ):
            self.callback_handler.on_general_response(response=agent_mes)
        self.current_agent_iteration = 0
        return agent_mes

    def handle_openai_llm_completions(
        self, messages: List[dict], tools_list: List[BaseTool], ava_llm: OpenaiLLM
    ):
        llm_resp = None
        if not self.streaming:
            t0 = time.time()
            llm_resp = ava_llm.ava_llm_completions(
                self.messages,
                self.converted_tools_list,
                is_function_based=self.is_function_based,
                streaming=self.streaming,
            )
            if self.logging:
                t1 = time.time() - t0
                print(
                    "AI responded in : {:.2f} milliseconds".format(t1 * 1000),
                    color="blue",
                )
                print("OPENAI LLM RESP:", llm_resp, "\n", color="green")
            return self.complete_normal_openai_llm_response(
                # agent_response=agent_response,
                llm_resp=llm_resp,
                messages=messages,
                tools_list=tools_list,
                ava_llm=ava_llm,
            )

        else:
            # HERE WE ARE HANDLING STREAMING RESPONSES
            t0 = time.time()
            llm_resp = ava_llm.ava_llm_completions(
                self.messages,
                self.converted_tools_list,
                is_function_based=self.is_function_based,
                streaming=self.streaming,
            )
            if self.logging:
                t1 = time.time() - t0
                print("AI responded in : {:.2f} milliseconds".format(t1 * 1000))
                print("OPENAI STREAMING LLM RESP:", llm_resp, "\n", color="green")
            return self.complete_streaming_openai_llm_response(
                # agent_response=agent_response,
                llm_resp=llm_resp,
                messages=messages,
                tools_list=tools_list,
                ava_llm=ava_llm,
            )
        
    def handle_openai_normal_llm_completions(
        self, messages: List[dict], tools_list: List[BaseTool], ava_llm: OpenaiLLM
    ):
        llm_resp = None
        if not self.streaming:
            t0 = time.time()
            llm_resp = ava_llm.ava_chain_llm_completions(
                self.messages,
                self.converted_tools_list,
                is_function_based=self.is_function_based,
                streaming=self.streaming,
            )
            if self.logging:
                t1 = time.time() - t0
                print(
                    "AI responded in : {:.2f} milliseconds".format(t1 * 1000),
                    color="blue",
                )
                print("OPENAI LLM RESP:", llm_resp, "\n", color="green")
            return self.complete_normal_openai_llm_response(
                # agent_response=agent_response,
                llm_resp=llm_resp,
                messages=messages,
                tools_list=tools_list,
                ava_llm=ava_llm,
            )

        else:
            # HERE WE ARE HANDLING STREAMING RESPONSES
            t0 = time.time()
            print("function_based", self.is_function_based)
            llm_resp = ava_llm.ava_chain_llm_completions(
                self.messages,
                self.converted_tools_list,
                is_function_based=self.is_function_based,
                streaming=self.streaming,
            )
            print("llm response in handle_openai_llm", llm_resp)
            if self.logging:
                t1 = time.time() - t0
                print("AI responded in : {:.2f} milliseconds".format(t1 * 1000))
                print("OPENAI STREAMING LLM RESP:", llm_resp, "\n", color="green")
            return self.complete_streaming_openai_llm_response(
                # agent_response=agent_response,
                llm_resp=llm_resp,
                messages=messages,
                tools_list=tools_list,
                ava_llm=ava_llm,
            )



    def complete_normal_openai_llm_response(
        self,
        llm_resp,
        messages: List[dict],
        tools_list: List[BaseTool],
        ava_llm: OpenaiLLM,
    ):
        agent_response = llm_resp.choices[0].message
        if not self.is_function_based:
            if llm_resp.choices[0].message.tool_calls:
                """THIS MEANS AGENT MADE TOOL CALL"""

                if self.isOpenaiLLM:
                    messages.append(
                        # {
                        #     "role": agent_response.role,
                        #     "content": "",
                        #     "tool_calls": [
                        #         tool_call.model_dump()
                        #         for tool_call in llm_resp.choices[0].message.tool_calls
                        #     ],
                        # }
                        agent_response
                    )
                for tool_call in llm_resp.choices[0].message.tool_calls:
                    name, params, tool_id = extract_function_info(
                        tool_call=tool_call, is_function_based=self.is_function_based
                    )
                    # print(f"Name: {name}")
                    # print(f"Params: {params}")

                    if name:
                        if self.logging:
                            print(
                                f"Executing tool '{name}' ... with param(s): ",
                                f"'{params}'",
                                "\n",
                                color="yellow",
                            )
                        if self.callback_handler and hasattr(
                            self.callback_handler, "on_tool_call"
                        ):
                            self.callback_handler.on_tool_call(
                                tool_name=name, tool_params=params
                            )

                        resp, is_direct = find_and_execute_tool(
                            tool_name=name,
                            tool_params=json.loads(params),
                            tools_list=tools_list,
                        )

                        if resp:
                            if self.logging:
                                print(
                                    f"Tool '{name}' response: ",
                                    resp,
                                    "\n",
                                    color="blue",
                                )

                            # messages.append({"role": "function", "name": name, "content": resp})
                            if isinstance(self.ava_llm, MistralAILLM):
                                messages.append(
                                    MistralChatMessage(
                                        role="tool", name=name, content=resp
                                    )
                                )
                            elif self.isOpenaiLLM:
                                messages.append(
                                    {
                                        # "role": "tool",
                                        # "content": resp
                                        "tool_call_id": tool_id,
                                        "role": "tool",
                                        "name": name,
                                        "content": resp,
                                    }
                                )
                            if self.use_system_prompt_as_context:
                                self.system_prompt_contexts_history += (
                                    f"\nTOOL({name}): {resp}"
                                )
                            # if
                            # Passin updated values in the recursive call
                            if is_direct:
                                if self.logging:
                                    print(
                                        "Returning tool response as direct message",
                                        "\n",
                                        color="magenta",
                                    )
                                    print(
                                        f"{self.agent_name_identifier.capitalize()} message: ",
                                        resp,
                                        color="yellow",
                                    )
                                    print()

                                self.appendToMessages(role="assistant", content=resp)
                                if self.use_system_prompt_as_context:
                                    if not self.include_message_timestap:
                                        self.system_prompt_contexts_history += (
                                            f"\nYOU:{resp}"
                                        )
                                    else:
                                        self.system_prompt_contexts_history += f"\nYOU(SystemNote:{get_current_timestamp()}):{resp}"
                                    self.generate_system_prompt_with_context(
                                        context_string=self.system_prompt_contexts_history
                                    )
                                    self.updateSysMessage()
                                return resp

                return self.maccaw_main_executor(
                    messages=messages, tools_list=tools_list, ava_llm=ava_llm
                )

            else:
                """THIS MEANS THE AGENT JUST REPLIED NORMALLY WITHOUT FUNCTION OR TOOL CALLING"""
                agent_mes = llm_resp.choices[0].message.content
                self.appendToMessages(role="assistant", content=agent_mes)
                if self.use_system_prompt_as_context:
                    if not self.include_message_timestap:
                        self.system_prompt_contexts_history += f"\nYOU:{agent_mes}"
                    else:
                        self.system_prompt_contexts_history += (
                            f"\nYOU(SystemNote:{get_current_timestamp()}):{agent_mes}"
                        )
                    self.generate_system_prompt_with_context(
                        context_string=self.system_prompt_contexts_history
                    )
                    self.updateSysMessage()
                if self.logging:
                    print(
                        f"{self.agent_name_identifier.capitalize()} message: ",
                        agent_mes,
                        color="yellow",
                    )
                    print()
                if self.callback_handler and hasattr(
                    self.callback_handler, "on_general_response"
                ):
                    self.callback_handler.on_general_response(response=agent_mes)
                self.current_agent_iteration = 0
                return agent_mes

        else:
            if llm_resp.choices[0].message.function_call:
                """THIS MEANS AGENT MADE FUNCTION CALL"""
                for function_call in llm_resp.choices[0].message.function_call:
                    name, params, tool_id = extract_function_info(
                        tool_call=llm_resp.choices[0].message.function_call,
                        is_function_based=self.is_function_based,
                    )
                    # print(f"Name: {name}")
                    # print(f"Params: {params}")

                    if name:
                        if self.logging:
                            print(
                                f"Executing function '{name}' ... with param(s): ",
                                f"'{params}'",
                                "\n",
                                color="yellow",
                            )
                        if self.callback_handler and hasattr(
                            self.callback_handler, "on_tool_call"
                        ):
                            self.callback_handler.on_tool_call(
                                tool_name=name, tool_params=params
                            )

                        resp, is_direct = find_and_execute_tool(
                            tool_name=name,
                            tool_params=json.loads(params),
                            tools_list=tools_list,
                        )

                        if resp:
                            if self.logging:
                                print(
                                    f"Returned From Function '{name}' response: ",
                                    resp,
                                    "\n",
                                    color="blue",
                                )

                            # messages.append({"role": "function", "name": name, "content": resp})
                            messages.append(
                                {
                                    # "role": "tool",
                                    # "content": resp
                                    # "tool_call_id": tool_call.id,
                                    "role": "function",
                                    "name": name,
                                    "content": resp,
                                }
                            )
                            if self.use_system_prompt_as_context:
                                self.system_prompt_contexts_history += (
                                    f"\nFUNCTION({name}): {resp}"
                                )
                            # if
                            # Passin updated values in the recursive call
                            if is_direct:
                                print(
                                    "Returning tool response as direct message",
                                    "\n",
                                    color="magenta",
                                )
                                self.appendToMessages(role="assistant", content=resp)
                                if self.use_system_prompt_as_context:
                                    if not self.include_message_timestap:
                                        self.system_prompt_contexts_history += (
                                            f"\nYOU:{resp}"
                                        )
                                    else:
                                        self.system_prompt_contexts_history += f"\nYOU(SystemNote:{get_current_timestamp()}):{resp}"
                                    self.generate_system_prompt_with_context(
                                        context_string=self.system_prompt_contexts_history
                                    )
                                    self.updateSysMessage()
                                return resp
                return self.maccaw_main_executor(
                    messages=messages, tools_list=tools_list, ava_llm=ava_llm
                )

            else:
                """THIS MEANS THE AGENT JUST REPLIED NORMALLY WITHOUT FUNCTION OR TOOL CALLING"""

                agent_mes = llm_resp.choices[0].message.content
                self.appendToMessages(role="assistant", content=agent_mes)
                if self.use_system_prompt_as_context:
                    if not self.include_message_timestap:
                        self.system_prompt_contexts_history += f"\nYOU:{agent_mes}"
                    else:
                        self.system_prompt_contexts_history += (
                            f"\nYOU(SystemNote:{get_current_timestamp()}):{agent_mes}"
                        )
                    self.generate_system_prompt_with_context(
                        context_string=self.system_prompt_contexts_history
                    )
                    self.updateSysMessage()

                if self.logging:
                    print(
                        f"{self.agent_name_identifier.capitalize()} message: ",
                        agent_mes,
                        color="yellow",
                    )
                    print()
                if self.callback_handler and hasattr(
                    self.callback_handler, "on_general_response"
                ):
                    self.callback_handler.on_general_response(response=agent_mes)
                self.current_agent_iteration = 0
                return agent_mes
            
   
    def complete_streaming_openai_llm_response(
        self,
        llm_resp,
        messages: List[dict],
        tools_list: List[BaseTool],
        ava_llm: OpenaiLLM,
    ):
        response_text: str = ""
        previous_chunk = ""
        stop_reason = None
        function_chunk = None
        tts_chunk: str = ""
        function_response = {
            "name": "",
            "arguments": "",
            "id": "",
            "tool_call_model_dump": {},
        }
        function_calls_list = []
        speech = True
        is_function_call = False

        for line in llm_resp:
            # if self.logging and self.deeper_logs:
            #     print("Openai LLM streaming line: ", line, line, color="green")
            #     print()
            chunk = None
            if line.choices[0].delta:
                chunk = line.choices[0].delta.content
                # tts_chunk = chunk
                function_chunk = line.choices[0].delta.tool_calls
            if function_chunk:
                is_function_call = True
                # print("TOOL CALLED!", function_chunk)
                for tool_call in function_chunk:
                    function_info = tool_call.function
                    if function_info.name:
                        function_response["name"] += function_info.name
                    if function_info.arguments:
                        function_response["arguments"] += function_info.arguments
                    if tool_call.id:
                        function_response["id"] += tool_call.id
                    if tool_call.model_dump():
                        function_response["tool_call_model_dump"].update(
                            tool_call.model_dump()
                        )
                        function_response["tool_call_model_dump"]["function"] = {
                            "arguments": function_response["name"],
                            "name": function_response["arguments"],
                        }
                    # function_response+=tool_call.

                print(
                    f"{self.agent_name_identifier.capitalize()} Tool call : ",
                    function_response,
                    color="yellow",
                    end="\r",
                    flush=True,
                )

            # elif line
            elif chunk and chunk != previous_chunk and function_chunk is None:
                is_function_call = False
                # print("CHUNK:", chunk)
                # yield chunk.encode('utf-8') + b'\n'
                response_text += chunk
                tts_chunk += chunk
                # Debugging statements
                # print("Received chunk:", chunk)
                # print("Updated response_text:", response_text)
                # Clear the previous line and print the updated text
                print(
                    f"{self.agent_name_identifier.capitalize()} message: ",
                    response_text,
                    color="yellow",
                    end="\r",
                    flush=True,
                )
                previous_chunk = chunk
                # Check if the chunk ends with a sentence-ending punctuation
                if tts_chunk.strip()[-1] in {".", "!", "?"}:
                    if self.tts_streaming == True:
                        # text_to_speech(chunk)
                        # play_tts_audio(
                        #     text=tts_chunk,
                        #     speaker="eva",
                        #     sampling_rate=10000,
                        #     speed_alpha=1.0,
                        #     reduce_latency=True,
                        #     authorization_token="M1jSmw004ffRB352M20OE1jGzncs0ualtgiIwXs_5nY"
                        # )
                        

                        tts_chunk = ""
            if line.choices[0].finish_reason:
                print("", end="\n", flush=True)
                stop_reason = line.choices[0].finish_reason
                if self.logging:
                    print(
                        "Steams ends with reason: ",
                        stop_reason,
                        "\nFUNCTION CALL: ",
                        is_function_call,
                    )
                break
        # Print a newline after the streaming is finished
        print("", end="\n", flush=True)
        if not is_function_call:
            agent_mes = response_text
            self.appendToMessages(role="assistant", content=agent_mes)
            if self.use_system_prompt_as_context:
                if not self.include_message_timestap:
                    self.system_prompt_contexts_history += f"\nYOU:{agent_mes}"
                else:
                    self.system_prompt_contexts_history += (
                        f"\nYOU(SystemNote:{get_current_timestamp()}):{agent_mes}"
                    )
                self.generate_system_prompt_with_context(
                    context_string=self.system_prompt_contexts_history
                )
                self.updateSysMessage()

            # if self.logging:
            #     print(f"{self.agent_name_identifier.capitalize()} message: ",
            #           agent_mes, color='yellow')
            #     print()
            if self.callback_handler and hasattr(
                self.callback_handler, "on_general_response"
            ):
                self.callback_handler.on_general_response(response=agent_mes)
            self.current_agent_iteration = 0
            return agent_mes

        else:
            """This means there is streaming function or say tool call"""

            if function_response:
                name = function_response["name"]
                params = function_response["arguments"]
                id = function_response["id"]
                tool_call_dump = function_response["tool_call_model_dump"]

                messages.append(
                    {"role": "assistant", "content": "", "tool_calls": [tool_call_dump]}
                )
                if name:
                    if self.logging:
                        print(
                            f"Executing tool '{name}' ... with param(s): ",
                            f"'{params}'",
                            "\n",
                            color="yellow",
                        )
                    if self.callback_handler and hasattr(
                        self.callback_handler, "on_tool_call"
                    ):
                        self.callback_handler.on_tool_call(
                            tool_name=name, tool_params=params
                        )

                    resp, is_direct = find_and_execute_tool(
                        tool_name=name,
                        tool_params=json.loads(params),
                        tools_list=tools_list,
                    )

                    if resp:
                        if self.logging:
                            print(f"Tool '{name}' response: ", resp, "\n", color="blue")

                        # messages.append({"role": "function", "name": name, "content": resp})
                        if isinstance(self.ava_llm, MistralAILLM):
                            messages.append(
                                MistralChatMessage(role="tool", name=name, content=resp)
                            )
                        elif self.isOpenaiLLM:
                            messages.append(
                                {
                                    # "role": "tool",
                                    # "content": resp
                                    "tool_call_id": id,
                                    "role": "tool",
                                    "name": name,
                                    "content": resp,
                                }
                            )
                        if self.use_system_prompt_as_context:
                            self.system_prompt_contexts_history += (
                                f"\nTOOL({name}): {resp}"
                            )
                        # if
                        # Passin updated values in the recursive call
                        if is_direct:
                            if self.logging:
                                print(
                                    "Returning tool response as direct message",
                                    "\n",
                                    color="magenta",
                                )
                                print(
                                    f"{self.agent_name_identifier.capitalize()} message: ",
                                    resp,
                                    color="yellow",
                                )
                                print()

                            self.appendToMessages(role="assistant", content=resp)
                            if self.use_system_prompt_as_context:
                                if not self.include_message_timestap:
                                    self.system_prompt_contexts_history += (
                                        f"\nYOU:{resp}"
                                    )
                                else:
                                    self.system_prompt_contexts_history += f"\nYOU(SystemNote:{get_current_timestamp()}):{resp}"
                                self.generate_system_prompt_with_context(
                                    context_string=self.system_prompt_contexts_history
                                )
                                self.updateSysMessage()
                            return resp
            return self.maccaw_main_executor(
                messages=messages, tools_list=tools_list, ava_llm=ava_llm
            )

    def prepare_conversation_history_summary(
        self,
    ):
        """This function is for preparing and conversation summary from the prior messages"""

        pass



    