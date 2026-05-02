import json
from typing import Any

from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel

from llm_client import Provider


class Agent:

    FAKE_TOOL_NAME = "structure_result"

    search_tool_openai = {
        "type": "function",
        "name": "search",
        "description": "Search for pages related to a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            },
            "required": ["query"],
        },
    }

    get_file_tool_openai = {
        "type": "function",
        "name": "get_file",
        "description": "Fetch the full content of a page by its title.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The page title to fetch",
                },
            },
            "required": ["filename"],
        },
    }

    search_tool_anthropic = {
        "name": "search",
        "description": "Search for pages related to a topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            },
            "required": ["query"],
        },
    }

    get_file_tool_anthropic = {
        "name": "get_file",
        "description": "Fetch the full content of a page by its title.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The page title to fetch",
                },
            },
            "required": ["filename"],
        },
    }

    def __init__(
        self,
        llm_client: OpenAI | Anthropic,
        provider: Provider,
        model: str,
        instructions: str,
        tools_impl: Any,
        output_type: type[BaseModel] | None = None,
    ):
        self.llm_client = llm_client
        self.provider = provider
        self.model = model
        self.instructions = instructions
        self.tools_impl = tools_impl
        self.output_type = output_type

        if provider == "openai":
            self.tools = [self.search_tool_openai, self.get_file_tool_openai]
            if output_type is not None:
                self.tools.append(self._make_fake_tool_openai(output_type))
        elif provider == "anthropic":
            self.tools = [self.search_tool_anthropic, self.get_file_tool_anthropic]
            if output_type is not None:
                self.tools.append(self._make_fake_tool_anthropic(output_type))

    # --- Fake tool (structured output) ---

    def _make_fake_tool_openai(self, output_type: type[BaseModel]) -> dict:
        schema = output_type.model_json_schema()
        schema["type"] = "object"
        schema["additionalProperties"] = False
        return {
            "type": "function",
            "name": self.FAKE_TOOL_NAME,
            "description": "Call when ready to return the final structured result.",
            "strict": True,
            "parameters": schema,
        }

    def _make_fake_tool_anthropic(self, output_type: type[BaseModel]) -> dict:
        schema = output_type.model_json_schema()
        return {
            "name": self.FAKE_TOOL_NAME,
            "description": "Call when ready to return the final structured result.",
            "input_schema": schema,
        }

    # --- Tool dispatch ---

    def _dispatch(self, name: str, arguments: dict) -> Any:
        if name == "search":
            return self.tools_impl.search(**arguments)
        if name == "get_file":
            return self.tools_impl.get_file(**arguments)
        raise ValueError(f'Unknown tool "{name}"')

    # --- OpenAI agentic loop ---

    def _loop_openai(self, message_history: list) -> tuple[list, Any]:
        iteration = 0
        while True:
            response = self.llm_client.responses.create(
                model=self.model,
                input=message_history,
                tools=self.tools,
            )
            print(f"iteration {iteration}...")

            has_tool_calls = False
            structured_output = None

            for message in response.output:
                if message.type == "function_call":
                    print(f"  calling {message.name}({message.arguments})")

                    if message.name == self.FAKE_TOOL_NAME:
                        structured_output = self.output_type.model_validate(
                            json.loads(message.arguments)
                        )
                        continue

                    message_history.append(message)
                    result = self._dispatch(message.name, json.loads(message.arguments))
                    message_history.append(
                        {
                            "type": "function_call_output",
                            "call_id": message.call_id,
                            "output": json.dumps(result),
                        }
                    )
                    has_tool_calls = True

                elif message.type == "message":
                    text = message.content[0].text
                    print("ASSISTANT:", text)
                    message_history.append(message)

            iteration += 1

            if structured_output is not None:
                message_history.append(
                    {"role": "assistant", "content": structured_output.answer}
                )
                return message_history, structured_output

            if not has_tool_calls:
                return message_history, None

    # --- Anthropic agentic loop ---

    def _loop_anthropic(self, message_history: list) -> tuple[list, Any]:
        iteration = 0
        while True:
            response = self.llm_client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.instructions,
                messages=message_history,
                tools=self.tools,
            )
            print(f"iteration {iteration}...")

            has_tool_calls = False
            structured_output = None
            tool_results = []

            for block in response.content:
                if block.type == "text":
                    print("ASSISTANT:", block.text)

                elif block.type == "tool_use":
                    print(f"  calling {block.name}({block.input})")

                    if block.name == self.FAKE_TOOL_NAME:
                        structured_output = self.output_type.model_validate(block.input)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": "done",
                            }
                        )
                        continue

                    result = self._dispatch(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    )
                    has_tool_calls = True

            # Append assistant turn then tool results as one user turn
            message_history.append({"role": "assistant", "content": response.content})
            if tool_results:
                message_history.append({"role": "user", "content": tool_results})

            iteration += 1

            if structured_output is not None:
                return message_history, structured_output

            if not has_tool_calls:
                return message_history, None

    # --- Public interface ---

    def loop(
        self, user_prompt: str, message_history: list | None = None
    ) -> tuple[list, Any]:
        if message_history is None:
            message_history = []
            if self.provider == "openai":
                message_history.append({"role": "system", "content": self.instructions})

        message_history.append({"role": "user", "content": user_prompt})

        if self.provider == "openai":
            return self._loop_openai(message_history)
        if self.provider == "anthropic":
            return self._loop_anthropic(message_history)

        raise ValueError(f"Unsupported provider: {self.provider}")

    def run_single_turn(self, user_prompt: str) -> Any:
        _, structured_output = self.loop(user_prompt)
        return structured_output

    def qna(self):
        message_history = []
        if self.provider == "openai":
            message_history.append({"role": "system", "content": self.instructions})

        while True:
            user_prompt = input("You: ").strip()
            if user_prompt.lower() in {"stop", "exit", "quit"}:
                break
            if not user_prompt:
                continue

            message_history, structured_output = self.loop(user_prompt, message_history)
            if structured_output is not None:
                print("STRUCTURED OUTPUT:", structured_output)
