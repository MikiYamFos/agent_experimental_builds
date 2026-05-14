from typing import Any, Dict
from dataclasses import dataclass

from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.run import AgentRun
from pydantic_ai.messages import FunctionToolCallEvent
from pydantic_ai._agent_graph import UserPromptNode, ModelRequestNode, CallToolsNode

from jaxn import JSONParserHandler, StreamingJSONParser

from tools import SearchTools
from models import RAGResponse

DEFAULT_INSTRUCTIONS = """
You're a documentation assistant.
Your user is a developer who is using Evidently and your task is to help them with it.

Answer the user question using only the documentation knowledge base.

CRITICAL: The documentation contains 'pip install' commands.
You MUST replace ALL 'pip install' with 'uv add' in your answers.


Scope:
- You only answer questions about Evidently, ML evaluation, LLM evaluation, data quality, data drift, model monitoring, and related documentation topics.
- If the user asks about an unrelated topic, perform one search to verify relevance.
- If the search results are unrelated or weak, do not call get_file.
- For unrelated questions, clearly state that the question is off-topic and that you only provide information about Evidently, ML evaluation, and LLM evaluation.
- For unrelated questions, return confidence <= 0.1.
- For unrelated questions, return no references.
- For unrelated questions, follow-up questions must redirect to Evidently, ML evaluation, or LLM evaluation.

Tool use:
1) First iteration:
   - Perform one search using the search tool to identify potentially relevant documents.

2) Second iteration:
   - Analyze the search results.
   - If the search results are relevant, perform:
       - Up to 2 additional search queries to refine or expand coverage, and
       - One or more get_file calls to retrieve the full content of the most relevant documents.
   - If the search results are not relevant, do not call get_file.

3) Third iteration:
   - If relevant documents were retrieved, synthesize the information into a final answer.
   - If no relevant documents were found, clearly inform the user.

Installation instructions:
- Always use uv commands for package installation.
- Use `uv add evidently`.
- Do not recommend `pip install`.

Code examples:
- Include code snippets when they are useful for answering the user's question.
- Do not include code snippets for off-topic questions.

Adapting to user context:
- Pay close attention to what the user already has: existing data, code, variables, setup, filenames, column names, and data structures.
- Tailor your answer to their specific situation.
- Reference their variable names, column names, and data structures when they provide them.
- Do not show generic from-scratch tutorials when the user has described their existing setup.
- Never create sample or toy DataFrames/example data when the user already has their own data.
- Use the user's existing data directly in all code examples.

Self-verification checklist:
Before submitting, verify each applicable rule and fill in the `checks` field.
If any check fails, fix your response first.
Only include relevant checks.

1. "No verbatim code from docs" — rewrite all code; do not copy code directly from documentation.
2. "uv package manager" — use `uv add`; never recommend Python pip installation commands.
3. "User context used" — use the user's variable names, data, columns, and setup instead of generic examples.
4. "Code formatting" — keyword arguments have no spaces around `=`, and multi-argument calls use hanging indentation.
5. "On-topic follow-ups" — follow-up questions must match the user's specific topic and stay within Evidently, ML evaluation, or LLM evaluation.

IMPORTANT:
- Use only facts found in the documentation knowledge base.
- Do not introduce outside knowledge or assumptions.
- If the answer cannot be found in the retrieved documents, clearly inform the user.

Additional notes:
- The knowledge base is entirely about Evidently, so you do not need to include the word "evidently" in search queries.
- Prefer retrieving and analyzing full documents via get_file before producing the final answer only when the search results are relevant.
""".strip()


@dataclass
class DocumentationAgentConfig:
    model: str = "openai:gpt-4o-mini"
    name: str = "search"
    instructions: str = DEFAULT_INSTRUCTIONS


def create_agent(
    config: DocumentationAgentConfig,
    search_tools: SearchTools,
) -> Agent:
    tools = [search_tools.search, search_tools.get_file]

    search_agent = Agent(
        name=config.name,
        model=config.model,
        instructions=config.instructions,
        tools=tools,
    )

    return search_agent


class NamedCallback:
    def __init__(self, agent):
        self.agent_name = agent.name

    async def print_function_calls(self, ctx, event):
        if hasattr(event, "__aiter__"):
            async for sub in event:
                await self.print_function_calls(ctx, sub)
            return

        if isinstance(event, FunctionToolCallEvent):
            tool_name = event.part.tool_name
            args = event.part.args
            print(f"TOOL CALL ({self.agent_name}): {tool_name}({args})")

    async def __call__(self, ctx, event):
        return await self.print_function_calls(ctx, event)


async def run_agent(
    agent: Agent,
    user_prompt: str,
    message_history=None,
) -> AgentRunResult:
    callback = NamedCallback(agent)

    if message_history is None:
        message_history = []

    result = await agent.run(
        user_prompt,
        event_stream_handler=callback,
        message_history=message_history,
        output_type=RAGResponse,
    )

    return result


class RAGResponseHandler(JSONParserHandler):
    def on_value_chunk(self, path: str, field_name: str, chunk: str) -> None:
        if path == "" and field_name == "answer":
            print(chunk, end="", flush=True)

    def on_field_end(
        self,
        path: str,
        field_name: str,
        value: str,
        parsed_value: Any = None,
    ) -> None:
        if path == "" and field_name == "answer_type":
            print("\nanswer type:", value)

    def on_array_item_end(
        self,
        path: str,
        field_name: str,
        item: Dict[str, Any] = None,
    ) -> None:
        if field_name == "followup_questions":
            print("follow up question:", item)


async def run_agent_stream(
    agent: Agent,
    user_prompt: str,
    message_history=None,
):
    runner = AgentStreamRunner(agent, RAGResponseHandler())
    return await runner.run(user_prompt, message_history)


class AgentStreamRunner:
    def __init__(self, agent: Agent, handler: JSONParserHandler):
        self.agent = agent
        self.handler = handler

    async def run(self, user_prompt: str, message_history=None):
        if message_history is None:
            message_history = []

        async with self.agent.iter(
            user_prompt,
            message_history=message_history,
            output_type=RAGResponse,
        ) as agent_run:
            async for node in agent_run:
                if Agent.is_user_prompt_node(node):
                    await self.process_user_node(node, agent_run)
                elif Agent.is_model_request_node(node):
                    await self.process_model_request_node(node, agent_run)
                elif Agent.is_call_tools_node(node):
                    await self.process_call_tools_node(node, agent_run)

            return agent_run.result

    async def process_user_node(
        self,
        node: UserPromptNode,
        agent_run: AgentRun,
    ):
        print(f"USER PROMPT ({self.agent.name}): {node.user_prompt}")

    async def process_model_request_node(
        self,
        node: ModelRequestNode,
        agent_run: AgentRun,
    ):
        args_so_far = ""
        parser = StreamingJSONParser(self.handler)

        async with node.stream(agent_run.ctx) as stream:
            async for response in stream.stream_responses():
                for part in response.parts:
                    if part.part_kind != "tool-call":
                        continue

                    if part.tool_name != "final_result":
                        continue

                    args_new = part.args
                    args_new_chunk = args_new[len(args_so_far) :]
                    args_so_far = args_new

                    parser.parse_incremental(args_new_chunk)

    async def process_call_tools_node(
        self,
        node: CallToolsNode,
        agent_run: AgentRun,
    ):
        async with node.stream(agent_run.ctx) as events:
            async for event in events:
                if not isinstance(event, FunctionToolCallEvent):
                    continue

                tool_name = event.part.tool_name
                args = event.part.args
                print(f"TOOL CALL ({self.agent.name}): {tool_name}({args})")
