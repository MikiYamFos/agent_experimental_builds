import asyncio
import sys

import dotenv

from doc_agent import (
    DocumentationAgentConfig,
    create_agent,
    run_agent_stream,
)

from tools import create_documentation_tools_cached

dotenv.load_dotenv()


def print_messages(messages):
    for m in messages:
        print(m.kind)
        for p in m.parts:
            part_kind = p.part_kind

            if part_kind == "user-prompt":
                print("  USER:", p.content)
            if part_kind == "tool-call":
                print("  TOOL CALL:", p.tool_name, p.args)
            if part_kind == "tool-return":
                print("  TOOL RETURN:", p.tool_name)
            if part_kind == "text":
                print("  ASSISTANT:", p.content)

        print()


async def run_agent_question(user_prompt: str):
    print(f"Running agent with question: {user_prompt}...")

    tools = create_documentation_tools_cached()
    agent_config = DocumentationAgentConfig()
    agent = create_agent(agent_config, tools)

    result = await run_agent_stream(agent, user_prompt)
    print_messages(result.new_messages())
    print(result.output)


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "LLM as a judge"
    asyncio.run(run_agent_question(prompt))
