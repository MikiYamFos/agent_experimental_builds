from pydantic_ai import Agent
#from pydantic_ai.mcp import MCPServerStdio
from toyaikit.chat.interface import StdOutputInterface
from toyaikit.chat.runners import PydanticAIRunner
from pathlib import Path
from pydantic_ai.mcp import MCPServerSSE

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from search_tools import developer_prompt

# mcp_client = MCPServerStdio(
#     command="uv", args=["run", "python", "main.py"], cwd="."
# )
mcp_client = MCPServerSSE(url="http://localhost:8000/sse")

agent = Agent(
    name="faq_agent",
    instructions=developer_prompt,
    toolsets=[mcp_client],
    model="gpt-4o-mini",
)


chat_interface = StdOutputInterface()
runner = PydanticAIRunner(chat_interface=chat_interface, agent=agent)


if __name__ == "__main__":
    import asyncio

    asyncio.run(runner.run())
