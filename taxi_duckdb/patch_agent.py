from pydantic_ai import Agent

from utils import capture_usage

_original_run = Agent.run


async def patched_run(self, *args, **kwargs):
    result = await _original_run(self, *args, **kwargs)
    capture_usage("openai:gpt-4o-mini", result)
    return result


Agent.run = patched_run
