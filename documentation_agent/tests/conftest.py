import pytest

from tests.utils import (
    display_total_usage,
    serialize_usage,
    merge_serialized_usage,
)
from tools import create_documentation_tools_cached
from doc_agent import (
    DocumentationAgentConfig,
    create_agent,
)


def pytest_sessionfinish(session, exitstatus):
    workeroutput = getattr(session.config, "workeroutput", None)

    if workeroutput is not None:
        workeroutput["usage"] = serialize_usage()
        return

    display_total_usage()


def pytest_testnodedown(node, error):
    worker_usage = node.workeroutput.get("usage", {})
    merge_serialized_usage(worker_usage)


@pytest.fixture(scope="session")
def agent():
    tools = create_documentation_tools_cached()
    agent_config = DocumentationAgentConfig()
    return create_agent(agent_config, tools)
