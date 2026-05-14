test:
	uv run pytest tests -s

test-agent:
	uv run pytest tests/test_agent.py -s

test-judge:
	uv run pytest tests/test_judge.py -s

test-one:
	uv run pytest tests/test_judge.py::test_agent_uses_tools -s

app:
	uv run streamlit run frontend_app.py

run:
	uv run python main.py "LLM as a judge"

.PHONY: tests run tests-judge

tests:
	uv run pytest

tests-judge:
	uv run pytest tests/test_judge.py -n 4