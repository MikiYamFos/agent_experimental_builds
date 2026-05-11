from typing import Dict, Any
from dataclasses import dataclass
from pydantic_ai import RunUsage
from collections import defaultdict

usages = defaultdict(RunUsage)


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]


def collect_tools(messages):
    tool_calls = []
    for m in messages:
        for p in m.parts:
            part_kind = p.part_kind

            if part_kind != "tool-call":
                continue

            if p.tool_name == "final_result":
                continue

            tool_calls.append(ToolCall(p.tool_name, p.args))
    return tool_calls


def capture_usage(model_name: str, result):
    usages[model_name] += result.usage()


def serialize_usage():
    return {
        model: {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        }
        for model, usage in usages.items()
    }


def merge_serialized_usage(serialized_usage):
    for model, usage_data in serialized_usage.items():
        usages[model].input_tokens += usage_data["input_tokens"]
        usages[model].output_tokens += usage_data["output_tokens"]


MODEL_PRICES = {
    "openai:gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai:gpt-4o": {"input": 2.50, "output": 10.00},
}


def calculate_cost(model_name, input_tokens, output_tokens):
    prices = MODEL_PRICES[model_name.lower()]
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    return input_cost + output_cost


def display_total_usage():
    print()

    total_cost = 0

    for model, usage in usages.items():
        cost = calculate_cost(model, usage.input_tokens, usage.output_tokens)
        print(f"{model}: ${cost:.6f}")
        total_cost += cost

    print(f"Total cost: ${total_cost:.6f}")
