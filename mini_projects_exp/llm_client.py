import os
from pathlib import Path
from typing import Literal

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

Provider = Literal["anthropic", "openai"]

REPO_ROOT = Path.cwd()
while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / "pyproject.toml").exists():
    REPO_ROOT = REPO_ROOT.parent

load_dotenv(REPO_ROOT / ".env")

# Prices in USD per million tokens, verified May 2026
MODEL_PRICES = {
    # Anthropic
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
    "claude-opus-4-7": {"input": 5.00, "output": 25.00},
    # OpenAI
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


def get_client(provider: Provider):
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY")
        return Anthropic(api_key=api_key)

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        return OpenAI(api_key=api_key)

    raise ValueError(f"Unsupported provider: {provider}")


def get_default_model(provider: Provider) -> str:
    if provider == "anthropic":
        return "claude-sonnet-4-6"
    if provider == "openai":
        return "gpt-4o"
    raise ValueError(f"Unsupported provider: {provider}")


def get_cheapest_model(provider: Provider) -> str:
    if provider == "anthropic":
        return "claude-haiku-4-5-20251001"
    if provider == "openai":
        return "gpt-4o-mini"
    raise ValueError(f"Unsupported provider: {provider}")


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = MODEL_PRICES.get(model)
    if prices is None:
        raise ValueError(f"No pricing data for model: {model}")
    return (input_tokens / 1_000_000) * prices["input"] + (
        output_tokens / 1_000_000
    ) * prices["output"]
