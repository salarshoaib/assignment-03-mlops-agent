"""LLM client abstraction. Messages use Anthropic-style content blocks everywhere."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class LLMResponse:
    content: list[dict[str, Any]]
    stop_reason: str = "end_turn"


class LLMClient(Protocol):
    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


class AnthropicClient:
    """Tool-calling via the Anthropic Messages API (API key read from the environment, never logged)."""

    def __init__(self, api_key: str, model: str, max_tokens: int = 2000):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key, timeout=60.0, max_retries=2)
        self.model, self.max_tokens = model, max_tokens

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        resp = self._client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                            messages=messages, tools=tools)
        blocks: list[dict[str, Any]] = []
        for b in resp.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": dict(b.input)})
        return LLMResponse(blocks, resp.stop_reason or "end_turn")
