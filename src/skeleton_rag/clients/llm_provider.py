from typing import Any, Protocol

from pydantic import BaseModel


class ProviderUsage(BaseModel):
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


class ProviderResponse(BaseModel):
    content: str
    usage: ProviderUsage


class LLMProvider(Protocol):
    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> ProviderResponse:
        """Send messages to an LLM and return its raw structured response."""