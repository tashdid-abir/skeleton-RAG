from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from skeleton_rag.core.exceptions import (
    LLMProviderRateLimitError,
    LLMProviderRequestError,
    LLMProviderTimeoutError,
    LLMProviderUnavailableError,
)
from skeleton_rag.clients.llm_provider import ProviderResponse, ProviderUsage


class OpenRouterProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float = 20.0,
    ) -> None:
        self._model = model
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> ProviderResponse:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "context_bound_answer",
                        "strict": True,
                        "schema": response_schema,
                    },
                },
            )

        except APITimeoutError as error:
            raise LLMProviderTimeoutError(
                "The LLM provider did not respond within the configured timeout."
            ) from error

        except RateLimitError as error:
            raise LLMProviderRateLimitError(
                "The LLM provider rate limit was reached."
            ) from error

        except APIConnectionError as error:
            raise LLMProviderUnavailableError(
                "Could not connect to the LLM provider."
            ) from error

        except APIStatusError as error:
            if error.status_code >= 500:
                raise LLMProviderUnavailableError(
                    "The LLM provider is currently unavailable."
                ) from error

            raise LLMProviderRequestError(
                f"The LLM provider rejected the request with status {error.status_code}."
            ) from error

        content = response.choices[0].message.content

        if content is None:
            raise LLMProviderRequestError(
                "The LLM provider returned no response content."
            )

        usage = response.usage

        return ProviderResponse(
            content=content,
            usage=ProviderUsage(
                input_tokens=usage.prompt_tokens if usage else None,
                output_tokens=usage.completion_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
            ),
        )
