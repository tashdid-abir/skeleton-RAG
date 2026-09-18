from typing import Any

from skeleton_rag.clients.llm_provider import ProviderResponse


class FakeProvider:
    def __init__(
        self,
        outcomes: list[ProviderResponse | Exception],
    ) -> None:
        self._outcomes = outcomes.copy()
        self.received_messages: list[list[dict[str, str]]] = []

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> ProviderResponse:
        self.received_messages.append(messages)

        if not self._outcomes:
            raise RuntimeError("FakeProvider has no configured outcome.")

        outcome = self._outcomes.pop(0)

        if isinstance(outcome, Exception):
            raise outcome

        return outcome
