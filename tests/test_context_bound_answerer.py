import json

import pytest

from skeleton_rag.clients.fake_provider import FakeProvider
from skeleton_rag.clients.llm_provider import ProviderResponse, ProviderUsage
from skeleton_rag.core.exceptions import LLMInvalidResponseError
from skeleton_rag.services.context_bound_answerer import ContextBoundAnswerer


PASSAGES = "[P3] Students cannot leave the examination hall within 45 minutes."


def provider_response(payload: dict[str, object]) -> ProviderResponse:
    return ProviderResponse(
        content=json.dumps(payload),
        usage=ProviderUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        ),
    )


def test_supported_question_returns_answered() -> None:
    provider = FakeProvider(
        outcomes=[
            provider_response(
                {
                    "status": "answered",
                    "answer": "No. You must wait 45 minutes.",
                    "citations": ["P3"],
                }
            )
        ]
    )

    result = ContextBoundAnswerer(provider).answer(
        question="Can I leave after 30 minutes?",
        passages=PASSAGES,
    )

    assert result.answer.status == "answered"
    assert result.answer.citations == ["P3"]


def test_unsupported_question_returns_unanswerable() -> None:
    provider = FakeProvider(
        outcomes=[
            provider_response(
                {
                    "status": "unanswerable",
                    "answer": None,
                    "citations": [],
                }
            )
        ]
    )

    result = ContextBoundAnswerer(provider).answer(
        question="What is the passing mark?",
        passages=PASSAGES,
    )

    assert result.answer.status == "unanswerable"


def test_invalid_first_response_uses_one_repair_attempt() -> None:
    provider = FakeProvider(
        outcomes=[
            provider_response(
                {
                    "status": "answered",
                    "answer": None,
                    "citations": ["P3"],
                }
            ),
            provider_response(
                {
                    "status": "answered",
                    "answer": "No. You must wait 45 minutes.",
                    "citations": ["P3"],
                }
            ),
        ]
    )

    result = ContextBoundAnswerer(provider).answer(
        question="Can I leave after 30 minutes?",
        passages=PASSAGES,
    )

    assert result.answer.status == "answered"
    assert len(provider.received_messages) == 2


def test_invalid_repaired_response_fails() -> None:
    provider = FakeProvider(
        outcomes=[
            provider_response(
                {
                    "status": "answered",
                    "answer": None,
                    "citations": ["P3"],
                }
            ),
            provider_response(
                {
                    "status": "answered",
                    "answer": None,
                    "citations": ["P3"],
                }
            ),
        ]
    )

    with pytest.raises(LLMInvalidResponseError):
        ContextBoundAnswerer(provider).answer(
            question="Can I leave after 30 minutes?",
            passages=PASSAGES,
        )
