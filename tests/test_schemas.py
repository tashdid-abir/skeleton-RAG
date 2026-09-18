import pytest
from pydantic import ValidationError

from skeleton_rag.schemas.answer import LLMResponseValidator


def test_answered_response_is_valid() -> None:
    response = LLMResponseValidator(
        status="answered",
        answer="You cannot leave within 45 minutes.",
        citations=["P3"],
    )

    assert response.status == "answered"


def test_unanswerable_response_is_valid() -> None:
    response = LLMResponseValidator(
        status="unanswerable",
        answer=None,
        citations=[],
    )

    assert response.answer is None


@pytest.mark.parametrize(
    "payload",
    [
        {
            "status": "answered",
            "answer": None,
            "citations": ["P3"],
        },
        {
            "status": "answered",
            "answer": "No.",
            "citations": [],
        },
        {
            "status": "unanswerable",
            "answer": "No.",
            "citations": [],
        },
        {
            "status": "unanswerable",
            "answer": None,
            "citations": ["P3"],
        },
    ],
)
def test_invalid_status_combinations_fail(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        LLMResponseValidator.model_validate(payload)
