from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LLMResponseValidator(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["answered", "unanswerable"] = Field(
        description="Whether the supplied passages support an answer."
    )
    answer: str | None = Field(
        description="A concise answer, or null when the answer is unavailable."
    )
    citations: list[str] = Field(
        description="Passage IDs that directly support the answer."
    )

    @model_validator(mode="after")
    def validate_answer_status(self) -> "LLMResponseValidator":
        if self.status == "answered":
            if not self.answer or not self.answer.strip():
                raise ValueError(
                    "An answered response must contain a non-empty answer."
                )

            if not self.citations:
                raise ValueError(
                    "An answered response must contain at least one citation."
                )

        if self.status == "unanswerable":
            if self.answer is not None:
                raise ValueError(
                    "An unanswerable response must have answer set to null."
                )

            if self.citations:
                raise ValueError(
                    "An unanswerable response must have no citations."
                )

        return self

class UsageMetadata(BaseModel):
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None

class AnswerResult(BaseModel):
    answer: LLMResponseValidator
    latency_ms: float
    usage: UsageMetadata
