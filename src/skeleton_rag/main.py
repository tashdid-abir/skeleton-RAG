import os

from dotenv import load_dotenv

from skeleton_rag.clients.openrouter_provider import OpenRouterProvider
from skeleton_rag.services.context_bound_answerer import ContextBoundAnswerer

PASSAGES = """
[P1] Cell phone or any other unauthorized electronic devices are not allowed in the examination hall.

[P2] Students shall not be allowed to sit for the examination if they are late by 30 minutes or more.

[P3] No examinee shall be allowed to leave the examination hall or submit an answer script within 45 minutes of the start of the examination.
"""

QUESTION = "Can I leave the examination hall 30 minutes after it begins?"


def main() -> None:
    load_dotenv()

    provider = OpenRouterProvider(
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=os.environ["OPENROUTER_MODEL"],
    )
    answerer = ContextBoundAnswerer(provider=provider)

    result = answerer.answer(
        question=QUESTION,
        passages=PASSAGES,
    )

    print(result.model_dump_json(indent=2))
