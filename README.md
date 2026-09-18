# Skeleton RAG

A learning project for the generation half of a RAG system. The application
answers a question only from passages supplied by the caller, returns a typed
response, cites passage IDs, and refuses unsupported or conflicting questions.

This is not full RAG yet: there is no PDF ingestion, chunking, embedding, or
retrieval. Those stages will be added in later weeks.

## Current capabilities

- OpenRouter-hosted LLM provider selected through environment variables
- JSON-schema structured LLM output
- Pydantic validation for answer status, answer text, and citations
- `answered` and `unanswerable` response states
- One bounded repair attempt for invalid model output
- Custom exceptions for timeout, rate-limit, connection, request, and invalid-response failures
- Provider `Protocol`, real OpenRouter provider, and deterministic fake provider
- Offline tests using the fake provider

## Project structure

```text
.
├── src/skeleton_rag/
│   ├── clients/       # Provider contract and implementations
│   ├── core/          # Application exceptions
│   ├── schemas/       # Pydantic response models
│   ├── services/      # Context-bound answering workflow
│   └── main.py        # Package command entry point
├── tests/             # Offline unit tests
├── .env.example       # Safe environment-variable template
├── pyproject.toml
└── uv.lock
```

## Setup

### 1. Install dependencies

```powershell
uv sync
```

### 2. Configure environment variables

Copy the template:

```powershell
Copy-Item .env.example .env
```

Then replace the placeholder API key in `.env`:

```env
OPENROUTER_API_KEY=your_actual_openrouter_api_key
OPENROUTER_MODEL=dots-studio/dots-3-note-preview:free
```

`.env` is ignored by Git. Never commit or share an API key.

## Run the example

```powershell
uv run skeleton-rag
```

The current example asks whether a student may leave an examination hall after
30 minutes. The result is structured JSON containing the answer, citations,
latency, and available token usage.

You can also use the compatibility launcher:

```powershell
uv run python main.py
```

## Run tests

```powershell
uv run pytest
```

The tests use `FakeProvider`, so they do not require an API key or internet
access.

## Current limitations

- Passages are manually supplied by the caller.
- Citations have a validated shape, but citation IDs are not yet checked against a caller-provided passage-ID allowlist.
- The configured free preview model may be withdrawn or changed by its provider; choose another OpenRouter model in `.env` if needed.
