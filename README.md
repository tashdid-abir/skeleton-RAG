# Skeleton RAG

A learning project for building a context-bound LLM answerer before adding
document retrieval. It answers questions from supplied university examination
policy passages, refuses unsupported questions, and cites the passage IDs used
as evidence.

## Current milestone

The current script demonstrates the first building block of a RAG system:

```text
Policy passages + question -> grounded answer + citations
```

It currently:

- sends requests to an OpenRouter-hosted LLM;
- provides manually selected policy passages as context;
- instructs the model to use that context only;
- returns `unanswerable` when the context does not support an answer; and
- asks the model to cite supporting passage IDs such as `P3`.

This is not retrieval-augmented generation yet: passages are supplied manually.
PDF extraction, chunking, embeddings, and retrieval will be added later.

## Setup

### 1. Install dependencies

```powershell
uv sync
```

### 2. Create `.env`

Create a file named `.env` in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=dots-studio/dots-3-note-preview:free
```

`.env` is ignored by Git. Never commit or share the API key.

### 3. Run the example

```powershell
uv run python main.py
```

The sample question asks whether a student may leave an examination hall after
30 minutes. The answer should be supported by passage `P3`.

## Example result

```text
No. You cannot leave the examination hall within the first 45 minutes.
Citations: P3
```

For a question that the supplied passages do not answer, the expected result is:

```text
unanswerable
```

## Project structure

```text
.
├── main.py                 # Current context-bound answering experiment
├── src/skeleton_rag/       # Package location for the evolving application
├── pyproject.toml          # Project metadata and dependencies
└── .env                    # Local API key and model selection; not committed
```

## Next steps

1. Return structured answers validated with Pydantic.
2. Handle malformed model output and provider failures.
3. Add a provider contract and deterministic test provider.
4. Test grounded, unsupported, conflicting, and adversarial-context questions.
5. Replace manually supplied passages with a document retrieval pipeline.
