# Internal RAG Assistant — Demo

This repository contains a minimal, local Retrieval-Augmented Generation (RAG) demo built with Streamlit, TF-IDF retrieval, and a small open-source model (`google/flan-t5-small`). It’s designed to showcase how content discovery, retrieval, and generation fit together, plus how Sentry provides observability across the flow.

## What’s included

- **Streamlit UI** (`app.py`) for asking questions and inspecting retrieved context.
- **RAG engine** (`rag.py`) using TF-IDF retrieval and a text-to-text generation pipeline.
- **Mock company docs** (`data/docs/*.md`) to serve as the knowledge base.
- **Pinned dependencies** (`requirements.txt`) and an **example environment file** (`.env.example`).

## Quickstart

1. Create a virtual environment and install dependencies (including Streamlit):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. (Optional) Configure Sentry:

   ```bash
   cp .env.example .env
   # Edit .env to set SENTRY_DSN if you want telemetry
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

## How it works (demo flow)

1. **Discovery**
   - The app loads markdown files from `data/docs/` and builds a TF-IDF index.
2. **Retrieval**
   - A query is embedded via TF-IDF and matched against the indexed docs.
   - The app shows **top-k** results and their similarity scores.
3. **Generation**
   - Retrieved context is passed to `google/flan-t5-small` via the transformers pipeline.
   - The model is instructed to answer **only** from the provided context.

## Demo callouts

When presenting, highlight the following behaviors:

- **Discovery → Retrieval → Generation**
  - Ask a question like “What is our PTO policy?” and point out:
    - The retrieved doc(s) in the right column.
    - The generated answer grounded in the retrieved text.

- **Low-confidence fallback**
  - Try a question that isn’t covered in the docs (e.g., “What’s our hardware reimbursement policy?”).
  - If the top similarity score is below the configured threshold, the app returns an “I don’t know / clarify” response.

- **Sentry observability**
  - Set `SENTRY_DSN` in `.env` to enable telemetry.
  - Use the **Simulate slow response** toggle to create a slow transaction.
  - Use the **Simulate exception** toggle to emit an error event.
  - In Sentry, point out spans/tags for retrieval and generation, plus query-level tags like top-k and min confidence.

## Notes

- This is a CPU-friendly demo using a small open-source model.
- The mock docs intentionally include an ambiguous policy to demonstrate clarification behavior.
