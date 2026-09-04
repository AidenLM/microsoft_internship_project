# microsoft_internship_project

Local RAG assistant built for the Microsoft summer internship project. The idea is
a small offline Q&A chatbot: it retrieves relevant chunks from a local document
set (SQLite + embeddings) and passes them to a local LLM through Foundry Local,
so answers stay grounded in the actual documents instead of the model just
guessing.

Everything runs on-device, no API keys or internet connection needed at
inference time.

## Status

Week 1 - setting up Foundry Local and the project skeleton.

- [x] Repo + basic project structure
- [x] Foundry Local installed and "hello model" test working
- [ ] Embeddings + SQLite storage
- [ ] Ingestion pipeline (chunking + embedding + storing documents)
- [ ] Retrieval function (top-k relevant chunks for a query)
- [ ] LLM integration (answer_query)
- [ ] CLI interface
- [ ] Test cases + docs

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Foundry Local itself also needs to be installed separately (it manages and runs
the actual models). See the official quickstart:
https://learn.microsoft.com/azure/ai-foundry/foundry-local/get-started

Once it's installed, run:

```bash
python main.py
```

This loads a small model (`qwen3-0.6b` by default) and sends it a test
prompt, just to confirm the local runtime is working before building the
actual RAG pipeline on top of it. Went with a 0.6B model here instead of
something like phi-3.5-mini mainly because it's lighter on RAM and answers
in a couple seconds - good enough for a sanity check. Worth trying a bigger
chat model later once the pipeline itself works, if the machine can handle it.

## Why Foundry Local + SQLite

Foundry Local handles running the LLM (and the embedding model) locally, with
no cloud dependency. SQLite is just a single file, so it's an easy way to
store document chunks and their embedding vectors without needing a real
database server - fine for the small document sets this project targets.

## Reference

Based on the Microsoft Tech Community walkthrough:
https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-your-first-local-rag-application-with-foundry-local/4501968
