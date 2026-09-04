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
- [x] Embeddings + SQLite storage
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

This drops you into a simple chat loop with `mistral-7b-v0.2` running
locally - type a question, get an answer, `exit` to quit. Went through a
few sizes here: qwen3-0.6b was fast but got basic facts wrong, qwen3-4b
was noticeably better, mistral-7b-v0.2 (~7B, ~4.2GB on disk) gives the
best answers so far and still runs comfortably - a few seconds per
response, no memory issues.

## Why Foundry Local + SQLite

Foundry Local handles running the LLM (and the embedding model) locally, with
no cloud dependency. SQLite is just a single file, so it's an easy way to
store document chunks and their embedding vectors without needing a real
database server - fine for the small document sets this project targets.

`db.py` has the storage/search logic (SQLite table + cosine similarity
search over all stored chunks - fine at this scale, wouldn't scale to a huge
document set but that's not what this project needs). `demo_search.py` is a
small script that stores a few sample sentences and searches them, mostly
just to prove the embedding + SQLite + cosine similarity pipeline actually
works end to end before building the real ingestion pipeline on top of it.

## Reference

Based on the Microsoft Tech Community walkthrough:
https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-your-first-local-rag-application-with-foundry-local/4501968
