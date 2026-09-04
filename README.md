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
- [x] Ingestion pipeline (chunking + embedding + storing documents)
- [x] Retrieval function (top-k relevant chunks for a query)
- [x] LLM integration (answer_query)
- [x] CLI interface
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

This drops you into a chat loop that answers from `docs/` - each question
gets embedded, matched against the stored document chunks, and the top
matches get passed to `mistral-7b-v0.2` as context before it answers.
`exit` to quit. Went through a few chat model sizes to land on Mistral:
qwen3-0.6b was fast but got basic facts wrong, qwen3-4b was noticeably
better, mistral-7b-v0.2 (~7B, ~4.2GB on disk) gives the best answers so far
and still runs comfortably - a few seconds per response, no memory issues.

Run `python ingest.py` first if `documents.db` doesn't exist yet (or if
`docs/` changed) - `main.py` reads from that database, it doesn't ingest
documents itself.

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

## Ingestion & retrieval

`docs/` holds the source documents (currently just the original project
plan, saved as plain text). `ingest.py` reads every `.txt` file in there,
splits it into chunks (one paragraph/bullet per chunk), embeds each one, and
stores it in `documents.db`:

```bash
python ingest.py
```

`retrieval.py` embeds a question and returns the top-k most similar chunks:

```bash
python retrieval.py "What is RAG?"
```

Real document chunks score noticeably better than the short one-line test
sentences from `demo_search.py` - asking "What is RAG?" against the actual
project plan returns the two paragraphs that directly define RAG, both
above 0.79 similarity.

## Reference

Based on the Microsoft Tech Community walkthrough:
https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-your-first-local-rag-application-with-foundry-local/4501968
