from foundry_local_sdk import Configuration, EmbeddingsSession, FoundryLocalManager, Request, TextItem

import numpy as np

from db import init_db, search, store_chunk

EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"

SAMPLE_DOCS = [
    ("notes.txt", "Foundry Local lets you run AI models on your own device, without needing the cloud."),
    ("notes.txt", "SQLite is a lightweight database that stores everything in a single file."),
    ("notes.txt", "RAG retrieves relevant text and gives it to the model as context before it answers."),
    ("notes.txt", "Cosine similarity measures how close two vectors are in direction, not length."),
]


def load_embedding_model():
    manager = FoundryLocalManager(Configuration(app_name="local-rag-assistant"))
    model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
    if not model.is_cached:
        print(f"Downloading {EMBEDDING_MODEL_ALIAS}...")
        model.download()
    model.load()
    return model


def embed(session, text):
    request = Request().add_item(TextItem(text))
    with session.process_request(request) as response:
        tensor = response.get_item(0)
        return np.frombuffer(tensor.data, dtype=np.float32)


def main():
    model = load_embedding_model()
    conn = init_db()
    conn.execute("DELETE FROM documents")  # keep this demo repeatable

    with EmbeddingsSession(model) as session:
        for source, content in SAMPLE_DOCS:
            store_chunk(conn, source, content, embed(session, content))

        query = "how do I avoid the model making things up"
        print(f"query: {query!r}\n")

        query_embedding = embed(session, query)
        results = search(conn, query_embedding, top_k=len(SAMPLE_DOCS))

        for score, row_id, source, content in results:
            print(f"score={score:.4f}  [{source}]  {content}")


if __name__ == "__main__":
    main()
