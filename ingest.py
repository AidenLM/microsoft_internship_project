import glob
import os

from foundry_local_sdk import Configuration, EmbeddingsSession, FoundryLocalManager, Request, TextItem

from db import init_db, store_chunk

EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"
DOCS_DIR = "docs"
MIN_CHUNK_LENGTH = 30  # skip near-empty lines/headers, not worth embedding


def load_embedding_model():
    manager = FoundryLocalManager(Configuration(app_name="local-rag-assistant"))
    model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
    if not model.is_cached:
        print(f"Downloading {EMBEDDING_MODEL_ALIAS}...")
        model.download()
    model.load()
    return model


def embed(session, text):
    import numpy as np

    request = Request().add_item(TextItem(text))
    with session.process_request(request) as response:
        tensor = response.get_item(0)
        return np.frombuffer(tensor.data, dtype=np.float32)


def chunk_document(path):
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    return [line for line in lines if len(line) >= MIN_CHUNK_LENGTH]


def main():
    model = load_embedding_model()
    conn = init_db()
    conn.execute("DELETE FROM documents")  # re-run ingestion cleanly each time

    doc_paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt")))
    if not doc_paths:
        print(f"No .txt files found in {DOCS_DIR}/")
        return

    with EmbeddingsSession(model) as session:
        total = 0
        for path in doc_paths:
            source = os.path.basename(path)
            chunks = chunk_document(path)
            for chunk in chunks:
                store_chunk(conn, source, chunk, embed(session, chunk))
                total += 1
            print(f"{source}: {len(chunks)} chunks")

    print(f"\nDone. Stored {total} chunks total.")


if __name__ == "__main__":
    main()
