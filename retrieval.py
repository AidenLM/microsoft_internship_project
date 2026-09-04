from foundry_local_sdk import Configuration, EmbeddingsSession, FoundryLocalManager

from db import init_db, search
from ingest import EMBEDDING_MODEL_ALIAS, embed


def get_top_chunks(manager, query, top_k=3):
    model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
    model.load()

    conn = init_db()
    with EmbeddingsSession(model) as session:
        query_embedding = embed(session, query)

    return search(conn, query_embedding, top_k=top_k)


if __name__ == "__main__":
    import sys

    manager = FoundryLocalManager(Configuration(app_name="local-rag-assistant"))
    query = " ".join(sys.argv[1:]) or "What is RAG?"
    print(f"query: {query!r}\n")
    for score, row_id, source, content in get_top_chunks(manager, query):
        print(f"score={score:.4f}  [{source}]  {content}\n")
