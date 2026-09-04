from foundry_local_sdk import Configuration, EmbeddingsSession, FoundryLocalManager, Request, TextItem

from db import init_db, search
from ingest import EMBEDDING_MODEL_ALIAS, embed, load_embedding_model


def get_top_chunks(query, top_k=3):
    manager = FoundryLocalManager(Configuration(app_name="local-rag-assistant"))
    model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
    model.load()

    conn = init_db()
    with EmbeddingsSession(model) as session:
        query_embedding = embed(session, query)

    return search(conn, query_embedding, top_k=top_k)


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What is RAG?"
    print(f"query: {query!r}\n")
    for score, row_id, source, content in get_top_chunks(query):
        print(f"score={score:.4f}  [{source}]  {content}\n")
