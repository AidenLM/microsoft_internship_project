import sqlite3

import numpy as np

DB_PATH = "documents.db"


def init_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            source TEXT,
            content TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    return conn


def store_chunk(conn, source, content, embedding):
    conn.execute(
        "INSERT INTO documents (source, content, embedding) VALUES (?, ?, ?)",
        (source, content, embedding.astype(np.float32).tobytes()),
    )
    conn.commit()


def get_all_chunks(conn):
    rows = conn.execute("SELECT id, source, content, embedding FROM documents").fetchall()
    return [
        (row_id, source, content, np.frombuffer(embedding, dtype=np.float32))
        for row_id, source, content, embedding in rows
    ]


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search(conn, query_embedding, top_k=3):
    chunks = get_all_chunks(conn)
    scored = [
        (cosine_similarity(query_embedding, embedding), row_id, source, content)
        for row_id, source, content, embedding in chunks
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]
