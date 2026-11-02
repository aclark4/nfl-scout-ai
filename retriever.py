"""
retriever.py - Vector search over QB documents using Chroma.

Build the index once, then query it:
    python retriever.py --rebuild
    python retriever.py "Who was best at avoiding sacks under pressure?"

Chroma's default embedding model (all-MiniLM-L6-v2) downloads automatically on
first use (~80 MB) and runs locally, so no API key is needed for retrieval.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import chromadb

from documents import DEFAULT_STATS_PATH, build_documents, load_stats

DEFAULT_DB_PATH = Path("data/chroma")
COLLECTION_NAME = "qb_2024"


def get_collection(db_path: Path = DEFAULT_DB_PATH, embedding_function: Optional[Any] = None):
    client = chromadb.PersistentClient(path=str(db_path))
    kwargs = {"metadata": {"hnsw:space": "cosine"}}
    if embedding_function is not None:
        kwargs["embedding_function"] = embedding_function
    return client, client.get_or_create_collection(COLLECTION_NAME, **kwargs)


def build_index(
    stats_path: Path = DEFAULT_STATS_PATH,
    db_path: Path = DEFAULT_DB_PATH,
    embedding_function: Optional[Any] = None,
) -> int:
    """(Re)build the collection from the stats JSON. Returns the document count."""
    client = chromadb.PersistentClient(path=str(db_path))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # nothing to delete on first build

    _, collection = get_collection(db_path, embedding_function)
    docs = build_documents(load_stats(stats_path))
    collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[d["metadata"] for d in docs],
    )
    return len(docs)


class Retriever:
    """Thin wrapper so the API and agent can share one open collection."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH, embedding_function: Optional[Any] = None):
        self._client, self.collection = get_collection(db_path, embedding_function)
        if self.collection.count() == 0:
            raise RuntimeError("Index is empty. Run: python retriever.py --rebuild")

    def retrieve(self, question: str, k: int = 5, where: Optional[dict] = None) -> list[dict]:
        """Return the k most relevant documents, best match first."""
        result = self.collection.query(query_texts=[question], n_results=k, where=where)
        return [
            {"id": doc_id, "text": text, "metadata": meta, "distance": round(dist, 4)}
            for doc_id, text, meta, dist in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query the QB vector index.")
    parser.add_argument("question", nargs="?", help="Question to search for")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the index from the stats JSON")
    parser.add_argument("-k", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    if args.rebuild:
        count = build_index()
        print(f"Indexed {count} documents into {DEFAULT_DB_PATH}/")
    if args.question:
        for i, hit in enumerate(Retriever().retrieve(args.question, k=args.k), start=1):
            print(f"{i}. [{hit['id']}] distance={hit['distance']}\n   {hit['text'][:220]}...\n")
    if not (args.rebuild or args.question):
        parser.print_help()


if __name__ == "__main__":
    main()
