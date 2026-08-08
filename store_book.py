#!/usr/bin/env python3
import hashlib
import sys
import time
from pathlib import Path

import psycopg

from extract_text import LIBRARY_DIR, extractor_for
from chunk_text import chunk_pages
from embed_client import embed

from db import DB_DSN


def source_type_for(path):
    return {"pdf": "pdf", "epub": "epub"}[path.suffix.lower().lstrip(".")]


def book_title_for(path):
    return path.stem


def embedding_text_for(chunk, book_title):
    """Contextual retrieval: prepend cheap, deterministic context (no LLM
    call) before embedding, so the embedding captures what book/section a
    chunk belongs to instead of just its isolated text. Verified necessary
    against a real case: a table-of-contents chunk scored only ~0.44 on the
    literal query "table of contents" because its raw text is mostly bare
    numbers and fragments with little natural language for the embedding
    model to anchor on. Only the text fed to the embedding model changes --
    the stored/returned `content` stays the original chunk, unprefixed, so
    search results don't repeat the book title on every line."""
    if chunk.get("chunk_type") == "structural":
        return f"Table of contents for {book_title}:\n\n{chunk['text']}"
    return f"{book_title}:\n\n{chunk['text']}"


def store_book(path, extractor=None):
    if extractor is None:
        extractor = extractor_for(path)
    pages = extractor(path)
    chunks = chunk_pages(pages)
    source_path = str(path.relative_to(LIBRARY_DIR))
    source_type = source_type_for(path)
    book_title = book_title_for(path)

    print(f"{path.name}: {len(pages)} pages/sections -> {len(chunks)} chunks")

    inserted = 0
    skipped = 0
    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            for i, chunk in enumerate(chunks):
                content_hash = hashlib.sha256(chunk["text"].encode()).hexdigest()

                cur.execute(
                    "SELECT 1 FROM rag_library_chunks WHERE source_path = %s AND chunk_index = %s AND content_hash = %s",
                    (source_path, i, content_hash),
                )
                if cur.fetchone():
                    skipped += 1
                    continue

                vector = embed(embedding_text_for(chunk, book_title))
                cur.execute(
                    """
                    INSERT INTO rag_library_chunks
                        (content, embedding, source_path, source_type, chunk_index, page_index, content_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (source_path, chunk_index)
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        page_index = EXCLUDED.page_index,
                        content_hash = EXCLUDED.content_hash,
                        updated_at = now()
                    """,
                    (chunk["text"], vector, source_path, source_type, i, chunk["page_index"], content_hash),
                )
                inserted += 1
            conn.commit()

    print(f"  inserted/updated: {inserted}, unchanged (skipped): {skipped}")
    return {"total_chunks": len(chunks), "inserted": inserted, "skipped": skipped}


def main():
    """Ingest one or more files: python store_book.py FILE [FILE ...]"""
    paths = [Path(a) for a in sys.argv[1:]]
    if not paths:
        print("usage: python store_book.py FILE [FILE ...]", file=sys.stderr)
        return 1
    started = time.time()
    for path in paths:
        store_book(path)
    print(f"\ndone in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
