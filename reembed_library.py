#!/usr/bin/env python3
"""Re-embed every stored chunk with the current embedding model.

Needed whenever the embedding model changes: vectors from different models are
not comparable, so a partially-migrated table silently degrades retrieval
rather than failing loudly. Progress is tracked in rag_library_chunks.embed_model
so the job is resumable and the table's true state is always self-describing.

Re-embeds summary rows too (cheap: embedding only, no LLM), which avoids
throwing away RAPTOR summaries that would otherwise cost hours to regenerate.

Uses Ollama's batch embedding API -- measured ~91 embeddings/s batched vs
~11/s one at a time, so batching is what makes a full-corpus re-embed practical.
"""
import argparse
import json
import sys
import time
import urllib.request

import psycopg

from chunk_text import is_structural_page
from embed_client import EMBED_DIMENSIONS, EMBED_MODEL, OLLAMA_URL

from db import DB_DSN
BATCH_SIZE = 64


def embed_batch(texts):
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps({
            "model": EMBED_MODEL,
            "input": texts,
            "dimensions": EMBED_DIMENSIONS,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        raw = json.loads(resp.read())["embeddings"]
    # Cast every component to float: JSON decodes an exact value like 0 as int,
    # and psycopg refuses a mixed int/float list ("cannot dump lists of mixed
    # types"). embed_client.embed() already does this for the single-vector
    # path; the batch path needs the same treatment.
    return [[float(x) for x in vector] for vector in raw]


def book_title_for(source_path):
    return source_path.rsplit(".", 1)[0]


def embedding_text_for(content, source_path, summary_level):
    """Reproduces the contextual prefixes the original ingestion applied, so
    re-embedded vectors stay consistent with how the corpus was built.

    chunk_type was never persisted, so leaf chunks are reclassified with the
    same is_structural_page heuristic that produced them -- table-of-contents
    chunks are exactly the dotted-leader/chapter-line-dense shape it detects,
    which is why that prefix existed in the first place (a TOC chunk scored
    only ~0.44 on the literal query "table of contents" without it)."""
    book_title = book_title_for(source_path)
    if summary_level > 0:
        return f"{book_title} (summary):\n\n{content}"
    if is_structural_page(content):
        return f"Table of contents for {book_title}:\n\n{content}"
    return f"{book_title}:\n\n{content}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, help="stop after N rows (for a trial run)")
    args = parser.parse_args()

    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM rag_library_chunks WHERE embed_model IS DISTINCT FROM %s",
                (EMBED_MODEL,),
            )
            pending_total = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM rag_library_chunks")
            grand_total = cur.fetchone()[0]

    print(f"re-embedding with {EMBED_MODEL} @ {EMBED_DIMENSIONS} dims", flush=True)
    print(f"{grand_total} rows total, {pending_total} need re-embedding", flush=True)
    if pending_total == 0:
        print("nothing to do", flush=True)
        return

    started = time.time()
    processed = 0
    while True:
        with psycopg.connect(DB_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, content, source_path, summary_level
                    FROM rag_library_chunks
                    WHERE embed_model IS DISTINCT FROM %s
                    ORDER BY id
                    LIMIT %s
                    """,
                    (EMBED_MODEL, args.batch_size),
                )
                rows = cur.fetchall()
                if not rows:
                    break

                texts = [embedding_text_for(c, sp, lvl) for _, c, sp, lvl in rows]
                try:
                    vectors = embed_batch(texts)
                except Exception as e:
                    print(f"  batch failed ({e}); retrying once after 5s", file=sys.stderr, flush=True)
                    time.sleep(5)
                    vectors = embed_batch(texts)

                # executemany pipelines the round-trips in psycopg3; the naive
                # per-row execute loop made DB writes, not embedding, the
                # bottleneck (~21 rows/s against ~91/s available from the model).
                cur.executemany(
                    "UPDATE rag_library_chunks SET embedding = %s, embed_model = %s, updated_at = now() WHERE id = %s",
                    [(vector, EMBED_MODEL, row_id)
                     for (row_id, _, _, _), vector in zip(rows, vectors)],
                )
                conn.commit()

        processed += len(rows)
        if processed % (args.batch_size * 10) == 0 or processed >= pending_total:
            elapsed = time.time() - started
            rate = processed / elapsed if elapsed else 0
            remaining = (pending_total - processed) / rate if rate else 0
            print(
                f"  {processed}/{pending_total} ({100*processed/pending_total:.1f}%) "
                f"{rate:.0f}/s, ETA {remaining/60:.1f} min",
                flush=True,
            )
        if args.limit and processed >= args.limit:
            print(f"stopping at --limit {args.limit}", flush=True)
            break

    elapsed = time.time() - started
    print(f"\ndone: {processed} rows in {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
