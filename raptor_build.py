#!/usr/bin/env python3
"""Build a RAPTOR-style summary hierarchy on top of existing rag_library_chunks
leaf chunks, book by book. Clusters a book's leaf-chunk embeddings, summarizes
each cluster with Maria's own chat model (qwen3.6 via Ollama), embeds the
summary with the same embedding model used for leaf chunks (nomic-embed-text),
and stores it back as a new row -- so retrieval can surface a broad synthesis
for a broad question, or a precise leaf chunk for a precise one, from the same
table and the same search path.

Scoped per-book, not whole-corpus: a corpus-wide tree would let an unrelated
book's chunks end up clustered together, and a per-book scope matches how the
existing embedding_text_for() convention already anchors context to one book.

Idempotent: re-running a book deletes its existing summary rows first, so a
rerun after new leaf chunks land (e.g. from the nightly ingestion job) always
reflects current content instead of accumulating stale duplicates.
"""
import argparse
import hashlib
import sys
import time

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from sklearn.cluster import KMeans

from embed_client import EMBED_MODEL, embed
from summarize_client import summarize

from db import DB_DSN

MIN_CHUNKS_TO_SUMMARIZE = 6  # below this, a book has too little content to meaningfully cluster
# Benchmarked on a real 150-chunk file, qwen3.6 warm on GPU:
#   8 -> 146s | 12 -> 107s | 16 -> 98s | 20 -> 87s | 24 -> 191s
# Cost is one LLM call per cluster, so larger clusters mean fewer calls -- but
# prompt prefill grows superlinearly with cluster size, so the curve is U-shaped
# and collapses past ~20. 16 takes most of the available speedup (~33% faster
# than 8) while leaving margin below the cliff, since k-means cluster sizes vary
# around the target and an oversized cluster on some other book would otherwise
# land in the expensive region. Summary quality was verified unchanged at 16.
TARGET_CLUSTER_SIZE = 16
MAX_LEVELS = 3  # leaf(0) -> level 1 -> level 2 -> level 3, whichever collapses to <=1 node first
MIN_NODES_TO_KEEP_SUMMARIZING = 3  # stop recursing once a level has this few nodes or fewer

SUMMARIZE_SYSTEM_PROMPT = (
    "You write concise, information-dense summaries of technical book excerpts "
    "for a retrieval system. Preserve concrete facts, terms, and claims. Do not "
    "add commentary, opinions, or meta-text like 'this section discusses'. "
    "3-6 sentences."
)


def fetch_book_titles(cur):
    cur.execute("SELECT DISTINCT source_path FROM rag_library_chunks WHERE summary_level = 0 ORDER BY source_path")
    return [row[0] for row in cur.fetchall()]


def fetch_books_with_summaries(cur):
    """Resume state is derived from the database, not a sidecar checkpoint file.
    This corpus has a documented history of workspace files being wiped while the
    the database volume survived intact, so the durable layer is the only trustworthy
    source of "what is already done" -- and it cannot drift out of sync with
    reality the way a separate JSON file can."""
    cur.execute("SELECT DISTINCT source_path FROM rag_library_chunks WHERE summary_level > 0")
    return {row[0] for row in cur.fetchall()}


def fetch_level_nodes(cur, source_path, level):
    cur.execute(
        """
        SELECT id, content, embedding, page_index
        FROM rag_library_chunks
        WHERE source_path = %s AND summary_level = %s
        ORDER BY chunk_index
        """,
        (source_path, level),
    )
    return cur.fetchall()  # [(id, content, embedding, page_index), ...]


def delete_existing_summaries(cur, source_path):
    cur.execute(
        "DELETE FROM rag_library_chunks WHERE source_path = %s AND summary_level > 0",
        (source_path,),
    )


def cluster_nodes(nodes, target_cluster_size):
    """K-means over node embeddings with a heuristic k. Simpler than the RAPTOR
    paper's GMM+BIC soft clustering -- a deliberate simplification: this corpus
    is technical non-fiction with fairly distinct topical sections, hard
    clustering is a reasonable fit, and GMM+BIC adds real complexity for a
    marginal quality gain not worth it as a first version."""
    n = len(nodes)
    k = max(2, min(n // target_cluster_size, n - 1))
    if k >= n:
        k = max(1, n - 1)
    embeddings = np.array([node[2].to_numpy() for node in nodes])
    labels = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(embeddings)
    clusters = {}
    for node, label in zip(nodes, labels):
        clusters.setdefault(int(label), []).append(node)
    return list(clusters.values())


def book_title_for(source_path):
    return source_path.rsplit(".", 1)[0]


def summarize_cluster(cluster_nodes_list, book_title):
    texts = [n[1] for n in cluster_nodes_list]
    joined = "\n\n---\n\n".join(texts)
    prompt = f"Summarize the following excerpts from \"{book_title}\":\n\n{joined}"
    return summarize(prompt, system=SUMMARIZE_SYSTEM_PROMPT)


def build_summary_level(cur, source_path, level, source_type, target_cluster_size=TARGET_CLUSTER_SIZE):
    nodes = fetch_level_nodes(cur, source_path, level - 1)
    if len(nodes) < MIN_NODES_TO_KEEP_SUMMARIZING:
        return 0
    if level == 1 and len(nodes) < MIN_CHUNKS_TO_SUMMARIZE:
        return 0

    book_title = book_title_for(source_path)
    clusters = cluster_nodes(nodes, target_cluster_size)

    inserted = 0
    for i, cluster in enumerate(clusters):
        if len(cluster) < 2:
            continue  # singleton "cluster" -- nothing to summarize, would just restate the one chunk
        summary_text = summarize_cluster(cluster, book_title)
        embed_input = f"{book_title} (summary):\n\n{summary_text}"
        vector = embed(embed_input)
        source_ids = [n[0] for n in cluster]
        min_page = min(n[3] for n in cluster)
        content_hash = hashlib.sha256(summary_text.encode()).hexdigest()
        # Negative chunk_index keeps summary rows collision-free against real
        # leaf chunk_index values (which start at 0) under the existing
        # UNIQUE(source_path, chunk_index) constraint, without altering it.
        chunk_index = -(level * 10_000 + i + 1)

        cur.execute(
            """
            INSERT INTO rag_library_chunks
                (content, embedding, source_path, source_type, chunk_index, page_index,
                 content_hash, summary_level, summary_source_ids, embed_model)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_path, chunk_index)
            DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                page_index = EXCLUDED.page_index,
                content_hash = EXCLUDED.content_hash,
                summary_source_ids = EXCLUDED.summary_source_ids,
                embed_model = EXCLUDED.embed_model,
                updated_at = now()
            """,
            # embed_model must be stamped here. Omitting it left 4,537 of 4,841
            # summary rows NULL, and reembed_library.py resumes by asking
            # `WHERE embed_model IS DISTINCT FROM <target>`. NULL is DISTINCT
            # FROM everything, so those rows would be needlessly re-embedded on
            # any forward migration -- and, migrating back to a model they
            # already hold, silently SKIPPED, leaving two vector spaces mixed in
            # one index. That failure shows up as bad results, never an error.
            (summary_text, vector, source_path, source_type, chunk_index, min_page,
             content_hash, level, source_ids, EMBED_MODEL),
        )
        inserted += 1
    return inserted


def build_book(source_path, target_cluster_size=TARGET_CLUSTER_SIZE):
    with psycopg.connect(DB_DSN) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT source_type FROM rag_library_chunks WHERE source_path = %s LIMIT 1", (source_path,))
            row = cur.fetchone()
            if not row:
                print(f"  SKIP {source_path}: no leaf chunks found", flush=True)
                return
            source_type = row[0]

            delete_existing_summaries(cur, source_path)

            # One transaction per book, committed only after every level finishes.
            # Resume state is "does this book have any summary rows", so a book must
            # be all-or-nothing: committing per level would let a crash between
            # levels leave a half-built book that resume then skips as complete.
            # Level N reads level N-1 rows through this same connection, so
            # read-your-own-writes makes the uncommitted parent level visible.
            for level in range(1, MAX_LEVELS + 1):
                started = time.time()
                inserted = build_summary_level(cur, source_path, level, source_type, target_cluster_size)
                elapsed = time.time() - started
                if inserted == 0:
                    if level == 1:
                        print(f"  {source_path}: too few chunks to summarize, skipped", flush=True)
                    break
                print(f"  {source_path}: level {level} -> {inserted} summary node(s) ({elapsed:.1f}s)", flush=True)
            conn.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("books", nargs="*", help="specific source_path values to process; default: all books")
    parser.add_argument("--limit", type=int, help="only process the first N books (for a pilot run)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild books that already have summaries instead of skipping them",
    )
    parser.add_argument(
        "--cluster-size",
        type=int,
        default=TARGET_CLUSTER_SIZE,
        help=f"target leaf chunks per cluster (default {TARGET_CLUSTER_SIZE}); "
             "larger means fewer, broader summaries and proportionally fewer LLM calls",
    )
    args = parser.parse_args()

    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            all_books = fetch_book_titles(cur)
            already_done = set() if args.force else fetch_books_with_summaries(cur)

    books = args.books if args.books else all_books
    pending = [b for b in books if b not in already_done]
    skipped = len(books) - len(pending)
    if args.limit:
        pending = pending[: args.limit]

    print(f"{len(books)} book(s) in scope, {skipped} already summarized, {len(pending)} to build", flush=True)
    started = time.time()
    failures = []
    for i, source_path in enumerate(pending, start=1):
        print(f"[{i}/{len(pending)}] {source_path}", flush=True)
        try:
            build_book(source_path, args.cluster_size)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr, flush=True)
            failures.append(source_path)
    print(f"\ndone in {time.time() - started:.1f}s ({len(pending) - len(failures)} built, {len(failures)} failed)", flush=True)
    if failures:
        print("failed books:", flush=True)
        for name in failures:
            print(f"  - {name}", flush=True)


if __name__ == "__main__":
    main()
