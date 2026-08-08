#!/usr/bin/env python3
"""Score retrieval configurations against a fixed evaluation set.

Answers the question the README had to admit it could not: what does each
retrieval stage actually buy? Ranking here was tuned by reading results and
judging them sensible, which is a real method but an unfalsifiable one -- it
cannot tell you that reranking helps, only that the output looked fine.

Four configurations, measured on identical queries:

  vector  vector lane alone (HNSW over the halfvec cast)
  fts     Postgres full-text alone
  rrf     both lanes fused by Reciprocal Rank Fusion, no reranking
  full    rrf + cross-encoder rerank -- what production actually runs

The stage wiring is duplicated here rather than called through search.search(),
because the intermediate configurations do not exist as callable functions --
production only exposes the finished pipeline. To keep that duplication from
drifting into a different system, the *components* are imported from the real
modules (embed, rerank, RRF_K, the pool sizes). If someone retunes RRF_K in
search.py, this harness retunes with it.

Metrics are computed against the chunk the query was generated from:
  recall@k  fraction of queries where the gold chunk appears in the top k
  MRR       mean reciprocal rank of the gold chunk (0 if outside top 10)

Read the deltas, not the absolutes -- see eval_build_set.py on why the absolute
numbers are optimistic.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from embed_client import embed
from rerank import rerank
from search import (CANDIDATE_POOL_SIZE, DB_DSN, RRF_K, TITLE_LANE_LIMIT,
                    TITLE_LANE_WEIGHT, TITLE_SQL, _matching_files, _title_terms,
                    keyword_lane)

EVAL_DIR = Path(__file__).parent / "eval"
TOP_K = 10

VECTOR_SQL = """
    SELECT id FROM rag_library_chunks
    WHERE summary_level = 0
    ORDER BY embedding::halfvec(768) <=> %s::halfvec(768)
    LIMIT %s
"""



def rrf_fuse(*ranked_lists, weights=None):
    """Reciprocal Rank Fusion over any number of ranked id lists.

    `weights` matches production's weighted fusion -- the title lane counts
    double, because an exact filename match is a stronger statement of intent
    than one lane's top guess. Without it every lane's rank-1 hit ties at
    1/(RRF_K+1) and ordering falls to insertion order.
    """
    scores = {}
    for i, ids in enumerate(ranked_lists):
        w = 1.0 if weights is None else weights[i]
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0) + w / (RRF_K + rank)
    return sorted(scores, key=lambda c: scores[c], reverse=True)


def title_lane(cur, query, vector):
    """The known-item lane, as production runs it."""
    terms = _title_terms(query)
    if len(terms) < 2:
        return []
    files = _matching_files(cur, terms)
    if not files:
        return []
    cur.execute(TITLE_SQL, (files, vector, TITLE_LANE_LIMIT))
    return [r[0] for r in cur.fetchall()]


def retrieve(cur, query, vector):
    """Return {config_name: [chunk_id, ...]} for one query."""
    cur.execute(VECTOR_SQL, (vector, CANDIDATE_POOL_SIZE))
    vec_ids = [r[0] for r in cur.fetchall()]

    # production's keyword lane, imported rather than reimplemented
    fts_ids = [r[0] for r in keyword_lane(cur, query)]

    fused = rrf_fuse(vec_ids, fts_ids)
    title_ids = title_lane(cur, query, vector)
    fused3 = rrf_fuse(vec_ids, fts_ids, title_ids, weights=[1.0, 1.0, TITLE_LANE_WEIGHT])

    # Rerank the whole fused pool, exactly as production does -- slicing to
    # top_k before reranking would discard the candidates reranking exists to
    # rescue, and would measure a pipeline nobody runs.
    if fused:
        cur.execute("SELECT id, content FROM rag_library_chunks WHERE id = ANY(%s)", (fused,))
        by_id = dict(cur.fetchall())
        texts = [by_id.get(cid, "") for cid in fused]
        scores = rerank(query, texts)
        reranked = [cid for cid, _ in sorted(zip(fused, scores), key=lambda p: p[1], reverse=True)]
    else:
        reranked = []

    return {"vector": vec_ids, "fts": fts_ids, "rrf": fused,
            "full": reranked, "rrf+title": fused3}


def score(results, gold_id):
    """(hit@1, hit@5, hit@10, reciprocal_rank) for one ranked list."""
    top = results[:TOP_K]
    if gold_id in top:
        rank = top.index(gold_id) + 1
        return (rank == 1, rank <= 5, True, 1.0 / rank)
    return (False, False, False, 0.0)


def main():
    ap = argparse.ArgumentParser(description="Score retrieval configurations")
    ap.add_argument("query_set", nargs="?", help="path to a queries-*.jsonl file")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N queries")
    args = ap.parse_args()

    path = Path(args.query_set) if args.query_set else None
    if path is None:
        # Newest by mtime, NOT by name: alphabetical sorting put
        # queries-seed999-n6 after queries-seed20260808-n120 (string compare,
        # '9' > '2'), so the default silently picked a 6-query smoke set and
        # printed a confident table from it.
        sets = sorted(EVAL_DIR.glob("queries-*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not sets:
            print("no query set found; run eval_build_set.py first", file=sys.stderr)
            return 1
        path = sets[-1]
    queries = [json.loads(l) for l in path.open() if l.strip()]
    if args.limit:
        queries = queries[: args.limit]
    print(f"eval set: {path.name}  ({len(queries)} queries)", file=sys.stderr)

    configs = ["vector", "fts", "rrf", "full", "rrf+title"]
    acc = {c: {"h1": [], "h5": [], "h10": [], "rr": []} for c in configs}
    started = time.time()

    with psycopg.connect(DB_DSN) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for i, q in enumerate(queries, 1):
                vector = embed(q["query"])
                got = retrieve(cur, q["query"], vector)
                for c in configs:
                    h1, h5, h10, rr = score(got[c], q["gold_chunk_id"])
                    acc[c]["h1"].append(h1); acc[c]["h5"].append(h5)
                    acc[c]["h10"].append(h10); acc[c]["rr"].append(rr)
                if i % 20 == 0:
                    print(f"  {i}/{len(queries)}", file=sys.stderr)

    print(f"\n{'config':8} {'recall@1':>9} {'recall@5':>9} {'recall@10':>10} {'MRR@10':>8}")
    print("-" * 48)
    for c in configs:
        a = acc[c]
        print(f"{c:8} {statistics.mean(a['h1'])*100:8.1f}% {statistics.mean(a['h5'])*100:8.1f}% "
              f"{statistics.mean(a['h10'])*100:9.1f}% {statistics.mean(a['rr']):8.3f}")
    print(f"\n{len(queries)} queries in {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
