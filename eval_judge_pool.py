#!/usr/bin/env python3
"""Build relevance judgments for real queries by pooling and judging.

The synthetic set gets its label for free: the passage that generated the
question is by construction the right answer. Real queries have no such label
-- nobody recorded which passage was correct for "The Art Of War".

This uses the standard IR answer, pooled judgments (the method TREC has used
since 1992): run every configuration, take the union of what they all return,
judge each (query, passage) pair once, then score every configuration against
that shared pool. The property that matters is that no configuration is
favoured by the labelling -- a passage only vector search finds is judged on
the same terms as one only full-text finds. Judging one system's output and
scoring another against it would bake in exactly the bias this is meant to
avoid.

Two limits worth stating rather than discovering later:

1. Unjudged passages count as irrelevant. Anything no configuration surfaced
   is invisible, so absolute recall is unmeasurable -- these numbers compare
   systems, they do not describe the corpus.
2. The judge is a local model, not a person. It is a consistent rater, not a
   correct one. Its verdicts are written out per pair so they can be reviewed
   or overridden by hand.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from embed_client import embed
from eval_run import VECTOR_SQL, rrf_fuse, title_lane
from rerank import rerank
from db import DB_DSN
from search import CANDIDATE_POOL_SIZE, TITLE_LANE_WEIGHT, keyword_lane

EVAL_DIR = Path(__file__).parent / "eval"
POOL_DEPTH = 10  # top-N from each configuration enters the pool

JUDGE_PROMPT = (
    "A user searched a personal technical library and got this passage back.\n\n"
    "SEARCH QUERY: {query}\n\n"
    "PASSAGE (from the file \"{source}\"):\n---\n{passage}\n---\n\n"
    "Would this passage satisfy what the user was looking for?\n\n"
    "Judge intent, not vocabulary. Many of these searches are known-item "
    "lookups -- someone hunting for a specific book, chapter, or table of "
    "contents. For those, a passage from the right work is relevant even if it "
    "shares few words with the query, and a passage from a different work is "
    "irrelevant even if it shares many.\n\n"
    "Answer with exactly one word: RELEVANT or IRRELEVANT"
)


def pool_candidates(cur, query, vector):
    """Union of the top POOL_DEPTH from every configuration, with each
    configuration's own ranking retained for later scoring."""
    cur.execute(VECTOR_SQL, (vector, CANDIDATE_POOL_SIZE))
    vec_ids = [r[0] for r in cur.fetchall()]
    fts_ids = [r[0] for r in keyword_lane(cur, query)]
    fused = rrf_fuse(vec_ids, fts_ids)
    title_ids = title_lane(cur, query, vector)
    fused3 = rrf_fuse(vec_ids, fts_ids, title_ids, weights=[1.0, 1.0, TITLE_LANE_WEIGHT])

    reranked = []
    if fused:
        cur.execute("SELECT id, content FROM rag_library_chunks WHERE id = ANY(%s)", (fused,))
        by_id = dict(cur.fetchall())
        scores = rerank(query, [by_id.get(c, "") for c in fused])
        reranked = [c for c, _ in sorted(zip(fused, scores), key=lambda p: p[1], reverse=True)]

    runs = {
        "vector": vec_ids[:POOL_DEPTH],
        "fts": fts_ids[:POOL_DEPTH],
        "rrf": fused[:POOL_DEPTH],
        "full": reranked[:POOL_DEPTH],
        "rrf+title": fused3[:POOL_DEPTH],
    }
    pool = []
    for ids in runs.values():
        for cid in ids:
            if cid not in pool:
                pool.append(cid)
    return runs, pool


def judge(cur, query, chunk_ids, verdict_cache):
    """Judge each (query, chunk) once. Cached across configurations because the
    same passage routinely appears in several runs and judging it twice would
    only add disagreement with itself."""
    from summarize_client import summarize
    out = {}
    if not chunk_ids:
        return out
    cur.execute("SELECT id, source_path, content FROM rag_library_chunks WHERE id = ANY(%s)",
                (chunk_ids,))
    rows = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    for cid in chunk_ids:
        key = (query, cid)
        if key in verdict_cache:
            out[cid] = verdict_cache[key]
            continue
        source, content = rows.get(cid, ("", ""))
        try:
            verdict = summarize(JUDGE_PROMPT.format(
                query=query, source=Path(source).name, passage=content[:2200]))
        except Exception as e:
            print(f"    judge failed for chunk {cid}: {e}", file=sys.stderr)
            continue
        rel = 1 if "IRRELEVANT" not in verdict.upper() and "RELEVANT" in verdict.upper() else 0
        verdict_cache[key] = rel
        out[cid] = rel
    return out


def main():
    ap = argparse.ArgumentParser(description="Pool and judge real queries")
    ap.add_argument("--queries", default=str(EVAL_DIR / "real-queries.jsonl"))
    ap.add_argument("--out", default=str(EVAL_DIR / "qrels-real.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    queries = [json.loads(l)["query"] for l in open(args.queries) if l.strip()]
    if args.limit:
        queries = queries[: args.limit]

    out_path = Path(args.out)
    verdict_cache = {}
    started = time.time()
    written = 0

    with psycopg.connect(DB_DSN) as conn:
        register_vector(conn)
        with conn.cursor() as cur, out_path.open("w") as f:
            for i, q in enumerate(queries, 1):
                vector = embed(q)
                runs, pool = pool_candidates(cur, q, vector)
                verdicts = judge(cur, q, pool, verdict_cache)
                f.write(json.dumps({
                    "query": q,
                    "runs": {k: v for k, v in runs.items()},
                    "judgments": {str(cid): rel for cid, rel in verdicts.items()},
                }) + "\n")
                f.flush()
                written += 1
                rel_n = sum(verdicts.values())
                print(f"  [{i}/{len(queries)}] pooled {len(pool)}, {rel_n} relevant — {q[:52]}",
                      file=sys.stderr)

    print(f"wrote {written} judged queries -> {out_path} in {time.time()-started:.0f}s",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
