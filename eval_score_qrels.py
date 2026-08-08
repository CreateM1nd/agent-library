#!/usr/bin/env python3
"""Score retrieval configurations against pooled relevance judgments.

eval_run.py scores the synthetic set, where each query has exactly one correct
answer (the passage it was generated from), so recall@k is the natural metric.
Real queries do not work that way: "The Art Of War" has no single right chunk,
it has some number of relevant ones, and how many differs per query. Metrics
that assume one gold answer would be measuring the wrong shape.

So this reports:

  P@5      precision at 5 -- of the top five, what fraction were judged
           relevant. The metric that matches how results are actually consumed.
  nDCG@10  rewards putting relevant results higher, and normalises per query
           so a query with 8 relevant passages does not swamp one with 1.
  MRR      how high the first relevant result lands.

Judgments come from eval_judge_pool.py, which pools every configuration's
candidates before judging, so no configuration is scored against labels
derived from its own output. Anything unjudged counts as irrelevant -- the
standard pooling assumption, and the reason these numbers compare systems
rather than describe the corpus.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent / "eval"
CONFIGS = ["vector", "fts", "rrf", "full", "rrf+title"]


def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at(ranked, judgments, k):
    gains = [judgments.get(str(cid), 0) for cid in ranked[:k]]
    ideal = sorted(judgments.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else None


def precision_at(ranked, judgments, k):
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(judgments.get(str(cid), 0) for cid in top) / len(top)


def reciprocal_rank(ranked, judgments):
    for i, cid in enumerate(ranked, start=1):
        if judgments.get(str(cid), 0):
            return 1.0 / i
    return 0.0


def main():
    ap = argparse.ArgumentParser(description="Score configurations against pooled judgments")
    ap.add_argument("qrels", nargs="?", default=str(EVAL_DIR / "qrels-real.jsonl"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.qrels) if l.strip()]
    if not rows:
        print("no judgments found", file=sys.stderr)
        return 1

    # A query where nothing was judged relevant cannot discriminate between
    # configurations -- every one scores zero. Counted and excluded, because
    # leaving them in drags every score toward zero equally and hides the
    # differences this exists to measure.
    usable = [r for r in rows if any(r["judgments"].values())]
    barren = len(rows) - len(usable)

    acc = {c: {"p5": [], "ndcg": [], "rr": []} for c in CONFIGS}
    for r in usable:
        j = r["judgments"]
        for c in CONFIGS:
            ranked = r["runs"].get(c, [])
            acc[c]["p5"].append(precision_at(ranked, j, 5))
            n = ndcg_at(ranked, j, 10)
            if n is not None:
                acc[c]["ndcg"].append(n)
            acc[c]["rr"].append(reciprocal_rank(ranked, j))

    print(f"\nreal-query eval: {len(usable)} scoreable of {len(rows)} judged "
          f"({barren} had no relevant passage in the pool)")
    print(f"\n{'config':8} {'P@5':>8} {'nDCG@10':>9} {'MRR':>8}")
    print("-" * 36)
    for c in CONFIGS:
        a = acc[c]
        print(f"{c:8} {statistics.mean(a['p5'])*100:7.1f}% "
              f"{statistics.mean(a['ndcg']):9.3f} {statistics.mean(a['rr']):8.3f}")

    rel_counts = [sum(r["judgments"].values()) for r in rows]
    print(f"\nrelevant passages per query: median {statistics.median(rel_counts):.0f}, "
          f"max {max(rel_counts)}, zero for {barren}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
