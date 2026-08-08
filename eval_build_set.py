#!/usr/bin/env python3
"""Build a fixed, reusable evaluation set for retrieval.

The ground-truth problem: measuring retrieval needs queries with known-correct
answers, and hand-labelling 70,801 passages is not happening. The standard
alternative is to invert the problem -- sample a passage, have a model write a
question that passage answers, and treat that passage as the gold answer. The
label comes for free because you already know which chunk produced the question.

WHAT THIS BIASES, stated plainly because it decides how the numbers should be
read: a question generated from a passage inherits that passage's vocabulary,
which flatters lexical matching (the full-text lane) more than a real user's
phrasing would. The generation prompt below pushes against it -- paraphrase,
no rare strings copied verbatim -- but it cannot remove it. Treat the absolute
scores as optimistic and the *differences between configurations* as the real
signal, since every configuration is measured against the identical set.

The set is written once and committed. Regenerating it changes the yardstick,
so a run against a new set is not comparable to an older run -- the filename
records the seed and size for that reason.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import psycopg

from db import DB_DSN
from summarize_client import summarize

OUT_DIR = Path(__file__).parent / "eval"
DEFAULT_SEED = 20260808
DEFAULT_SIZE = 120

# One question per sampled passage. Kept deliberately short: a long generated
# question drifts into summarising the passage, which makes retrieval trivially
# easy and measures nothing.
PROMPT = (
    "Below is a passage from a technical book.\n\n"
    "Write ONE natural question that this passage answers -- the kind of "
    "question someone would actually type into a search box, not an exam "
    "question about the text.\n\n"
    "Rules:\n"
    "- Paraphrase. Do NOT copy distinctive phrases, rare terms, or exact "
    "wording from the passage.\n"
    "- Under 15 words.\n"
    "- No preamble, no quotes, output the question only.\n"
    "- If the passage is boilerplate (index, copyright, table of contents, "
    "code with no prose), output exactly: SKIP\n\n"
    "PASSAGE:\n{passage}\n"
)

MIN_CONTENT_CHARS = 600  # below this a chunk is usually front/back matter


def sample_chunks(cur, size, seed):
    """Stratified by file: at most a couple per source, so the set is not
    dominated by whichever book happens to have the most chunks."""
    cur.execute(
        """
        SELECT id, source_path, content
        FROM (
            SELECT id, source_path, content,
                   row_number() OVER (PARTITION BY source_path ORDER BY md5(id::text || %s)) AS rn
            FROM rag_library_chunks
            WHERE summary_level = 0 AND length(content) >= %s
        ) t
        WHERE rn <= 2
        """,
        (str(seed), MIN_CONTENT_CHARS),
    )
    rows = cur.fetchall()
    random.Random(seed).shuffle(rows)
    return rows[: size * 2]  # oversample; SKIPs get dropped below


def main():
    ap = argparse.ArgumentParser(description="Generate a retrieval evaluation set")
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"queries-seed{args.seed}-n{args.size}.jsonl"
    if out_path.exists():
        print(f"{out_path} already exists -- refusing to overwrite a committed "
              f"yardstick. Delete it deliberately if you mean to.", file=sys.stderr)
        return 1

    with psycopg.connect(DB_DSN) as conn, conn.cursor() as cur:
        candidates = sample_chunks(cur, args.size, args.seed)

    print(f"{len(candidates)} candidate passages sampled; generating questions", file=sys.stderr)
    written = skipped = 0
    with out_path.open("w") as f:
        for chunk_id, source_path, content in candidates:
            if written >= args.size:
                break
            try:
                q = summarize(PROMPT.format(passage=content[:2500])).strip()
            except Exception as e:
                print(f"  generation failed for {chunk_id}: {e}", file=sys.stderr)
                continue
            q = q.strip().strip('"').strip()
            if not q or q.upper().startswith("SKIP") or len(q.split()) > 25:
                skipped += 1
                continue
            # gold_source (the filename) is deliberately NOT written. Scoring
            # uses gold_chunk_id alone, so recording the filename would publish
            # a list of what the corpus contains for no functional gain.
            f.write(json.dumps({
                "query": q,
                "gold_chunk_id": chunk_id,
            }) + "\n")
            f.flush()  # so a long run's progress is visible in the file, not just at the end
            written += 1
            if written % 20 == 0:
                print(f"  {written}/{args.size}", file=sys.stderr)

    print(f"wrote {written} queries ({skipped} skipped) -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
