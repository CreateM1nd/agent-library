#!/usr/bin/env python3
"""Integration checks -- these need a populated database and a running Ollama.

Separate from test_units.py because they are slow, environment-dependent, and
will fail on a fresh clone with nothing ingested. That is not a defect; a test
for "does retrieval work against a real corpus" cannot be hermetic.

    python test_integration.py

They assert on invariants rather than exact results, because the corpus grows
and rankings move. A test that pins today's top hit would fail every time a
file is added, which trains people to ignore it.
"""
import sys

import psycopg

from search import DB_DSN, find_files, search

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def check_true(label, cond, detail=""):
    if not cond:
        FAILURES.append(f"{label}: expected true {detail}".rstrip())


def corpus_ready():
    try:
        with psycopg.connect(DB_DSN) as c, c.cursor() as cur:
            cur.execute("SELECT count(*) FROM rag_library_chunks WHERE summary_level = 0")
            return cur.fetchone()[0] > 0
    except Exception:
        return False


# --- absence: the regression this suite exists for --------------------------

def test_absence_is_reliable():
    """find_files must return 0 for a work that is not present, regardless of
    how many words the query has.

    REGRESSION (2026-08-08): the threshold was `if hits:` -- any single stemmed
    lexeme in common. `min_hits=2` was declared as a parameter and never
    applied. A three-word query for an absent work matched 21 of 257 files on
    the word "guide" alone. Single-word queries passed only by accident, which
    is why manual checking missed it for hours.
    """
    absent = [
        "hitchhiker",
        "hitchhiker guide galaxy",
        "advanced malware forensics handbook",
        "the complete guide to underwater basket weaving",
        "guide handbook manual introduction",   # all common filename words
    ]
    for q in absent:
        r = find_files(q)
        check(f"absent query {q!r}", r["matched"], 0)


def test_presence_is_found():
    """A file that exists must be findable from its own filename tokens."""
    with psycopg.connect(DB_DSN) as c, c.cursor() as cur:
        cur.execute("""SELECT source_path FROM rag_library_chunks
                       WHERE summary_level = 0 GROUP BY 1 ORDER BY count(*) DESC LIMIT 1""")
        path = cur.fetchone()[0]
    import re
    terms = [w for w in re.split(r"[^A-Za-z0-9]+", path) if len(w) >= 3][:4]
    r = find_files(" ".join(terms))
    check_true("a real file finds itself", path in r["files"],
               f"({len(terms)} terms, {r['matched']} matched)")
    check_true("inventory total is the whole corpus", r["total"] > 0)


# --- retrieval invariants ---------------------------------------------------

def test_search_returns_both_kinds():
    results = search("penetration testing methodology", top_k=5, summary_results=2)
    kinds = {k for _, _, _, _, k in results}
    check_true("returns passages", "passage" in kinds)
    check_true("summary lane runs", "summary" in kinds, f"(kinds={kinds})")


def test_search_ranking_is_ordered():
    results = search("buffer overflow exploitation", top_k=8, summary_results=0)
    scores = [s for _, _, _, s, _ in results]
    check_true("scores descend", scores == sorted(scores, reverse=True), f"({scores[:3]})")
    check_true("no duplicate chunks", len({c for _, _, c, _, _ in results}) == len(results))


def test_known_item_beats_topic_drift():
    """A query naming a file should return that file, not merely its subject.
    This is what the title lane exists for; without it an exact title returned
    an unrelated file first."""
    with psycopg.connect(DB_DSN) as c, c.cursor() as cur:
        cur.execute("""SELECT source_path FROM rag_library_chunks
                       WHERE summary_level = 0 GROUP BY 1
                       HAVING count(*) > 100 ORDER BY 1 LIMIT 1""")
        path = cur.fetchone()[0]
    import re
    terms = [w for w in re.split(r"[^A-Za-z0-9]+", path) if len(w) >= 3][:5]
    results = search(" ".join(terms), top_k=5, summary_results=0)
    from_file = sum(1 for s, _, _, _, _ in results if s == path)
    check_true("named file dominates its own title query", from_file >= 3,
               f"({from_file}/5 from the named file)")


def test_empty_and_degenerate_queries():
    for q in ("", "   ", "the of and a"):
        try:
            search(q, top_k=3, summary_results=0)
        except Exception as e:
            FAILURES.append(f"query {q!r} raised {type(e).__name__}: {e}")


def main():
    if not corpus_ready():
        print("no corpus available -- skipping integration checks", file=sys.stderr)
        return 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
        except Exception as e:
            FAILURES.append(f"{t.__name__} raised {type(e).__name__}: {e}")
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s):", file=sys.stderr)
        for f in FAILURES:
            print(f"  {f}", file=sys.stderr)
        return 1
    print(f"{len(tests)} integration tests, all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
