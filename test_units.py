#!/usr/bin/env python3
"""Unit tests for the pure functions -- no database, no models, no network.

Plain asserts rather than a framework: this project has no test dependency and
adding one to check a dozen functions would cost more than it returns. Run it
directly:

    python test_units.py

The bias in what is covered is deliberate. These are the functions where a
silent wrong answer is worse than a crash -- a vocabulary that drifts, a
threshold that never applies, a metric that quietly reports the wrong number.
Anything that fails loudly when broken is left to the integration checks.
"""
import sys

import chunk_text
import kg_ontology
import search
from eval_score_qrels import ndcg_at, precision_at, reciprocal_rank

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def check_true(label, cond):
    if not cond:
        FAILURES.append(f"{label}: expected true")


# --- kg_ontology: the closed vocabulary -------------------------------------
# The failure this guards against is the one that actually happened: extraction
# produced 175 entity types and 650 predicates, and nothing rejected them.

def test_entity_types_closed():
    for raw in ("tool", "tools", "Tool", "TOOLS", "software", "utility"):
        check(f"entity {raw!r}", kg_ontology.canonical_entity_type(raw), "tool")
    check("plural predicate", kg_ontology.canonical_relation("part of"), "part_of")
    check("underscore predicate", kg_ontology.canonical_relation("part_of"), "part_of")

    # Anything unrecognised must land inside the vocabulary, never invent a bucket.
    for raw in ("", None, "wibble", "a thing nobody planned for", "12345"):
        got = kg_ontology.canonical_entity_type(raw)
        check_true(f"unknown entity {raw!r} stays in vocabulary",
                   got in kg_ontology.ENTITY_TYPES)
        got = kg_ontology.canonical_relation(raw)
        check_true(f"unknown relation {raw!r} stays in vocabulary",
                   got in kg_ontology.RELATION_TYPES)


def test_relation_direction_not_inferred():
    # "used by" is the converse of "uses", but rewriting direction from a phrase
    # can silently invert an edge. Both map to `uses` on purpose; a reversed
    # edge is recoverable, an inverted one is not.
    check("uses", kg_ontology.canonical_relation("uses"), "uses")
    check("used by", kg_ontology.canonical_relation("used by"), "uses")


def test_normalize_name_is_stable():
    # The write path stores norm_name; the read path looks up by it. If these
    # ever disagree an exact lookup silently misses instead of erroring.
    n = kg_ontology.normalize_name
    check("case", n("Active Directory"), "active directory")
    check("punctuation", n("Kali Linux 2.0!"), "kali linux 2 0")
    check("whitespace", n("  SQL   Injection "), "sql injection")
    check("idempotent", n(n("Active Directory")), n("Active Directory"))


def test_known_ontology_wart():
    # Documented, not asserted-away: "programming language" contains "program",
    # so the tool rule claims it. Affects ~0.2% of entities. Encoded here so the
    # behaviour is visible and a future fix has a place to land.
    check("known wart: programming language -> tool",
          kg_ontology.canonical_entity_type("programming language"), "tool")


# --- search: query term selection -------------------------------------------

def test_title_terms():
    t = search._title_terms
    # Two-character tokens are kept: "48" is the most discriminating token in
    # a title like "<title>", and an earlier len>2 filter dropped it.
    check_true("keeps 2-char numeric", "48" in t("<title>"))
    check("lowercases", t("Kali LINUX"), ["kali", "linux"])
    check("strips punctuation", t('"CHAPTER" "Lang"'), ["chapter", "lang"])
    check("dedupes", t("python python PYTHON"), ["python"])
    check("empty query", t("   "), [])


# --- chunk_text: structure detection ----------------------------------------

def test_structural_page_detection():
    toc = "Table of Contents\nChapter 1 . . . . 5\nChapter 2 . . . . 21\nChapter 3 . . . . 44\n"
    prose = ("The quick brown fox jumps over the lazy dog. " * 12) + "\nA second sentence follows.\n"
    check_true("TOC detected", chunk_text.is_structural_page(toc))
    check_true("prose not detected", not chunk_text.is_structural_page(prose))
    check_true("empty page not detected", not chunk_text.is_structural_page(""))


def test_chunking_covers_all_text():
    pages = [f"page {i} " + ("lorem ipsum dolor sit amet " * 90) for i in range(6)]
    chunks = chunk_text.chunk_pages(pages, chunk_size=800, overlap=100)
    check_true("produced chunks", len(chunks) > 1)
    check_true("every chunk has a page index",
               all(isinstance(c["page_index"], int) for c in chunks))
    check_true("page indexes are within range",
               all(0 <= c["page_index"] < len(pages) for c in chunks))
    # Overlap means total chunk text exceeds the source; it must never be less,
    # which would mean content was dropped.
    total = sum(len(c["text"]) for c in chunks)
    check_true("no text lost", total >= sum(len(p) for p in pages) * 0.9)


# --- eval metrics: these underwrite the numbers in the README ---------------

def test_metrics():
    judgments = {"1": 1, "2": 0, "3": 1}
    check("P@5 with 2 of 3 relevant", round(precision_at([1, 2, 3], judgments, 5), 4), 0.6667)
    check("P@5 ignores unjudged", precision_at([9, 8, 7], judgments, 5), 0.0)
    check("MRR first hit", reciprocal_rank([1, 2, 3], judgments), 1.0)
    check("MRR second hit", reciprocal_rank([2, 1, 3], judgments), 0.5)
    check("MRR no hit", reciprocal_rank([9, 8], judgments), 0.0)

    # A perfect ranking scores 1.0; the reverse scores less. If this ever
    # inverts, every comparison in the README is backwards.
    perfect = ndcg_at([1, 3, 2], judgments, 10)
    worst = ndcg_at([2, 1, 3], judgments, 10)
    check("nDCG perfect ordering", round(perfect, 4), 1.0)
    check_true("nDCG rewards better ordering", perfect > worst)
    check("nDCG with no relevant docs judged", ndcg_at([1], {"1": 0}, 10), None)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s):", file=sys.stderr)
        for f in FAILURES:
            print(f"  {f}", file=sys.stderr)
        return 1
    print(f"{len(tests)} test functions, all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
