#!/usr/bin/env python3
import json
import re
import os
import statistics
import sys
import time
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from embed_client import embed
from rerank import rerank

from db import DB_DSN


# Ordering casts to halfvec (float16) so the HNSW halfvec indexes are used.
# Measured 2026-08-08 on the real corpus: index 296MB -> 148MB, recall@10 vs the
# float32 index 100% with rank-1 identical across 20 probes, warm p90 1.07ms ->
# 0.58ms. The size is the point, not the latency -- shared_buffers is 128MB, so
# the float32 index could never be resident, and page cache here competes
# directly with Ollama's ~35GB of resident models.
#
# The `similarity` value in the SELECT stays float32 on purpose: it is reported,
# not ranked on (RRF fuses by rank position), so there is no reason to round it.
RRF_K = 60  # a commonly cited default for Reciprocal Rank Fusion
CANDIDATE_POOL_SIZE = 30

# Circuit breaker state, added 2026-08-04 -- a real diagnostic pass on this
# file found zero error handling for degenerate results (empty candidate
# pool, all-identical rerank scores). A single bad query is normal (nothing
# relevant in the library); the same failure shape repeating across
# consecutive real queries is a systemic signal (Ollama down, DB down, a
# broken rerank model) that a one-off empty result can't be told apart from
# without tracking state across calls -- so this persists a small counter
# on disk, same principle as the queue-runner's circuit breaker, scaled down
# to a synchronous per-query function instead of a batch loop.
CIRCUIT_STATE_FILE = Path(__file__).parent / ".search-circuit-state.json"
CIRCUIT_BREAKER_THRESHOLD = 3
UNIFORM_SCORE_STDEV_THRESHOLD = 0.01  # near-zero spread across a real pool is suspicious, not meaningful

# RAPTOR summary nodes are retrieved in their own lane rather than competing
# with verbatim passages in one ranking. Measured on the real corpus: the best
# summary for a broad methodology query scored 0.658 cosine against 0.745 for
# raw passages, so summaries lost the candidate cutoff and never reached the
# reranker at all; when forced to compete head-to-head the cross-encoder still
# ranked every passage above every summary. That is expected rather than a
# defect -- passages share literal vocabulary with the query while summaries
# are abstractive, and ms-marco rerankers are trained on passage relevance.
# Scoring a 1800-char passage against a book-level synthesis with one function
# compares different granularities, so the two are ranked separately and both
# returned, letting the caller pick the altitude the question needs.
SUMMARY_CANDIDATE_POOL_SIZE = 15
DEFAULT_SUMMARY_RESULTS = 2


# Title lane, added 2026-08-08. Real queries against this corpus are dominated
# by known-item lookups -- "<title> <author>", '"CHAPTER" "<author>"' -- and both existing lanes handled them badly:
#
#   * the full-text index covers `content` only, so a filename was not
#     searchable at all;
#   * plainto_tsquery ANDs every term, so "<author>" (absent from the
#     filename "<title>.pdf") made the whole query match nothing,
#     even though five of seven terms matched.
#
# Measured before this existed: searching that exact title returned a
# unrelated file first and the correct one third, with the top three
# scores within 2% of each other -- fusion was barely discriminating.
#
# So: OR the terms, rank files by how much of the title they match, then order
# within a matched file by vector similarity, which answers "the most relevant
# passage from the book you named" rather than "some passage from it".
TITLE_LANE_LIMIT = 15

# Weighted RRF. With RRF_K=60 every lane's rank-1 hit scores 1/61 = 0.0164, so
# an exact filename match tied with an ordinary semantic hit and lost on
# insertion order -- searching "<title> <author>" returned a
# unrelated file above the correct one. A filename matching several
# query terms is a far stronger signal of intent than one lane's top guess, so
# it is worth 2x: enough to beat any single-lane hit, not enough to beat a
# candidate that both the vector and full-text lanes independently ranked
# first. Prose queries are unaffected -- the lane requires >= 2 distinct query
# terms in a filename before it returns anything at all.
TITLE_LANE_WEIGHT = 2.0
_TERM_RE = re.compile(r"[^a-z0-9]+")


def _title_terms(query):
    """Distinct alphanumeric terms from a free-text query.

    Length >= 2, not > 2: the first version dropped "48", which is the single
    most discriminating token in "<title>". Two-character numbers
    and initials carry real signal in titles."""
    terms = [_TERM_RE.sub("", w.lower()) for w in query.split()]
    return list(dict.fromkeys(w for w in terms if len(w) >= 2))


def _title_tsquery(query):
    """OR-joined tsquery, or None if nothing usable. Terms are stripped to
    alphanumerics because to_tsquery treats punctuation as operators and would
    raise a syntax error on a quoted query."""
    return " | ".join(_title_terms(query)) or None


# Ranked by how many DISTINCT query terms the filename contains, not by
# ts_rank. Measured: for "<title> <author>", ts_rank scored
# three unrelated titles at an identical 0.01520 -- it does not discriminate between short
# titles matching different numbers of terms, so the ordering was arbitrary.
# Raw overlap count does exactly what a known-item lookup wants: the file whose
# name accounts for most of what you typed.
TITLE_SQL = """
    SELECT id, source_path, page_index, content
    FROM rag_library_chunks
    WHERE summary_level = 0 AND source_path = ANY(%s)
    ORDER BY embedding::halfvec(768) <=> %s::halfvec(768)
    LIMIT %s
"""

# Filename lexemes, cached per process. There are 257 distinct filenames and
# they change only when a file is ingested, but the first two attempts at this
# did the stemming inside the per-query SQL: once per chunk (70,801 tsvector
# calls, 671ms) and then via a DISTINCT CTE that the planner materialised and
# re-joined (1736ms). Fetching them once and intersecting in Python is a few
# milliseconds and does the identical comparison, because the stemming still
# happens in Postgres -- just once per file rather than once per query.
_FILE_LEXEMES = None


def _file_lexemes(cur):
    global _FILE_LEXEMES
    if _FILE_LEXEMES is None:
        cur.execute("""
            SELECT source_path,
                   tsvector_to_array(to_tsvector('english',
                       regexp_replace(source_path, '[^a-zA-Z0-9]+', ' ', 'g')))
            FROM (SELECT DISTINCT source_path FROM rag_library_chunks
                  WHERE summary_level = 0) f
        """)
        _FILE_LEXEMES = [(path, set(lex)) for path, lex in cur.fetchall()]
    return _FILE_LEXEMES


def _matching_files(cur, terms, min_hits=2, limit=3):
    """Filenames sharing at least min_hits stemmed lexemes with the query,
    most overlap first."""
    cur.execute("SELECT tsvector_to_array(to_tsvector('english', %s))",
                (" ".join(terms),))
    qlex = set(cur.fetchone()[0] or [])
    if not qlex:
        return []
    scored = [(len(qlex & lex), path) for path, lex in _file_lexemes(cur)]
    scored = [(h, p) for h, p in scored if h >= min_hits]
    scored.sort(reverse=True)
    return [p for _, p in scored[:limit]]


# Corpus lexeme frequencies, built by build_lexeme_df.py and refreshed nightly.
# Used to pick which query terms are worth searching on, modelled on RAGflow's
# term-weighting in rag/nlp/query.py -- it sorts terms by weight and applies
# minimum_should_match=0.6 rather than demanding every term. Postgres has no
# minimum_should_match, so the same intent is expressed as: AND only the rarest
# terms, and relax progressively if that finds nothing.
#
# Measured on this corpus: one query's common term appears in 5,887 passages,
# its rare term in 206.
# ANDing all of "<chapter> <topic> <author>" matched zero passages; ANDing
# just its rare terms matches few enough to score quickly and lands in the
# right book.
FTS_MAX_TERMS = 6          # rarest N terms considered; beyond that adds noise
FTS_COMMON_DF_RATIO = 0.10  # a lexeme in >10% of passages carries no signal

RARITY_SQL = """
    SELECT l.lexeme, COALESCE(d.ndoc, 0) AS ndoc
    FROM unnest(tsvector_to_array(to_tsvector('english', %s))) AS l(lexeme)
    LEFT JOIN rag_library_lexeme_df d ON d.lexeme = l.lexeme
    ORDER BY COALESCE(d.ndoc, 0) ASC
"""


_CORPUS_SIZE = None


def _corpus_size(cur):
    """Passage count, fetched once per process. Counting 70k rows on every
    search cost more than the search -- and it is only used to compute a 10%
    ceiling, so a count that drifts by a nightly ingest changes nothing."""
    global _CORPUS_SIZE
    if _CORPUS_SIZE is None:
        cur.execute("SELECT count(*) FROM rag_library_chunks WHERE summary_level = 0")
        _CORPUS_SIZE = cur.fetchone()[0]
    return _CORPUS_SIZE


def _rare_terms(cur, query, corpus_size):
    """Query lexemes, rarest first, with corpus-wide stopwords dropped."""
    cur.execute(RARITY_SQL, (query,))
    rows = cur.fetchall()
    ceiling = corpus_size * FTS_COMMON_DF_RATIO
    # ndoc == 0 means the lexeme is ABSENT FROM THE TABLE, not absent from the
    # corpus -- which is exactly what a newly ingested file's distinctive words
    # look like until the nightly rebuild runs. Dropping them would discard the
    # rarest, most discriminating terms in a new book, so an unknown lexeme is
    # treated as maximally rare instead.
    kept = [lex for lex, ndoc in rows if ndoc <= ceiling]
    # If every term is common, keep the rarest few anyway -- a weak keyword
    # lane still beats an empty one, and the vector lane is there to disagree.
    if not kept:
        kept = [lex for lex, _ in rows]
    return kept[:FTS_MAX_TERMS]


KEYWORD_SQL = """
    SELECT id, source_path, page_index, content,
           ts_rank(to_tsvector('english', content), to_tsquery('english', %s)) AS rank
    FROM rag_library_chunks
    WHERE summary_level = 0
      AND to_tsvector('english', content) @@ to_tsquery('english', %s)
    ORDER BY rank DESC
    LIMIT %s
"""


def keyword_lane(cur, query):
    """AND over the rarest query terms, relaxing until something matches.

    Lives as a function so the evaluation harness measures THIS, not a copy.
    The first eval run after this lane changed still reported the old
    full-text numbers, because eval_run.py held its own plainto_tsquery -- a
    harness quietly scoring a pipeline nobody runs is worse than no harness.
    """
    rare = _rare_terms(cur, query, _corpus_size(cur))
    for cutoff in range(len(rare), 0, -1):
        tsq = " & ".join(rare[:cutoff])
        if not tsq:
            break
        cur.execute(KEYWORD_SQL, (tsq, tsq, CANDIDATE_POOL_SIZE))
        rows = cur.fetchall()
        if rows:
            return rows
    return []


def _read_circuit_state():
    if not CIRCUIT_STATE_FILE.exists():
        return {"consecutive_degenerate": 0}
    try:
        return json.loads(CIRCUIT_STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {"consecutive_degenerate": 0}


def _write_circuit_state(state):
    try:
        CIRCUIT_STATE_FILE.write_text(json.dumps(state))
    except OSError:
        pass  # best-effort only -- never let state tracking break a real query


def _note_query_outcome(degenerate, reason=""):
    state = _read_circuit_state()
    if degenerate:
        state["consecutive_degenerate"] = state.get("consecutive_degenerate", 0) + 1
        if state["consecutive_degenerate"] >= CIRCUIT_BREAKER_THRESHOLD:
            print(
                f"CIRCUIT BREAKER: {state['consecutive_degenerate']} consecutive degenerate "
                f"search results ({reason}) -- this looks systemic (Ollama/DB down, broken "
                f"rerank model?), not just queries with no real match. Check the underlying "
                f"services before trusting further search results.",
                file=sys.stderr,
            )
    else:
        state["consecutive_degenerate"] = 0
    _write_circuit_state(state)


def search(query, top_k=5, summary_results=DEFAULT_SUMMARY_RESULTS):
    """Hybrid retrieval:
    1. Vector search (meaning-based) + keyword search (exact-match-based),
       combined via Reciprocal Rank Fusion -- needed because pure vector
       search can miss short, literal queries. Verified against a real
       case where "table of contents" scored only ~0.44 similarity
       against the actual table-of-contents chunk, since that chunk is
       mostly bare numbers with little natural language to embed well.
       Measured best-of-four on both a synthetic and a real-query eval.
    2. A separate summary lane appends the best RAPTOR summary nodes, ranked
       among themselves rather than against passages.

    3. Cross-encoder reranking over the fused pool. Both lanes and both evals
       agree this is the best configuration -- but only since the keyword lane
       was fixed; see the comment at that stage for why it was briefly removed.

    Returns (source_path, page_index, content, score, kind) tuples, where kind
    is "passage" for verbatim book text or "summary" for a synthesized node.
    Both kinds carry a cross-encoder logit (unbounded, higher is better), but
    they are ranked within their own lane -- compare scores within a kind."""
    # An empty or stopword-only query has nothing to embed: Ollama returns an
    # empty embeddings list and indexing it raises IndexError deep in the
    # client. Fail closed here instead, where the caller can see why.
    if not query or not query.strip():
        return []
    vector = embed(query)
    # plainto_tsquery ANDs every term, so the keyword lane returns NOTHING
    # unless one passage contains every word typed. Measured on 19 real queries
    # from this system's own logs, 10 of 19 matched zero passages -- more than
    # half of all searches were silently running on the vector lane alone while
    # appearing to be hybrid.
    #
    # Fixing that by ORing every term outright was tried and reverted: it made
    # the median query 261ms -> 4820ms, because OR matches a large fraction of
    # the corpus and ts_rank must then score all of it. AND stays the fast path;
    # the OR query runs only when AND found nothing, where the alternative is
    # not a slower answer but no answer at all.
    fts_or_query = " | ".join(_title_terms(query)) or None
    with psycopg.connect(DB_DSN) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, source_path, page_index, content,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM rag_library_chunks
                WHERE summary_level = 0
                ORDER BY embedding::halfvec(768) <=> %s::halfvec(768)
                LIMIT %s
                """,
                (vector, vector, CANDIDATE_POOL_SIZE),
            )
            vector_rows = cur.fetchall()

            keyword_rows = keyword_lane(cur, query)

            # Third lane: files whose NAME matches the query. Only runs when
            # the query yields usable terms, so it costs nothing on queries
            # that are pure prose.
            title_rows = []
            terms = _title_terms(query)
            if len(terms) >= 2:
                files = _matching_files(cur, terms)
                if files:
                    cur.execute(TITLE_SQL, (files, vector, TITLE_LANE_LIMIT))
                    title_rows = cur.fetchall()

    # Circuit breaker check 1: empty candidate pool. A single query with no
    # real match is normal; tracked below as part of the consecutive-failure
    # count rather than treated as an error here, since raising would break
    # legitimate "nothing in the library matches this" cases.
    if not vector_rows and not keyword_rows and not title_rows:
        _note_query_outcome(degenerate=True, reason="empty candidate pool from both vector and keyword search")
        return []

    scores = {}
    rows_by_id = {}
    for rank, row in enumerate(vector_rows, start=1):
        chunk_id = row[0]
        rows_by_id[chunk_id] = row
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (RRF_K + rank)
    for rank, row in enumerate(keyword_rows, start=1):
        chunk_id = row[0]
        rows_by_id.setdefault(chunk_id, row)
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (RRF_K + rank)
    for rank, row in enumerate(title_rows, start=1):
        chunk_id = row[0]
        rows_by_id.setdefault(chunk_id, row)
        scores[chunk_id] = scores.get(chunk_id, 0) + TITLE_LANE_WEIGHT / (RRF_K + rank)

    # Cross-encoder reranking over the whole fused pool (not a top_k slice --
    # a candidate the first stage under-ranked is exactly what reranking exists
    # to rescue).
    #
    # This stage was REMOVED earlier on 2026-08-08 and RESTORED the same day.
    # Worth recording both numbers, because the reversal is the lesson:
    #
    #                        P@5     nDCG@10   MRR
    #   rrf (before)        62.9%    0.728     0.821
    #   + rerank (before)   54.3%    0.609     0.667   <- removed on this
    #   rrf (after)         56.2%    0.630     0.724
    #   + rerank (after)    58.8%    0.713     0.802   <- best, restored
    #
    # Nothing about the reranker changed between those measurements. What
    # changed was upstream: the keyword lane had been silently returning
    # NOTHING for 10 of 19 real queries (plainto_tsquery ANDs every term), so
    # the pool being reranked was almost pure vector search. Reordering bad
    # candidates cannot produce good results. Once _rare_terms fixed the
    # keyword lane, the same reranker became the best configuration on both
    # the synthetic and the real-query evals.
    #
    # The measurement was not wrong; it inherited a defect upstream of it.
    # Re-derive with eval_run.py and eval_score_qrels.py.
    pool_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    candidate_texts = [rows_by_id[cid][3] for cid in pool_ids]
    rerank_scores = rerank(query, candidate_texts) if pool_ids else []

    # Circuit breaker: near-zero variance across a real pool means the rerank
    # model failed silently and returned a constant, not that every candidate
    # is equally relevant.
    if len(rerank_scores) >= 5:
        spread = statistics.pstdev(rerank_scores)
        if spread < UNIFORM_SCORE_STDEV_THRESHOLD:
            _note_query_outcome(
                degenerate=True,
                reason=f"rerank scores suspiciously uniform (stdev={spread:.5f} across {len(rerank_scores)} candidates)",
            )
        else:
            _note_query_outcome(degenerate=False)
    else:
        _note_query_outcome(degenerate=False)

    reranked = sorted(zip(pool_ids, rerank_scores), key=lambda pair: pair[1], reverse=True)
    results = [
        (rows_by_id[cid][1], rows_by_id[cid][2], rows_by_id[cid][3], rerank_score, "passage")
        for cid, rerank_score in reranked[:top_k]
    ]
    results.extend(search_summaries(query, vector, summary_results))
    return results


def search_summaries(query, vector, limit):
    """Rank RAPTOR summary nodes among themselves and return the best few.

    Deliberately a separate lane from the passage search: see the comment on
    SUMMARY_CANDIDATE_POOL_SIZE for the measurements showing summaries lose
    both the vector cutoff and the cross-encoder against verbatim passages,
    which would otherwise make the whole summary tree unreachable. Reranking
    still happens, just within the summary set, so the best summary wins
    against other summaries rather than against a different granularity."""
    if limit <= 0:
        return []
    with psycopg.connect(DB_DSN) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_path, page_index, content
                FROM rag_library_chunks
                WHERE summary_level > 0
                ORDER BY embedding::halfvec(768) <=> %s::halfvec(768)
                LIMIT %s
                """,
                (vector, SUMMARY_CANDIDATE_POOL_SIZE),
            )
            rows = cur.fetchall()
    if not rows:
        return []
    scores = rerank(query, [row[2] for row in rows])
    ranked = sorted(zip(rows, scores), key=lambda pair: pair[1], reverse=True)
    return [
        (row[0], row[1], row[2], score, "summary")
        for row, score in ranked[:limit]
    ]


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "how do you download a file with requests"
    print(f"query: {query!r}\n")
    for source_path, page_index, content, score, kind in search(query):
        print(f"[{score:.4f}] ({kind}) {source_path} (page/section {page_index})")
        print(" ", content[:250].replace("\n", "\n  "))
        print()


if __name__ == "__main__":
    main()


def find_files(pattern, limit=30):
    """Files in the corpus whose name matches `pattern`. The corpus inventory,
    not a content search.

    Exists because the agent had no way to answer "do you have book X?" and was
    reasoning from the tool description's topic list instead -- which happens to
    work for an obviously out-of-domain title and cannot work for a plausible
    one. Asked for a book that is not here, content search returns its closest
    guesses and nothing signals absence.

    Matches on substring OR stemmed lexeme overlap, so a half-remembered title
    or a wrong separator still lands: the corpus really does contain files whose
    names differ from their titles only by hyphens.
    """
    with psycopg.connect(DB_DSN) as conn, conn.cursor() as cur:
        files = _file_lexemes(cur)
        needle = (pattern or "").strip().lower()
        if not needle:
            return {"total": len(files), "matched": len(files),
                    "files": sorted(p for p, _ in files)[:limit]}
        terms = _title_terms(needle)
        qlex = set()
        if terms:
            cur.execute("SELECT tsvector_to_array(to_tsvector('english', %s))",
                        (" ".join(terms),))
            qlex = set(cur.fetchone()[0] or [])
        # Threshold scales with query length -- the same intent as
        # Elasticsearch's minimum_should_match, which Postgres has no operator
        # for. Two fixes were needed here, both found by tests:
        #
        #   `if hits:`        any single overlap. A 3-word query for an absent
        #                     work matched 21 of 257 files on "guide" alone.
        #   `min(2, len)`     better, but a 7-word query still only needed two
        #                     overlaps, so nonsense with two common words passed.
        #
        # Now: one term needs one hit, two or three need two, and beyond that
        # half the query must match. A real title query survives because its
        # extra words (author, subtitle) are a minority of the terms; nonsense
        # does not, because its overlaps are incidental.
        import math
        n = len(qlex)
        min_hits = 1 if n == 1 else max(2, math.ceil(n * 0.5))
        scored = []
        for path, lex in files:
            hits = len(qlex & lex)
            substring = needle in path.lower()
            if substring:
                hits += 10  # a literal substring beats any amount of stem overlap
            if hits >= min_hits or substring:
                scored.append((hits, path))
        scored.sort(reverse=True)
        return {"total": len(files), "matched": len(scored),
                "files": [p for _, p in scored[:limit]]}
