# agent-library — a document corpus an AI agent can actually use

![License](https://img.shields.io/badge/license-MIT-black?style=flat-square)
![Python](https://img.shields.io/badge/python-3.12-black?style=flat-square)
![Postgres](https://img.shields.io/badge/postgres-15%20%2B%20pgvector-black?style=flat-square)
![Local](https://img.shields.io/badge/inference-fully%20local-black?style=flat-square)

**Status:** in daily use on one machine since August 2026. Graph extraction runs
nightly and coverage grows per run; the numbers below are live figures, not
targets.

Not a library in the packaging sense — a *library* of documents, and the
retrieval an agent needs to use one: hybrid search, cross-encoder reranking,
RAPTOR hierarchical summarisation, a queryable knowledge graph, and an inventory
lookup so the agent can say what it does *not* have. Served over MCP. No API
keys, no external services; the models run on the same machine as the database.

It is a reference implementation, not a framework. There is no config file and
nothing is pluggable — the value is in the reasoning recorded beside each
decision, and in two evaluations that show what happens when the reasoning is
wrong.

Built in August 2026 on a single mini PC (AMD Ryzen AI Max+ 395, 128 GB unified
memory) running Ollama + PostgreSQL/pgvector.

## Why this exists

This is the library a self-hosted AI assistant actually reads from.

The assistant runs entirely on one machine — its models, its database, and this
corpus all share the same memory. That constraint shapes every decision here:
retrieval competes with the models for RAM, which is why the vector index is
half-precision; the interface is MCP because the assistant calls these as tools
rather than a human calling an API; and there is no cloud fallback, so a failure
is a wrong answer rather than a timeout.

It also explains why the evaluation set is built from real logs. The queries in
`eval_real_set.py` are what someone actually asked their assistant over a week —
mostly hunting for a specific book or chapter, rarely the well-formed questions
a benchmark assumes. Tuning for the second workload and shipping to the first is
how a system scores well and feels wrong.

## Where it stands

| | |
|---|---|
| Corpus | 257 files |
| Verbatim passages | 70,801 |
| RAPTOR summary nodes | 4,841, across 236 files |
| Graph entities / relations | 6,724 / 3,599 (extraction covers 107 files so far) |
| Embedding | `qwen3-embedding:0.6b`, 768 dims |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, CPU |
| Vector index | HNSW over a halfvec cast — 148 MB, 100% recall@10 vs float32 |
| Entity types / relation types | 14 / 11, closed vocabulary |
| Retrieval, real queries | P@5 58.8%, nDCG@10 0.713, MRR 0.802 |

## What it looks like

Asking the knowledge graph about a concept — what kind of thing it is, and what
the corpus connects it to:

```
>>> library_entity("Kerberos")

{"found": true, "name": "Kerberos", "type": "protocol",
 "mentions": 17, "source_count": 9}

  out  uses         GSS_NEGOTIATE   (protocol)
  in   related_to   Bloodhound      (tool)
  in   related_to   Rubeus.exe      (tool)
  in   targets      Hashcat         (tool)
  in   uses         DCSync          (technique)
```

Every type there — `protocol`, `tool`, `technique` — comes from the closed
vocabulary in section 2. Edge direction is preserved rather than inferred, and
the relations are aggregated across every file that asserts them.

Asking whether the corpus contains something at all, which is a different
question from searching it:

```
>>> library_files("hitchhiker")    ->  matched 0 of 257
>>> library_files("nextcloud")     ->  matched 1 of 257
```

A `matched` of 0 is a definitive negative. Content search cannot produce one:
ask it for something absent and it returns its closest guesses, which read
exactly like an answer. See section 6.

## Pipeline

```
PDF / EPUB
   │  extract_text.py          PyMuPDF, EbookLib
   │  ocr_extract.py           Tesseract fallback for image-only PDFs
   ▼
structure-aware chunking       chunk_text.py
   │                           TOC-like pages kept whole, prose sliding-window
   ▼
embed + store                  embed_client.py → store_book.py → Postgres
   │                           contextual prefix: filename prepended before embedding
   ▼
RAPTOR tree                    raptor_build.py
   │                           k-means clusters → LLM summary → embed → repeat
   ▼
knowledge graph                kg_extract.py + kg_ontology.py
   │                           entities/relations, closed vocabulary
   ▼
graph queries                  kg_query.py
                               entity lookup, single-hop traversal

query ──► search.py ──► MCP ──► agent
          vector + keyword (rarest terms) + filename → weighted RRF
          → cross-encoder rerank → results
          summaries ranked in a separate lane
          library_files answers "is it here at all?" separately
```

## Six things worth reading

### 1. The summaries were invisible for a week

`raptor_build.py` built 4,841 summary nodes. Search never returned a single one.

Two independent causes, both worth knowing about:

**The index.** An HNSW index finds the k nearest neighbours *first* and applies
the `WHERE` clause *after*. With passages outnumbering summaries about 15:1, a
`WHERE summary_level > 0` filter against the main index reliably matched
nothing in the global top-k. The fix is a partial index over just the summary
rows (`schema.sql`) — a different index, not a faster one.

**The ranking.** Even with candidates retrieved, summaries lost. Measured on
the real corpus: for a broad methodology query, the best summary scored 0.658
cosine against 0.745 for raw passages, and when forced to compete head-to-head
the cross-encoder ranked *every* passage above *every* summary. That is correct
behaviour, not a bug — passages share literal vocabulary with the query while
summaries are abstractive, and MS MARCO rerankers are trained on passage
relevance. Comparing an 1,800-character passage to a file-level synthesis with
one scoring function compares different granularities.

So summaries get their own retrieval lane, ranked among themselves, and both
are returned. The caller picks the altitude the question needs.

### 2. An ontology nobody enforced

The first extraction pass asked the model for entity types by naming examples
in the prompt. Across the 93 files extracted at that point it produced **175
distinct entity types and 650 distinct relation predicates** — including `tool`
(980 rows) alongside `tools` (325), and `part of` (71) alongside `part_of` (42).

This does not fail loudly. It returns a confidently incomplete answer: a query
filtering `type = 'tool'` silently missed a quarter of the tools.

The fix was a closed vocabulary enforced in code at insert time
(`kg_ontology.py`) with the original string preserved in `type_raw`, plus a
migration over existing rows:

| | before | after |
|---|---|---|
| Entity types | 175 | 14 |
| Relation predicates | 650 | 11 |
| Rows matching `type = 'tool'` | 980 | 1,411 |

Direction is deliberately *not* inferred during normalisation — `used by` and
`uses` both map to `uses`, because rewriting direction from a phrase can
silently invert an edge. A possibly-reversed edge is recoverable; an inverted
one is not.

The general lesson: a prompt is a request, code is the enforcement. The same
system's TypeScript-side vocabulary never drifted, because the compiler
rejected anything outside the union.

### 3. Hybrid retrieval, because vectors miss literal queries

Pure vector search failed a case that looked trivial: the query "table of
contents" scored ~0.44 against the actual table-of-contents chunk, because that
chunk is mostly bare numbers and dotted leaders with little natural language to
embed. Full-text search finds it instantly.

So stage one runs both lanes and fuses them with Reciprocal Rank Fusion, and
stage two reranks the *entire* fused pool rather than a top-k slice — a
candidate the first stage under-ranked is precisely the case reranking exists
to fix.

Related: chunking is structure-aware. Contiguous TOC-like pages are kept
together instead of being sliced by the fixed-size window, and that detection
is restricted to front matter, because an answer-key section deep in the back
of a textbook produces an identical "Chapter N + short lines + numbers"
signature. Density alone cannot tell them apart; position can.

### 4. A measurement that was right, and the decision from it that was wrong

Ranking here was originally tuned by reading results and judging them sensible.
`eval_run.py` replaced that with numbers: four configurations, one fixed query
set, gold answers known. It immediately said the cross-encoder reranker was
dead weight — so the reranker was removed.

That was a mistake, and finding out why is the useful part.

A second evaluation set was built from **real** queries, taken from the
deployment's own logs rather than generated. They looked nothing like the
synthetic ones:

```
<title> <author>
"CHAPTER" "<author>"
bash scripting basics
```

(Query examples are generalised throughout. The corpus is private, so titles and
authors are replaced with placeholders — the shapes are real, the works are not
named.)

Overwhelmingly known-item lookups — find *this* book, *this* chapter — where
the synthetic set was full-sentence questions. Scoring those needs pooled
relevance judgments (`eval_judge_pool.py`), since a real query has no single
gold answer: every configuration's candidates go into one pool, each is judged
once, and no configuration is scored against labels derived from its own output.

That set exposed the actual defect, and it was not in the reranker at all.
**`plainto_tsquery` ANDs every term**, so the keyword lane returned *nothing*
unless one passage contained every word typed — true for 10 of 19 real queries.
More than half of all searches had been running on the vector lane alone while
appearing to be hybrid.

Fixing it needed the idea RAGflow implements as `minimum_should_match`: match
most terms, not all. Postgres has no such operator, so the same intent became a
corpus-frequency table (`build_lexeme_df.py`) and a relaxation ladder — AND the
rarest terms, drop one and retry if nothing matches. On this corpus the common
term of one query appears in 5,887 passages and the rare one in 206, so the
rare term carries the search and the common one is noise.

Then the same reranker was re-measured on the same queries:

| | P@5 | nDCG@10 | MRR |
|---|--:|--:|--:|
| fused, before the keyword fix | 62.9% | 0.728 | 0.821 |
| + reranker, before | 54.3% | 0.609 | 0.667 |
| fused, after | 56.2% | 0.630 | 0.724 |
| **+ reranker, after** | **58.8%** | **0.713** | **0.802** |

Nothing about the reranker changed. It went from worst to best because the pool
it reorders stopped being garbage. **The measurement was never wrong; it
inherited a defect upstream of it.** Reordering bad candidates cannot produce
good results, and a benchmark cannot tell you that the thing it is measuring is
being starved.

Two smaller lessons from the same work, both recorded in comments where they
happened:

- A 6-query smoke run of this harness reported full-text beating vector, and
  reranking actively harmful. Both reversed at 120 queries. A confident table is
  easy to produce and easy to believe.
- After the keyword lane changed, the harness kept printing the *old* full-text
  numbers, because it held its own copy of the query. A harness quietly scoring
  a pipeline nobody runs is worse than having no harness. The lane now lives in
  one function that both import.

### 5. Known-item lookup needs its own lane

The real-query set showed most searches naming a specific work, and both
existing lanes handled that badly: the full-text index covered `content` only,
so a filename was not searchable at all.

A third lane matches the query against filenames, on stemmed lexemes rather
than raw strings — so a singular term in the query still matches a plural in the
filename, which a literal comparison misses. It fuses at 2x weight, and the
weight is load-bearing:
with `RRF_K = 60` every lane's rank-1 hit scores 1/61, so an exact filename
match tied with an ordinary semantic hit and lost on insertion order.

Getting it fast took three attempts, and the two failures are more instructive
than the fix. Computing the overlap in SQL ran the stemmer once per *chunk* —
70,801 times instead of 257 — at 671ms. Rewriting that as a `DISTINCT` CTE made
it *worse*, 1736ms, because the planner materialised and re-joined it. The
version that works caches 257 filenames once per process and intersects in
Python; Postgres still does the stemming, just once per file rather than once
per query.

### 6. An agent cannot prove absence with a search

Asked whether the corpus contained a particular work, the agent answered
correctly — and by reasoning from its tool description, which happened to list
the corpus topics. The work was obviously outside them. For a *plausible* title
the same reasoning has nothing to go on, and content search offers no help:
query for something absent and it returns its closest guesses, with no signal
distinguishing "here is what you asked for" from "here is the nearest thing I
have".

So absence needed its own tool. `library_files` matches against filenames and
returns `{"total", "matched", "files"}`; a `matched` of 0 is a definitive
negative, and the tool description says so explicitly, because the failure mode
is an agent softening a real absence into an approximation.

Worth noting what surfaced once it could actually check: searching for the
absent work returned passages from *other* files that quoted it. "The corpus
mentions this" and "the corpus contains this" are different claims, and
conflating them is exactly what makes an absence hard to prove.

## Running it

```bash
createdb library && psql library -f schema.sql
pip install -r requirements.txt

export LIBRARY_DIR=./library
export LIBRARY_DB_DSN="host=127.0.0.1 port=5432 dbname=library user=library password=..."

ollama pull qwen3-embedding:0.6b
ollama pull <a local chat model>       # summarisation + graph extraction,
                                       # set RAPTOR_CHAT_MODEL to match

python store_library.py                # ingest everything under LIBRARY_DIR
python raptor_build.py                 # build the summary tree
python kg_extract.py                   # extract entities/relations (incremental)
python search.py "your question here"
python kg_query.py "Active Directory"  # inspect the graph
python eval_run.py                     # score retrieval configurations

python test_units.py                   # no database needed
python test_integration.py             # needs a populated corpus; skips without one
```

Serve it to an agent:

```bash
podman build -t library-search .
podman run -p 8050:8080 -e DB_PASSWORD="$DB_PASSWORD" library-search
```

No credentials are committed. Everything reads from the environment (`db.py`).

## Reading order

If you only read three files: `search.py`, `kg_ontology.py`, `chunk_text.py`.

| File | |
|---|---|
| `search.py` | three retrieval lanes, weighted fusion, reranking, circuit breaker |
| `kg_ontology.py` | the closed vocabulary and why it is enforced in code |
| `chunk_text.py` | structure-aware chunking, TOC detection |
| `raptor_build.py` | clustering + summarisation, atomic per-file transactions |
| `kg_extract.py` | incremental extraction with a per-run chunk budget |
| `kg_query.py` | graph read path: name resolution, edge aggregation |
| `store_book.py` | contextual prefix before embedding, content-hash skip |
| `embed_client.py` | Matryoshka truncation 1024 → 768 |
| `reembed_library.py` | resumable migration between embedding models |
| `library_mcp_server.py` | the MCP surface: inventory, search, entity lookup, traversal |
| `eval_run.py` | scores four retrieval configurations against a fixed query set |
| `eval_build_set.py` | builds that set, and documents what it biases |
| `eval_real_set.py` | extracts real queries from a deployment's own logs |
| `eval_judge_pool.py` | pooled relevance judgments for queries with no gold answer |
| `eval_score_qrels.py` | P@5 / nDCG / MRR against those judgments |
| `build_lexeme_df.py` | corpus term-frequency table the keyword lane ranks on |
| `test_units.py` | pure-function tests; no database or models needed |
| `test_integration.py` | invariants against a real corpus; skips cleanly without one |

The comments are denser than usual and often record a measurement or a wrong
turn rather than restating the code. That is deliberate — most of the
non-obvious decisions here came from something being benchmarked and coming
back the opposite of what was expected.

## How this was built

Built with an AI coding assistant (Claude, via Claude Code) over about a week,
directed by me. I chose the architecture, decided what to build and in what
order, and rejected a fair amount of what came back; the assistant did most of
the typing and much of the benchmarking.

Several designs in here exist *because* a measurement contradicted an
assumption, and more than one feature was built, tested, and then removed once
it turned out to do more harm than good. Those decisions are recorded in the
comments alongside the code they shaped.

## Limitations

- **Graph coverage is partial.** 107 of 257 files extracted. Extraction is
  incremental and resumable, so coverage grows per run rather than all at once.
- **Graph access is read-only and shallow.** Entity lookup and single-hop
  traversal are exposed; multi-hop paths, subgraph extraction, and aggregate
  queries are not.
- **Only the synthetic query set ships.** The real-query set is generated from
  a deployment's private search history, so it is gitignored; `eval_real_set.py`
  rebuilds an equivalent one from your own logs. The real-query figures quoted
  above are therefore reported, not reproducible from this repo alone.
- **Test coverage is partial.** `test_units.py` covers the pure functions where
  a silent wrong answer is worse than a crash — the closed vocabulary, term
  selection, chunk-structure detection, the evaluation metrics. Ingestion,
  embedding, and RAPTOR construction are exercised only by running them.
- **Relevance judgments come from a local model, not a person.** A consistent
  rater, not a correct one, and only 16 of 19 real queries had any relevant
  passage in the pool at all.
- **Absence is only detectable by filename.** `library_files` answers "is this
  work here at all", but only against file names — a work present under an
  unrecognisable filename reads as absent. Content search still cannot signal
  absence on its own: a relevance threshold was tried and rejected, because
  measured against the judgments it silenced good results and passed bad ones.
- **Single machine.** Embedding, summarisation, reranking, and Postgres all
  share one box's memory. Model choice is constrained by what fits alongside
  everything else.

## License

MIT.
