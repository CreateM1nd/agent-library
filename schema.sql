-- Schema for the local document RAG corpus.
-- Requires pgvector >= 0.7 for halfvec. Tested on PostgreSQL 15 + pgvector 0.8.4.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per chunk. RAPTOR summary nodes live in this same table rather than a
-- separate one: a summary is a chunk with summary_level > 0 and the ids of the
-- chunks it was built from. Keeping them together means one index, one search
-- path, and no join to answer "what is the best thing in this corpus for X".
CREATE TABLE rag_library_chunks (
    id                 bigserial PRIMARY KEY,
    content            text        NOT NULL,
    embedding          vector(768),
    source_path        text        NOT NULL,
    source_type        text        NOT NULL,   -- 'pdf' | 'epub'
    chunk_index        integer     NOT NULL,
    page_index         integer     NOT NULL,
    content_hash       text        NOT NULL,   -- lets re-ingest skip unchanged chunks
    created_at         timestamptz NOT NULL DEFAULT now(),
    updated_at         timestamptz NOT NULL DEFAULT now(),
    summary_level      integer     NOT NULL DEFAULT 0,  -- 0 = verbatim passage
    summary_source_ids bigint[],                        -- set only when summary_level > 0
    embed_model        text                             -- which model produced `embedding`
);

CREATE UNIQUE INDEX uq_rag_library_chunks_source_chunk
    ON rag_library_chunks (source_path, chunk_index);

CREATE INDEX ix_rag_library_chunks_source_path
    ON rag_library_chunks (source_path);

-- Keyword lane of the hybrid search.
CREATE INDEX ix_rag_library_chunks_content_fts
    ON rag_library_chunks USING gin (to_tsvector('english', content));

-- Vector lane, verbatim passages. Indexed on a halfvec (float16) cast rather
-- than the float32 column: measured on the real corpus, this halves the index
-- (296MB -> 148MB) at 100% recall@10 against the float32 equivalent, with the
-- rank-1 result identical across every probe. Queries must apply the same cast
-- or the planner will not use this index -- see search.py.
--
-- Size is the point, not speed. Warm query latency barely moves, because an
-- embedding call and a cross-encoder rerank dominate a search by two orders of
-- magnitude. What matters is that shared_buffers is smaller than the float32
-- index ever was, and on a single-box deployment page cache competes directly
-- with the local models' resident memory.
CREATE INDEX ix_rag_library_chunks_embedding_hnsw_half
    ON rag_library_chunks USING hnsw ((embedding::halfvec(768)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Vector lane, summaries only. This partial index is load-bearing, not an
-- optimisation: an HNSW index finds the k nearest neighbours FIRST and applies
-- the WHERE clause afterwards, so a summary_level > 0 filter against the main
-- index returns whatever survives the filter out of the global top-k -- which,
-- for a corpus where verbatim passages outnumber summaries ~15:1, is routinely
-- nothing. Summaries were unreachable until this index existed.
CREATE INDEX ix_rag_library_chunks_summary_hnsw_half
    ON rag_library_chunks USING hnsw ((embedding::halfvec(768)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE summary_level > 0;


-- Knowledge graph -----------------------------------------------------------

-- norm_name is the merge key (lowercased, punctuation-stripped); `name` keeps
-- the first-seen surface form for display. `type` is constrained to
-- kg_ontology.ENTITY_TYPES in application code -- see the note in that file on
-- why this is not a CHECK constraint.
CREATE TABLE rag_library_entities (
    id           bigserial   PRIMARY KEY,
    name         text        NOT NULL,
    norm_name    text        NOT NULL UNIQUE,
    type         text,
    type_raw     text,                          -- what the extractor originally said
    mentions     integer     NOT NULL DEFAULT 0,
    source_paths text[]      NOT NULL DEFAULT '{}',
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX ix_rag_library_entities_norm ON rag_library_entities (norm_name);

CREATE TABLE rag_library_relations (
    id          bigserial   PRIMARY KEY,
    from_id     bigint      NOT NULL REFERENCES rag_library_entities (id),
    rel         text        NOT NULL,
    to_id       bigint      NOT NULL REFERENCES rag_library_entities (id),
    rel_raw     text,                           -- what the extractor originally said
    source_path text        NOT NULL,
    mentions    integer     NOT NULL DEFAULT 1,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (from_id, rel, to_id, source_path)
);

CREATE INDEX ix_rag_library_relations_from ON rag_library_relations (from_id);
CREATE INDEX ix_rag_library_relations_to   ON rag_library_relations (to_id);

-- Resume state for the incremental extractor: which files are done, so a
-- nightly run picks up where the last one stopped.
CREATE TABLE rag_library_kg_progress (
    source_path    text        PRIMARY KEY,
    chunks_done    integer     NOT NULL DEFAULT 0,
    entities_found integer     NOT NULL DEFAULT 0,
    completed_at   timestamptz NOT NULL DEFAULT now()
);


-- Corpus term frequencies, rebuilt by build_lexeme_df.py after every ingest.
-- The keyword lane picks which query terms to search on by how rare they are,
-- which is this project's stand-in for Elasticsearch's minimum_should_match:
-- AND the rare terms, ignore the common ones. Postgres has no such operator.
--
-- Not a view: ts_stat over 70k passages takes ~11s, which is fine nightly and
-- not fine per query.
CREATE TABLE rag_library_lexeme_df (
    lexeme text PRIMARY KEY,
    ndoc   integer NOT NULL,  -- passages containing the lexeme
    nentry integer NOT NULL   -- total occurrences
);
-- Populate with:
--   INSERT INTO rag_library_lexeme_df
--   SELECT word, ndoc, nentry
--   FROM ts_stat($$SELECT to_tsvector('english', content)
--                  FROM rag_library_chunks WHERE summary_level = 0$$);
