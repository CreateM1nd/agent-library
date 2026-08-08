#!/usr/bin/env python3
"""(Re)build the corpus lexeme-frequency table used by the keyword lane.

search.py picks which query terms to search on by how rare they are, so those
frequencies have to reflect the corpus as it currently stands. This is cheap --
about 11 seconds over 70,000 passages -- and belongs immediately after
ingestion, since a new file's distinctive vocabulary is precisely what the
keyword lane most wants to know about.

Safe to run while searches are in flight: the table is rebuilt under a
different name and swapped in one transaction, so a query never sees it
half-built or missing.

If this has not run since new files were ingested, search still works --
unknown lexemes are treated as maximally rare rather than dropped (see
_rare_terms in search.py). Stale means slightly worse ranking, never silence.
"""
import sys
import time

import psycopg

from db import DB_DSN


def main():
    started = time.time()
    conn = psycopg.connect(DB_DSN)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS rag_library_lexeme_df_new")
            cur.execute("""
                CREATE TABLE rag_library_lexeme_df_new AS
                SELECT word AS lexeme, ndoc, nentry
                FROM ts_stat($$SELECT to_tsvector('english', content)
                               FROM rag_library_chunks WHERE summary_level = 0$$)
            """)
            cur.execute("CREATE UNIQUE INDEX ix_lexeme_df_new ON rag_library_lexeme_df_new (lexeme)")
            cur.execute("SELECT count(*) FROM rag_library_lexeme_df_new")
            n = cur.fetchone()[0]
            if n == 0:
                conn.rollback()
                print("refusing to swap in an empty frequency table", file=sys.stderr)
                return 1
            cur.execute("DROP TABLE IF EXISTS rag_library_lexeme_df")
            cur.execute("ALTER TABLE rag_library_lexeme_df_new RENAME TO rag_library_lexeme_df")
            cur.execute("ALTER INDEX ix_lexeme_df_new RENAME TO ix_lexeme_df")
        conn.commit()
        print(f"rebuilt: {n:,} lexemes in {time.time() - started:.0f}s", file=sys.stderr)
        return 0
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
