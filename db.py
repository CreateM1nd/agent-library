#!/usr/bin/env python3
"""Single source of truth for the Postgres connection string.

Read entirely from the environment. Nothing here has a committed password --
set LIBRARY_DB_DSN, or the individual DB_* variables, before running anything.

    export LIBRARY_DB_DSN="host=127.0.0.1 port=5432 dbname=library user=library password=..."

The pieces are kept separate as a fallback because the container image only
needs to override the host (DB_HOST=host.containers.internal) while everything
else stays the same.
"""
import os

DB_DSN = os.environ.get("LIBRARY_DB_DSN") or " ".join(
    f"{key}={value}"
    for key, value in (
        ("host", os.environ.get("DB_HOST", "127.0.0.1")),
        ("port", os.environ.get("DB_PORT", "5432")),
        ("dbname", os.environ.get("DB_NAME", "library")),
        ("user", os.environ.get("DB_USER", "library")),
        ("password", os.environ.get("DB_PASSWORD", "")),
    )
)
