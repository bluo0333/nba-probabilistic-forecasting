from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import duckdb

@contextmanager
def get_connection() -> Iterator[duckdb.DuckDBPyConnection]:
    conn = duckdb.connect()
    try:
        yield conn
    finally:
        conn.close()
