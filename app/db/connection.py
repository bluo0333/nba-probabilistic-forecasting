from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "nba.duckdb"


def get_db_path() -> Path:
    return Path(os.getenv("NBA_DUCKDB_PATH", str(DEFAULT_DB_PATH)))


@contextmanager
def get_connection() -> Iterator[duckdb.DuckDBPyConnection]:
    db_path = get_db_path()
    if not db_path.is_file():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

    conn = duckdb.connect(str(db_path))
    try:
        yield conn
    finally:
        conn.close()
