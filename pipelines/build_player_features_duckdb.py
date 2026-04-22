from __future__ import annotations

from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "data" / "nba.duckdb"
SQL_PATH = REPO_ROOT / "sql" / "build_player_features.sql"

REQUIRED_COLUMNS = {"player_id", "game_id", "points", "rebounds", "assists"}
FEATURE_COLUMNS = {
    "avg_pts_last5",
    "avg_pts_last10",
    "avg_reb_last5",
    "avg_ast_last5",
    "avg_pts_season",
}


def _table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND table_name = ?
        """,
        [table_name],
    ).fetchone()
    return bool(row and row[0] > 0)


def _table_columns(conn: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    rows = conn.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'main'
          AND table_name = ?
        """,
        [table_name],
    ).fetchall()
    return {str(row[0]) for row in rows}


def validate_source(conn: duckdb.DuckDBPyConnection) -> None:
    if not _table_exists(conn, "player_game_stats"):
        raise RuntimeError("Required source table 'player_game_stats' was not found.")

    present = _table_columns(conn, "player_game_stats")
    missing = sorted(REQUIRED_COLUMNS.difference(present))
    if missing:
        raise RuntimeError(
            f"Source table 'player_game_stats' is missing required columns: {missing}"
        )


def build_player_features(conn: duckdb.DuckDBPyConnection) -> None:
    if not SQL_PATH.is_file():
        raise FileNotFoundError(f"SQL script not found: {SQL_PATH}")
    sql = SQL_PATH.read_text(encoding="utf-8")
    conn.execute(sql)


def validate_output(conn: duckdb.DuckDBPyConnection) -> tuple[int, int]:
    if not _table_exists(conn, "player_features"):
        raise RuntimeError("Expected output table 'player_features' was not created.")

    present = _table_columns(conn, "player_features")
    missing = sorted(FEATURE_COLUMNS.difference(present))
    if missing:
        raise RuntimeError(
            f"Output table 'player_features' is missing expected feature columns: {missing}"
        )

    row_count = conn.execute("SELECT COUNT(*) FROM player_features").fetchone()[0]
    null_count = conn.execute("""
        SELECT COUNT(*)
        FROM player_features
        WHERE avg_pts_last5 IS NULL
           OR avg_pts_last10 IS NULL
           OR avg_reb_last5 IS NULL
           OR avg_ast_last5 IS NULL
           OR avg_pts_season IS NULL
        """).fetchone()[0]
    return int(row_count), int(null_count)


def main() -> None:
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"DuckDB database not found: {DB_PATH}")

    print(f"Connecting to DuckDB: {DB_PATH}")
    conn = duckdb.connect(str(DB_PATH))
    try:
        print("Validating source table...")
        validate_source(conn)

        print("Building player_features table from SQL...")
        build_player_features(conn)

        row_count, null_count = validate_output(conn)
        if null_count != 0:
            raise RuntimeError(
                f"Output validation failed: found {null_count} NULL feature rows."
            )
        print(
            f"Built player_features with {row_count} model-ready rows (0 NULL feature rows)."
        )
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: player feature pipeline failed: {exc}")
        raise SystemExit(1)
