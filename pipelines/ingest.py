from pathlib import Path

import duckdb


TABLES_TO_INGEST = ("game", "line_score", "team", "other_stats")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sqlite_path = repo_root / "data" / "raw" / "kaggle" / "nba.sqlite"
    duckdb_path = repo_root / "data" / "nba.duckdb"

    if not sqlite_path.is_file():
        raise FileNotFoundError(f"SQLite source not found: {sqlite_path}")

    print(f"Connecting to DuckDB: {duckdb_path}")
    conn = duckdb.connect(str(duckdb_path))
    in_transaction = False

    try:
        print(f"Attaching SQLite source: {sqlite_path}")
        conn.execute("INSTALL sqlite;")
        conn.execute("LOAD sqlite;")
        conn.execute(
            f"ATTACH '{str(sqlite_path)}' AS kaggle (TYPE sqlite);"
        )

        conn.execute("BEGIN TRANSACTION;")
        in_transaction = True

        for table in TABLES_TO_INGEST:
            print(f"Ingesting table: {table}")
            conn.execute(f"DROP TABLE IF EXISTS {table};")
            conn.execute(f"CREATE TABLE {table} AS SELECT * FROM kaggle.{table};")

        conn.execute("COMMIT;")
        in_transaction = False
        conn.execute("DETACH kaggle;")
        print("Ingestion complete.")

    except Exception:
        if in_transaction:
            conn.execute("ROLLBACK;")
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"ERROR: Ingestion failed: {exc}")
        raise SystemExit(1)
