from __future__ import annotations

from pathlib import Path

import duckdb


DB_RELATIVE_PATH = Path("data") / "nba.duckdb"

REQUIRED_TABLES: dict[str, set[str]] = {
    "play_by_play": {
        "game_id",
        "eventnum",
        "eventmsgtype",
        "player1_id",
        "player2_id",
    },
    "common_player_info": {"person_id", "display_first_last", "position", "team_id"},
    "player": {"id", "full_name"},
}

PLAYER_GAME_STATS_SQL_TEMPLATE = """
CREATE OR REPLACE TABLE player_game_stats AS
WITH normalized_pbp AS (
    -- Keep only required columns and normalize types/values for stat derivation.
    SELECT
        CAST(game_id AS BIGINT) AS game_id,
        CAST(eventnum AS BIGINT) AS eventnum,
        CAST(eventmsgtype AS INTEGER) AS eventmsgtype,
        NULLIF(CAST(player1_id AS BIGINT), 0) AS player1_id, -- scorer / rebounder
        NULLIF(CAST(player2_id AS BIGINT), 0) AS player2_id, -- assister
        UPPER(COALESCE({description_expr}, '')) AS description
    FROM play_by_play
    WHERE game_id IS NOT NULL
),
deduped_pbp AS (
    -- Guard against duplicate events to avoid double counting stats.
    SELECT game_id, eventnum, eventmsgtype, player1_id, player2_id, description
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY game_id, eventnum
                ORDER BY eventnum
            ) AS row_rank
        FROM normalized_pbp
    )
    WHERE row_rank = 1
),
point_events AS (
    -- Points credited to player1:
    -- eventmsgtype=1 => made FG (3 if description contains 3PT, else 2)
    -- eventmsgtype=3 => made FT only (exclude MISS)
    SELECT
        game_id,
        player1_id AS player_id,
        CASE
            WHEN eventmsgtype = 1 AND description LIKE '%3PT%' THEN 3
            WHEN eventmsgtype = 1 THEN 2
            WHEN eventmsgtype = 3 AND description NOT LIKE '%MISS%' THEN 1
            ELSE 0
        END AS points,
        0 AS rebounds,
        0 AS assists
    FROM deduped_pbp
    WHERE eventmsgtype IN (1, 3)
      AND player1_id IS NOT NULL
),
rebound_events AS (
    -- Rebounds credited to player1 on rebound events.
    SELECT
        game_id,
        player1_id AS player_id,
        0 AS points,
        1 AS rebounds,
        0 AS assists
    FROM deduped_pbp
    WHERE eventmsgtype = 4
      AND player1_id IS NOT NULL
),
assist_events AS (
    -- Assists credited to player2 for made field goals.
    SELECT
        game_id,
        player2_id AS player_id,
        0 AS points,
        0 AS rebounds,
        1 AS assists
    FROM deduped_pbp
    WHERE eventmsgtype = 1
      AND player2_id IS NOT NULL
),
stat_events AS (
    -- One row per stat contribution, then aggregate at (game_id, player_id).
    SELECT * FROM point_events WHERE points > 0
    UNION ALL
    SELECT * FROM rebound_events
    UNION ALL
    SELECT * FROM assist_events
),
aggregated_stats AS (
    SELECT
        game_id,
        player_id,
        SUM(points) AS points,
        SUM(rebounds) AS rebounds,
        SUM(assists) AS assists
    FROM stat_events
    GROUP BY game_id, player_id
),
player_metadata AS (
    SELECT
        CAST(person_id AS BIGINT) AS player_id,
        display_first_last AS player_name,
        position,
        CAST(team_id AS BIGINT) AS team_id
    FROM common_player_info
),
player_game_enriched AS (
    SELECT
        s.player_id,
        s.game_id,
        s.points,
        s.rebounds,
        s.assists,
        COALESCE(m.player_name, p.full_name) AS player_name,
        m.position,
        m.team_id,
        {order_game_date_expr} AS game_date
    FROM aggregated_stats s
    LEFT JOIN player_metadata m
        ON s.player_id = m.player_id
    LEFT JOIN player p
        ON s.player_id = CAST(p.id AS BIGINT)
    {game_join_clause}
)
SELECT
    player_id,
    game_id,
    points,
    rebounds,
    assists,
    AVG(points) OVER (
        PARTITION BY player_id
        ORDER BY game_date NULLS LAST, game_id
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS rolling_points_5,
    AVG(rebounds) OVER (
        PARTITION BY player_id
        ORDER BY game_date NULLS LAST, game_id
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS rolling_rebounds_5,
    AVG(assists) OVER (
        PARTITION BY player_id
        ORDER BY game_date NULLS LAST, game_id
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS rolling_assists_5,
    game_date,
    player_name,
    position,
    team_id
FROM player_game_enriched;
"""


def _table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    row = conn.execute(
        """
        SELECT COUNT(*) AS table_count
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


def validate_sources(conn: duckdb.DuckDBPyConnection) -> set[str]:
    play_by_play_columns: set[str] | None = None
    for table_name, required_columns in REQUIRED_TABLES.items():
        if not _table_exists(conn, table_name):
            raise RuntimeError(
                f"Required source table '{table_name}' was not found in DuckDB."
            )

        present_columns = _table_columns(conn, table_name)
        missing = sorted(required_columns.difference(present_columns))
        if missing:
            raise RuntimeError(
                f"Source table '{table_name}' is missing required columns: {missing}"
            )

        if table_name == "play_by_play":
            play_by_play_columns = present_columns

    if play_by_play_columns is None:
        raise RuntimeError("Could not inspect play_by_play table columns.")
    return play_by_play_columns


def resolve_description_expression(play_by_play_columns: set[str]) -> str:
    if "description" in play_by_play_columns:
        return "description"

    text_sources = [
        col
        for col in ["homedescription", "neutraldescription", "visitordescription"]
        if col in play_by_play_columns
    ]
    if text_sources:
        return ", ".join(text_sources)

    raise RuntimeError(
        "play_by_play needs either a 'description' column or at least one of "
        "'homedescription', 'neutraldescription', 'visitordescription'."
    )


def resolve_game_date_sql(conn: duckdb.DuckDBPyConnection) -> tuple[str, str]:
    if not _table_exists(conn, "game"):
        return ("", "CAST(NULL AS TIMESTAMP)")

    game_columns = _table_columns(conn, "game")
    if {"game_id", "game_date"}.issubset(game_columns):
        game_join_clause = """
    LEFT JOIN (
        SELECT game_id, game_date
        FROM (
            SELECT
                CAST(game_id AS BIGINT) AS game_id,
                TRY_CAST(game_date AS TIMESTAMP) AS game_date,
                ROW_NUMBER() OVER (
                    PARTITION BY CAST(game_id AS BIGINT)
                    ORDER BY TRY_CAST(game_date AS TIMESTAMP) DESC NULLS LAST
                ) AS row_rank
            FROM game
        )
        WHERE row_rank = 1
    ) g
        ON s.game_id = g.game_id
"""
        return (game_join_clause, "g.game_date")

    return ("", "CAST(NULL AS TIMESTAMP)")


def build_player_game_stats_sql(
    play_by_play_columns: set[str],
    game_join_clause: str,
    order_game_date_expr: str,
) -> str:
    description_expr = resolve_description_expression(play_by_play_columns)
    return PLAYER_GAME_STATS_SQL_TEMPLATE.format(
        description_expr=description_expr,
        game_join_clause=game_join_clause,
        order_game_date_expr=order_game_date_expr,
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    db_path = repo_root / DB_RELATIVE_PATH
    if not db_path.is_file():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

    print(f"Connecting to DuckDB: {db_path}")
    conn = duckdb.connect(str(db_path))
    try:
        print("Validating required source tables...")
        play_by_play_columns = validate_sources(conn)
        game_join_clause, order_game_date_expr = resolve_game_date_sql(conn)

        print("Building player_game_stats table...")
        conn.execute(
            build_player_game_stats_sql(
                play_by_play_columns=play_by_play_columns,
                game_join_clause=game_join_clause,
                order_game_date_expr=order_game_date_expr,
            )
        )

        row_count = conn.execute("SELECT COUNT(*) FROM player_game_stats").fetchone()[0]
        print(f"Built player_game_stats with {row_count} rows.")
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: player game stats pipeline failed: {exc}")
        raise SystemExit(1)
