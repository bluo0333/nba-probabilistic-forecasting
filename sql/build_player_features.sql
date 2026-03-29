-- Build model-ready player features from historical player_game_stats.
-- Leakage guard: every rolling/expanding feature excludes the current game.

CREATE OR REPLACE TABLE player_features AS
WITH base_stats AS (
    -- Normalize types and keep only required columns at player-game grain.
    SELECT
        CAST(player_id AS BIGINT) AS player_id,
        CAST(game_id AS BIGINT) AS game_id,
        CAST(points AS DOUBLE) AS points,
        CAST(rebounds AS DOUBLE) AS rebounds,
        CAST(assists AS DOUBLE) AS assists
    FROM player_game_stats
    WHERE player_id IS NOT NULL
      AND game_id IS NOT NULL
),
deduped_stats AS (
    -- Defensive dedupe in case duplicate player-game rows exist upstream.
    SELECT
        player_id,
        game_id,
        points,
        rebounds,
        assists
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY player_id, game_id
                ORDER BY game_id
            ) AS row_rank
        FROM base_stats
    )
    WHERE row_rank = 1
),
windowed_features AS (
    SELECT
        player_id,
        game_id,
        points,
        rebounds,
        assists,

        -- Number of games before current game for this player.
        ROW_NUMBER() OVER (
            PARTITION BY player_id
            ORDER BY game_id
        ) - 1 AS games_played_so_far,

        -- Rolling averages excluding current row (no leakage).
        AVG(points) OVER (
            PARTITION BY player_id
            ORDER BY game_id
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS avg_pts_last5,
        COUNT(points) OVER (
            PARTITION BY player_id
            ORDER BY game_id
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS cnt_pts_last5,
        AVG(points) OVER (
            PARTITION BY player_id
            ORDER BY game_id
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
        ) AS avg_pts_last10,
        COUNT(points) OVER (
            PARTITION BY player_id
            ORDER BY game_id
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
        ) AS cnt_pts_last10,
        AVG(rebounds) OVER (
            PARTITION BY player_id
            ORDER BY game_id
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS avg_reb_last5,
        COUNT(rebounds) OVER (
            PARTITION BY player_id
            ORDER BY game_id
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS cnt_reb_last5,
        AVG(assists) OVER (
            PARTITION BY player_id
            ORDER BY game_id
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS avg_ast_last5,
        COUNT(assists) OVER (
            PARTITION BY player_id
            ORDER BY game_id
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS cnt_ast_last5,

        -- Expanding season-to-date average excluding current row.
        AVG(points) OVER (
            PARTITION BY player_id
            ORDER BY game_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS avg_pts_season
    FROM deduped_stats
),
model_ready AS (
    -- Keep rows only when all required rolling features are fully available.
    SELECT
        player_id,
        game_id,
        points,
        rebounds,
        assists,
        games_played_so_far,
        avg_pts_last5,
        avg_pts_last10,
        avg_reb_last5,
        avg_ast_last5,
        avg_pts_season
    FROM windowed_features
    WHERE avg_pts_last5 IS NOT NULL
      AND avg_pts_last10 IS NOT NULL
      AND avg_reb_last5 IS NOT NULL
      AND avg_ast_last5 IS NOT NULL
      AND avg_pts_season IS NOT NULL
      AND cnt_pts_last5 = 5
      AND cnt_pts_last10 = 10
      AND cnt_reb_last5 = 5
      AND cnt_ast_last5 = 5
)
SELECT
    player_id,
    game_id,
    points,
    rebounds,
    assists,
    games_played_so_far,
    avg_pts_last5,
    avg_pts_last10,
    avg_reb_last5,
    avg_ast_last5,
    avg_pts_season
FROM model_ready;
