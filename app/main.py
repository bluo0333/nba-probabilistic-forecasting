from __future__ import annotations

from datetime import date
import json
import math
from html import escape
from pathlib import Path
import sys
from typing import Any
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

import duckdb
import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines import predict_matchup as pm

DB_PATH = REPO_ROOT / "data" / "nba.duckdb"
MODEL_PATH = REPO_ROOT / "models" / "logistic_model.pkl"
HOST = "127.0.0.1"
PORT = 8000
EASTERN_CONFERENCE = {
    "ATL",
    "BKN",
    "BOS",
    "CHA",
    "CHI",
    "CLE",
    "DET",
    "IND",
    "MIA",
    "MIL",
    "NYK",
    "ORL",
    "PHI",
    "TOR",
    "WAS",
}
PLAYER_FEATURES_PATH = REPO_ROOT / "data" / "player_features.csv"
PLAYER_STATS_PATH = REPO_ROOT / "data" / "player_game_stats.csv"
COMMON_PLAYER_INFO_PATH = REPO_ROOT / "data" / "raw" / "kaggle" / "csv" / "common_player_info.csv"
PLAYER_PROP_MODEL_PATHS = {
    "points": REPO_ROOT / "models" / "points_model.pkl",
    "rebounds": REPO_ROOT / "models" / "rebounds_model.pkl",
    "3ps": REPO_ROOT / "models" / "threes_model.pkl",
}
PLAYER_PROP_MEAN_COLUMN_CANDIDATES = {
    "points": ["rolling_points_10", "points"],
    "rebounds": ["rolling_rebounds_10", "rebounds"],
    "3ps": ["rolling_3pm_10", "rolling_fg3m_10", "rolling_threes_10", "threes_made", "fg3m"],
}
PLAYER_PROP_STAT_COLUMN_CANDIDATES = {
    "points": ["points", "pts"],
    "rebounds": ["rebounds", "reb", "trb"],
    "3ps": ["threes_made", "fg3m", "three_pointers_made", "three_pointers"],
}
PLAYER_PROP_DEFAULT_STD = {"points": 6.0, "rebounds": 3.0, "3ps": 1.5}
PLAYER_PROP_TYPES = ("points", "rebounds", "3ps")
PLAYER_PROP_LABELS = {"points": "Points", "rebounds": "Rebounds", "3ps": "3PM"}
_PLAYER_PROP_CONTEXT: dict[str, Any] | None = None


def team_logo_url(team_id: str) -> str:
    return f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"


def team_conference(team_abbreviation: str) -> str:
    return "East" if team_abbreviation in EASTERN_CONFERENCE else "West"


def load_teams() -> list[dict[str, str]]:
    conn = duckdb.connect(str(DB_PATH))
    try:
        rows = conn.execute(
            """
            SELECT id, abbreviation, full_name
            FROM team
            ORDER BY abbreviation
            """
        ).fetchall()
        return [
            {
                "id": str(team_id),
                "abbreviation": str(abbrev),
                "full_name": str(full_name),
                "conference": team_conference(str(abbrev)),
                "logo_url": team_logo_url(str(team_id)),
            }
            for team_id, abbrev, full_name in rows
        ]
    finally:
        conn.close()


def format_display_date(value: date) -> str:
    return pd.Timestamp(value).strftime("%B %d, %Y").replace(" 0", " ")


def load_tonights_games(today: date | None = None) -> dict[str, Any]:
    target_date = today or date.today()
    conn = duckdb.connect(str(DB_PATH))
    try:
        query = """
            SELECT
                game_id,
                CAST(game_date AS DATE) AS game_date,
                season_type,
                team_id_away,
                team_abbreviation_away,
                team_name_away,
                pts_away,
                team_id_home,
                team_abbreviation_home,
                team_name_home,
                pts_home
            FROM game
            WHERE CAST(game_date AS DATE) = ?
            ORDER BY game_id
        """
        rows = conn.execute(query, [target_date]).fetchall()
        source = "today"
        source_date = target_date
        note = None

        if not rows:
            latest_row = conn.execute("SELECT MAX(CAST(game_date AS DATE)) FROM game").fetchone()
            latest_date = latest_row[0] if latest_row else None
            if latest_date is not None:
                rows = conn.execute(query, [latest_date]).fetchall()
                source = "latest"
                source_date = latest_date
                note = (
                    f"No games found for {target_date.isoformat()} in local data. "
                    f"Showing latest available slate."
                )
            else:
                note = "No games are available in the local dataset."

        games: list[dict[str, Any]] = []
        for (
            game_id,
            game_date_value,
            season_type,
            away_id,
            away_abbreviation,
            away_name,
            away_points,
            home_id,
            home_abbreviation,
            home_name,
            home_points,
        ) in rows:
            games.append(
                {
                    "game_id": str(game_id),
                    "game_date": str(game_date_value),
                    "season_type": str(season_type),
                    "away_abbreviation": str(away_abbreviation),
                    "away_name": str(away_name),
                    "away_logo_url": team_logo_url(str(away_id)),
                    "away_points": int(away_points) if away_points is not None else None,
                    "home_abbreviation": str(home_abbreviation),
                    "home_name": str(home_name),
                    "home_logo_url": team_logo_url(str(home_id)),
                    "home_points": int(home_points) if home_points is not None else None,
                }
            )

        return {
            "display_date": format_display_date(source_date),
            "source": source,
            "note": note,
            "games": games,
        }
    finally:
        conn.close()


def normalize_name(value: str) -> str:
    return str(value or "").strip().lower()


def american_to_implied_probability(odds: float) -> float:
    if odds == 0:
        raise ValueError("Odds cannot be zero.")
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def normal_cdf(value: float, mean: float, std_dev: float) -> float:
    if std_dev <= 0:
        return 1.0 if value >= mean else 0.0
    z = (value - mean) / (std_dev * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def pick_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _load_player_prop_context() -> dict[str, Any]:
    global _PLAYER_PROP_CONTEXT
    if _PLAYER_PROP_CONTEXT is not None:
        return _PLAYER_PROP_CONTEXT

    features_df: pd.DataFrame | None = None
    stats_df: pd.DataFrame | None = None
    models: dict[str, Any] = {}
    player_names: set[str] = set()

    if PLAYER_FEATURES_PATH.is_file():
        features_df = pd.read_csv(PLAYER_FEATURES_PATH)
        if "player_name" in features_df.columns:
            features_df["player_name"] = features_df["player_name"].astype(str).str.strip()
            features_df["player_name_norm"] = features_df["player_name"].map(normalize_name)
            player_names.update(features_df["player_name"].dropna().tolist())
        if "date" in features_df.columns:
            features_df["date"] = pd.to_datetime(features_df["date"], errors="coerce")

    if PLAYER_STATS_PATH.is_file():
        stats_df = pd.read_csv(PLAYER_STATS_PATH)
        if "player_name" in stats_df.columns:
            stats_df["player_name"] = stats_df["player_name"].astype(str).str.strip()
            stats_df["player_name_norm"] = stats_df["player_name"].map(normalize_name)
            player_names.update(stats_df["player_name"].dropna().tolist())
        if "date" in stats_df.columns:
            stats_df["date"] = pd.to_datetime(stats_df["date"], errors="coerce")

    if not player_names and COMMON_PLAYER_INFO_PATH.is_file():
        info = pd.read_csv(COMMON_PLAYER_INFO_PATH, usecols=["display_first_last"])
        values = info["display_first_last"].dropna().astype(str).str.strip()
        player_names.update([name for name in values if name])

    for prop_type, model_path in PLAYER_PROP_MODEL_PATHS.items():
        if model_path.is_file():
            try:
                models[prop_type] = joblib.load(model_path)
            except Exception:
                continue

    sorted_players = sorted(player_names)
    name_map = {normalize_name(name): name for name in sorted_players}
    _PLAYER_PROP_CONTEXT = {
        "features_df": features_df,
        "stats_df": stats_df,
        "models": models,
        "players": sorted_players,
        "name_map": name_map,
    }
    return _PLAYER_PROP_CONTEXT


def load_player_names() -> list[str]:
    context = _load_player_prop_context()
    return context["players"]


def _latest_feature_row(features_df: pd.DataFrame, player_norm: str) -> pd.Series | None:
    rows = features_df[features_df["player_name_norm"] == player_norm].copy()
    if rows.empty:
        return None
    order_cols = [col for col in ["date", "game_id"] if col in rows.columns]
    if order_cols:
        rows = rows.sort_values(order_cols, ascending=True)
    return rows.iloc[-1]


def _predict_mean_from_model(model: Any, row: pd.Series) -> float | None:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        feature_names = ["rolling_points_10", "rolling_assists_10", "rolling_rebounds_10", "rolling_minutes_10"]

    values: dict[str, float] = {}
    for feature_name in feature_names:
        if feature_name not in row.index:
            return None
        value = row.get(feature_name)
        if value is None or pd.isna(value):
            return None
        values[str(feature_name)] = float(value)

    frame = pd.DataFrame([values], columns=list(feature_names))
    return float(model.predict(frame)[0])


def _extract_stat_series(stats_df: pd.DataFrame, player_norm: str, prop_type: str) -> pd.Series | None:
    stat_col = pick_existing_column(stats_df, PLAYER_PROP_STAT_COLUMN_CANDIDATES[prop_type])
    if stat_col is None:
        return None
    rows = stats_df[stats_df["player_name_norm"] == player_norm].copy()
    if rows.empty:
        return None
    order_cols = [col for col in ["date", "game_id"] if col in rows.columns]
    if order_cols:
        rows = rows.sort_values(order_cols, ascending=True)
    series = pd.to_numeric(rows[stat_col], errors="coerce").dropna()
    if series.empty:
        return None
    return series


def predict_player_prop(player: str, prop_type: str, side: str, line: float, odds: float) -> dict[str, Any]:
    context = _load_player_prop_context()
    prop_type_norm = str(prop_type).strip().lower()
    side_norm = str(side).strip().lower()
    if prop_type_norm not in PLAYER_PROP_TYPES:
        raise ValueError("Line type must be one of: points, rebounds, 3ps.")
    if side_norm not in {"over", "under"}:
        raise ValueError("Side must be over or under.")

    player_norm = normalize_name(player)
    canonical_name = context["name_map"].get(player_norm)
    if not canonical_name:
        raise ValueError(f"Player '{player}' not found.")

    features_df = context["features_df"]
    stats_df = context["stats_df"]
    models = context["models"]
    predicted_mean: float | None = None
    mean_source = ""

    if features_df is not None:
        latest_row = _latest_feature_row(features_df, player_norm)
        if latest_row is not None:
            model = models.get(prop_type_norm)
            if model is not None:
                predicted_mean = _predict_mean_from_model(model, latest_row)
                if predicted_mean is not None:
                    mean_source = "model"
            if predicted_mean is None:
                mean_col = pick_existing_column(features_df, PLAYER_PROP_MEAN_COLUMN_CANDIDATES[prop_type_norm])
                if mean_col is not None:
                    value = latest_row.get(mean_col)
                    if value is not None and not pd.isna(value):
                        predicted_mean = float(value)
                        mean_source = f"rolling ({mean_col})"

    player_series = None if stats_df is None else _extract_stat_series(stats_df, player_norm, prop_type_norm)
    if predicted_mean is None and player_series is not None and len(player_series) > 0:
        predicted_mean = float(player_series.tail(10).mean())
        mean_source = "recent average (last 10)"

    if predicted_mean is None:
        raise ValueError(
            "Could not compute this player prop yet. Missing player feature/stat data. "
            "Run player ingestion/build/train pipelines first."
        )

    std_dev: float | None = None
    if player_series is not None and len(player_series) >= 2:
        std_dev = float(player_series.std(ddof=1))
    if (std_dev is None or not math.isfinite(std_dev) or std_dev <= 0) and stats_df is not None:
        global_col = pick_existing_column(stats_df, PLAYER_PROP_STAT_COLUMN_CANDIDATES[prop_type_norm])
        if global_col is not None:
            global_std = float(pd.to_numeric(stats_df[global_col], errors="coerce").dropna().std(ddof=1))
            if math.isfinite(global_std) and global_std > 0:
                std_dev = global_std

    if std_dev is None or not math.isfinite(std_dev) or std_dev <= 0:
        std_dev = PLAYER_PROP_DEFAULT_STD[prop_type_norm]

    prob_over = float(1.0 - normal_cdf(line, predicted_mean, std_dev))
    hit_probability = prob_over if side_norm == "over" else 1.0 - prob_over
    implied_probability = float(american_to_implied_probability(odds))
    edge = hit_probability - implied_probability

    return {
        "player": canonical_name,
        "line_type": prop_type_norm,
        "line_type_label": PLAYER_PROP_LABELS[prop_type_norm],
        "side": side_norm,
        "line": float(line),
        "odds": float(odds),
        "predicted_mean": predicted_mean,
        "std_dev": std_dev,
        "hit_probability": hit_probability,
        "implied_probability": implied_probability,
        "edge": edge,
        "mean_source": mean_source or "fallback",
    }


def predict(home: str, away: str, game_date_text: str | None) -> dict[str, Any]:
    conn = duckdb.connect(str(DB_PATH))
    try:
        model = joblib.load(MODEL_PATH)
        home_id = pm.resolve_team_id(conn, home)
        away_id = pm.resolve_team_id(conn, away)
        if home_id == away_id:
            raise ValueError("Home and away teams must be different.")

        teams = conn.execute(
            "SELECT id, full_name, abbreviation FROM team WHERE id IN (?, ?)",
            [home_id, away_id],
        ).df()
        home_meta = teams[teams["id"] == home_id].iloc[0]
        away_meta = teams[teams["id"] == away_id].iloc[0]

        home_state = pm.latest_team_state(conn, home_id)
        away_state = pm.latest_team_state(conn, away_id)
        home_last_game = pm.last_game_date(conn, home_id)
        away_last_game = pm.last_game_date(conn, away_id)

        if game_date_text:
            game_date = pd.Timestamp(game_date_text)
        else:
            game_date = max(home_last_game, away_last_game) + pd.Timedelta(days=1)

        home_rest_days = max((game_date - home_last_game).days, 0)
        away_rest_days = max((game_date - away_last_game).days, 0)

        x = pm.build_feature_row(home_state, away_state, home_rest_days, away_rest_days)
        home_win_prob = float(model.predict_proba(x)[:, 1][0])
        away_win_prob = 1.0 - home_win_prob

        predicted_winner = home_meta["full_name"] if home_win_prob >= 0.5 else away_meta["full_name"]
        matchup = (
            f"{away_meta['full_name']} ({away_meta['abbreviation']}) at "
            f"{home_meta['full_name']} ({home_meta['abbreviation']})"
        )
        return {
            "matchup": matchup,
            "home_team_name": str(home_meta["full_name"]),
            "away_team_name": str(away_meta["full_name"]),
            "home_team_abbreviation": str(home_meta["abbreviation"]),
            "away_team_abbreviation": str(away_meta["abbreviation"]),
            "home_logo_url": team_logo_url(str(home_id)),
            "away_logo_url": team_logo_url(str(away_id)),
            "game_date": str(game_date.date()),
            "home_rest_days": home_rest_days,
            "away_rest_days": away_rest_days,
            "home_win_prob": home_win_prob,
            "away_win_prob": away_win_prob,
            "predicted_winner": predicted_winner,
        }
    finally:
        conn.close()


def build_team_grid_html(teams: list[dict[str, str]], target: str, conference: str) -> str:
    tiles: list[str] = []
    for team in teams:
        if team["conference"] != conference:
            continue
        tiles.append(
            f"""
            <button
              type="button"
              class="team-tile"
              data-target="{escape(target)}"
              data-team-id="{escape(team["id"])}"
            >
              <img
                src="{escape(team["logo_url"])}"
                alt="{escape(team["full_name"])} logo"
                loading="lazy"
                decoding="async"
                fetchpriority="low"
                width="34"
                height="34"
              />
              <span class="team-tile-abbrev">{escape(team["abbreviation"])}</span>
              <span class="team-tile-name">{escape(team["full_name"])}</span>
            </button>
            """
        )
    return "\n".join(tiles)


def build_team_selector_html(
    *,
    teams: list[dict[str, str]],
    target: str,
    label: str,
    selected_value: str,
) -> str:
    east_tiles = build_team_grid_html(teams, target, "East")
    west_tiles = build_team_grid_html(teams, target, "West")
    return f"""
    <section class="team-selector" data-target="{escape(target)}">
      <div class="team-selector-head">
        <h3>{escape(label)}</h3>
        <p id="{escape(target)}-selected" class="selected-team">No team selected</p>
      </div>
      <input id="{escape(target)}-input" type="hidden" name="{escape(target)}" value="{escape(selected_value)}" />
      <div class="conference-split">
        <section class="conference-panel">
          <h4>Eastern Conference</h4>
          <div class="team-grid">
            {east_tiles}
          </div>
        </section>
        <section class="conference-panel">
          <h4>Western Conference</h4>
          <div class="team-grid">
            {west_tiles}
          </div>
        </section>
      </div>
    </section>
    """


def build_tonights_games_html(tonight_games: dict[str, Any]) -> str:
    game_cards: list[str] = []
    for game in tonight_games["games"]:
        if game["away_points"] is not None and game["home_points"] is not None:
            score_or_status_html = (
                f'<p class="slate-score">{game["away_points"]} - {game["home_points"]}</p>'
            )
        else:
            score_or_status_html = '<p class="slate-status">Scheduled</p>'

        game_cards.append(
            f"""
            <article class="slate-game">
              <div class="slate-team">
                <img src="{escape(game["away_logo_url"])}" alt="{escape(game["away_name"])} logo" loading="lazy" decoding="async" width="26" height="26" />
                <span>{escape(game["away_abbreviation"])}</span>
              </div>
              <p class="slate-at">@</p>
              <div class="slate-team">
                <img src="{escape(game["home_logo_url"])}" alt="{escape(game["home_name"])} logo" loading="lazy" decoding="async" width="26" height="26" />
                <span>{escape(game["home_abbreviation"])}</span>
              </div>
              {score_or_status_html}
            </article>
            """
        )

    games_html = "\n".join(game_cards) if game_cards else '<p class="slate-empty">No games available.</p>'
    note_html = f'<p class="slate-note">{escape(tonight_games["note"])}</p>' if tonight_games["note"] else ""
    return f"""
    <section class="section-block slate-block">
      <div class="slate-head">
        <h2>Tonight&apos;s Games</h2>
        <p>{escape(tonight_games["display_date"])}</p>
      </div>
      {note_html}
      <div class="slate-grid">
        {games_html}
      </div>
    </section>
    """


def build_player_prop_section_html(
    *,
    player_options_html: str,
    prop_input: dict[str, str],
    prop_result: dict[str, Any] | None,
    prop_error: str | None,
) -> str:
    if prop_result:
        prop_result_html = f"""
        <section class="prop-result">
          <h3>Prop Probability Result</h3>
          <p><strong>Player:</strong> {escape(prop_result["player"])}</p>
          <p><strong>Market:</strong> {escape(prop_result["line_type_label"])} {escape(prop_result["side"].title())} {prop_result["line"]:.1f}</p>
          <p><strong>Odds:</strong> {prop_result["odds"]:+.0f}</p>
          <p><strong>Hit Probability:</strong> {prop_result["hit_probability"] * 100:.1f}%</p>
          <p><strong>Implied Probability:</strong> {prop_result["implied_probability"] * 100:.1f}%</p>
          <p><strong>Model Edge:</strong> {prop_result["edge"] * 100:+.1f}%</p>
          <p><strong>Projection Mean:</strong> {prop_result["predicted_mean"]:.2f} (source: {escape(prop_result["mean_source"])})</p>
        </section>
        """
    elif prop_error:
        prop_result_html = f"""
        <section class="prop-result prop-result-error">
          <h3>Prop Probability Result</h3>
          <p>{escape(prop_error)}</p>
        </section>
        """
    else:
        prop_result_html = """
        <section class="prop-result">
          <h3>Prop Probability Result</h3>
          <p>Pick player + line details to estimate hit probability.</p>
        </section>
        """

    return f"""
    <section class="section-block prop-block">
      <h2>Player Props</h2>
      <form method="get" action="/" class="prop-form">
        <label>Player
          <input list="player-list" name="prop_player" value="{escape(prop_input["player"])}" placeholder="Select player" required />
        </label>
        <label>Line Type
          <select name="prop_type">
            <option value="points" {"selected" if prop_input["type"] == "points" else ""}>Points</option>
            <option value="rebounds" {"selected" if prop_input["type"] == "rebounds" else ""}>Rebounds</option>
            <option value="3ps" {"selected" if prop_input["type"] == "3ps" else ""}>3PM</option>
          </select>
        </label>
        <label>Side
          <select name="prop_side">
            <option value="over" {"selected" if prop_input["side"] == "over" else ""}>Over</option>
            <option value="under" {"selected" if prop_input["side"] == "under" else ""}>Under</option>
          </select>
        </label>
        <label>Odds (American)
          <input type="number" step="1" name="prop_odds" value="{escape(prop_input["odds"])}" placeholder="-110" required />
        </label>
        <label>Line
          <input type="number" step="0.5" name="prop_line" value="{escape(prop_input["line"])}" placeholder="e.g. 24.5" required />
        </label>
        <button type="submit" class="prop-submit">Calculate Prop Probability</button>
      </form>
      {prop_result_html}
    </section>
    <datalist id="player-list">
      {player_options_html}
    </datalist>
    """


def render_page(
    *,
    teams: list[dict[str, str]],
    team_lookup_json: str,
    team_meta_by_id_json: str,
    tonight_games: dict[str, Any],
    player_options_html: str,
    prop_input: dict[str, str],
    prop_result: dict[str, Any] | None,
    prop_error: str | None,
    home: str,
    away: str,
    game_date: str,
    result: dict[str, Any] | None,
    error: str | None,
) -> str:
    if result:
        result_html = f"""
        <section class="section-block result-block">
          <h2>Forecast Result</h2>
          <div class="team-logos">
            <div class="team-logo-card">
              <img
                src="{escape(result["away_logo_url"])}"
                alt="{escape(result["away_team_name"])} logo"
                loading="lazy"
                decoding="async"
                width="72"
                height="72"
              />
              <span>{escape(result["away_team_name"])} ({escape(result["away_team_abbreviation"])})</span>
            </div>
            <div class="team-logo-card">
              <img
                src="{escape(result["home_logo_url"])}"
                alt="{escape(result["home_team_name"])} logo"
                loading="lazy"
                decoding="async"
                width="72"
                height="72"
              />
              <span>{escape(result["home_team_name"])} ({escape(result["home_team_abbreviation"])})</span>
            </div>
          </div>
          <p><strong>Matchup:</strong> {escape(result["matchup"])}</p>
          <p><strong>Assumed game date:</strong> {escape(result["game_date"])}</p>
          <p><strong>Rest:</strong> Home {result["home_rest_days"]} days | Away {result["away_rest_days"]} days</p>
          <div class="prob-grid">
            <div>
              <span>Home Win Probability</span>
              <strong>{result["home_win_prob"] * 100:.1f}%</strong>
            </div>
            <div>
              <span>Away Win Probability</span>
              <strong>{result["away_win_prob"] * 100:.1f}%</strong>
            </div>
          </div>
          <p><strong>Predicted winner:</strong> {escape(result["predicted_winner"])}</p>
        </section>
        """
    elif error:
        result_html = f"""
        <section class="section-block error-block">
          <h2>Could Not Generate Forecast</h2>
          <p>{escape(error)}</p>
        </section>
        """
    else:
        result_html = """
        <section class="section-block result-block">
          <h2>Forecast Result</h2>
          <p>Enter a home team and away team to generate a matchup forecast.</p>
        </section>
        """

    home_selector_html = build_team_selector_html(
        teams=teams,
        target="home",
        label="Home Team",
        selected_value=home,
    )
    away_selector_html = build_team_selector_html(
        teams=teams,
        target="away",
        label="Away Team",
        selected_value=away,
    )
    tonight_games_html = build_tonights_games_html(tonight_games)
    player_prop_html = build_player_prop_section_html(
        player_options_html=player_options_html,
        prop_input=prop_input,
        prop_result=prop_result,
        prop_error=prop_error,
    )

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>NBA Probabilistic Forecaster</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Roboto+Condensed:wght@400;500;700&display=swap" rel="stylesheet" />
    <style>
      :root {{
        color-scheme: dark;
        --bg-0: #0f0705;
        --bg-1: #211009;
        --bg-2: #34160a;
        --text: #f7ede3;
        --muted: #d2b49a;
        --card: rgba(40, 20, 10, 0.88);
        --line: rgba(242, 140, 40, 0.35);
        --line-strong: rgba(255, 169, 86, 0.8);
        --accent-red: #ca2f2f;
        --accent-orange: #f28c28;
        --accent-amber: #ffba75;
        --shadow: 0 16px 40px rgba(0, 0, 0, 0.45);
      }}
      * {{
        box-sizing: border-box;
        font-family: "Roboto Condensed", "Arial Narrow", Arial, sans-serif;
      }}
      body {{
        margin: 0;
        min-height: 100vh;
        background:
          radial-gradient(circle at 18% 14%, rgba(242, 140, 40, 0.2), transparent 34%),
          radial-gradient(circle at 85% 10%, rgba(202, 47, 47, 0.18), transparent 36%),
          linear-gradient(150deg, var(--bg-0), var(--bg-1) 55%, var(--bg-2));
        color: var(--text);
        position: relative;
      }}
      body::before {{
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background:
          repeating-linear-gradient(
            90deg,
            rgba(255, 172, 96, 0.06) 0 1px,
            transparent 1px 95px
          );
      }}
      body::after {{
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background: linear-gradient(to bottom, rgba(4, 2, 1, 0.15), rgba(6, 3, 2, 0.76));
      }}
      .wrap {{
        max-width: min(1700px, calc(100vw - 24px));
        margin: 0 auto;
        padding: 20px 12px 28px;
        position: relative;
        z-index: 1;
      }}
      .title-block {{
        margin-bottom: 14px;
      }}
      .kicker {{
        margin: 0 0 4px;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.22em;
        color: var(--accent-orange);
        font-weight: 700;
      }}
      h1 {{
        margin: 0;
        font-family: "Oswald", "Roboto Condensed", sans-serif;
        font-size: clamp(1.9rem, 2.7vw, 2.7rem);
        text-transform: uppercase;
        letter-spacing: 0.03em;
        background: linear-gradient(90deg, #fff3e3, #ffbf7c 54%, #ff9e4c);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
      }}
      .deck {{
        margin: 6px 0 0;
        font-size: 1.02rem;
        color: var(--muted);
      }}
      .section-block {{
        padding: 18px 0;
        border-top: 1px solid var(--line);
      }}
      .section-block:last-of-type {{
        border-bottom: 1px solid var(--line);
      }}
      .form-block {{
        border-top-color: var(--line-strong);
      }}
      .slate-block {{
        margin-bottom: 0;
      }}
      .slate-head {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 12px;
      }}
      .slate-head h2 {{
        margin: 0;
        font-family: "Oswald", "Roboto Condensed", sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 1.2rem;
      }}
      .slate-head p {{
        margin: 0;
        color: var(--muted);
        font-size: 0.95rem;
      }}
      .slate-note {{
        margin: 10px 0 0;
        color: #f5be85;
        font-size: 0.9rem;
      }}
      .slate-grid {{
        margin-top: 12px;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 10px;
      }}
      .slate-game {{
        border-left: 3px solid var(--accent-orange);
        border-top: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        background: rgba(38, 17, 9, 0.72);
        padding: 10px;
      }}
      .slate-team {{
        display: flex;
        align-items: center;
        gap: 8px;
      }}
      .slate-team img {{
        object-fit: contain;
      }}
      .slate-team span {{
        font-weight: 700;
        letter-spacing: 0.04em;
      }}
      .slate-at {{
        margin: 5px 0;
        color: #f6c894;
        font-weight: 700;
      }}
      .slate-score {{
        margin: 7px 0 0;
        font-size: 1.02rem;
        font-weight: 700;
        color: #ffe8ce;
      }}
      .slate-status {{
        margin: 7px 0 0;
        color: #f5be85;
        text-transform: uppercase;
        font-size: 0.82rem;
        letter-spacing: 0.04em;
      }}
      .slate-empty {{
        margin: 0;
        color: var(--muted);
      }}
      form {{
        display: grid;
        gap: 16px;
      }}
      #matchup-form > label,
      #matchup-form > button[type="submit"] {{
        grid-column: 1 / -1;
      }}
      .prop-form {{
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 12px;
        align-items: end;
      }}
      .prop-form > label {{
        display: grid;
        gap: 6px;
      }}
      .prop-form > button {{
        align-self: end;
      }}
      .team-selector {{
        border-top: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        padding: 14px 0;
        content-visibility: auto;
        contain-intrinsic-size: 460px;
      }}
      .team-selector-head {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 10px;
        margin-bottom: 10px;
      }}
      .team-selector h3 {{
        margin: 0;
        font-size: 1.05rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }}
      .selected-team {{
        margin: 0;
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--accent-amber);
      }}
      .conference-split {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }}
      .conference-panel {{
        border-top: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        padding: 10px;
        background: rgba(36, 16, 9, 0.45);
        content-visibility: auto;
        contain-intrinsic-size: 280px;
      }}
      .conference-panel h4 {{
        margin: 0 0 8px;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #e7bc8d;
      }}
      .team-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
        gap: 8px;
      }}
      label {{
        font-size: 0.9rem;
        font-weight: 700;
        color: #f2d8bf;
      }}
      .team-tile {{
        border: 1px solid var(--line);
        border-radius: 2px;
        background: linear-gradient(160deg, rgba(79, 35, 16, 0.85), rgba(43, 19, 10, 0.95));
        padding: 8px 6px;
        cursor: pointer;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        gap: 4px;
        transition: transform 70ms ease-out, border-color 70ms linear, background-color 70ms linear;
      }}
      .team-tile:hover {{
        border-color: var(--line-strong);
        transform: translateY(-2px);
      }}
      .team-tile.active {{
        border-color: var(--accent-orange);
        box-shadow: 0 0 0 1px rgba(255, 186, 117, 0.65);
        background:
          linear-gradient(155deg, rgba(202, 47, 47, 0.36), rgba(242, 140, 40, 0.36)),
          rgba(46, 20, 10, 0.96);
      }}
      .team-tile img {{
        width: 34px;
        height: 34px;
        object-fit: contain;
      }}
      .team-tile-abbrev {{
        font-weight: 700;
        font-size: 0.86rem;
        letter-spacing: 0.04em;
      }}
      .team-tile-name {{
        font-size: 0.72rem;
        line-height: 1.2;
        color: #f0d8c1;
      }}
      input[type="date"] {{
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 10px 12px;
        font-size: 0.98rem;
        color: var(--text);
        background: rgba(40, 18, 10, 0.92);
      }}
      input[type="number"],
      input[list],
      select {{
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 10px 12px;
        font-size: 0.98rem;
        color: var(--text);
        background: rgba(40, 18, 10, 0.92);
      }}
      button[type="submit"] {{
        width: fit-content;
        border: none;
        background: linear-gradient(90deg, var(--accent-red), var(--accent-orange));
        color: #fff0df;
        border-radius: 999px;
        padding: 11px 20px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        cursor: pointer;
        box-shadow: 0 8px 18px rgba(168, 63, 29, 0.45);
        transition: transform 150ms ease, box-shadow 150ms ease;
      }}
      button[type="submit"]:hover {{
        transform: translateY(-1px);
        box-shadow: 0 12px 22px rgba(168, 63, 29, 0.62);
      }}
      .prop-result {{
        margin-top: 14px;
        padding-top: 10px;
        border-top: 1px solid var(--line);
      }}
      .prop-result h3 {{
        margin: 0 0 8px;
        font-family: "Oswald", "Roboto Condensed", sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 1.02rem;
      }}
      .prop-result p {{
        margin: 4px 0;
        color: #efd5bc;
      }}
      .prop-result-error p {{
        color: #ffc29e;
      }}
      .prob-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin: 14px 0;
      }}
      .team-logos {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        margin: 8px 0 14px;
      }}
      .team-logo-card {{
        border: 1px solid var(--line);
        border-radius: 2px;
        padding: 12px;
        background: rgba(67, 29, 14, 0.56);
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        gap: 8px;
      }}
      .team-logo-card img {{
        width: 72px;
        height: 72px;
        object-fit: contain;
      }}
      .team-logo-card span {{
        font-size: 0.9rem;
        font-weight: 600;
      }}
      .prob-grid > div {{
        border: 1px solid var(--line);
        border-radius: 2px;
        padding: 10px;
        background: rgba(52, 24, 13, 0.56);
      }}
      .prob-grid span {{
        display: block;
        font-size: 0.85rem;
        color: #f2d4b1;
      }}
      .prob-grid strong {{
        font-size: 1.45rem;
        font-family: "Oswald", "Roboto Condensed", sans-serif;
        color: #fff2de;
      }}
      .result-block h2,
      .error-block h2 {{
        margin-top: 0;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-family: "Oswald", "Roboto Condensed", sans-serif;
        font-size: 1.1rem;
      }}
      .result-block p {{
        color: #efd5bc;
      }}
      .error-block {{
        border-color: rgba(202, 47, 47, 0.75);
        background: linear-gradient(160deg, rgba(97, 22, 16, 0.9), rgba(52, 12, 8, 0.95));
      }}
      @media (prefers-reduced-motion: reduce) {{
        * {{
          animation: none !important;
          transition: none !important;
        }}
      }}
      @media (max-width: 680px) {{
        .slate-head {{
          flex-direction: column;
          align-items: flex-start;
          gap: 4px;
        }}
        .prop-form {{
          grid-template-columns: 1fr;
        }}
        .team-selector-head {{
          flex-direction: column;
          align-items: flex-start;
        }}
        .conference-split {{
          grid-template-columns: 1fr;
        }}
        .team-logos {{
          grid-template-columns: 1fr;
        }}
        .prob-grid {{
          grid-template-columns: 1fr;
        }}
      }}
      @media (min-width: 1200px) {{
        #matchup-form {{
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }}
        .team-grid {{
          grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
        }}
      }}
    </style>
  </head>
  <body>
    <main class="wrap">
      <header class="title-block">
        <h1>NBA Probabilistic Forecaster</h1>
        <p class="deck">Game-edge modeling powered by Elo, form, and four factors.</p>
      </header>
      {tonight_games_html}
      <section class="section-block form-block">
        <form id="matchup-form" method="get" action="/">
          {home_selector_html}
          {away_selector_html}
          <label>Game Date (optional)
            <input type="date" name="game_date" value="{escape(game_date)}" />
          </label>
          <button type="submit">Generate Forecast</button>
        </form>
      </section>
      {player_prop_html}
      {result_html}
    </main>
    <script>
      const TEAM_LOOKUP = {team_lookup_json};
      const TEAM_META_BY_ID = {team_meta_by_id_json};
      const TILES_BY_TARGET = {{
        home: Array.from(document.querySelectorAll('.team-tile[data-target="home"]')),
        away: Array.from(document.querySelectorAll('.team-tile[data-target="away"]')),
      }};

      function normalizeTeamValue(value) {{
        return value.trim().toLowerCase();
      }}

      function resolveTeamId(value) {{
        const key = normalizeTeamValue(value);
        return TEAM_LOOKUP[key] || "";
      }}

      function setSelection(target, teamId) {{
        const input = document.getElementById(target + "-input");
        const selectedLabel = document.getElementById(target + "-selected");
        const team = TEAM_META_BY_ID[teamId];

        if (!input || !selectedLabel) return;

        if (team) {{
          input.value = team.abbreviation;
          selectedLabel.textContent = team.full_name + " (" + team.abbreviation + ")";
        }} else {{
          input.value = "";
          selectedLabel.textContent = "No team selected";
        }}

        const buttons = TILES_BY_TARGET[target] || [];
        buttons.forEach(function(button) {{
          button.classList.toggle("active", button.dataset.teamId === teamId);
        }});
      }}

      function bindTeamTiles() {{
        document.addEventListener("click", function(event) {{
          const button = event.target.closest(".team-tile");
          if (!button) return;
          const target = button.dataset.target;
          const teamId = button.dataset.teamId;
          if (!target || !teamId) return;
          setSelection(target, teamId);
        }});
      }}

      function hydrateInitialSelections() {{
        const homeInput = document.getElementById("home-input");
        const awayInput = document.getElementById("away-input");
        if (homeInput) {{
          setSelection("home", resolveTeamId(homeInput.value));
        }}
        if (awayInput) {{
          setSelection("away", resolveTeamId(awayInput.value));
        }}
      }}

      function bindSubmitValidation() {{
        const form = document.getElementById("matchup-form");
        if (!form) return;
        form.addEventListener("submit", function(event) {{
          const homeValue = document.getElementById("home-input")?.value || "";
          const awayValue = document.getElementById("away-input")?.value || "";
          if (!homeValue || !awayValue) {{
            event.preventDefault();
            window.alert("Select both a home and away team.");
          }}
        }});
      }}

      bindTeamTiles();
      hydrateInitialSelections();
      bindSubmitValidation();
    </script>
  </body>
</html>
"""


def application(environ: dict[str, Any], start_response: Any) -> list[bytes]:
    params = parse_qs(environ.get("QUERY_STRING", ""), keep_blank_values=True)
    home = params.get("home", [""])[0].strip()
    away = params.get("away", [""])[0].strip()
    game_date = params.get("game_date", [""])[0].strip()
    prop_player = params.get("prop_player", [""])[0].strip()
    prop_type = params.get("prop_type", ["points"])[0].strip().lower() or "points"
    prop_side = params.get("prop_side", ["over"])[0].strip().lower() or "over"
    prop_odds = params.get("prop_odds", ["-110"])[0].strip()
    prop_line = params.get("prop_line", [""])[0].strip()

    matchup_error = None
    matchup_result = None
    prop_error = None
    prop_result = None
    teams: list[dict[str, str]] = []
    player_names: list[str] = []
    tonight_games: dict[str, Any] = {"display_date": "Tonight", "source": "today", "note": None, "games": []}

    try:
        teams = load_teams()
    except Exception as exc:  # pragma: no cover - startup data guard
        matchup_error = f"Could not load team list: {exc}"

    try:
        tonight_games = load_tonights_games()
    except Exception as exc:  # pragma: no cover - startup data guard
        tonight_games = {
            "display_date": "Tonight",
            "source": "today",
            "note": f"Could not load games: {exc}",
            "games": [],
        }

    try:
        player_names = load_player_names()
    except Exception:
        player_names = []

    if home and away and matchup_error is None:
        try:
            matchup_result = predict(home, away, game_date or None)
        except Exception as exc:
            matchup_error = str(exc)

    prop_input = {
        "player": prop_player,
        "type": prop_type if prop_type in PLAYER_PROP_TYPES else "points",
        "side": prop_side if prop_side in {"over", "under"} else "over",
        "odds": prop_odds,
        "line": prop_line,
    }
    prop_submitted = any([prop_player, prop_line, prop_odds, "prop_type" in params, "prop_side" in params])
    if prop_submitted:
        if not prop_player or not prop_line or not prop_odds:
            prop_error = "Enter player, line, and odds to calculate probability."
        else:
            try:
                line_value = float(prop_line)
                odds_value = float(prop_odds)
                prop_result = predict_player_prop(
                    prop_player,
                    prop_input["type"],
                    prop_input["side"],
                    line_value,
                    odds_value,
                )
            except Exception as exc:
                prop_error = str(exc)

    team_lookup: dict[str, str] = {}
    team_meta_by_id: dict[str, dict[str, str]] = {}
    for team in teams:
        team_lookup[team["abbreviation"].lower()] = team["id"]
        team_lookup[team["full_name"].lower()] = team["id"]
        team_meta_by_id[team["id"]] = {
            "abbreviation": team["abbreviation"],
            "full_name": team["full_name"],
        }
    team_lookup_json = json.dumps(team_lookup, separators=(",", ":"))
    team_meta_by_id_json = json.dumps(team_meta_by_id, separators=(",", ":"))
    player_options_html = "\n".join(
        f'<option value="{escape(name)}"></option>' for name in player_names
    )
    page = render_page(
        teams=teams,
        team_lookup_json=team_lookup_json,
        team_meta_by_id_json=team_meta_by_id_json,
        tonight_games=tonight_games,
        player_options_html=player_options_html,
        prop_input=prop_input,
        prop_result=prop_result,
        prop_error=prop_error,
        home=home,
        away=away,
        game_date=game_date,
        result=matchup_result,
        error=matchup_error,
    )
    body = page.encode("utf-8")
    start_response(
        "200 OK",
        [
            ("Content-Type", "text/html; charset=utf-8"),
            ("Content-Length", str(len(body))),
        ],
    )
    return [body]


def main() -> None:
    print(f"Serving NBA Probabilistic Forecaster at http://{HOST}:{PORT}")
    with make_server(HOST, PORT, application) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
