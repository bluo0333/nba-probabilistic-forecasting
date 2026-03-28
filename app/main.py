from __future__ import annotations

from datetime import date
import json
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
    <section class="card slate-card">
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


def render_page(
    *,
    teams: list[dict[str, str]],
    team_lookup_json: str,
    team_meta_by_id_json: str,
    tonight_games: dict[str, Any],
    home: str,
    away: str,
    game_date: str,
    result: dict[str, Any] | None,
    error: str | None,
) -> str:
    if result:
        result_html = f"""
        <section class="card result-card">
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
        <section class="card error-card">
          <h2>Could Not Generate Forecast</h2>
          <p>{escape(error)}</p>
        </section>
        """
    else:
        result_html = """
        <section class="card result-card">
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
      .slate-card {{
        margin-bottom: 16px;
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
        border: 1px solid var(--line);
        border-radius: 10px;
        background: linear-gradient(160deg, rgba(64, 28, 13, 0.95), rgba(38, 17, 9, 0.95));
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
      .card {{
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px;
        box-shadow: var(--shadow);
      }}
      .card + .card {{
        margin-top: 16px;
      }}
      form {{
        display: grid;
        gap: 16px;
      }}
      #matchup-form > label,
      #matchup-form > button[type="submit"] {{
        grid-column: 1 / -1;
      }}
      .team-selector {{
        border: 1px solid var(--line);
        border-radius: 14px;
        background: linear-gradient(170deg, rgba(55, 24, 12, 0.92), rgba(33, 14, 8, 0.92));
        padding: 14px;
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
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 10px;
        background: rgba(36, 16, 9, 0.8);
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
        border-radius: 10px;
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
        border-radius: 10px;
        padding: 12px;
        background: linear-gradient(160deg, rgba(67, 29, 14, 0.92), rgba(35, 15, 9, 0.94));
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
        border-radius: 10px;
        padding: 10px;
        background: rgba(52, 24, 13, 0.92);
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
      .result-card h2,
      .error-card h2 {{
        margin-top: 0;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-family: "Oswald", "Roboto Condensed", sans-serif;
        font-size: 1.1rem;
      }}
      .result-card p {{
        color: #efd5bc;
      }}
      .error-card {{
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
        <p class="kicker">NBA x Tech Forecast Lab</p>
        <h1>NBA Probabilistic Forecaster</h1>
        <p class="deck">Game-edge modeling powered by Elo, form, and four factors.</p>
      </header>
      {tonight_games_html}
      <section class="card">
        <form id="matchup-form" method="get" action="/">
          {home_selector_html}
          {away_selector_html}
          <label>Game Date (optional)
            <input type="date" name="game_date" value="{escape(game_date)}" />
          </label>
          <button type="submit">Generate Forecast</button>
        </form>
      </section>
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

    error = None
    result = None
    teams = []
    tonight_games: dict[str, Any] = {"display_date": "", "source": "today", "note": None, "games": []}

    try:
        teams = load_teams()
        tonight_games = load_tonights_games()
    except Exception as exc:  # pragma: no cover - startup data guard
        error = f"Could not load team list: {exc}"

    if home and away and error is None:
        try:
            result = predict(home, away, game_date or None)
        except Exception as exc:
            error = str(exc)

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
    page = render_page(
        teams=teams,
        team_lookup_json=team_lookup_json,
        team_meta_by_id_json=team_meta_by_id_json,
        tonight_games=tonight_games,
        home=home,
        away=away,
        game_date=game_date,
        result=result,
        error=error,
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
