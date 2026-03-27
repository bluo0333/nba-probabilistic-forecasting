from __future__ import annotations

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


def team_logo_url(team_id: str) -> str:
    return f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"


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
            {"id": str(team_id), "abbreviation": str(abbrev), "full_name": str(full_name)}
            for team_id, abbrev, full_name in rows
        ]
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


def render_page(
    *,
    team_options: str,
    team_logo_lookup_json: str,
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
              <img src="{escape(result["away_logo_url"])}" alt="{escape(result["away_team_name"])} logo" loading="lazy" />
              <span>{escape(result["away_team_name"])} ({escape(result["away_team_abbreviation"])})</span>
            </div>
            <div class="team-logo-card">
              <img src="{escape(result["home_logo_url"])}" alt="{escape(result["home_team_name"])} logo" loading="lazy" />
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

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>NBA Probabilistic Forecaster</title>
    <style>
      :root {{
        color-scheme: light;
        --bg-a: #f6f8fb;
        --bg-b: #dce6f7;
        --text: #121a26;
        --card: #ffffff;
        --line: #d5deea;
        --accent: #0a5fb4;
      }}
      * {{
        box-sizing: border-box;
        font-family: "Segoe UI", Arial, sans-serif;
      }}
      body {{
        margin: 0;
        min-height: 100vh;
        background: linear-gradient(135deg, var(--bg-a), var(--bg-b));
        color: var(--text);
      }}
      .wrap {{
        max-width: 860px;
        margin: 0 auto;
        padding: 32px 16px 40px;
      }}
      h1 {{
        margin: 0 0 18px;
        font-size: 2rem;
      }}
      .card {{
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.07);
      }}
      .card + .card {{
        margin-top: 16px;
      }}
      form {{
        display: grid;
        gap: 12px;
      }}
      .team-field {{
        display: grid;
        gap: 8px;
      }}
      .team-input-wrap {{
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 10px;
        align-items: center;
      }}
      label {{
        font-size: 0.9rem;
        font-weight: 600;
      }}
      input {{
        width: 100%;
        margin-top: 6px;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 0.95rem;
      }}
      .logo-preview {{
        width: 52px;
        height: 52px;
        border: 1px solid var(--line);
        border-radius: 10px;
        background: #fbfdff;
        display: grid;
        place-items: center;
      }}
      .logo-preview img {{
        width: 38px;
        height: 38px;
        object-fit: contain;
        display: none;
      }}
      .logo-preview.has-logo img {{
        display: block;
      }}
      button {{
        width: fit-content;
        border: none;
        background: var(--accent);
        color: #fff;
        border-radius: 8px;
        padding: 10px 16px;
        font-weight: 600;
        cursor: pointer;
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
        background: #fbfdff;
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
        border-radius: 8px;
        padding: 10px;
        background: #fbfdff;
      }}
      .prob-grid span {{
        display: block;
        font-size: 0.85rem;
      }}
      .prob-grid strong {{
        font-size: 1.3rem;
      }}
      .error-card {{
        border-color: #f3c5c5;
        background: #fff5f5;
      }}
      @media (max-width: 680px) {{
        .team-logos {{
          grid-template-columns: 1fr;
        }}
        .prob-grid {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <main class="wrap">
      <h1>NBA Probabilistic Forecaster</h1>
      <section class="card">
        <form method="get" action="/">
          <label class="team-field">Home Team
            <div class="team-input-wrap">
              <input id="home-input" list="team-list" name="home" value="{escape(home)}" placeholder="e.g. IND or Indiana Pacers" required />
              <div id="home-logo-preview" class="logo-preview" aria-hidden="true">
                <img id="home-logo-img" alt="" loading="lazy" />
              </div>
            </div>
          </label>
          <label class="team-field">Away Team
            <div class="team-input-wrap">
              <input id="away-input" list="team-list" name="away" value="{escape(away)}" placeholder="e.g. LAL or Los Angeles Lakers" required />
              <div id="away-logo-preview" class="logo-preview" aria-hidden="true">
                <img id="away-logo-img" alt="" loading="lazy" />
              </div>
            </div>
          </label>
          <label>Game Date (optional)
            <input type="date" name="game_date" value="{escape(game_date)}" />
          </label>
          <button type="submit">Generate Forecast</button>
        </form>
      </section>
      {result_html}
    </main>
    <datalist id="team-list">
      {team_options}
    </datalist>
    <script>
      const TEAM_LOGO_LOOKUP = {team_logo_lookup_json};

      function resolveTeamLogoUrl(inputValue) {{
        const key = inputValue.trim().toLowerCase();
        if (!key) return "";
        const teamId = TEAM_LOGO_LOOKUP[key];
        return teamId ? "https://cdn.nba.com/logos/nba/" + teamId + "/global/L/logo.svg" : "";
      }}

      function updateLogoPreview(inputId, previewId, imageId) {{
        const input = document.getElementById(inputId);
        const preview = document.getElementById(previewId);
        const image = document.getElementById(imageId);
        if (!input || !preview || !image) return;

        const logoUrl = resolveTeamLogoUrl(input.value);
        if (logoUrl) {{
          image.src = logoUrl;
          preview.classList.add("has-logo");
        }} else {{
          image.removeAttribute("src");
          preview.classList.remove("has-logo");
        }}
      }}

      function bindTeamPreview(inputId, previewId, imageId) {{
        const input = document.getElementById(inputId);
        if (!input) return;
        const refresh = function() {{
          updateLogoPreview(inputId, previewId, imageId);
        }};
        input.addEventListener("input", refresh);
        input.addEventListener("change", refresh);
        refresh();
      }}

      bindTeamPreview("home-input", "home-logo-preview", "home-logo-img");
      bindTeamPreview("away-input", "away-logo-preview", "away-logo-img");
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

    try:
        teams = load_teams()
    except Exception as exc:  # pragma: no cover - startup data guard
        error = f"Could not load team list: {exc}"

    if home and away and error is None:
        try:
            result = predict(home, away, game_date or None)
        except Exception as exc:
            error = str(exc)

    team_options = "\n".join(
        f'<option value="{escape(team["abbreviation"])}">{escape(team["full_name"])}</option>' for team in teams
    )
    team_logo_lookup: dict[str, str] = {}
    for team in teams:
        team_logo_lookup[team["abbreviation"].lower()] = team["id"]
        team_logo_lookup[team["full_name"].lower()] = team["id"]
    team_logo_lookup_json = json.dumps(team_logo_lookup, separators=(",", ":"))
    page = render_page(
        team_options=team_options,
        team_logo_lookup_json=team_logo_lookup_json,
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
