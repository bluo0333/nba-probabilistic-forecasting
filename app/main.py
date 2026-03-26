from __future__ import annotations

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


def load_teams() -> list[tuple[str, str]]:
    conn = duckdb.connect(str(DB_PATH))
    try:
        rows = conn.execute(
            """
            SELECT abbreviation, full_name
            FROM team
            ORDER BY abbreviation
            """
        ).fetchall()
        return [(str(abbrev), str(full_name)) for abbrev, full_name in rows]
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
          <label>Home Team
            <input list="team-list" name="home" value="{escape(home)}" placeholder="e.g. IND or Indiana Pacers" required />
          </label>
          <label>Away Team
            <input list="team-list" name="away" value="{escape(away)}" placeholder="e.g. LAL or Los Angeles Lakers" required />
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
        f'<option value="{escape(abbrev)}">{escape(full_name)}</option>' for abbrev, full_name in teams
    )
    page = render_page(
        team_options=team_options,
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
