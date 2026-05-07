import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const TEAM_LOGOS = {
  "Atlanta Hawks": "https://a.espncdn.com/i/teamlogos/nba/500/atl.png",
  "Boston Celtics": "https://a.espncdn.com/i/teamlogos/nba/500/bos.png",
  "Brooklyn Nets": "https://a.espncdn.com/i/teamlogos/nba/500/bkn.png",
  "Charlotte Hornets": "https://a.espncdn.com/i/teamlogos/nba/500/cha.png",
  "Chicago Bulls": "https://a.espncdn.com/i/teamlogos/nba/500/chi.png",
  "Cleveland Cavaliers": "https://a.espncdn.com/i/teamlogos/nba/500/cle.png",
  "Dallas Mavericks": "https://a.espncdn.com/i/teamlogos/nba/500/dal.png",
  "Denver Nuggets": "https://a.espncdn.com/i/teamlogos/nba/500/den.png",
  "Detroit Pistons": "https://a.espncdn.com/i/teamlogos/nba/500/det.png",
  "Golden State Warriors": "https://a.espncdn.com/i/teamlogos/nba/500/gsw.png",
  "Houston Rockets": "https://a.espncdn.com/i/teamlogos/nba/500/hou.png",
  "Indiana Pacers": "https://a.espncdn.com/i/teamlogos/nba/500/ind.png",
  "Los Angeles Clippers": "https://a.espncdn.com/i/teamlogos/nba/500/lac.png",
  "LA Clippers": "https://a.espncdn.com/i/teamlogos/nba/500/lac.png",
  "Los Angeles Lakers": "https://a.espncdn.com/i/teamlogos/nba/500/lal.png",
  "Memphis Grizzlies": "https://a.espncdn.com/i/teamlogos/nba/500/mem.png",
  "Miami Heat": "https://a.espncdn.com/i/teamlogos/nba/500/mia.png",
  "Milwaukee Bucks": "https://a.espncdn.com/i/teamlogos/nba/500/mil.png",
  "Minnesota Timberwolves": "https://a.espncdn.com/i/teamlogos/nba/500/min.png",
  "New Orleans Pelicans": "https://a.espncdn.com/i/teamlogos/nba/500/no.png",
  "New York Knicks": "https://a.espncdn.com/i/teamlogos/nba/500/ny.png",
  "Oklahoma City Thunder": "https://a.espncdn.com/i/teamlogos/nba/500/okc.png",
  "Orlando Magic": "https://a.espncdn.com/i/teamlogos/nba/500/orl.png",
  "Philadelphia 76ers": "https://a.espncdn.com/i/teamlogos/nba/500/phi.png",
  "Phoenix Suns": "https://a.espncdn.com/i/teamlogos/nba/500/phx.png",
  "Portland Trail Blazers": "https://a.espncdn.com/i/teamlogos/nba/500/por.png",
  "Sacramento Kings": "https://a.espncdn.com/i/teamlogos/nba/500/sac.png",
  "San Antonio Spurs": "https://a.espncdn.com/i/teamlogos/nba/500/sa.png",
  "Toronto Raptors": "https://a.espncdn.com/i/teamlogos/nba/500/tor.png",
  "Utah Jazz": "https://a.espncdn.com/i/teamlogos/nba/500/utah.png",
  "Washington Wizards": "https://a.espncdn.com/i/teamlogos/nba/500/wsh.png",
};

const PROP_TYPES = [
  { value: "points", label: "Points" },
  { value: "rebounds", label: "Rebounds" },
  { value: "3ps", label: "3PM" },
];

function edgeColor(edge) {
  if (edge > 0.05) return "positive";
  if (edge < -0.05) return "negative";
  return "neutral";
}

function getInitials(teamName) {
  const words = teamName.split(" ").filter(Boolean);
  if (words.length === 0) return "NBA";
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return `${words[0][0]}${words[words.length - 1][0]}`.toUpperCase();
}

function Spinner({ size = 18 }) {
  return (
    <span
      className="spinner"
      style={{ width: size, height: size }}
      aria-hidden="true"
    />
  );
}

function TeamLogo({ team, className = "team-logo-circle", brokenLogos, onBroken }) {
  const logoUrl = TEAM_LOGOS[team];
  const showLogo = Boolean(logoUrl && team && !brokenLogos[team]);
  return (
    <span className={className} aria-hidden="true">
      {team && showLogo ? (
        <img
          src={logoUrl}
          alt=""
          onError={() => onBroken(team)}
        />
      ) : (
        <span className="team-initials">{team ? getInitials(team) : "?"}</span>
      )}
    </span>
  );
}

function PlayerPropsSection() {
  // Player search & selection
  const [playerQuery, setPlayerQuery] = useState("");
  const [players, setPlayers] = useState([]); // from /props/players
  const [selectedPlayer, setSelectedPlayer] = useState("");
  const [playerDropdownOpen, setPlayerDropdownOpen] = useState(false);
  const [loadingPlayers, setLoadingPlayers] = useState(false);
  const [playerLoadError, setPlayerLoadError] = useState("");

  // Prop form
  const [propType, setPropType] = useState("points");
  const [side, setSide] = useState("over");
  const [line, setLine] = useState("");
  const [odds, setOdds] = useState("");

  // Results
  const [result, setResult] = useState(null);
  const [loadingResult, setLoadingResult] = useState(false);
  const [error, setError] = useState("");

  const [recentGames, setRecentGames] = useState(null);
  const [loadingGames, setLoadingGames] = useState(false);
  const [recentGamesError, setRecentGamesError] = useState("");

  const dropdownRef = useRef(null);

  // Load player list from our backend
  useEffect(() => {
    let cancelled = false;

    const loadPlayers = async () => {
      setLoadingPlayers(true);
      setPlayerLoadError("");
      try {
        const response = await fetch(`${API_BASE}/props/players`);
        if (!response.ok) {
          throw new Error("Failed to load players.");
        }
        const data = await response.json();
        if (!cancelled) {
          setPlayers(Array.isArray(data) ? data : []);
        }
      } catch {
        if (!cancelled) {
          setPlayers([]);
          setPlayerLoadError("Could not load players. Check that the API server is running.");
        }
      } finally {
        if (!cancelled) {
          setLoadingPlayers(false);
        }
      }
    };

    loadPlayers();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!playerQuery.trim()) {
      setSelectedPlayer("");
      return;
    }

    const exactMatch = players.find(
      (player) => player.toLowerCase() === playerQuery.trim().toLowerCase()
    );
    setSelectedPlayer(exactMatch || "");
  }, [players, playerQuery]);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setPlayerDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const filteredPlayers = useMemo(() => {
    const q = playerQuery.trim().toLowerCase();
    if (!q) return players;
    return players.filter((p) => p.toLowerCase().includes(q));
  }, [players, playerQuery]);

  const resolvedPlayer = useMemo(() => {
    if (selectedPlayer) return selectedPlayer;
    const query = playerQuery.trim().toLowerCase();
    if (!query) return "";
    return players.find((player) => player.toLowerCase() === query) || "";
  }, [playerQuery, players, selectedPlayer]);

  const handleSelectPlayer = async (name) => {
    setSelectedPlayer(name);
    setPlayerQuery(name);
    setPlayerDropdownOpen(false);
    setResult(null);
    setError("");
  };

  useEffect(() => {
    if (!resolvedPlayer) {
      setRecentGames(null);
      setRecentGamesError("");
      return;
    }

    const controller = new AbortController();
    setLoadingGames(true);
    setRecentGamesError("");

    const params = new URLSearchParams({
      player: resolvedPlayer,
      line_type: propType,
      limit: "10",
    });

    fetch(`${API_BASE}/props/recent-games?${params.toString()}`, {
      signal: controller.signal,
    })
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "Failed to load recent games.");
        }
        setRecentGames(Array.isArray(payload) ? payload : []);
      })
      .catch((err) => {
        if (err.name === "AbortError") return;
        setRecentGames(null);
        setRecentGamesError(err.message || "Failed to load recent games.");
      })
      .finally(() => setLoadingGames(false));

    return () => controller.abort();
  }, [propType, resolvedPlayer]);

  const handlePredict = async () => {
    setError("");
    setResult(null);

    const playerName = resolvedPlayer;
    if (!playerName) { setError("Enter a player name."); return; }
    const lineNum = parseFloat(line);
    const oddsNum = parseFloat(odds);
    if (isNaN(lineNum) || lineNum <= 0) { setError("Enter a valid line (e.g. 24.5)."); return; }
    if (isNaN(oddsNum)) { setError("Enter valid American odds (e.g. -110 or +120)."); return; }

    setLoadingResult(true);
    try {
      const resp = await fetch(`${API_BASE}/props/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          player: playerName,
          line_type: propType,
          side,
          line: lineNum,
          odds: oddsNum,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "Prediction failed.");
      setResult(data);
    } catch (e) {
      setError(e.message || "Failed to fetch prediction.");
    } finally {
      setLoadingResult(false);
    }
  };

  const canSubmit = Boolean(resolvedPlayer && line && odds);
  const lineNum = parseFloat(line) || 0;

  return (
    <section className="props-section">
      <div className="props-header">
        <div className="section-eyebrow">Player Props</div>
        <h2 className="section-title">Prop Analyzer</h2>
        <p className="section-sub">Edge detection against the posted line</p>
      </div>

      <div className="props-body">
        <div className="props-form-col">

          {/* Player search */}
          <div className="form-group" ref={dropdownRef}>
            <label className="form-label">Player</label>
            <div className="player-search-wrap">
              <input
                className="form-input"
                type="text"
                placeholder={loadingPlayers ? "Loading players..." : "Search player..."}
                value={playerQuery}
                onChange={(e) => {
                  setPlayerQuery(e.target.value);
                  setSelectedPlayer("");
                  setPlayerDropdownOpen(true);
                }}
                onFocus={() => setPlayerDropdownOpen(true)}
                disabled={loadingPlayers}
                autoComplete="off"
              />
              {playerLoadError && (
                <p className="field-help error">{playerLoadError}</p>
              )}
              {playerDropdownOpen && playerQuery.trim() && (
                <ul className="player-dropdown">
                  {filteredPlayers.length > 0 ? (
                    filteredPlayers.slice(0, 12).map((p) => (
                      <li
                        key={p}
                        className={`player-option${p === selectedPlayer ? " active" : ""}`}
                        onMouseDown={() => handleSelectPlayer(p)}
                      >
                        {p}
                      </li>
                    ))
                  ) : (
                    <li className="player-option more">
                      {loadingPlayers ? "Loading players..." : "No players found"}
                    </li>
                  )}
                  {filteredPlayers.length > 12 ? (
                    <li className="player-option more">
                      +{filteredPlayers.length - 12} more - keep typing
                    </li>
                  ) : null}
                </ul>
              )}
            </div>
          </div>

          {/* Prop type */}
          <div className="form-group">
            <label className="form-label">Stat</label>
            <div className="pill-row">
              {PROP_TYPES.map((pt) => (
                <button
                  key={pt.value}
                  type="button"
                  className={`pill-btn${propType === pt.value ? " active" : ""}`}
                  onClick={() => setPropType(pt.value)}
                >
                  {pt.label}
                </button>
              ))}
            </div>
          </div>

          {/* Side */}
          <div className="form-group">
            <label className="form-label">Side</label>
            <div className="pill-row">
              <button
                type="button"
                className={`pill-btn over${side === "over" ? " active" : ""}`}
                onClick={() => setSide("over")}
              >
                Over
              </button>
              <button
                type="button"
                className={`pill-btn under${side === "under" ? " active" : ""}`}
                onClick={() => setSide("under")}
              >
                Under
              </button>
            </div>
          </div>

          {/* Line + Odds */}
          <div className="form-row-2">
            <div className="form-group">
              <label className="form-label">Line</label>
              <input
                className="form-input"
                type="number"
                step="0.5"
                min="0"
                placeholder="24.5"
                value={line}
                onChange={(e) => setLine(e.target.value)}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Odds (American)</label>
              <input
                className="form-input"
                type="number"
                step="1"
                placeholder="-110"
                value={odds}
                onChange={(e) => setOdds(e.target.value)}
              />
            </div>
          </div>

          {error && <p className="error-msg">{error}</p>}

          <button
            type="button"
            className="predict-btn"
            onClick={handlePredict}
            disabled={!canSubmit || loadingResult}
          >
            {loadingResult ? (
              <><Spinner size={15} /> Analyzing...</>
            ) : (
              "Analyze Prop"
            )}
          </button>

          {result && (
            <div className="prop-result-card">
              <div className="prop-result-top">
                <div>
                  <div className="prop-result-player">{result.player}</div>
                  <div className="prop-result-line">
                    {result.side === "over" ? "Over" : "Under"}{" "}
                    {result.line} {result.line_type_label}
                  </div>
                </div>
                <div className={`edge-badge ${edgeColor(result.edge)}`}>
                  {result.edge >= 0 ? "+" : ""}
                  {(result.edge * 100).toFixed(1)}% edge
                </div>
              </div>

              <div className="prop-stats-grid">
                <div className="prop-stat">
                  <span className="prop-stat-label">Hit Probability</span>
                  <span className="prop-stat-value hit">
                    {(result.hit_probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="prop-stat">
                  <span className="prop-stat-label">Implied Prob</span>
                  <span className="prop-stat-value">
                    {(result.implied_probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="prop-stat">
                  <span className="prop-stat-label">Predicted Mean</span>
                  <span className="prop-stat-value">
                    {result.predicted_mean.toFixed(1)}
                  </span>
                </div>
                <div className="prop-stat">
                  <span className="prop-stat-label">Std Dev</span>
                  <span className="prop-stat-value">
                    +-{result.std_dev.toFixed(1)}
                  </span>
                </div>
              </div>

              {/* Probability bar */}
              <div className="prop-bar-wrap">
                <div className="prop-bar-track">
                  <div
                    className="prop-bar-fill"
                    style={{ width: `${(result.hit_probability * 100).toFixed(1)}%` }}
                  />
                  {/* Line marker */}
                  <div
                    className="prop-bar-marker"
                    style={{
                      left: `${(result.implied_probability * 100).toFixed(1)}%`,
                    }}
                    title={`Implied: ${(result.implied_probability * 100).toFixed(1)}%`}
                  />
                </div>
                <div className="prop-bar-labels">
                  <span>Model: {(result.hit_probability * 100).toFixed(1)}%</span>
                  <span>Implied: {(result.implied_probability * 100).toFixed(1)}%</span>
                </div>
              </div>

              <div className="prop-source">Source: {result.mean_source}</div>
            </div>
          )}
        </div>

        <div className="props-log-col">
          <div className="log-header">
            <span className="log-title">Recent Game Log</span>
            {resolvedPlayer && (
              <span className="log-stat-label">
                {PROP_TYPES.find((p) => p.value === propType)?.label}
              </span>
            )}
          </div>

          {!resolvedPlayer && !playerQuery.trim() && (
            <div className="log-empty">Select a player to see recent games</div>
          )}

          {resolvedPlayer && loadingGames && (
            <div className="log-loading">
              <Spinner /> Loading game log...
            </div>
          )}

          {resolvedPlayer && !loadingGames && recentGamesError && (
            <div className="log-empty">{recentGamesError}</div>
          )}

          {resolvedPlayer && !loadingGames && !recentGamesError && recentGames && (
            <div className="game-log">
              {recentGames.length === 0 && (
                <div className="log-empty">No recent games found.</div>
              )}
              {recentGames.map((g, i) => {
                const isOver = lineNum > 0 && g.value > lineNum;
                const isUnder = lineNum > 0 && g.value < lineNum;
                const isPush = lineNum > 0 && g.value === lineNum;
                return (
                  <div key={i} className="game-log-row">
                    <div className="game-log-left">
                      <span className="game-log-date">{g.date}</span>
                      <span className="game-log-opp">vs {g.opponent}</span>
                    </div>
                    <div className="game-log-right">
                      <span className="game-log-mins">{g.mins} min</span>
                      <span
                        className={`game-log-val${isOver ? " over" : isUnder ? " under" : isPush ? " push" : ""}`}
                      >
                        {g.value}
                      </span>
                    </div>
                  </div>
                );
              })}

              {/* Mini distribution chart */}
              {recentGames.length > 0 && (
                <MiniBarChart
                  games={recentGames}
                  line={lineNum}
                  propLabel={PROP_TYPES.find((p) => p.value === propType)?.label}
                />
              )}
            </div>
          )}

          {resolvedPlayer && !loadingGames && !recentGamesError && !recentGames && (
            <div className="log-empty">No recent games loaded.</div>
          )}
        </div>
      </div>
    </section>
  );
}

function MiniBarChart({ games, line, propLabel }) {
  const values = games.map((g) => g.value);
  const max = Math.max(...values, line || 0) * 1.15 || 1;

  return (
    <div className="mini-chart">
      <div className="mini-chart-title">Last {games.length} games - {propLabel}</div>
      <div className="mini-chart-bars">
        {[...games].reverse().map((g, i) => {
          const pct = (g.value / max) * 100;
          const isOver = line > 0 && g.value > line;
          return (
            <div key={i} className="mini-bar-col" title={`${g.date}: ${g.value}`}>
              <div className="mini-bar-track">
                <div
                  className={`mini-bar-fill${isOver ? " over" : " under"}`}
                  style={{ height: `${pct}%` }}
                />
                {line > 0 && (
                  <div
                    className="mini-bar-line"
                    style={{ bottom: `${(line / max) * 100}%` }}
                  />
                )}
              </div>
              <div className="mini-bar-label">{g.value}</div>
            </div>
          );
        })}
      </div>
      {line > 0 && (
        <div className="mini-chart-legend">
          <span className="legend-over">- Over {line}</span>
          <span className="legend-under">- Under {line}</span>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState("matchup");

  // Matchup state
  const [teams, setTeams] = useState([]);
  const [selectedHome, setSelectedHome] = useState("");
  const [selectedAway, setSelectedAway] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [loadingTeams, setLoadingTeams] = useState(false);
  const [isSlowLoad, setIsSlowLoad] = useState(false);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [error, setError] = useState("");
  const [selectionError, setSelectionError] = useState("");
  const [modalOpen, setModalOpen] = useState(false);
  const [selectingType, setSelectingType] = useState(null);
  const [teamSearch, setTeamSearch] = useState("");
  const [brokenLogos, setBrokenLogos] = useState({});

  useEffect(() => {
    const loadTeams = async () => {
      setLoadingTeams(true);
      setError("");
      try {
        const response = await fetch(`${API_BASE}/teams/`);
        if (!response.ok) throw new Error();
        const payload = await response.json();
        setTeams(payload);
      } catch {
        setError("Failed to fetch teams");
      } finally {
        setLoadingTeams(false);
      }
    };
    loadTeams();
  }, []);

  useEffect(() => {
    if (!loadingTeams) { setIsSlowLoad(false); return; }
    const timer = setTimeout(() => setIsSlowLoad(true), 2000);
    return () => clearTimeout(timer);
  }, [loadingTeams]);

  const filteredTeams = useMemo(() => {
    const query = teamSearch.trim().toLowerCase();
    if (!query) return teams;
    return teams.filter((team) => team.toLowerCase().includes(query));
  }, [teams, teamSearch]);

  const selectedInModal = selectingType === "home" ? selectedHome : selectedAway;

  const openTeamModal = (type) => {
    setSelectingType(type);
    setSelectionError("");
    setTeamSearch("");
    setModalOpen(true);
  };

  const closeTeamModal = () => {
    setModalOpen(false);
    setSelectingType(null);
    setTeamSearch("");
  };

  const handleSelectTeam = (team) => {
    if (selectingType === "home" && team === selectedAway) {
      setSelectionError("Home and away teams must be different.");
      return;
    }
    if (selectingType === "away" && team === selectedHome) {
      setSelectionError("Home and away teams must be different.");
      return;
    }
    if (selectingType === "home") setSelectedHome(team);
    if (selectingType === "away") setSelectedAway(team);
    setSelectionError("");
    closeTeamModal();
  };

  const handlePredict = async () => {
    setError("");
    setPrediction(null);
    setSelectionError("");
    if (!selectedHome || !selectedAway) { setError("Select Team A and Team B."); return; }
    if (selectedHome === selectedAway) { setError("Home and away teams must be different."); return; }

    setLoadingPrediction(true);
    try {
      const url = `${API_BASE}/predict/quick?home=${encodeURIComponent(selectedHome)}&away=${encodeURIComponent(selectedAway)}`;
      const response = await fetch(url);
      const payload = await response.json();
      if (!response.ok) throw new Error();
      setPrediction(payload);
    } catch {
      setError("Failed to fetch prediction");
    } finally {
      setLoadingPrediction(false);
    }
  };

  const asPercent = (value) => `${(value * 100).toFixed(1)}%`;
  const canPredict = Boolean(selectedHome && selectedAway);
  const predictionHomeLeads = prediction?.home_win_probability >= prediction?.away_win_probability;
  const homeBarWidth = prediction ? `${(prediction.home_win_probability * 100).toFixed(1)}%` : "50%";
  const onBroken = (team) => setBrokenLogos((prev) => ({ ...prev, [team]: true }));

  return (
    <main className="page-shell">
      <section className="nba-app">
        <svg className="court-lines" viewBox="0 0 500 600" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <rect x="50" y="50" width="400" height="500" fill="none" />
          <line x1="50" y1="300" x2="450" y2="300" />
          <circle cx="250" cy="300" r="60" fill="none" />
          <circle cx="250" cy="300" r="6" />
          <rect x="150" y="50" width="200" height="140" fill="none" />
          <rect x="150" y="410" width="200" height="140" fill="none" />
          <path d="M 160 190 A 90 90 0 0 1 340 190" fill="none" />
          <path d="M 160 410 A 90 90 0 0 0 340 410" fill="none" />
          <circle cx="250" cy="115" r="20" fill="none" />
          <circle cx="250" cy="485" r="20" fill="none" />
        </svg>

        <header className="app-header">
          <div className="app-logo">Probabilistic Forecasting</div>
          <h1 className="app-title">NBA Predictor</h1>
          <p className="app-subtitle">Model-driven analytics engine</p>
        </header>

        {/* Tab nav */}
        <nav className="tab-nav" aria-label="Sections">
          <button
            className={`tab-btn${activeTab === "matchup" ? " active" : ""}`}
            type="button"
            onClick={() => setActiveTab("matchup")}
          >
            Matchup
          </button>
          <button
            className={`tab-btn${activeTab === "props" ? " active" : ""}`}
            type="button"
            onClick={() => setActiveTab("props")}
          >
            Player Props
          </button>
        </nav>

        <div className="app-body">
          {activeTab === "matchup" && (
            <>
              {loadingTeams ? (
                <div className="teams-loading" role="status" aria-live="polite">
                  <span className="teams-loading__spinner" aria-hidden="true" />
                  <p className="teams-loading__message">
                    {isSlowLoad
                      ? "Waking up server (first request may take ~10-20 seconds)"
                      : "Loading teams..."}
                  </p>
                </div>
              ) : (
                <>
                  <div className="matchup-builder">
                    <button
                      className={`team-slot${selectedHome ? " selected" : ""}`}
                      type="button"
                      onClick={() => openTeamModal("home")}
                      disabled={loadingTeams}
                      aria-label="Select home team"
                    >
                      <span className="team-slot-label">Home</span>
                      <TeamLogo team={selectedHome} brokenLogos={brokenLogos} onBroken={onBroken} />
                      <span className={selectedHome ? "team-name-selected" : "team-placeholder"}>
                        {selectedHome || "Select team"}
                      </span>
                    </button>

                    <div className="vs-badge">VS</div>

                    <button
                      className={`team-slot${selectedAway ? " selected" : ""}`}
                      type="button"
                      onClick={() => openTeamModal("away")}
                      disabled={loadingTeams}
                      aria-label="Select away team"
                    >
                      <span className="team-slot-label">Away</span>
                      <TeamLogo team={selectedAway} brokenLogos={brokenLogos} onBroken={onBroken} />
                      <span className={selectedAway ? "team-name-selected" : "team-placeholder"}>
                        {selectedAway || "Select team"}
                      </span>
                    </button>
                  </div>

                  <button
                    className="predict-btn"
                    type="button"
                    onClick={handlePredict}
                    disabled={loadingPrediction || loadingTeams || !canPredict}
                  >
                    {loadingPrediction ? "Predicting..." : "Run Prediction"}
                  </button>

                  {selectionError && <p className="error-msg">{selectionError}</p>}
                  {error && <p className="error-msg">{error}</p>}

                  {loadingPrediction && (
                    <div className="prediction-loading" role="status" aria-live="polite">
                      <span className="teams-loading__spinner" aria-hidden="true" />
                      <span>Computing win probabilities...</span>
                    </div>
                  )}

                  {prediction && (
                    <section className="result-card" aria-label={prediction.matchup}>
                      <div className="result-header">
                        <span className="result-header-label">Win Probability</span>
                        <span className="winner-badge">{prediction.predicted_winner}</span>
                      </div>
                      <div className="result-body">
                        <div className="prob-row">
                          <div className="prob-team">
                            <div className="prob-team-name">{prediction.home_team}</div>
                            <div className={`prob-value${predictionHomeLeads ? " leader" : ""}`}>
                              {asPercent(prediction.home_win_probability)}
                            </div>
                          </div>
                          <div className="prob-divider"><span>vs</span></div>
                          <div className="prob-team away">
                            <div className="prob-team-name">{prediction.away_team}</div>
                            <div className={`prob-value${!predictionHomeLeads ? " leader" : ""}`}>
                              {asPercent(prediction.away_win_probability)}
                            </div>
                          </div>
                        </div>
                        <div className="prob-bar-container" aria-hidden="true">
                          <div className="prob-bar-fill" style={{ width: homeBarWidth }} />
                        </div>
                        <div className="prob-bar-labels">
                          <span>{prediction.home_team}</span>
                          <span>{prediction.away_team}</span>
                        </div>
                        <div className="winner-line">
                          Predicted winner: <span className="winner-name">{prediction.predicted_winner}</span>
                        </div>
                      </div>
                    </section>
                  )}
                </>
              )}
            </>
          )}

          {activeTab === "props" && <PlayerPropsSection />}
        </div>
      </section>

      {/* Team selector modal */}
      {modalOpen && (
        <div className="modal-overlay" onClick={closeTeamModal}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 className="modal-title">
                {selectingType === "home" ? "Select Home Team" : "Select Away Team"}
              </h2>
              <button className="modal-close" type="button" onClick={closeTeamModal} aria-label="Close team selector">
                x
              </button>
            </div>
            <div className="modal-search-wrap">
              <input
                className="modal-search"
                type="text"
                value={teamSearch}
                onChange={(e) => setTeamSearch(e.target.value)}
                placeholder="Search teams..."
              />
            </div>
            {selectionError && <p className="sel-err">{selectionError}</p>}
            <div className="modal-grid">
              {filteredTeams.map((team) => (
                <button
                  key={team}
                  className={`team-card${selectedInModal === team ? " selected-card" : ""}`}
                  type="button"
                  onClick={() => handleSelectTeam(team)}
                >
                  <TeamLogo team={team} className="team-card-logo" brokenLogos={brokenLogos} onBroken={onBroken} />
                  <span className="team-card-name">{team}</span>
                </button>
              ))}
            </div>
            {filteredTeams.length === 0 && <p className="empty-teams">No teams found.</p>}
          </div>
        </div>
      )}
    </main>
  );
}
