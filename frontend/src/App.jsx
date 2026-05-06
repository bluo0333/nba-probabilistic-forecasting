import { useEffect, useMemo, useState } from "react";
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

function getInitials(teamName) {
  const words = teamName.split(" ").filter(Boolean);
  if (words.length === 0) return "NBA";
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return `${words[0][0]}${words[words.length - 1][0]}`.toUpperCase();
}

export default function App() {
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
        if (!response.ok) {
          throw new Error();
        }
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
    if (!loadingTeams) {
      setIsSlowLoad(false);
      return;
    }

    const timer = setTimeout(() => {
      setIsSlowLoad(true);
    }, 2000);

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

    if (selectingType === "home") {
      setSelectedHome(team);
    }
    if (selectingType === "away") {
      setSelectedAway(team);
    }

    setSelectionError("");
    closeTeamModal();
  };

  const handlePredict = async () => {
    setError("");
    setPrediction(null);
    setSelectionError("");

    if (!selectedHome || !selectedAway) {
      setError("Select Team A and Team B.");
      return;
    }
    if (selectedHome === selectedAway) {
      setError("Home and away teams must be different.");
      return;
    }

    setLoadingPrediction(true);
    try {
      const url = `${API_BASE}/predict/quick?home=${encodeURIComponent(selectedHome)}&away=${encodeURIComponent(selectedAway)}`;
      const response = await fetch(url);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error();
      }
      setPrediction(payload);
    } catch {
      setError("Failed to fetch prediction");
    } finally {
      setLoadingPrediction(false);
    }
  };

  const asPercent = (value) => `${(value * 100).toFixed(1)}%`;

  const canPredict = Boolean(selectedHome && selectedAway);
  const predictionHomeLeads =
    prediction?.home_win_probability >= prediction?.away_win_probability;
  const homeBarWidth = prediction
    ? `${(prediction.home_win_probability * 100).toFixed(1)}%`
    : "50%";

  const renderTeamLogo = (team, className = "team-logo-circle") => {
    const logoUrl = TEAM_LOGOS[team];
    const showLogo = Boolean(logoUrl && !brokenLogos[team]);

    return (
      <span className={className} aria-hidden="true">
        {team && showLogo ? (
          <img
            src={logoUrl}
            alt=""
            onError={() => setBrokenLogos((prev) => ({ ...prev, [team]: true }))}
          />
        ) : (
          <span className="team-initials">{team ? getInitials(team) : "?"}</span>
        )}
      </span>
    );
  };

  return (
    <main className="page-shell">
      <section className="nba-app">
        <svg
          className="court-lines"
          viewBox="0 0 500 600"
          xmlns="http://www.w3.org/2000/svg"
          aria-hidden="true"
        >
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
          <h1 className="app-title">NBA Matchup Predictor</h1>
          <p className="app-subtitle">Model-driven win probability engine</p>
        </header>

        <div className="app-body">
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
                  {renderTeamLogo(selectedHome)}
                  <span
                    className={
                      selectedHome ? "team-name-selected" : "team-placeholder"
                    }
                  >
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
                  {renderTeamLogo(selectedAway)}
                  <span
                    className={
                      selectedAway ? "team-name-selected" : "team-placeholder"
                    }
                  >
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

              {selectionError ? (
                <p className="error-msg">{selectionError}</p>
              ) : null}

              {error ? <p className="error-msg">{error}</p> : null}

              {loadingPrediction ? (
                <div
                  className="prediction-loading"
                  role="status"
                  aria-live="polite"
                >
                  <span className="teams-loading__spinner" aria-hidden="true" />
                  <span>Computing win probabilities...</span>
                </div>
              ) : null}

              {prediction ? (
                <section className="result-card" aria-label={prediction.matchup}>
                  <div className="result-header">
                    <span className="result-header-label">Win Probability</span>
                    <span className="winner-badge">
                      {prediction.predicted_winner}
                    </span>
                  </div>
                  <div className="result-body">
                    <div className="prob-row">
                      <div className="prob-team">
                        <div className="prob-team-name">
                          {prediction.home_team}
                        </div>
                        <div
                          className={`prob-value${
                            predictionHomeLeads ? " leader" : ""
                          }`}
                        >
                          {asPercent(prediction.home_win_probability)}
                        </div>
                      </div>
                      <div className="prob-divider">
                        <span>vs</span>
                      </div>
                      <div className="prob-team away">
                        <div className="prob-team-name">
                          {prediction.away_team}
                        </div>
                        <div
                          className={`prob-value${
                            !predictionHomeLeads ? " leader" : ""
                          }`}
                        >
                          {asPercent(prediction.away_win_probability)}
                        </div>
                      </div>
                    </div>
                    <div className="prob-bar-container" aria-hidden="true">
                      <div
                        className="prob-bar-fill"
                        style={{ width: homeBarWidth }}
                      />
                    </div>
                    <div className="prob-bar-labels">
                      <span>{prediction.home_team}</span>
                      <span>{prediction.away_team}</span>
                    </div>
                    <div className="winner-line">
                      Predicted winner:{" "}
                      <span className="winner-name">
                        {prediction.predicted_winner}
                      </span>
                    </div>
                  </div>
                </section>
              ) : null}
            </>
          )}
        </div>
      </section>

      {modalOpen ? (
        <div className="modal-overlay" onClick={closeTeamModal}>
          <div className="modal" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <h2 className="modal-title">
                {selectingType === "home"
                  ? "Select Home Team"
                  : "Select Away Team"}
              </h2>
              <button
                className="modal-close"
                type="button"
                onClick={closeTeamModal}
                aria-label="Close team selector"
              >
                x
              </button>
            </div>

            <div className="modal-search-wrap">
              <input
                className="modal-search"
                type="text"
                value={teamSearch}
                onChange={(event) => setTeamSearch(event.target.value)}
                placeholder="Search teams..."
              />
            </div>

            {selectionError ? (
              <p className="sel-err">{selectionError}</p>
            ) : null}

            <div className="modal-grid">
              {filteredTeams.map((team) => {
                const isSelected = selectedInModal === team;

                return (
                  <button
                    key={team}
                    className={`team-card${isSelected ? " selected-card" : ""}`}
                    type="button"
                    onClick={() => handleSelectTeam(team)}
                  >
                    {renderTeamLogo(team, "team-card-logo")}
                    <span className="team-card-name">{team}</span>
                  </button>
                );
              })}
            </div>

            {filteredTeams.length === 0 ? (
              <p className="empty-teams">No teams found.</p>
            ) : null}
          </div>
        </div>
      ) : null}
    </main>
  );
}
