import { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const TEAM_ABBREVIATIONS = {
  "Atlanta Hawks": "atl",
  "Boston Celtics": "bos",
  "Brooklyn Nets": "bkn",
  "Charlotte Hornets": "cha",
  "Chicago Bulls": "chi",
  "Cleveland Cavaliers": "cle",
  "Dallas Mavericks": "dal",
  "Denver Nuggets": "den",
  "Detroit Pistons": "det",
  "Golden State Warriors": "gs",
  "Houston Rockets": "hou",
  "Indiana Pacers": "ind",
  "LA Clippers": "lac",
  "Los Angeles Clippers": "lac",
  "Los Angeles Lakers": "lal",
  "Memphis Grizzlies": "mem",
  "Miami Heat": "mia",
  "Milwaukee Bucks": "mil",
  "Minnesota Timberwolves": "min",
  "New Orleans Pelicans": "no",
  "New York Knicks": "ny",
  "Oklahoma City Thunder": "okc",
  "Orlando Magic": "orl",
  "Philadelphia 76ers": "phi",
  "Phoenix Suns": "phx",
  "Portland Trail Blazers": "por",
  "Sacramento Kings": "sac",
  "San Antonio Spurs": "sa",
  "Toronto Raptors": "tor",
  "Utah Jazz": "uta",
  "Washington Wizards": "wsh",
};

const TEAM_LOGOS = Object.fromEntries(
  Object.entries(TEAM_ABBREVIATIONS).map(([teamName, code]) => [
    teamName,
    `https://a.espncdn.com/i/teamlogos/nba/500/${code}.png`,
  ]),
);

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
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [error, setError] = useState("");
  const [selectionError, setSelectionError] = useState("");
  const [buttonHover, setButtonHover] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [selectingType, setSelectingType] = useState(null);
  const [hoveredTeam, setHoveredTeam] = useState("");
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

  const openTeamModal = (type) => {
    setSelectingType(type);
    setSelectionError("");
    setModalOpen(true);
  };

  const closeTeamModal = () => {
    setModalOpen(false);
    setSelectingType(null);
    setHoveredTeam("");
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

  const buttonBackground = loadingPrediction
    ? "#94a3b8"
    : buttonHover
      ? "#1d4ed8"
      : "#2563eb";

  return (
    <main
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "#f3f4f6",
        padding: 16,
        fontFamily:
          "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      }}
    >
      <section
        style={{
          width: "100%",
          maxWidth: 400,
          backgroundColor: "#ffffff",
          borderRadius: 14,
          boxShadow: "0 10px 30px rgba(15, 23, 42, 0.10)",
          padding: 24,
        }}
      >
        <h1
          style={{
            margin: 0,
            textAlign: "center",
            fontSize: 24,
            color: "#111827",
          }}
        >
          NBA Matchup Predictor
        </h1>
        <p
          style={{
            marginTop: 8,
            marginBottom: 20,
            textAlign: "center",
            color: "#6b7280",
            fontSize: 14,
          }}
        >
          Select two teams and run a prediction
        </p>

        <div style={{ display: "grid", gap: 12 }}>
          <div style={{ display: "grid", gap: 6 }}>
            <span style={{ fontSize: 13, color: "#374151", fontWeight: 600 }}>
              Team A (Home)
            </span>
            <button
              type="button"
              onClick={() => openTeamModal("home")}
              disabled={loadingTeams}
              style={{
                width: "100%",
                textAlign: "left",
                padding: "10px 12px",
                borderRadius: 8,
                border: "1px solid #d1d5db",
                backgroundColor: "#ffffff",
                color: selectedHome ? "#111827" : "#9ca3af",
                fontSize: 14,
                cursor: loadingTeams ? "not-allowed" : "pointer",
              }}
            >
              {selectedHome || "Select Team"}
            </button>
          </div>

          <div style={{ display: "grid", gap: 6 }}>
            <span style={{ fontSize: 13, color: "#374151", fontWeight: 600 }}>
              Team B (Away)
            </span>
            <button
              type="button"
              onClick={() => openTeamModal("away")}
              disabled={loadingTeams}
              style={{
                width: "100%",
                textAlign: "left",
                padding: "10px 12px",
                borderRadius: 8,
                border: "1px solid #d1d5db",
                backgroundColor: "#ffffff",
                color: selectedAway ? "#111827" : "#9ca3af",
                fontSize: 14,
                cursor: loadingTeams ? "not-allowed" : "pointer",
              }}
            >
              {selectedAway || "Select Team"}
            </button>
          </div>

          <button
            type="button"
            onClick={handlePredict}
            disabled={loadingPrediction || loadingTeams}
            onMouseEnter={() => setButtonHover(true)}
            onMouseLeave={() => setButtonHover(false)}
            style={{
              width: "100%",
              border: "none",
              borderRadius: 8,
              padding: "11px 12px",
              backgroundColor: buttonBackground,
              color: "#ffffff",
              fontSize: 14,
              fontWeight: 600,
              cursor:
                loadingPrediction || loadingTeams ? "not-allowed" : "pointer",
              transition: "background-color 120ms ease-in-out",
              marginTop: 2,
            }}
          >
            {loadingPrediction ? "Predicting..." : "Predict"}
          </button>
        </div>

        {loadingTeams ? (
          <p style={{ marginTop: 12, color: "#6b7280", textAlign: "center" }}>
            Loading teams...
          </p>
        ) : null}

        {selectionError ? (
          <p
            style={{
              marginTop: 12,
              marginBottom: 0,
              color: "#dc2626",
              textAlign: "center",
              fontSize: 13,
            }}
          >
            {selectionError}
          </p>
        ) : null}

        {error ? (
          <p
            style={{
              marginTop: 12,
              marginBottom: 0,
              color: "#dc2626",
              textAlign: "center",
              fontSize: 14,
            }}
          >
            {error}
          </p>
        ) : null}

        {prediction ? (
          <section
            style={{
              marginTop: 16,
              backgroundColor: "#f3f4f6",
              borderRadius: 10,
              padding: 14,
              textAlign: "center",
            }}
          >
            <h2
              style={{
                margin: 0,
                fontSize: 18,
                color: "#111827",
              }}
            >
              {prediction.matchup}
            </h2>
            <p style={{ margin: "10px 0 0", color: "#374151", fontSize: 14 }}>
              {prediction.home_team}:{" "}
              {asPercent(prediction.home_win_probability)}
            </p>
            <p style={{ margin: "6px 0 0", color: "#374151", fontSize: 14 }}>
              {prediction.away_team}:{" "}
              {asPercent(prediction.away_win_probability)}
            </p>
            <p style={{ margin: "10px 0 0", color: "#111827", fontSize: 14 }}>
              Predicted winner: <strong>{prediction.predicted_winner}</strong>
            </p>
          </section>
        ) : null}
      </section>

      {modalOpen ? (
        <div
          onClick={closeTeamModal}
          style={{
            position: "fixed",
            inset: 0,
            backgroundColor: "rgba(15, 23, 42, 0.55)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: 16,
            zIndex: 1000,
          }}
        >
          <div
            onClick={(event) => event.stopPropagation()}
            style={{
              width: "100%",
              maxWidth: 720,
              maxHeight: "80vh",
              overflowY: "auto",
              backgroundColor: "#ffffff",
              borderRadius: 12,
              padding: 18,
            }}
          >
            <h3
              style={{
                marginTop: 0,
                marginBottom: 14,
                textAlign: "center",
                color: "#111827",
              }}
            >
              Select Team
            </h3>

            {selectionError ? (
              <p
                style={{
                  marginTop: 0,
                  marginBottom: 12,
                  color: "#dc2626",
                  textAlign: "center",
                  fontSize: 13,
                }}
              >
                {selectionError}
              </p>
            ) : null}

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
                gap: 10,
              }}
            >
              {teams.map((team) => {
                const logoUrl = TEAM_LOGOS[team];
                const showLogo = logoUrl && !brokenLogos[team];
                const isHovered = hoveredTeam === team;

                return (
                  <button
                    key={team}
                    type="button"
                    onClick={() => handleSelectTeam(team)}
                    onMouseEnter={() => setHoveredTeam(team)}
                    onMouseLeave={() => setHoveredTeam("")}
                    style={{
                      border: "1px solid #e5e7eb",
                      borderRadius: 10,
                      padding: 10,
                      backgroundColor: isHovered ? "#f3f4f6" : "#ffffff",
                      cursor: "pointer",
                      textAlign: "center",
                      display: "grid",
                      gap: 8,
                      justifyItems: "center",
                      minHeight: 90,
                    }}
                  >
                    {showLogo ? (
                      <img
                        src={logoUrl}
                        alt={`${team} logo`}
                        width={34}
                        height={34}
                        style={{ objectFit: "contain" }}
                        onError={() =>
                          setBrokenLogos((prev) => ({ ...prev, [team]: true }))
                        }
                      />
                    ) : (
                      <div
                        style={{
                          width: 34,
                          height: 34,
                          borderRadius: "50%",
                          backgroundColor: "#e5e7eb",
                          color: "#374151",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: 11,
                          fontWeight: 700,
                        }}
                      >
                        {getInitials(team)}
                      </div>
                    )}
                    <span style={{ fontSize: 12, color: "#111827" }}>
                      {team}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}
