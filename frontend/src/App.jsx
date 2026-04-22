import { useEffect, useMemo, useState } from "react";

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
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [error, setError] = useState("");
  const [selectionError, setSelectionError] = useState("");
  const [buttonHover, setButtonHover] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [selectingType, setSelectingType] = useState(null);
  const [hoveredTeam, setHoveredTeam] = useState("");
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
    setHoveredTeam("");
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
          maxWidth: 420,
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
            backgroundColor: "rgba(15, 23, 42, 0.58)",
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
              maxWidth: 860,
              maxHeight: "82vh",
              overflowY: "auto",
              backgroundColor: "#ffffff",
              borderRadius: 14,
              padding: 20,
              boxShadow: "0 18px 40px rgba(15, 23, 42, 0.24)",
            }}
          >
            <h3
              style={{
                marginTop: 0,
                marginBottom: 12,
                textAlign: "center",
                color: "#111827",
                fontSize: 22,
              }}
            >
              Select Team
            </h3>

            <input
              type="text"
              value={teamSearch}
              onChange={(event) => setTeamSearch(event.target.value)}
              placeholder="Search teams..."
              style={{
                width: "100%",
                boxSizing: "border-box",
                border: "1px solid #d1d5db",
                borderRadius: 10,
                padding: "10px 12px",
                fontSize: 14,
                color: "#111827",
                marginBottom: 14,
                outline: "none",
              }}
            />

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
                gridTemplateColumns: "repeat(auto-fit, minmax(135px, 1fr))",
                gap: 12,
              }}
            >
              {filteredTeams.map((team) => {
                const logoUrl = TEAM_LOGOS[team];
                const showLogo = Boolean(logoUrl && !brokenLogos[team]);
                const isHovered = hoveredTeam === team;
                const isSelected = selectedInModal === team;

                return (
                  <button
                    key={team}
                    type="button"
                    onClick={() => handleSelectTeam(team)}
                    onMouseEnter={() => setHoveredTeam(team)}
                    onMouseLeave={() => setHoveredTeam("")}
                    style={{
                      border: isSelected ? "2px solid #2563eb" : "1px solid #e5e7eb",
                      borderRadius: 12,
                      padding: 12,
                      backgroundColor: isSelected ? "#eff6ff" : "#ffffff",
                      cursor: "pointer",
                      textAlign: "center",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: 10,
                      minHeight: 120,
                      boxShadow:
                        isHovered || isSelected
                          ? "0 8px 18px rgba(15, 23, 42, 0.12)"
                          : "0 1px 2px rgba(15, 23, 42, 0.04)",
                      transform: isHovered ? "scale(1.02)" : "scale(1)",
                      transition:
                        "transform 120ms ease, box-shadow 120ms ease, background-color 120ms ease, border-color 120ms ease",
                    }}
                  >
                    {showLogo ? (
                      <img
                        src={logoUrl}
                        alt={`${team} logo`}
                        width={50}
                        height={50}
                        style={{ objectFit: "contain" }}
                        onError={() =>
                          setBrokenLogos((prev) => ({ ...prev, [team]: true }))
                        }
                      />
                    ) : (
                      <div
                        style={{
                          width: 50,
                          height: 50,
                          borderRadius: "50%",
                          backgroundColor: "#e5e7eb",
                          color: "#374151",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: 14,
                          fontWeight: 700,
                        }}
                      >
                        {getInitials(team)}
                      </div>
                    )}
                    <span
                      style={{
                        fontSize: 13,
                        color: "#111827",
                        fontWeight: 600,
                        lineHeight: 1.25,
                      }}
                    >
                      {team}
                    </span>
                  </button>
                );
              })}
            </div>

            {filteredTeams.length === 0 ? (
              <p
                style={{
                  marginTop: 14,
                  marginBottom: 0,
                  color: "#6b7280",
                  textAlign: "center",
                  fontSize: 14,
                }}
              >
                No teams found.
              </p>
            ) : null}
          </div>
        </div>
      ) : null}
    </main>
  );
}
