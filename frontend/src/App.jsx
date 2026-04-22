import { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function App() {
  const [teams, setTeams] = useState([]);
  const [home, setHome] = useState("");
  const [away, setAway] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [loadingTeams, setLoadingTeams] = useState(false);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [error, setError] = useState("");
  const [buttonHover, setButtonHover] = useState(false);

  useEffect(() => {
    const loadTeams = async () => {
      setLoadingTeams(true);
      setError("");
      try {
        const response = await fetch(`${API_BASE}/teams/`);
        if (!response.ok) {
          throw new Error(`Failed to load teams (${response.status})`);
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

  const getPrediction = async () => {
    setError("");
    setPrediction(null);

    if (!home || !away) {
      setError("Select Team A and Team B.");
      return;
    }
    if (home === away) {
      setError("Team A and Team B must be different.");
      return;
    }

    setLoadingPrediction(true);
    try {
      const url = `${API_BASE}/predict/quick?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`;
      const response = await fetch(url);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to fetch prediction");
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
          <label style={{ display: "grid", gap: 6 }}>
            <span style={{ fontSize: 13, color: "#374151", fontWeight: 600 }}>
              Team A (Home)
            </span>
            <select
              value={home}
              onChange={(event) => setHome(event.target.value)}
              disabled={loadingTeams}
              style={{
                width: "100%",
                padding: "10px 12px",
                borderRadius: 8,
                border: "1px solid #d1d5db",
                backgroundColor: "#ffffff",
                color: "#111827",
                fontSize: 14,
              }}
            >
              <option value="">Select team</option>
              {teams.map((team) => (
                <option key={team} value={team}>
                  {team}
                </option>
              ))}
            </select>
          </label>

          <label style={{ display: "grid", gap: 6 }}>
            <span style={{ fontSize: 13, color: "#374151", fontWeight: 600 }}>
              Team B (Away)
            </span>
            <select
              value={away}
              onChange={(event) => setAway(event.target.value)}
              disabled={loadingTeams}
              style={{
                width: "100%",
                padding: "10px 12px",
                borderRadius: 8,
                border: "1px solid #d1d5db",
                backgroundColor: "#ffffff",
                color: "#111827",
                fontSize: 14,
              }}
            >
              <option value="">Select team</option>
              {teams.map((team) => (
                <option key={team} value={team}>
                  {team}
                </option>
              ))}
            </select>
          </label>

          <button
            type="button"
            onClick={getPrediction}
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
    </main>
  );
}
