import { useEffect, useState } from "react";
import PredictionCard from "./components/PredictionCard";
import TeamSelector from "./components/TeamSelector";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function App() {
  const [teams, setTeams] = useState([]);
  const [home, setHome] = useState("");
  const [away, setAway] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [loadingTeams, setLoadingTeams] = useState(true);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const loadTeams = async () => {
      try {
        const response = await fetch(`${API_BASE}/teams/`);
        if (!response.ok) {
          throw new Error(`Failed to load teams (${response.status})`);
        }
        const payload = await response.json();
        setTeams(payload);
      } catch (err) {
        setError(err.message || "Failed to load teams");
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
        throw new Error(payload.detail || "Prediction request failed");
      }
      setPrediction(payload);
    } catch (err) {
      setError(err.message || "Prediction request failed");
    } finally {
      setLoadingPrediction(false);
    }
  };

  return (
    <main
      style={{
        maxWidth: 640,
        margin: "40px auto",
        padding: "0 16px",
        fontFamily: "sans-serif",
      }}
    >
      <h1 style={{ marginBottom: 8 }}>NBA Matchup Predictor</h1>
      <p style={{ marginTop: 0, color: "#666" }}>
        Select two teams and run a quick prediction.
      </p>

      {loadingTeams ? <p>Loading teams...</p> : null}

      {!loadingTeams ? (
        <section style={{ display: "grid", gap: 12 }}>
          <TeamSelector
            label="Team A (Home)"
            teams={teams}
            value={home}
            onChange={setHome}
          />
          <TeamSelector
            label="Team B (Away)"
            teams={teams}
            value={away}
            onChange={setAway}
          />
          <button
            type="button"
            onClick={getPrediction}
            disabled={loadingPrediction}
          >
            {loadingPrediction ? "Predicting..." : "Predict"}
          </button>
        </section>
      ) : null}

      {error ? (
        <p style={{ color: "#b00020", marginTop: 16 }}>{error}</p>
      ) : null}
      <PredictionCard prediction={prediction} />
    </main>
  );
}
