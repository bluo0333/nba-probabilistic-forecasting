function asPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

export default function PredictionCard({ prediction }) {
  if (!prediction) {
    return null;
  }

  return (
    <section
      style={{
        marginTop: 24,
        padding: 16,
        border: "1px solid #d9d9d9",
        borderRadius: 8,
      }}
    >
      <h2 style={{ marginTop: 0 }}>{prediction.matchup}</h2>
      <p style={{ margin: "8px 0" }}>
        <strong>{prediction.home_team}</strong>:{" "}
        {asPercent(prediction.home_win_probability)}
      </p>
      <p style={{ margin: "8px 0" }}>
        <strong>{prediction.away_team}</strong>:{" "}
        {asPercent(prediction.away_win_probability)}
      </p>
      <p style={{ margin: "8px 0" }}>
        Predicted winner: <strong>{prediction.predicted_winner}</strong>
      </p>
    </section>
  );
}
