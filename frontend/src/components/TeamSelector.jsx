export default function TeamSelector({ label, teams, value, onChange }) {
  return (
    <label style={{ display: "grid", gap: 6 }}>
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">Select team</option>
        {teams.map((team) => (
          <option key={team} value={team}>
            {team}
          </option>
        ))}
      </select>
    </label>
  );
}

