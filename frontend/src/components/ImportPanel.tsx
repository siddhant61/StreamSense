import { useState } from "react";
import { api, type E4Summary } from "../api";

export function ImportPanel() {
  const [path, setPath] = useState("");
  const [summary, setSummary] = useState<E4Summary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const doImport = async () => {
    if (!path.trim()) return;
    setBusy(true);
    setError(null);
    try {
      setSummary(await api.importE4(path.trim()));
    } catch (e) {
      setSummary(null);
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel">
      <h2>⌚ Import E4 session</h2>
      <p className="muted">Offline only — Empatica withdrew live streaming. Folder or .zip.</p>
      <div className="import-row">
        <input
          type="text"
          placeholder="/path/to/E4 session"
          value={path}
          onChange={(e) => setPath(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && doImport()}
        />
        <button disabled={busy || !path.trim()} onClick={doImport}>Import</button>
      </div>
      {error && <small className="off">{error}</small>}
      {summary && (
        <table className="e4">
          <tbody>
            {Object.entries(summary.signals).map(([name, s]) => (
              <tr key={name}>
                <td>{name}</td>
                <td className="muted">{s.sample_rate} Hz × {s.channels}</td>
                <td className="muted">{s.duration}s</td>
              </tr>
            ))}
            <tr><td>IBI</td><td className="muted" colSpan={2}>{summary.ibi_count} beats</td></tr>
            <tr><td>Tags</td><td className="muted" colSpan={2}>{summary.tags.length}</td></tr>
          </tbody>
        </table>
      )}
    </div>
  );
}
