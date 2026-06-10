import { useEffect, useState } from "react";
import type { RecordingState } from "../api";

function fmt(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = String(Math.floor(s / 3600)).padStart(2, "0");
  const m = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const sec = String(s % 60).padStart(2, "0");
  return `${h}:${m}:${sec}`;
}

export function SessionBar({ recording, busy, onStart, onStop }: {
  recording: RecordingState;
  busy: boolean;
  onStart: () => void;
  onStop: () => void;
}) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!recording.active || recording.started_at == null) {
      setElapsed(0);
      return;
    }
    const started = recording.started_at;
    const tick = () => setElapsed(Date.now() / 1000 - started);
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [recording.active, recording.started_at]);

  return (
    <div className="session-bar">
      {recording.active ? (
        <button className="rec stop" disabled={busy} onClick={onStop}>⏹ Stop recording</button>
      ) : (
        <button className="rec start" disabled={busy} onClick={onStart}>🔴 Start recording</button>
      )}
      {recording.active && (
        <span className="session-info">
          <span className="rec-dot" /> {recording.session_id} · {fmt(elapsed)}
          {recording.output_folder && <small className="muted"> → {recording.output_folder}</small>}
        </span>
      )}
    </div>
  );
}
