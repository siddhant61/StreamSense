import { useEffect, useState, useCallback } from "react";
import {
  api, connectWebSocket,
  type Device, type SystemStatus, type WsEvent,
} from "./api";

const DEVICE_ICON: Record<string, string> = {
  muse: "🧠", bitalino: "💓", kinect: "🎥", e4: "⌚",
};

export default function App() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try { setStatus(await api.status()); } catch (e) { pushLog(`status error: ${e}`); }
  }, []);

  const pushLog = (line: string) =>
    setLog((l) => [`${new Date().toLocaleTimeString()}  ${line}`, ...l].slice(0, 100));

  useEffect(() => {
    refresh();
    const ws = connectWebSocket((e: WsEvent) => {
      if (e.type === "status") setStatus(e.payload as SystemStatus);
      else if (e.type === "device_update") refresh();
      else if (e.type === "recording") refresh();
      else if (e.type === "log") {
        const p = e.payload as { level: string; message: string };
        pushLog(`[${p.level}] ${p.message}`);
      }
    });
    ws.onclose = () => pushLog("websocket closed");
    return () => ws.close();
  }, [refresh]);

  const run = async (label: string, fn: () => Promise<unknown>) => {
    setBusy(true);
    try { await fn(); pushLog(label); } catch (e) { pushLog(`${label} failed: ${e}`); }
    finally { setBusy(false); await refresh(); }
  };

  const rec = status?.recording;

  return (
    <div className="app">
      <header>
        <h1>🧠 StreamSense</h1>
        <span className="subtitle">Multi-Device Recording Platform</span>
      </header>

      <section className="controls">
        <button disabled={busy} onClick={() => run("discover", () => api.discover())}>
          🔍 Discover devices
        </button>
        {rec?.active ? (
          <button className="rec stop" disabled={busy}
            onClick={() => run("stop recording", api.stopRecording)}>
            ⏹ Stop recording
          </button>
        ) : (
          <button className="rec start" disabled={busy}
            onClick={() => run("start recording", api.startRecording)}>
            🔴 Start recording
          </button>
        )}
        {rec?.active && <span className="session">● {rec.session_id}</span>}
      </section>

      <section className="grid">
        <div className="panel">
          <h2>Devices</h2>
          {(status?.devices ?? []).length === 0 && <p className="muted">No devices yet — discover.</p>}
          {(status?.devices ?? []).map((d) => (
            <DeviceCard key={d.id} d={d} busy={busy}
              onConnect={() => run(`connect ${d.name}`, () => api.connect(d.id))}
              onDisconnect={() => run(`disconnect ${d.name}`, () => api.disconnect(d.id))} />
          ))}
        </div>

        <div className="panel">
          <h2>Modalities</h2>
          {Object.entries(status?.driver_availability ?? {}).map(([t, a]) => (
            <div key={t} className="modality">
              <span>{DEVICE_ICON[t] ?? "📟"} {t}</span>
              <span className={a.available ? "ok" : "off"}>
                {a.available ? "available" : (a.live ? "unavailable" : "import-only")}
              </span>
              {!a.available && <small className="muted">{a.reason}</small>}
            </div>
          ))}
        </div>

        <div className="panel">
          <h2>Activity</h2>
          <pre className="log">{log.join("\n")}</pre>
        </div>
      </section>
    </div>
  );
}

function DeviceCard({ d, busy, onConnect, onDisconnect }: {
  d: Device; busy: boolean; onConnect: () => void; onDisconnect: () => void;
}) {
  const connected = d.state === "connected";
  return (
    <div className={`device ${d.state}`}>
      <div className="device-head">
        <span>{DEVICE_ICON[d.type] ?? "📟"} <strong>{d.name}</strong></span>
        <span className={`badge ${d.state}`}>{d.state}</span>
      </div>
      <div className="device-meta">
        <span className="muted">{d.address}</span>
        <span>SQ: {d.signal_quality == null ? "—" : `${Math.round(d.signal_quality * 100)}%`}</span>
      </div>
      {connected ? (
        <button disabled={busy} onClick={onDisconnect}>Disconnect</button>
      ) : (
        <button disabled={busy} onClick={onConnect}>Connect</button>
      )}
      {d.detail && <small className="muted">{d.detail}</small>}
    </div>
  );
}
