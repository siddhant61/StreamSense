import type { Device } from "../api";
import { SignalQuality } from "./SignalQuality";

const ICON: Record<string, string> = { muse: "🧠", bitalino: "💓", kinect: "🎥", e4: "⌚" };

export function DeviceCard({ d, busy, onConnect, onDisconnect }: {
  d: Device;
  busy: boolean;
  onConnect: () => void;
  onDisconnect: () => void;
}) {
  const connected = d.state === "connected";
  return (
    <div className={`device ${d.state}`}>
      <div className="device-head">
        <span>{ICON[d.type] ?? "📟"} <strong>{d.name}</strong></span>
        <span className={`badge ${d.state}`}>{d.state}</span>
      </div>
      <div className="device-meta">
        <span className="muted">{d.address || d.type}</span>
        {d.streams.length > 0 && <span className="muted">{d.streams.length} streams</span>}
      </div>
      <SignalQuality q={d.signal_quality} />
      {connected ? (
        <button disabled={busy} onClick={onDisconnect}>Disconnect</button>
      ) : (
        <button disabled={busy} onClick={onConnect}>Connect</button>
      )}
      {d.detail && <small className="muted">{d.detail}</small>}
    </div>
  );
}
