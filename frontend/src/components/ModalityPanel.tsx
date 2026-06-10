import type { DriverAvailability } from "../api";

const ICON: Record<string, string> = { muse: "🧠", bitalino: "💓", kinect: "🎥", e4: "⌚" };

export function ModalityPanel({ availability }: {
  availability: Record<string, DriverAvailability>;
}) {
  return (
    <div className="panel">
      <h2>Modalities</h2>
      {Object.entries(availability).map(([t, a]) => (
        <div key={t} className="modality">
          <span className="modality-row">
            <span>{ICON[t] ?? "📟"} {t}</span>
            <span className={a.available ? "ok" : "off"}>
              {a.available ? "available" : a.live ? "unavailable" : "import-only"}
            </span>
          </span>
          {!a.available && <small className="muted">{a.reason}</small>}
        </div>
      ))}
    </div>
  );
}
