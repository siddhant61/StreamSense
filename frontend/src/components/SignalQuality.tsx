import { qualityView } from "../api";

export function SignalQuality({ q }: { q: number | null }) {
  const v = qualityView(q);
  return (
    <div className="sq" title="signal quality">
      <div className="sq-track">
        <div className="sq-fill" style={{ width: `${v.pct}%`, background: v.color }} />
      </div>
      <span className="sq-label" style={{ color: v.color }}>
        {q == null ? "—" : `${v.pct}% ${v.label}`}
      </span>
    </div>
  );
}
