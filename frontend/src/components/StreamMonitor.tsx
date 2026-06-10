export function StreamMonitor({ streams }: { streams: string[] }) {
  return (
    <div className="panel">
      <h2>Active streams <span className="count">{streams.length}</span></h2>
      {streams.length === 0 ? (
        <p className="muted">No LSL streams resolved.</p>
      ) : (
        <ul className="streams">
          {streams.map((s) => (
            <li key={s}><span className="dot" /> {s}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
