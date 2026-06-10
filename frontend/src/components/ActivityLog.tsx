export function ActivityLog({ lines }: { lines: string[] }) {
  return (
    <div className="panel">
      <h2>Activity</h2>
      <pre className="log">{lines.join("\n")}</pre>
    </div>
  );
}
