import { useCallback, useEffect, useState } from "react";
import {
  api, connectWebSocket,
  type SystemStatus, type WsEvent, type JointsPayload, type LogPayload,
} from "./api";
import type { Point } from "./skeleton";
import { DeviceCard } from "./components/DeviceCard";
import { ModalityPanel } from "./components/ModalityPanel";
import { StreamMonitor } from "./components/StreamMonitor";
import { SessionBar } from "./components/SessionBar";
import { SkeletonCanvas } from "./components/SkeletonCanvas";
import { ActivityLog } from "./components/ActivityLog";

export default function App() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [streams, setStreams] = useState<string[]>([]);
  const [joints, setJoints] = useState<Point[] | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [online, setOnline] = useState(false);

  const pushLog = useCallback((line: string) => {
    setLog((l) => [`${new Date().toLocaleTimeString()}  ${line}`, ...l].slice(0, 200));
  }, []);

  const refresh = useCallback(async () => {
    try {
      setStatus(await api.status());
    } catch (e) {
      pushLog(`status error: ${e}`);
    }
  }, [pushLog]);

  const refreshStreams = useCallback(async () => {
    try {
      const r = await api.streams();
      setStreams(r.streams);
    } catch {
      /* transient */
    }
  }, []);

  useEffect(() => {
    refresh();
    refreshStreams();
    const ws = connectWebSocket((e: WsEvent) => {
      switch (e.type) {
        case "status":
          setStatus(e.payload as SystemStatus);
          setOnline(true);
          break;
        case "device_update":
        case "recording":
          refresh();
          break;
        case "joints":
          setJoints((e.payload as JointsPayload).points);
          break;
        case "log": {
          const p = e.payload as LogPayload;
          pushLog(`[${p.level}] ${p.message}`);
          break;
        }
      }
    });
    ws.onopen = () => setOnline(true);
    ws.onclose = () => {
      setOnline(false);
      pushLog("websocket closed");
    };
    const streamPoll = setInterval(refreshStreams, 4000);
    return () => {
      clearInterval(streamPoll);
      ws.close();
    };
  }, [refresh, refreshStreams, pushLog]);

  const run = async (label: string, fn: () => Promise<unknown>) => {
    setBusy(true);
    try {
      await fn();
      pushLog(label);
    } catch (e) {
      pushLog(`${label} failed: ${e}`);
    } finally {
      setBusy(false);
      await refresh();
      await refreshStreams();
    }
  };

  const devices = status?.devices ?? [];
  const recording = status?.recording ?? { active: false, session_id: null, output_folder: null, started_at: null };
  const hasKinect = devices.some((d) => d.type === "kinect");

  return (
    <div className="app">
      <header>
        <h1>🧠 StreamSense</h1>
        <span className="subtitle">Multi-Device Recording Platform</span>
        <span className={`conn ${online ? "up" : "down"}`}>{online ? "● live" : "○ offline"}</span>
      </header>

      <section className="controls">
        <button disabled={busy} onClick={() => run("discover", () => api.discover())}>
          🔍 Discover devices
        </button>
        <SessionBar
          recording={recording}
          busy={busy}
          onStart={() => run("start recording", api.startRecording)}
          onStop={() => run("stop recording", api.stopRecording)}
        />
      </section>

      <section className="grid">
        <div className="panel">
          <h2>Devices <span className="count">{devices.length}</span></h2>
          {devices.length === 0 && <p className="muted">No devices yet — discover.</p>}
          {devices.map((d) => (
            <DeviceCard
              key={d.id}
              d={d}
              busy={busy}
              onConnect={() => run(`connect ${d.name}`, () => api.connect(d.id))}
              onDisconnect={() => run(`disconnect ${d.name}`, () => api.disconnect(d.id))}
            />
          ))}
        </div>

        <div className="col">
          <ModalityPanel availability={status?.driver_availability ?? {}} />
          <StreamMonitor streams={streams} />
        </div>

        <div className="col">
          {hasKinect && (
            <div className="panel">
              <h2>Kinect preview</h2>
              <SkeletonCanvas points={joints} />
            </div>
          )}
          <ActivityLog lines={log} />
        </div>
      </section>
    </div>
  );
}
