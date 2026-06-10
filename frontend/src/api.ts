// Typed client for the StreamSense Platform API.

export type ConnectionState =
  | "discovered" | "connecting" | "connected"
  | "disconnecting" | "disconnected" | "error";

export interface Device {
  id: string;
  name: string;
  type: string;
  address: string;
  state: ConnectionState;
  signal_quality: number | null;
  detail: string | null;
  streams: string[];
}

export interface RecordingState {
  active: boolean;
  session_id: string | null;
  output_folder: string | null;
  started_at: number | null;
}

export interface DriverAvailability {
  available: boolean;
  reason: string;
  live: boolean;
}

export interface SystemStatus {
  devices: Device[];
  recording: RecordingState;
  driver_availability: Record<string, DriverAvailability>;
}

async function detail(res: Response): Promise<string> {
  return (await res.json().catch(() => ({} as { detail?: string }))).detail ?? res.statusText;
}

async function jget<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

async function jpost<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export interface E4Summary {
  source: string;
  start_time: number | null;
  signals: Record<string, { sample_rate: number; channels: number; n: number; duration: number }>;
  ibi_count: number;
  tags: number[];
}

export const api = {
  status: (): Promise<SystemStatus> => jget("/api/status"),
  streams: (): Promise<{ streams: string[] }> => jget("/api/streams"),
  discover: (types?: string[]): Promise<Device[]> => jpost("/api/discover", { types }),
  importE4: (path: string): Promise<E4Summary> => jpost("/api/import/e4", { path }),
  connect: (id: string) => jpost(`/api/devices/${encodeURIComponent(id)}/connect`),
  disconnect: (id: string) => jpost(`/api/devices/${encodeURIComponent(id)}/disconnect`),
  startRecording: () => jpost("/api/recording/start"),
  stopRecording: () => jpost("/api/recording/stop"),
};

// Live event stream over WebSocket.
export type WsEventType = "status" | "device_update" | "recording" | "log" | "joints";

export interface WsEvent {
  type: WsEventType;
  payload: unknown;
  ts: number;
}

export interface JointsPayload {
  device_id: string;
  points: [number, number, number][]; // [x, y, confidence] per joint
}

export interface LogPayload {
  level: string;
  message: string;
}

// Map a 0..1 quality (or null) to a label + colour for the UI.
export function qualityView(q: number | null): { label: string; color: string; pct: number } {
  if (q == null) return { label: "unknown", color: "var(--muted)", pct: 0 };
  const pct = Math.round(q * 100);
  if (q >= 0.8) return { label: "good", color: "var(--ok)", pct };
  if (q >= 0.5) return { label: "fair", color: "var(--warn)", pct };
  return { label: "poor", color: "var(--off)", pct };
}

export function connectWebSocket(onEvent: (e: WsEvent) => void): WebSocket {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (m) => onEvent(JSON.parse(m.data));
  return ws;
}
