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

async function jpost<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
  return res.json();
}

export const api = {
  status: (): Promise<SystemStatus> => fetch("/api/status").then((r) => r.json()),
  discover: (types?: string[]): Promise<Device[]> => jpost("/api/discover", { types }),
  connect: (id: string) => jpost(`/api/devices/${encodeURIComponent(id)}/connect`),
  disconnect: (id: string) => jpost(`/api/devices/${encodeURIComponent(id)}/disconnect`),
  startRecording: () => jpost("/api/recording/start"),
  stopRecording: () => jpost("/api/recording/stop"),
};

// Live event stream over WebSocket.
export interface WsEvent {
  type: "status" | "device_update" | "recording" | "log";
  payload: unknown;
  ts: number;
}

export function connectWebSocket(onEvent: (e: WsEvent) => void): WebSocket {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (m) => onEvent(JSON.parse(m.data));
  return ws;
}
