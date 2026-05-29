# StreamSense Frontend (Platform v2)

Vite + React + TypeScript web UI for the StreamSense platform. Talks to the FastAPI
backend (`api/app.py`) over REST (`/api/*`) and a WebSocket (`/ws`).

## Run (development)

```bash
# 1) Backend (from repo root)
pip install -r requirements-dev.txt          # or: pip install fastapi "uvicorn[standard]"
uvicorn api.app:app --reload --port 8000

# 2) Frontend (this folder)
npm install
npm run dev          # http://localhost:5173  (proxies /api and /ws to :8000)
```

## Build / typecheck

```bash
npm run build        # tsc -b && vite build  -> dist/
npm run typecheck
```

## Layout

- `src/api.ts` — typed REST client + WebSocket helper
- `src/App.tsx` — dashboard (discover, connect/disconnect, recording, live status, modality availability, activity log)
