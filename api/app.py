"""FastAPI application exposing the DeviceManager over REST + WebSocket.

REST endpoints are declared as sync `def` so FastAPI runs the (blocking) device calls in
a worker thread, keeping the event loop free for the WebSocket. The WebSocket bridges
DeviceManager events into the asyncio loop via ``loop.call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from core import DeviceManager, DeviceManagerError


class DiscoverRequest(BaseModel):
    types: Optional[List[str]] = None


def create_app(manager: Optional[DeviceManager] = None) -> FastAPI:
    app = FastAPI(title="StreamSense Platform API", version="2.0.0-dev")
    app.state.manager = manager or DeviceManager()

    def mgr() -> DeviceManager:
        return app.state.manager

    @app.exception_handler(DeviceManagerError)
    async def _dm_error(_request, exc: DeviceManagerError):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/api/health")
    def health():
        return {"status": "ok", "ts": time.time()}

    @app.get("/api/devices")
    def list_devices():
        return [d.to_dict() for d in mgr().devices.values()]

    @app.post("/api/discover")
    def discover(req: DiscoverRequest):
        return [d.to_dict() for d in mgr().discover(req.types)]

    @app.post("/api/devices/{device_id}/connect")
    def connect(device_id: str):
        ok = mgr().connect(device_id)
        return {"ok": ok, "device": mgr().devices[device_id].to_dict()}

    @app.post("/api/devices/{device_id}/disconnect")
    def disconnect(device_id: str):
        ok = mgr().disconnect(device_id)
        return {"ok": ok, "device": mgr().devices[device_id].to_dict()}

    @app.post("/api/recording/start")
    def start_recording():
        ok = mgr().start_recording()
        return {"ok": ok, "recording": mgr().recording.to_dict()}

    @app.post("/api/recording/stop")
    def stop_recording():
        ok = mgr().stop_recording()
        return {"ok": ok, "recording": mgr().recording.to_dict()}

    @app.get("/api/status")
    def status():
        return mgr().get_status().to_dict()

    @app.get("/api/streams")
    def streams():
        return {"streams": mgr().list_streams()}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def listener(event: dict) -> None:
            # Called from device/recorder threads -> hop back onto the event loop.
            loop.call_soon_threadsafe(queue.put_nowait, event)

        mgr().add_listener(listener)
        try:
            await websocket.send_json(
                {"type": "status", "payload": mgr().get_status().to_dict(), "ts": time.time()}
            )
            while True:
                event = await queue.get()
                await websocket.send_json(event)
        except WebSocketDisconnect:
            pass
        finally:
            mgr().remove_listener(listener)

    return app


# Module-level app for `uvicorn api.app:app`
app = create_app()
