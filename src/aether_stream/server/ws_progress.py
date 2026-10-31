"""WebSocket progress manager for live generation monitoring."""

from typing import List
import json
from fastapi import WebSocket


class ProgressManager:
    """Broadcaster tracking active clients and broadcasting generation progress."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_progress(self, step: int, total_steps: int, stage: str) -> None:
        message = json.dumps({
            "step": step,
            "total_steps": total_steps,
            "progress_percent": round((step / max(1, total_steps)) * 100, 1),
            "stage": stage,
        })
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        for dead in disconnected:
            self.disconnect(dead)
