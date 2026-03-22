"""
Async API with WebSocket Support for Real-Time Training

Provides:
- Async endpoints for non-blocking operations
- WebSocket connections for real-time training progress
- Background task processing
- Real-time collaboration features
"""

import asyncio
import json
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import torch
import torch.nn as nn
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    BackgroundTasks,
    HTTPException,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from dataclasses import dataclass, asdict
import uuid


@dataclass
class TrainingJob:
    """Represents a training job."""

    job_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0.0 to 100.0
    current_epoch: int
    total_epochs: int
    current_loss: Optional[float]
    metrics: Dict[str, List[float]]
    created_at: str
    completed_at: Optional[str]
    error_message: Optional[str]
    model_config: Optional[Dict]


class TrainingJobManager:
    """Manages async training jobs."""

    def __init__(self):
        self.jobs: Dict[str, TrainingJob] = {}
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self.websocket_connections: Dict[str, List[WebSocket]] = {}

    def create_job(self, model_config: Dict, total_epochs: int) -> str:
        """Create a new training job."""
        job_id = str(uuid.uuid4())
        job = TrainingJob(
            job_id=job_id,
            status="pending",
            progress=0.0,
            current_epoch=0,
            total_epochs=total_epochs,
            current_loss=None,
            metrics={"train_loss": [], "val_loss": [], "accuracy": []},
            created_at=datetime.now().isoformat(),
            completed_at=None,
            error_message=None,
            model_config=model_config,
        )
        self.jobs[job_id] = job
        self.websocket_connections[job_id] = []
        return job_id

    async def start_training(
        self,
        job_id: str,
        model: nn.Module,
        train_data: torch.utils.data.DataLoader,
        val_data: Optional[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epochs: int,
    ):
        """Start training process asynchronously."""
        job = self.jobs[job_id]
        job.status = "running"

        try:
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_losses = []

                for batch_idx, (data, target) in enumerate(train_data):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    train_losses.append(loss.item())

                    # Update progress every 10 batches
                    if batch_idx % 10 == 0:
                        progress = (
                            (epoch * len(train_data) + batch_idx)
                            / (epochs * len(train_data))
                        ) * 100
                        job.progress = progress
                        job.current_loss = loss.item()
                        await self._notify_clients(
                            job_id,
                            {
                                "type": "progress",
                                "epoch": epoch,
                                "batch": batch_idx,
                                "loss": loss.item(),
                                "progress": progress,
                            },
                        )

                avg_train_loss = np.mean(train_losses)
                job.metrics["train_loss"].append(avg_train_loss)
                job.current_epoch = epoch + 1

                # Validation phase
                if val_data:
                    model.eval()
                    val_losses = []
                    correct = 0
                    total = 0

                    with torch.no_grad():
                        for data, target in val_data:
                            output = model(data)
                            val_loss = criterion(output, target)
                            val_losses.append(val_loss.item())

                            _, predicted = torch.max(output.data, 1)
                            total += target.size(0)
                            correct += (predicted == target).sum().item()

                    avg_val_loss = np.mean(val_losses)
                    accuracy = 100 * correct / total

                    job.metrics["val_loss"].append(avg_val_loss)
                    job.metrics["accuracy"].append(accuracy)

                    await self._notify_clients(
                        job_id,
                        {
                            "type": "epoch_complete",
                            "epoch": epoch + 1,
                            "train_loss": avg_train_loss,
                            "val_loss": avg_val_loss,
                            "accuracy": accuracy,
                            "progress": ((epoch + 1) / epochs) * 100,
                        },
                    )

                # Small delay to prevent blocking
                await asyncio.sleep(0.01)

            job.status = "completed"
            job.progress = 100.0
            job.completed_at = datetime.now().isoformat()

            await self._notify_clients(
                job_id,
                {
                    "type": "complete",
                    "message": "Training completed successfully",
                    "final_metrics": job.metrics,
                },
            )

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            await self._notify_clients(job_id, {"type": "error", "message": str(e)})

    async def _notify_clients(self, job_id: str, message: Dict):
        """Notify all WebSocket clients for a job."""
        if job_id in self.websocket_connections:
            disconnected = []
            for websocket in self.websocket_connections[job_id]:
                try:
                    await websocket.send_json(message)
                except:
                    disconnected.append(websocket)

            # Remove disconnected clients
            for ws in disconnected:
                self.websocket_connections[job_id].remove(ws)

    def register_websocket(self, job_id: str, websocket: WebSocket):
        """Register a WebSocket connection for a job."""
        if job_id not in self.websocket_connections:
            self.websocket_connections[job_id] = []
        self.websocket_connections[job_id].append(websocket)

    def unregister_websocket(self, job_id: str, websocket: WebSocket):
        """Unregister a WebSocket connection."""
        if job_id in self.websocket_connections:
            if websocket in self.websocket_connections[job_id]:
                self.websocket_connections[job_id].remove(websocket)

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get job by ID."""
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[TrainingJob]:
        """List all jobs."""
        return list(self.jobs.values())

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        if job_id in self.active_jobs:
            self.active_jobs[job_id].cancel()
            self.jobs[job_id].status = "cancelled"
            return True
        return False


# Global job manager
job_manager = TrainingJobManager()


# Pydantic models for API
class AsyncTrainRequest(BaseModel):
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 10
    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32


class AsyncPredictRequest(BaseModel):
    job_id: str
    data: List[List[float]]


# Async API routes
from fastapi import APIRouter

router = APIRouter()


@router.post("/async/train")
async def async_train(request: AsyncTrainRequest, background_tasks: BackgroundTasks):
    """
    Start async training job.

    Returns immediately with job_id. Use WebSocket to track progress.
    """
    job_id = job_manager.create_job(
        model_config={
            "input_dim": request.input_dim,
            "hidden_dim": request.hidden_dim,
            "output_dim": request.output_dim,
        },
        total_epochs=request.epochs,
    )

    # Create simple model and data for demo
    model = nn.Sequential(
        nn.Linear(request.input_dim, request.hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(request.hidden_dim, request.output_dim),
    )

    # Create dummy data loaders
    train_data = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(1000, request.input_dim),
            torch.randint(0, request.output_dim, (1000,)),
        ),
        batch_size=request.batch_size,
        shuffle=True,
    )

    val_data = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(200, request.input_dim),
            torch.randint(0, request.output_dim, (200,)),
        ),
        batch_size=request.batch_size,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=request.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Start training in background
    task = asyncio.create_task(
        job_manager.start_training(
            job_id, model, train_data, val_data, optimizer, criterion, request.epochs
        )
    )
    job_manager.active_jobs[job_id] = task

    return {
        "job_id": job_id,
        "status": "started",
        "message": "Training started. Connect to WebSocket to track progress.",
        "websocket_url": f"/ws/train/{job_id}",
    }


@router.get("/async/jobs")
async def list_jobs():
    """List all training jobs."""
    jobs = job_manager.list_jobs()
    return {"jobs": [asdict(job) for job in jobs], "count": len(jobs)}


@router.get("/async/jobs/{job_id}")
async def get_job(job_id: str):
    """Get specific job details."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return asdict(job)


@router.post("/async/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    success = job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Job not found or not running")
    return {"message": "Job cancelled", "job_id": job_id}


# WebSocket endpoint
@router.websocket("/ws/train/{job_id}")
async def training_websocket(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time training updates.

    Connect to this endpoint to receive live training progress.
    """
    await websocket.accept()

    # Check if job exists
    job = job_manager.get_job(job_id)
    if not job:
        await websocket.send_json({"type": "error", "message": "Job not found"})
        await websocket.close()
        return

    # Register connection
    job_manager.register_websocket(job_id, websocket)

    try:
        # Send initial status
        await websocket.send_json(
            {
                "type": "connected",
                "job_id": job_id,
                "status": job.status,
                "current_epoch": job.current_epoch,
                "total_epochs": job.total_epochs,
            }
        )

        # Keep connection open and listen for client messages
        while True:
            try:
                # Wait for client messages (with timeout)
                message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)

                data = json.loads(message)
                action = data.get("action")

                if action == "ping":
                    await websocket.send_json({"type": "pong"})
                elif action == "get_status":
                    job = job_manager.get_job(job_id)
                    await websocket.send_json(
                        {
                            "type": "status",
                            "status": job.status,
                            "progress": job.progress,
                            "current_epoch": job.current_epoch,
                            "metrics": job.metrics,
                        }
                    )

            except asyncio.TimeoutError:
                # No message received, check if job is still active
                job = job_manager.get_job(job_id)
                if job.status in ["completed", "failed", "cancelled"]:
                    # Send final status and close
                    await websocket.send_json(
                        {
                            "type": "status",
                            "status": job.status,
                            "progress": job.progress,
                            "message": job.error_message
                            if job.status == "failed"
                            else "Training complete",
                        }
                    )
                    break
                continue

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        job_manager.unregister_websocket(job_id, websocket)
        await websocket.close()


# Collaborative workspace WebSocket
class CollaborationSession:
    """Manages collaborative editing sessions."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.clients: Dict[str, WebSocket] = {}
        self.messages: List[Dict] = []
        self.created_at = datetime.now().isoformat()

    async def broadcast(self, message: Dict, exclude: Optional[str] = None):
        """Broadcast message to all clients."""
        disconnected = []
        for client_id, websocket in self.clients.items():
            if client_id != exclude:
                try:
                    await websocket.send_json(message)
                except:
                    disconnected.append(client_id)

        for client_id in disconnected:
            del self.clients[client_id]

    def add_client(self, client_id: str, websocket: WebSocket):
        """Add client to session."""
        self.clients[client_id] = websocket

    def remove_client(self, client_id: str):
        """Remove client from session."""
        if client_id in self.clients:
            del self.clients[client_id]


# Collaboration session manager
collaboration_sessions: Dict[str, CollaborationSession] = {}


@router.websocket("/ws/collaborate/{session_id}")
async def collaboration_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time collaboration.

    Multiple researchers can connect to collaborate on model development.
    """
    await websocket.accept()

    # Get or create session
    if session_id not in collaboration_sessions:
        collaboration_sessions[session_id] = CollaborationSession(session_id)

    session = collaboration_sessions[session_id]
    client_id = str(uuid.uuid4())
    session.add_client(client_id, websocket)

    try:
        # Notify others of new participant
        await session.broadcast(
            {
                "type": "user_joined",
                "client_id": client_id,
                "total_clients": len(session.clients),
            },
            exclude=client_id,
        )

        # Send session history
        await websocket.send_json(
            {
                "type": "connected",
                "client_id": client_id,
                "session_id": session_id,
                "history": session.messages,
            }
        )

        # Listen for messages
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            # Store message
            data["timestamp"] = datetime.now().isoformat()
            data["client_id"] = client_id
            session.messages.append(data)

            # Broadcast to all clients
            await session.broadcast(data)

    except WebSocketDisconnect:
        pass
    finally:
        session.remove_client(client_id)
        await session.broadcast(
            {
                "type": "user_left",
                "client_id": client_id,
                "total_clients": len(session.clients),
            }
        )


# Export
__all__ = [
    "job_manager",
    "TrainingJob",
    "TrainingJobManager",
    "router",
    "AsyncTrainRequest",
    "CollaborationSession",
]
