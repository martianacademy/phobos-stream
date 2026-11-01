"""Tests for FastAPI HTTP gateway endpoints."""

from pathlib import Path
from fastapi.testclient import TestClient
import pytest
from phobos_stream.server.app import create_app
from phobos_stream.storage.layout import ModelShardsManifest
from phobos_stream.types import BackendType


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    manifest = ModelShardsManifest(
        model_name="test-server-dit",
        architecture="wan2.1",
        num_blocks=2,
        hidden_size=64,
    )
    app = create_app(manifest=manifest, backend_type=BackendType.CPU, output_dir=tmp_path)
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["architecture"] == "wan2.1"


def test_models_endpoint(client: TestClient) -> None:
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert "data" in data
    model_ids = [m["id"] for m in data["data"]]
    assert "wan2.1" in model_ids
    assert "cogvideox" in model_ids


def test_image_generation_endpoint(client: TestClient) -> None:
    payload = {
        "prompt": "Cyberpunk city neon rain",
        "model": "flux",
        "size": "128x128",
        "steps": 2,
    }
    res = client.post("/v1/images/generations", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "data" in data
    assert len(data["data"]) == 1
    assert data["data"][0]["b64_json"] is not None
    # Verify the generated PNG file is actually served from /outputs
    img_res = client.get(data["data"][0]["url"])
    assert img_res.status_code == 200
    assert img_res.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_video_generation_endpoint(client: TestClient) -> None:
    payload = {
        "prompt": "Soaring hawk over canyons",
        "model": "wan2.1",
        "height": 128,
        "width": 128,
        "num_frames": 5,
        "steps": 2,
    }
    res = client.post("/v1/videos/generations", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert "video_url" in data
    vid_res = client.get(data["video_url"])
    assert vid_res.status_code == 200
