"""Live integration test testing PhobosStream API gateway."""

import subprocess
import sys
import time
import httpx


def test_live_server() -> None:
    port = 8089
    print(f"[*] Starting PhobosStream API server on port {port}...")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "phobos_stream.server.app:create_app",
            "--factory",
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for server to boot
        time.sleep(2)
        base_url = f"http://127.0.0.1:{port}"

        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            # 1. Health check
            print("\n1. Testing GET /health...")
            res = client.get("/health")
            print(f"   Status: {res.status_code} | Payload: {res.json()}")
            assert res.status_code == 200

            # 2. Models list
            print("\n2. Testing GET /v1/models...")
            res = client.get("/v1/models")
            models = res.json()["data"]
            print(f"   Available models: {[m['id'] for m in models]}")
            assert len(models) >= 3

            # 3. Image Generation
            print("\n3. Testing POST /v1/images/generations...")
            img_payload = {
                "prompt": "Cybernetic crystal lotus blooming on Mars, neon bioluminescence",
                "model": "flux-schnell",
                "size": "512x512",
                "steps": 2,
            }
            res = client.post("/v1/images/generations", json=img_payload)
            print(f"   Status: {res.status_code} | Image response: {res.json()}")
            assert res.status_code == 200

            # 4. Video Generation
            print("\n4. Testing POST /v1/videos/generations...")
            vid_payload = {
                "prompt": "Orbital space station descending through Jupiter's storm clouds",
                "model": "wan2.1",
                "height": 256,
                "width": 256,
                "num_frames": 5,
                "steps": 2,
            }
            res = client.post("/v1/videos/generations", json=vid_payload)
            print(f"   Status: {res.status_code} | Video response: {res.json()}")
            assert res.status_code == 200

        print("\n" + "=" * 60)
        print("✅ ALL LIVE SERVER INTEGRATION TESTS PASSED PERFECTLY!")
        print("=" * 60)

    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    test_live_server()
