import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_root_endpoint() -> None:
    """Test the root diagnostic health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "VoiceShield AI Core Running"}


def test_websocket_stream_audio_chunks() -> None:
    """Test WebSocket audio streaming with silence skipping and active speech processing."""
    import numpy as np

    # 1. Silence chunk (all zeros)
    silence_chunk = b"\x00\x00" * 8000

    # 2. Active speech chunk (sine tone at 440 Hz)
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    sine = (np.sin(2 * np.pi * 440.0 * t) * 0.5 * 32767).astype(np.int16)
    speech_chunk = sine.tobytes()

    with client.websocket_connect("/ws/stream") as websocket:
        # First chunk: Silence
        websocket.send_bytes(silence_chunk)
        payload_1 = websocket.receive_json()

        assert payload_1["chunk_id"] == 1
        assert payload_1["vad_active"] is False
        assert payload_1["spoof_score"] == 0.0
        assert payload_1["status"] == "SILENCE"
        assert isinstance(payload_1["latency_ms"], float)
        assert payload_1["latency_ms"] < 50.0  # Well within latency budget

        # Second chunk: Active Speech
        websocket.send_bytes(speech_chunk)
        payload_2 = websocket.receive_json()

        assert payload_2["chunk_id"] == 2
        assert payload_2["vad_active"] is True
        assert isinstance(payload_2["spoof_score"], float)
        assert 0.0 <= payload_2["spoof_score"] <= 1.0
        assert payload_2["status"] in ["SAFE", "SYNTHETIC_ALERT"]
        assert isinstance(payload_2["latency_ms"], float)
        assert payload_2["latency_ms"] < 50.0

        # Assert metrics payload structure
        assert "metrics" in payload_2
        metrics = payload_2["metrics"]
        assert "rawnet2_score" in metrics
        assert "resnet_score" in metrics
        assert "fused_score" in metrics
        assert "consecutive_threats" in metrics
        assert "consecutive_safe" in metrics

