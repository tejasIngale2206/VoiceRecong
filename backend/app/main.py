from typing import Any, Dict
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
import io
import logging
import math
import statistics
import time
import torch
import torchaudio

from backend.app.dsp.feature_extractor import extract_features
from backend.app.dsp.preprocessor import extract_log_mel_spectrogram
from backend.app.dsp.vad import detect_voice_activity
from backend.app.engine.decision import HysteresisDecisionEngine
from backend.app.engine.inference import get_inference_engine

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("voiceshield.main")

# Initialize FastAPI application
app = FastAPI(
    title="VoiceShield AI Core",
    description="Real-Time Voice Anti-Spoofing & Deepfake Detection Engine (SIH26104)",
    version="1.0.0",
)

# Enable CORS for local frontend development environments
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, str]:
    """Root health-check and service diagnostic endpoint."""
    return {"status": "VoiceShield AI Core Running"}


@app.websocket("/ws/stream")
async def websocket_audio_stream(websocket: WebSocket) -> None:
    """Asynchronous WebSocket streaming endpoint for 16kHz binary PCM audio frames."""
    await websocket.accept()
    chunk_id: int = 0
    decision_engine = HysteresisDecisionEngine()
    inference_engine = get_inference_engine()
    logger.info("Client connected to /ws/stream audio pipeline.")

    try:
        while True:
            raw_audio_bytes: bytes = await websocket.receive_bytes()
            start_time: float = time.perf_counter()
            chunk_id += 1

            audio_buffer: io.BytesIO = io.BytesIO(raw_audio_bytes)

            is_speech, energy_db = detect_voice_activity(audio_buffer)

            if not is_speech:
                vad_active: bool = False
                spoof_score: float = 0.0
                status: str = "SILENCE"
                metrics: Dict[str, Any] = {
                    "energy_db": energy_db,
                    "rawnet2_score": 0.0,
                    "resnet_score": 0.0,
                    "fused_score": 0.0,
                    "consecutive_threats": decision_engine.consecutive_threats,
                    "consecutive_safe": decision_engine.consecutive_safe,
                }
            else:
                vad_active = True
                waveform_1d, log_mel = extract_features(audio_buffer)

                inference_result = inference_engine.infer(waveform_1d, log_mel)
                spoof_score = round(inference_result.fused_score, 4)

                decision_result = decision_engine.update(spoof_score)
                status = decision_result.status

                metrics = {
                    "energy_db": energy_db,
                    "rawnet2_score": round(inference_result.rawnet2_score, 4),
                    "resnet_score": round(inference_result.resnet_score, 4),
                    "fused_score": spoof_score,
                    "consecutive_threats": decision_result.consecutive_threats,
                    "consecutive_safe": decision_result.consecutive_safe,
                    "device": str(inference_result.device),
                }

            elapsed_time_ms: float = (time.perf_counter() - start_time) * 1000.0
            latency_ms: float = round(elapsed_time_ms, 2)

            response_payload: Dict[str, Any] = {
                "chunk_id": chunk_id,
                "vad_active": vad_active,
                "spoof_score": spoof_score,
                "status": status,
                "latency_ms": latency_ms,
                "metrics": metrics,
            }

            await websocket.send_json(response_payload)

    except WebSocketDisconnect:
        logger.info(
            f"Client disconnected gracefully after processing {chunk_id} chunks."
        )
    except Exception as exc:
        logger.error(f"Unexpected error in streaming session: {exc}", exc_info=True)
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/api/v1/scan-file")
async def scan_audio_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    if not filename.endswith((".wav", ".mp3", ".flac", ".ogg")):
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported format. Please upload a .wav, .mp3, .flac, or .ogg"
                " file."
            ),
        )

    try:
        contents = await file.read()
        buffer = io.BytesIO(contents)

        try:
            waveform, sample_rate = torchaudio.load(buffer)
        except Exception:
            buffer.seek(0)
            import soundfile as sf

            data, sample_rate = sf.read(buffer, dtype="float32")
            waveform = torch.from_numpy(data)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim == 2:
                waveform = waveform.T

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )
            waveform = resampler(waveform)

        waveform_1d = waveform.squeeze(0)

        chunk_size = 8000  # 0.5 seconds at 16kHz
        total_samples = waveform_1d.shape[0]

        if total_samples < chunk_size:
            pad_size = chunk_size - total_samples
            waveform_1d = torch.nn.functional.pad(waveform_1d, (0, pad_size))
            total_samples = chunk_size

        engine = get_inference_engine()
        rawnet_scores = []
        resnet_scores = []
        fused_scores = []
        timeline = []

        chunk_idx = 0
        for start_idx in range(0, total_samples - chunk_size + 1, chunk_size):
            chunk_1d = waveform_1d[start_idx : start_idx + chunk_size]
            timestamp_sec = round(chunk_idx * 0.5, 1)
            chunk_idx += 1

            rms_energy = torch.sqrt(torch.mean(chunk_1d**2))
            if rms_energy < 1e-4:
                timeline.append({
                    "timestamp_sec": timestamp_sec,
                    "threat_score": 0.0,
                    "is_silent": True,
                })
                continue

            spec_2d = extract_log_mel_spectrogram(chunk_1d)
            res = engine.infer_file(chunk_1d, spec_2d)

            # Convert 0.0-1.0 to 0.0-100.0%
            chunk_fused = round(
                (res.fused_score * 100.0 if res.fused_score <= 1.0 else res.fused_score),
                1
            )

            rawnet_scores.append(res.rawnet2_score)
            resnet_scores.append(res.resnet_score)
            fused_scores.append(res.fused_score)

            # APPEND CHUNK TO TIMELINE ARRAY
            timeline.append({
                "timestamp_sec": timestamp_sec,
                "threat_score": chunk_fused,
                "is_silent": False,
            })

        if not fused_scores:
            final_rawnet = 0.0
            final_resnet = 0.0
            final_fused = 0.0
        else:
            med_rawnet = statistics.median(rawnet_scores)
            med_resnet = statistics.median(resnet_scores)
            med_fused = statistics.median(fused_scores)

            avg_rawnet = sum(rawnet_scores) / len(rawnet_scores)
            avg_resnet = sum(resnet_scores) / len(resnet_scores)
            avg_fused = sum(fused_scores) / len(fused_scores)

            final_rawnet = (0.7 * med_rawnet) + (0.3 * avg_rawnet)
            final_resnet = (0.7 * med_resnet) + (0.3 * avg_resnet)
            final_fused = (0.7 * med_fused) + (0.3 * avg_fused)

        raw_rawnet = final_rawnet * 100.0 if final_rawnet <= 1.0 else final_rawnet
        raw_resnet = final_resnet * 100.0 if final_resnet <= 1.0 else final_resnet
        fused = final_fused * 100.0 if final_fused <= 1.0 else final_fused

        device_str = str(engine.device) if hasattr(engine, "device") else "cpu"

        return {
            "status": "SPOOF" if fused >= 50.0 else "SAFE",
            "fused_score": round(fused, 1),
            "rawnet2_score": round(raw_rawnet, 1),
            "resnet_score": round(raw_resnet, 1),
            "device": device_str,
            "timeline": timeline,  # SEND TIMELINE TO FRONTEND
        }

    except Exception as e:
        logger.error(f"Failed to process audio file: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process audio file: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)