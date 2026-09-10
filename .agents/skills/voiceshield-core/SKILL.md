---
name: voiceshield-core
description: Core engineering guidelines, non-negotiable tech stack requirements, and low-latency audio processing rules for VoiceShield AI (SIH26104). Trigger this skill whenever developing, refactoring, or writing code for audio streaming, PyTorch model inference, or React WebSockets.
---

# VoiceShield AI Development Guidelines & Constraints

## Goal
Guide the Antigravity AI agent to write bug-free, zero-latency, production-ready code on the first attempt without exceeding execution rate limits or hallucinating dependencies.

## Architecture & Tech Stack Rules
1. **Backend Framework**: Python 3.10+, FastAPI (Uvicorn server), PyTorch 2.x, Torchaudio, Librosa.
2. **Frontend Framework**: React 18, TypeScript, Tailwind CSS, WaveSurfer.js.
3. **Communication Protocol**: WebSockets over WSS (TLS 1.3) exchanging binary PCM audio chunks and JSON response objects.

## Non-Negotiable Engineering Constraints
- **Audio Stream Specification**: Strictly 16,000 Hz, mono-channel, 16-bit PCM arrays.
- **Buffer Size**: 500ms chunks (8,000 raw audio samples per payload).
- **Zero-Disk Storage Policy**: Audio data must reside strictly in volatile RAM (`io.BytesIO`). Never save, buffer, or record audio chunks to physical disk drives (`open()`, `save()`, etc.).
- **Latency Budget**: Feature extraction and PyTorch inference must complete within 50 ms per 500ms chunk.
- **Robust Exception Handling**: Wrap all WebSocket event loops in `try-except WebSocketDisconnect` blocks to ensure gracefully closed connections without server crashes.
- **State Machine Rules**: Require 3 consecutive frames with a spoof score >= 0.70 before changing the UI threat state from Safe (Green) to Synthetic Alert (Red).

## Antigravity Execution Instructions
1. Never rewrite existing working files entirely; use targeted edits (`replace_file_content`).
2. Verify all imported modules exist in `requirements.txt` or `package.json` before writing code.
3. Always include type hints in Python functions and explicit interfaces in TypeScript components.