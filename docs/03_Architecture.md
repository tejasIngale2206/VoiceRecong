# Document 3: Technical Architecture & System Blueprint
**Project Title:** VoiceShield AI  

### 1. System Pipeline
React AudioWorklet -> WebSocket (WSS) -> FastAPI Event Loop -> PyTorch Model Core -> Decision Hysteresis Engine

### 2. Score Fusion & Decision
- Combined Score = (0.6 * RawNet2) + (0.4 * ResNet18)
- Requires 3 consecutive frames with score >= 0.70 to trigger Alert State.
