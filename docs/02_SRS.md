# Document 2: System Requirement Specification (SRS)
**Project Title:** VoiceShield AI  

### 1. Ingestion Requirements
- Real-time 500ms audio chunks (8000 PCM samples per payload).
- Asynchronous WebSockets (`/ws/stream`).

### 2. Processing Engine
- Voice Activity Detection (VAD) noise gate (-45 dB energy threshold).
- Dual AI Inference Core: RawNet2 (1D Waveform) + ResNet18 (2D Mel-Spectrogram).
