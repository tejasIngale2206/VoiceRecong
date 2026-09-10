"""Digital Signal Processing (DSP) and Audio Feature Extraction Module."""
from backend.app.dsp.vad import detect_voice_activity, VoiceActivityDetector
from backend.app.dsp.feature_extractor import (
    AudioFeatureExtractor,
    extract_features,
)

__all__ = [
    "detect_voice_activity",
    "VoiceActivityDetector",
    "AudioFeatureExtractor",
    "extract_features",
]
