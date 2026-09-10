import io
import math
import numpy as np
import torch
import pytest
from backend.app.dsp.vad import (
    VoiceActivityDetector,
    compute_energy_dbfs,
    detect_voice_activity,
    NOISE_GATE_THRESHOLD_DB,
)
from backend.app.dsp.feature_extractor import (
    AudioFeatureExtractor,
    extract_features,
    N_MELS,
)


def _generate_mock_pcm(frequency_hz: float, amplitude: float = 0.5, num_samples: int = 8000) -> bytes:
    """Generate 16kHz 16-bit mono PCM bytes for a pure sine tone."""
    t = np.linspace(0, num_samples / 16000.0, num_samples, endpoint=False)
    sine = np.sin(2 * np.pi * frequency_hz * t) * amplitude
    pcm_int16 = (sine * 32767).astype(np.int16)
    return pcm_int16.tobytes()


def test_vad_silence() -> None:
    """Verify that pure silence yields energy well below the -45 dB threshold."""
    silence_pcm = b"\x00\x00" * 8000
    is_speech, energy_db = detect_voice_activity(silence_pcm)
    assert is_speech is False
    assert energy_db < NOISE_GATE_THRESHOLD_DB


def test_vad_active_speech() -> None:
    """Verify that audible speech-level audio is detected above the -45 dB threshold."""
    speech_pcm = _generate_mock_pcm(frequency_hz=440.0, amplitude=0.4)
    is_speech, energy_db = detect_voice_activity(speech_pcm)
    assert is_speech is True
    assert energy_db >= NOISE_GATE_THRESHOLD_DB


def test_vad_bytes_io_input() -> None:
    """Verify that VAD properly ingests pure RAM io.BytesIO stream."""
    speech_pcm = _generate_mock_pcm(frequency_hz=1000.0, amplitude=0.2)
    buffer = io.BytesIO(speech_pcm)
    is_speech, energy_db = detect_voice_activity(buffer)
    assert is_speech is True
    assert energy_db >= NOISE_GATE_THRESHOLD_DB


def test_feature_extractor_waveform_and_spectrogram() -> None:
    """Verify 1D normalized waveform tensor and 2D Log-Mel-Spectrogram shapes and ranges."""
    raw_pcm = _generate_mock_pcm(frequency_hz=500.0, amplitude=0.8, num_samples=8000)
    buffer = io.BytesIO(raw_pcm)

    waveform_1d, log_mel = extract_features(buffer)

    # 1D Tensor checks
    assert isinstance(waveform_1d, torch.Tensor)
    assert waveform_1d.dim() == 1
    assert waveform_1d.shape[0] == 8000
    assert waveform_1d.dtype == torch.float32
    assert waveform_1d.min().item() >= -1.0
    assert waveform_1d.max().item() <= 1.0

    # 2D Mel-Spectrogram checks
    assert isinstance(log_mel, torch.Tensor)
    assert log_mel.dim() == 2
    assert log_mel.shape[0] == N_MELS  # 128 mel frequency bins
    assert log_mel.dtype == torch.float32
