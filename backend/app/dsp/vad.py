import io
import math
from typing import Tuple, Union
import numpy as np

# SIH26104 Voice Activity Detection Noise Gate Threshold
NOISE_GATE_THRESHOLD_DB: float = -45.0
MIN_ENERGY_DB: float = -100.0


def compute_energy_dbfs(pcm_input: Union[bytes, io.BytesIO]) -> float:
    """
    Calculate the Root Mean Square (RMS) energy in decibels relative to full scale (dBFS)
    for 16kHz 16-bit mono PCM audio data in volatile RAM (Zero-Disk policy).

    :param pcm_input: Raw PCM bytes or volatile io.BytesIO stream.
    :return: RMS energy in dBFS (clamped to a floor of -100.0 dBFS).
    """
    raw_bytes: bytes = (
        pcm_input.getvalue() if isinstance(pcm_input, io.BytesIO) else pcm_input
    )
    if not raw_bytes:
        return MIN_ENERGY_DB

    # Convert 16-bit signed integer PCM (-32768 to 32767) normalized to [-1.0, 1.0]
    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Calculate mean of squared amplitudes
    mean_square: float = float(np.mean(samples**2))
    if mean_square <= 1e-10:
        return MIN_ENERGY_DB

    rms: float = math.sqrt(mean_square)
    energy_db: float = 20.0 * math.log10(rms)
    return max(energy_db, MIN_ENERGY_DB)


class VoiceActivityDetector:
    """
    Low-latency Voice Activity Detector (VAD) evaluating energy levels against a noise gate.
    """

    def __init__(self, threshold_db: float = NOISE_GATE_THRESHOLD_DB) -> None:
        self.threshold_db: float = threshold_db

    def process(self, pcm_input: Union[bytes, io.BytesIO]) -> Tuple[bool, float]:
        """
        Evaluate PCM audio chunk to determine if active human speech is present.

        :param pcm_input: Raw PCM bytes or volatile io.BytesIO buffer.
        :return: (is_speech: bool, energy_db: float).
        """
        energy_db: float = compute_energy_dbfs(pcm_input)
        is_speech: bool = energy_db >= self.threshold_db
        return is_speech, round(energy_db, 2)


# Pre-allocated default detector instance for zero-allocation reuse
_default_vad = VoiceActivityDetector()


def detect_voice_activity(
    pcm_input: Union[bytes, io.BytesIO],
    threshold_db: float = NOISE_GATE_THRESHOLD_DB,
) -> Tuple[bool, float]:
    """
    Convenience function for energy-based VAD evaluation.

    :param pcm_input: Raw PCM bytes or io.BytesIO stream.
    :param threshold_db: Noise gate threshold in dBFS (default: -45.0 dB).
    :return: (is_speech: bool, energy_db: float).
    """
    if threshold_db == NOISE_GATE_THRESHOLD_DB:
        return _default_vad.process(pcm_input)
    detector = VoiceActivityDetector(threshold_db=threshold_db)
    return detector.process(pcm_input)
