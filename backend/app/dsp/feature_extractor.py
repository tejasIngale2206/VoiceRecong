import io
from typing import Tuple, Union
import numpy as np
import torch
import torchaudio

# Standard audio pipeline parameters
SAMPLE_RATE: int = 16000
N_MELS: int = 128
N_FFT: int = 512
HOP_LENGTH: int = 160


class AudioFeatureExtractor:
    """
    Volatile RAM-based audio feature extractor.
    Enforces the zero-disk policy by operating strictly on memory buffers (io.BytesIO).
    Converts 16kHz binary PCM to 1D normalized PyTorch tensors and 2D Log-Mel-Spectrograms.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
    ) -> None:
        self.sample_rate: int = sample_rate
        self.n_mels: int = n_mels
        self.n_fft: int = n_fft
        self.hop_length: int = hop_length

        # Initialize TorchAudio Mel-Spectrogram transform pipeline once in memory
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            mel_scale="slaney",
            power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def pcm_to_tensor(self, pcm_input: Union[bytes, io.BytesIO]) -> torch.Tensor:
        """
        Convert incoming 500ms binary PCM buffer (8,000 samples) into a 1D PyTorch float tensor
        normalized to [-1.0, 1.0]. Enforces Zero-Disk volatile memory policy.

        :param pcm_input: Volatile io.BytesIO stream or raw PCM bytes.
        :return: 1D torch.Tensor of shape (samples,) normalized to [-1.0, 1.0].
        """
        raw_bytes: bytes = (
            pcm_input.getvalue() if isinstance(pcm_input, io.BytesIO) else pcm_input
        )
        if not raw_bytes:
            return torch.empty(0, dtype=torch.float32)

        # Ingest 16-bit PCM little-endian samples directly from memory
        samples_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        # Normalize to [-1.0, 1.0] by full-scale amplitude (32768.0)
        waveform_1d: torch.Tensor = torch.from_numpy(samples_np / 32768.0)
        return waveform_1d

    def extract_log_mel_spectrogram(self, waveform_1d: torch.Tensor) -> torch.Tensor:
        """
        Extract a 2D Log-Mel-Spectrogram matrix using Torchaudio.

        :param waveform_1d: 1D PyTorch float tensor of normalized audio samples.
        :return: 2D torch.Tensor of shape (n_mels, time_frames).
        """
        if waveform_1d.dim() == 1:
            waveform_in = waveform_1d.unsqueeze(0)
        else:
            waveform_in = waveform_1d

        mel_spectrogram = self.mel_transform(waveform_in)
        log_mel: torch.Tensor = self.amplitude_to_db(mel_spectrogram)
        return log_mel.squeeze(0)

    def extract_features(
        self, pcm_input: Union[bytes, io.BytesIO]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        End-to-end volatile memory feature extraction.
        Returns both the 1D normalized waveform and the 2D Log-Mel-Spectrogram tensor.

        :param pcm_input: Raw PCM bytes or volatile io.BytesIO buffer.
        :return: (waveform_1d, log_mel_spectrogram_2d).
        """
        waveform_1d = self.pcm_to_tensor(pcm_input)
        log_mel = self.extract_log_mel_spectrogram(waveform_1d)
        return waveform_1d, log_mel


# Shared instance for zero-allocation hot-path reuse
_default_extractor = AudioFeatureExtractor()


def extract_features(
    pcm_input: Union[bytes, io.BytesIO],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function for memory-based audio feature extraction.

    :param pcm_input: Raw PCM bytes or volatile io.BytesIO stream.
    :return: (waveform_1d, log_mel_spectrogram_2d).
    """
    return _default_extractor.extract_features(pcm_input)
