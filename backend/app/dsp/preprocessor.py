import torch
import torchaudio.transforms as T


def pre_emphasis_filter(waveform: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
    """
    Lightweight pre-emphasis high-pass filter: y[t] = x[t] - coeff * x[t-1].
    Flattens the natural -6dB/octave spectral tilt and suppresses low-frequency
    microphone rumble / room proximity effects.
    """
    if waveform.numel() <= 1:
        return waveform

    filtered = torch.cat(
        [waveform[..., :1], waveform[..., 1:] - coeff * waveform[..., :-1]],
        dim=-1,
    )
    return filtered


def z_score_normalize(waveform: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Z-score normalization (zero-mean, unit-variance) across time-domain chunks:
    z[t] = (x[t] - mean(x)) / (std(x) + eps).
    Eliminates microphone hardware gain variances and distance fluctuations.
    """
    mean = torch.mean(waveform)
    std = torch.std(waveform)
    if std < eps:
        return waveform - mean
    return (waveform - mean) / (std + eps)


def extract_log_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 512,
    hop_length: int = 160,
) -> torch.Tensor:
    """
    Extracts 2D Log-Mel Spectrogram representation for ResNet18 model ingestion.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )

    mel_spec = mel_transform(waveform)
    log_mel_spec = torch.log(mel_spec + 1e-6)
    return log_mel_spec