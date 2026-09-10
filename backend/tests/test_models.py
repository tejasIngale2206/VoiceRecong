import torch
import pytest
from backend.app.models.rawnet2 import RawNet2
from backend.app.models.resnet_spectrogram import ResNet18Spectrogram


def test_rawnet2_forward_and_prediction() -> None:
    """Verify RawNet2 processes 1D audio waveform tensors and bounds output in [0.0, 1.0]."""
    model = RawNet2()
    model.eval()

    # 500ms 16kHz chunk = 8000 samples
    x = torch.randn(8000, dtype=torch.float32)

    score = model.predict_spoof_prob(x)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

    # Batch forward pass check
    batch_x = torch.randn(2, 8000, dtype=torch.float32)
    out = model(batch_x)
    assert out.shape == (2, 2)
    probs = torch.softmax(out, dim=1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_resnet_spectrogram_forward_and_prediction() -> None:
    """Verify ResNet18Spectrogram processes 2D Mel-Spectrograms and bounds output in [0.0, 1.0]."""
    model = ResNet18Spectrogram()
    model.eval()

    # 128 mel frequency bins x 51 time steps
    spectrogram = torch.randn(128, 51, dtype=torch.float32)

    score = model.predict_spoof_prob(spectrogram)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

    # Batch forward pass check
    batch_spec = torch.randn(2, 128, 51, dtype=torch.float32)
    out = model(batch_spec)
    assert out.shape == (2, 1)
    assert (out >= 0.0).all() and (out <= 1.0).all()
