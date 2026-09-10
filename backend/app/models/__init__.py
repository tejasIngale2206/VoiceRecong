"""Neural network model definitions for anti-spoofing detection."""
from backend.app.models.rawnet2 import RawNet2
from backend.app.models.resnet_spectrogram import ResNet18Spectrogram

__all__ = ["RawNet2", "ResNet18Spectrogram"]
