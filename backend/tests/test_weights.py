import os
import tempfile
import torch
import pytest
from backend.app.models.rawnet2 import RawNet2
from backend.app.models.resnet_spectrogram import ResNet18Spectrogram
from backend.app.engine.inference import DualStreamInferenceEngine, WEIGHTS_DIR


def test_rawnet2_load_weights() -> None:
    """Verify RawNet2 load_weights logic with missing and valid checkpoints."""
    model = RawNet2()

    # 1. Non-existent path returns False
    assert model.load_weights("non_existent_file.pth") is False

    # 2. Save valid state dict to temporary file and load
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        torch.save(model.state_dict(), tmp_path)
        loaded = model.load_weights(tmp_path)
        assert loaded is True
        assert not model.training  # Enforces eval mode
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_resnet_load_weights() -> None:
    """Verify ResNet18Spectrogram load_weights logic with missing and valid checkpoints."""
    model = ResNet18Spectrogram()

    # 1. Non-existent path returns False
    assert model.load_weights("non_existent_file.pth") is False

    # 2. Save valid state dict to temporary file and load
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        torch.save(model.state_dict(), tmp_path)
        loaded = model.load_weights(tmp_path)
        assert loaded is True
        assert not model.training  # Enforces eval mode
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_inference_engine_weights_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify DualStreamInferenceEngine gracefully initializes with or without weights."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from pathlib import Path
        tmp_weights_dir = Path(tmp_dir)
        monkeypatch.setattr("backend.app.engine.inference.WEIGHTS_DIR", tmp_weights_dir)

        # 1. No weights present
        engine_no_weights = DualStreamInferenceEngine(device=torch.device("cpu"))
        assert not engine_no_weights.rawnet2.training
        assert not engine_no_weights.resnet.training

        # 2. Save mock checkpoints in temporary directory
        torch.save(engine_no_weights.rawnet2.state_dict(), str(tmp_weights_dir / "rawnet2_la.pth"))
        torch.save(engine_no_weights.resnet.state_dict(), str(tmp_weights_dir / "resnet18_spec.pth"))

        # Initialize engine with weights present in temp dir
        loaded_engine = DualStreamInferenceEngine(device=torch.device("cpu"))
        assert not loaded_engine.rawnet2.training
        assert not loaded_engine.resnet.training
