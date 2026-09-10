from collections import deque
import logging
import math
import os
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv
import torch
import torch.nn.functional as F
from pydantic import BaseModel

from backend.app.dsp.preprocessor import pre_emphasis_filter, z_score_normalize

load_dotenv()
logger = logging.getLogger("voiceshield.inference")

RAWNET2_WEIGHT: float = 0.6
RESNET_WEIGHT: float = 0.4
ALERT_THRESHOLD: float = 0.55
DEFAULT_TEMPERATURE: float = 3.0

WEIGHTS_DIR: Path = Path(__file__).resolve().parent.parent / "models" / "weights"


def classify_status(score: float) -> str:
    return "SYNTHETIC_ALERT" if score >= ALERT_THRESHOLD else "SAFE"


class InferenceResult(BaseModel):
    fused_score: float
    rawnet2_score: float
    resnet_score: float
    device: str
    status: str


class DualStreamInferenceEngine:

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        mic_logit_offset: Optional[float] = None,
    ):
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.temperature: float = float(
            os.getenv("TEMPERATURE", str(temperature))
        )

        from backend.app.models.rawnet2 import RawNet2
        from backend.app.models.resnet_spectrogram import ResNet18Spectrogram

        self.rawnet2 = RawNet2().to(self.device)
        self.resnet = ResNet18Spectrogram().to(self.device)

        self.has_rawnet_weights: bool = False
        self.has_resnet_weights: bool = False

        # Load RawNet2 weights
        rawnet2_candidates = [
            WEIGHTS_DIR / "pre_trained_DF_RawNet2.pth",
            WEIGHTS_DIR / "rawnet2_la.pth",
        ]
        for p in rawnet2_candidates:
            if p.exists():
                try:
                    checkpoint = torch.load(str(p), map_location=self.device)
                    state_dict = (
                        checkpoint.get("model_state_dict", checkpoint)
                        if isinstance(checkpoint, dict)
                        else checkpoint
                    )
                    cleaned_state_dict = {
                        k.replace("module.", ""): v
                        for k, v in state_dict.items()
                    }
                    self.rawnet2.load_state_dict(
                        cleaned_state_dict, strict=False
                    )
                    self.has_rawnet_weights = True
                    logger.info(f"Loaded RawNet2 weights from {p.name}")
                    break
                except Exception as e:
                    logger.warning(
                        f"Error loading RawNet2 weights from {p}: {e}"
                    )

        # Load ResNet weights
        resnet_candidates = [
            WEIGHTS_DIR / "resnet18_spec.pth",
        ]
        for p in resnet_candidates:
            if p.exists():
                try:
                    checkpoint = torch.load(str(p), map_location=self.device)
                    state_dict = (
                        checkpoint.get("model_state_dict", checkpoint)
                        if isinstance(checkpoint, dict)
                        else checkpoint
                    )
                    self.resnet.load_state_dict(state_dict, strict=False)
                    self.has_resnet_weights = True
                    logger.info(f"Loaded ResNet weights from {p.name}")
                    break
                except Exception as e:
                    logger.warning(
                        f"Error loading ResNet weights from {p}: {e}"
                    )

        self.rawnet2.eval()
        self.resnet.eval()

        # Calibration & Smoothing Controls (FOR LIVE STREAM ONLY)
        self.score_history: deque = deque(maxlen=10)
        self.smoothed_score: float = 0.0
        self.last_status: str = "SAFE"

    # =========================================================================
    # 1. STATELESS INFERENCE FOR OFFLINE FILE SCAN (No Memory Contamination)
    # =========================================================================
    @torch.no_grad()
    def infer_file(
        self,
        waveform_1d: torch.Tensor,
        spectrogram_2d: torch.Tensor,
    ) -> InferenceResult:
        wave_dev = waveform_1d.to(self.device)

        # Noise Dampening
        kernel_size = 7
        padding = kernel_size // 2
        kernel = (
            torch.ones(1, 1, kernel_size, device=self.device) / kernel_size
        )
        wave_padded = wave_dev.view(1, 1, -1)
        filtered_wave = F.conv1d(wave_padded, kernel, padding=padding).view_as(
            wave_dev
        )

        # Normalization
        max_val = torch.max(torch.abs(filtered_wave))
        norm_wave = (
            filtered_wave / max_val if max_val > 1e-6 else filtered_wave
        )
        norm_audio = z_score_normalize(norm_wave)
        if norm_audio.dim() == 1:
            norm_audio = norm_audio.unsqueeze(0)

        # RawNet2 Forward
        logits: torch.Tensor = self.rawnet2(norm_audio)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        logit_0 = float(logits[0][0].item())
        logit_1 = float(logits[0][1].item())

        t = max(0.1, self.temperature)
        diff = (logit_1 - logit_0) / t
        raw_prob = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, diff))))
        s_rawnet2 = max(0.0, min(1.0, (raw_prob - 0.20) / 0.60))

        # ResNet Forward
        if self.has_resnet_weights:
            try:
                s_resnet = self.resnet.predict_spoof_prob(
                    spectrogram_2d.to(self.device)
                )
            except Exception:
                s_resnet = s_rawnet2
        else:
            s_resnet = s_rawnet2

        # Pure Instantaneous Fused Score (No History/EMA Buffering)
        fused_score = (RAWNET2_WEIGHT * s_rawnet2) + (RESNET_WEIGHT * s_resnet)
        fused_score = max(0.0, min(1.0, fused_score))

        current_status = (
            "SYNTHETIC_ALERT" if fused_score >= ALERT_THRESHOLD else "SAFE"
        )

        return InferenceResult(
            fused_score=fused_score,
            rawnet2_score=s_rawnet2,
            resnet_score=s_resnet,
            device=str(self.device),
            status=current_status,
        )

    # =========================================================================
    # 2. STATEFUL INFERENCE FOR LIVE STREAMING (UNTOUCHED)
    # =========================================================================
    @torch.no_grad()
    def infer(
        self,
        waveform_1d: torch.Tensor,
        spectrogram_2d: torch.Tensor,
    ) -> InferenceResult:
        wave_dev = waveform_1d.to(self.device)

        # 1. Voice Activity Gate (Energy threshold)
        energy = torch.mean(wave_dev**2).item()
        if energy < 3.0e-4:
            self.smoothed_score = 0.0
            return InferenceResult(
                fused_score=0.0,
                rawnet2_score=0.0,
                resnet_score=0.0,
                device=str(self.device),
                status=self.last_status,
            )

        # 2. Mic High-Frequency Noise Dampening Filter
        kernel_size = 7
        padding = kernel_size // 2
        kernel = (
            torch.ones(1, 1, kernel_size, device=self.device) / kernel_size
        )
        wave_padded = wave_dev.view(1, 1, -1)
        filtered_wave = F.conv1d(wave_padded, kernel, padding=padding).view_as(
            wave_dev
        )

        # 3. Peak Normalization
        max_val = torch.max(torch.abs(filtered_wave))
        if max_val > 1e-6:
            norm_wave = filtered_wave / max_val
        else:
            norm_wave = filtered_wave

        norm_audio = z_score_normalize(norm_wave)
        if norm_audio.dim() == 1:
            norm_audio = norm_audio.unsqueeze(0)

        # 4. Model Forward Pass
        logits: torch.Tensor = self.rawnet2(norm_audio)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        logit_0 = float(logits[0][0].item())
        logit_1 = float(logits[0][1].item())

        # Raw Difference Ratio
        t = max(0.1, self.temperature)
        diff = (logit_1 - logit_0) / t

        # Softmax Probability mapping
        raw_prob = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, diff))))

        # Relative Scaling for Live Mic Input
        s_rawnet2 = max(0.0, min(1.0, (raw_prob - 0.20) / 0.60))

        # 5. Dual Stream Fusion
        if self.has_resnet_weights:
            try:
                s_resnet = self.resnet.predict_spoof_prob(
                    spectrogram_2d.to(self.device)
                )
            except Exception:
                s_resnet = s_rawnet2
        else:
            s_resnet = s_rawnet2

        raw_fused = (RAWNET2_WEIGHT * s_rawnet2) + (RESNET_WEIGHT * s_resnet)
        raw_fused = max(0.0, min(1.0, raw_fused))

        # 6. Temporal Median Queue + Strong EMA Smoothing
        self.score_history.append(raw_fused)
        sorted_scores = sorted(self.score_history)
        median_score = sorted_scores[len(sorted_scores) // 2]

        alpha = 0.15  # Heavy dampening to eliminate green/red flickering
        self.smoothed_score = (alpha * median_score) + (
            (1.0 - alpha) * self.smoothed_score
        )

        final_score = self.smoothed_score
        if final_score < 0.18:
            final_score = 0.0

        # 7. Decision Gate
        if final_score >= ALERT_THRESHOLD:
            current_status = "SYNTHETIC_ALERT"
        else:
            current_status = "SAFE"

        self.last_status = current_status

        return InferenceResult(
            fused_score=final_score,
            rawnet2_score=s_rawnet2,
            resnet_score=s_resnet,
            device=str(self.device),
            status=current_status,
        )


_default_engine: Optional[DualStreamInferenceEngine] = None


def get_inference_engine() -> DualStreamInferenceEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = DualStreamInferenceEngine()
    return _default_engine