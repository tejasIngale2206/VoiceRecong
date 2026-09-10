import torch
import pytest
from backend.app.engine.inference import (
    DualStreamInferenceEngine,
    RAWNET2_WEIGHT,
    RESNET_WEIGHT,
    ALERT_THRESHOLD,
    classify_status,
)
from backend.app.engine.decision import (
    HysteresisDecisionEngine,
    ThreatState,
    DecisionResult,
)


def test_score_fusion_weights() -> None:
    """Validate score fusion equation: S_final = (0.6 * S_RawNet2) + (0.4 * S_ResNet)."""
    assert RAWNET2_WEIGHT == 0.6
    assert RESNET_WEIGHT == 0.4

    engine = DualStreamInferenceEngine(device=torch.device("cpu"))
    waveform = torch.randn(8000, dtype=torch.float32)
    spectrogram = torch.randn(128, 51, dtype=torch.float32)

    result = engine.infer(waveform, spectrogram)

    expected_fused = (0.6 * result.rawnet2_score) + (0.4 * result.resnet_score)
    expected_fused = max(0.0, min(1.0, expected_fused))

    assert pytest.approx(result.fused_score, rel=1e-4) == expected_fused
    assert 0.0 <= result.fused_score <= 1.0
    assert result.status in ["SAFE", "SYNTHETIC_ALERT"]


def test_classification_threshold() -> None:
    """
    Verify threshold behavior:
    - spoof_prob >= 0.40 -> SYNTHETIC_ALERT
    - spoof_prob < 0.40 -> SAFE
    """
    assert ALERT_THRESHOLD == 0.40
    assert classify_status(0.40) == "SYNTHETIC_ALERT"
    assert classify_status(0.4001) == "SYNTHETIC_ALERT"
    assert classify_status(0.85) == "SYNTHETIC_ALERT"
    assert classify_status(0.3999) == "SAFE"
    assert classify_status(0.0) == "SAFE"


def test_hysteresis_state_machine_transitions() -> None:
    """
    Validate that:
    1. System starts in SAFE.
    2. Single or double frames >= 0.40 do not trigger ALERT.
    3. Exactly 3 consecutive frames >= 0.40 trigger SYNTHETIC_ALERT.
    4. S_alert recovers back to SAFE only after 3 consecutive safe frames.
    """
    engine = HysteresisDecisionEngine(
        spoof_threshold=0.40,
        trigger_frames=3,
        recovery_frames=3,
    )

    # Initial state
    assert engine.current_state == ThreatState.SAFE.value

    # Frame 1: Threat (1/3)
    res1 = engine.update(0.45)
    assert res1.status == "SAFE"
    assert res1.consecutive_threats == 1

    # Frame 2: Threat (2/3)
    res2 = engine.update(0.50)
    assert res2.status == "SAFE"
    assert res2.consecutive_threats == 2

    # Frame 3: Threat (3/3) -> ALERT TRIGGERED!
    res3 = engine.update(0.60)
    assert res3.status == "SYNTHETIC_ALERT"
    assert res3.is_alert is True
    assert res3.consecutive_threats == 3

    # Frame 4: Threat continues (at threshold)
    res4 = engine.update(0.40)
    assert res4.status == "SYNTHETIC_ALERT"
    assert res4.consecutive_threats == 4

    # Frame 5: Single safe frame arrives -> Still in SYNTHETIC_ALERT (hysteresis prevents chattering)
    res5 = engine.update(0.30)
    assert res5.status == "SYNTHETIC_ALERT"
    assert res5.consecutive_threats == 0
    assert res5.consecutive_safe == 1

    # Frame 6: Second safe frame
    res6 = engine.update(0.20)
    assert res6.status == "SYNTHETIC_ALERT"
    assert res6.consecutive_safe == 2

    # Frame 7: Third consecutive safe frame -> Recovers to SAFE
    res7 = engine.update(0.15)
    assert res7.status == "SAFE"
    assert res7.is_alert is False
    assert res7.consecutive_safe == 3


def test_hysteresis_intermittent_noise_rejection() -> None:
    """Verify that intermittent spikes (e.g. 2 threat frames interrupted by 1 safe frame) do not trigger an alert."""
    engine = HysteresisDecisionEngine(spoof_threshold=0.40, trigger_frames=3)

    engine.update(0.50)  # 1 threat
    engine.update(0.45)  # 2 threat
    res = engine.update(0.35)  # Safe frame resets streak! (0.35 < 0.40)

    assert res.status == "SAFE"
    assert res.consecutive_threats == 0
