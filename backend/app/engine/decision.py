from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


class ThreatState(str, Enum):
    SAFE = "SAFE"
    SYNTHETIC_ALERT = "SYNTHETIC_ALERT"


DEFAULT_SPOOF_THRESHOLD: float = 0.40
DEFAULT_TRIGGER_FRAMES: int = 3
DEFAULT_RECOVERY_FRAMES: int = 3


@dataclass(frozen=True)
class DecisionResult:
    """Represents the output of the hysteresis decision engine for a single frame."""

    status: str
    consecutive_threats: int
    consecutive_safe: int
    is_alert: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "consecutive_threats": self.consecutive_threats,
            "consecutive_safe": self.consecutive_safe,
            "is_alert": self.is_alert,
        }


class HysteresisDecisionEngine:
    """
    Hysteresis State Machine preventing UI flicker and noise oscillations.
    Requires 3 consecutive frames with spoof_score >= 0.70 before shifting from
    SAFE to SYNTHETIC_ALERT, and requires consecutive safe frames to recover.
    """

    def __init__(
        self,
        spoof_threshold: float = DEFAULT_SPOOF_THRESHOLD,
        trigger_frames: int = DEFAULT_TRIGGER_FRAMES,
        recovery_frames: int = DEFAULT_RECOVERY_FRAMES,
    ) -> None:
        self.spoof_threshold: float = spoof_threshold
        self.trigger_frames: int = trigger_frames
        self.recovery_frames: int = recovery_frames

        self._state: ThreatState = ThreatState.SAFE
        self._consecutive_threats: int = 0
        self._consecutive_safe: int = 0

    @property
    def current_state(self) -> str:
        return self._state.value

    @property
    def consecutive_threats(self) -> int:
        return self._consecutive_threats

    @property
    def consecutive_safe(self) -> int:
        return self._consecutive_safe

    def reset(self) -> None:
        """Reset the hysteresis tracker to initial SAFE baseline state."""
        self._state = ThreatState.SAFE
        self._consecutive_threats = 0
        self._consecutive_safe = 0

    def update(self, spoof_score: float) -> DecisionResult:
        """
        Evaluate a newly computed spoof score and transition system state according
        to hysteresis thresholds.

        :param spoof_score: Float confidence score in range [0.0, 1.0].
        :return: DecisionResult with updated system status and frame counters.
        """
        # Threshold check: If spoof_score >= 0.40 triggers threat frame, score < 0.40 is safe
        if spoof_score >= self.spoof_threshold:
            self._consecutive_threats += 1
            self._consecutive_safe = 0

            # Transition from SAFE -> SYNTHETIC_ALERT after 3 consecutive threat frames
            if (
                self._state == ThreatState.SAFE
                and self._consecutive_threats >= self.trigger_frames
            ):
                self._state = ThreatState.SYNTHETIC_ALERT
        else:
            self._consecutive_safe += 1
            # Reset threat streak if below threshold
            self._consecutive_threats = 0

            # Recover from SYNTHETIC_ALERT -> SAFE after sustained safe frames
            if (
                self._state == ThreatState.SYNTHETIC_ALERT
                and self._consecutive_safe >= self.recovery_frames
            ):
                self._state = ThreatState.SAFE

        return DecisionResult(
            status=self._state.value,
            consecutive_threats=self._consecutive_threats,
            consecutive_safe=self._consecutive_safe,
            is_alert=self._state == ThreatState.SYNTHETIC_ALERT,
        )
