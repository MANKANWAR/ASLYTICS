"""On-device speech recognition model adapters for NVIDIA Parakeet & Canary."""

from aslytics.models.parakeet_engine import (
    ParakeetOnDeviceEngine,
    TranscriptionSegment,
    WordTimestamp,
)

__all__ = ["ParakeetOnDeviceEngine", "TranscriptionSegment", "WordTimestamp"]
