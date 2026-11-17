"""
Visual, Prosodic, and Environmental Accessibility Analyzer for D/HoH Group Settings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AccessibilityAnnotations:
    """Visual and tactile metadata attached to every group conversation turn."""
    tone_label: str
    tone_icon: str
    triggered_watchwords: List[str] = field(default_factory=list)
    requires_visual_flash: bool = False
    haptic_pattern_ms: List[int] = field(default_factory=list)
    acoustic_events: List[str] = field(default_factory=list)
    pacing_status: str = "Comfortable Pace"
    direct_address_to_user: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tone_label": self.tone_label,
            "tone_icon": self.tone_icon,
            "triggered_watchwords": self.triggered_watchwords,
            "requires_visual_flash": self.requires_visual_flash,
            "haptic_pattern_ms": self.haptic_pattern_ms,
            "acoustic_events": self.acoustic_events,
            "pacing_status": self.pacing_status,
            "direct_address_to_user": self.direct_address_to_user,
        }


class DHHAccessibilityAnalyzer:
    """
    Analyzes transcribed text, translated text, speaking rate, and overlap state
    to emit rich non-auditory cues for Deaf and Hard-of-Hearing participants.
    """

    def __init__(self, watchwords: Optional[List[str]] = None) -> None:
        self.watchwords = [w.lower() for w in (watchwords or ["alex", "everyone", "question", "urgent"])]

    def analyze_turn(
        self,
        original_text: str,
        translated_text: str,
        words_per_minute: float,
        is_overlapping: bool = False,
        acoustic_events: Optional[List[str]] = None,
    ) -> AccessibilityAnnotations:
        combined = f"{original_text} {translated_text}".lower()
        clean_words = set(re.findall(r"\b\w+\b", combined))

        matched_watchwords = [
            w for w in self.watchwords if w in clean_words or w in combined
        ]

        # Determine prosodic / intent tone cue
        if original_text.strip().endswith("?") or translated_text.strip().endswith("?"):
            tone_label = "Question / Floor Open"
            tone_icon = "❓"
        elif any(k in combined for k in ("urgent", "critical", "blocker", "careful", "wait")):
            tone_label = "Urgent / High Priority"
            tone_icon = "⚡"
        elif original_text.strip().endswith("!") or translated_text.strip().endswith("!"):
            tone_label = "Enthusiastic / Emphatic"
            tone_icon = "✨"
        else:
            tone_label = "Conversational / Calm"
            tone_icon = "💬"

        events = list(acoustic_events or [])
        if is_overlapping and "OVERLAPPING SPEECH" not in events:
            events.append("OVERLAPPING SPEECH")

        if words_per_minute > 175:
            pacing_status = f"Fast Pace ({int(words_per_minute)} WPM)"
        elif words_per_minute < 105:
            pacing_status = f"Deliberate Pace ({int(words_per_minute)} WPM)"
        else:
            pacing_status = f"Clear Pace ({int(words_per_minute)} WPM)"

        direct_address = len(matched_watchwords) > 0
        requires_flash = direct_address or is_overlapping or tone_label.startswith("Urgent")

        # Web Vibration API compatible pattern (ms)
        if direct_address:
            haptic = [180, 80, 180]
        elif is_overlapping:
            haptic = [100, 60, 100]
        else:
            haptic = []

        return AccessibilityAnnotations(
            tone_label=tone_label,
            tone_icon=tone_icon,
            triggered_watchwords=matched_watchwords,
            requires_visual_flash=requires_flash,
            haptic_pattern_ms=haptic,
            acoustic_events=events,
            pacing_status=pacing_status,
            direct_address_to_user=direct_address,
        )
