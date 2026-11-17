"""
Group Conversation Speaker Diarization & Spatial Radar Tracker.

Designed around NVIDIA Sortformer (`nvidia/diar_sortformer_4spk-v1`) arrival-order
speaker diarization plus directional azimuth estimation so Deaf and Hard-of-Hearing
users immediately see:
  1. WHO is speaking (distinct high-contrast color badge + avatar + role)
  2. WHERE they are seated around the group table (azimuth angle 0°-360°)
  3. WHEN speakers overlap or interrupt each other (explicit visual overlap banner
     instead of garbled merged captions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# WCAG AAA high-contrast speaker palette safe for color-vision deficiency
HIGH_CONTRAST_SPEAKER_PALETTE = [
    {"color": "#38BDF8", "bg": "rgba(56, 189, 248, 0.16)", "name": "Cyan"},
    {"color": "#FBBF24", "bg": "rgba(251, 191, 36, 0.16)", "name": "Amber"},
    {"color": "#A78BFA", "bg": "rgba(167, 139, 250, 0.16)", "name": "Violet"},
    {"color": "#34D399", "bg": "rgba(52, 211, 153, 0.16)", "name": "Emerald"},
    {"color": "#FB7185", "bg": "rgba(251, 113, 133, 0.16)", "name": "Rose"},
    {"color": "#F97316", "bg": "rgba(249, 115, 22, 0.16)", "name": "Orange"},
]


@dataclass
class SpeakerProfile:
    """Represents an individual participant in a group conversation."""
    speaker_id: str
    display_name: str
    azimuth_deg: int  # 0 = North/Front, 90 = Right, 180 = South/User, 270 = Left
    primary_language: str = "en"
    color_hex: str = "#38BDF8"
    badge_bg: str = "rgba(56, 189, 248, 0.16)"
    role: str = "Participant"
    total_speaking_time_s: float = 0.0
    turn_count: int = 0

    @property
    def compass_direction(self) -> str:
        """Human-friendly directional cue so a D/HoH user knows where to look."""
        deg = self.azimuth_deg % 360
        if 330 <= deg or deg < 30:
            return "Across Table (12 o'clock)"
        if 30 <= deg < 75:
            return "Front-Right (2 o'clock)"
        if 75 <= deg < 115:
            return "Right Side (3 o'clock)"
        if 115 <= deg < 160:
            return "Back-Right (4 o'clock)"
        if 160 <= deg < 205:
            return "Near You (6 o'clock)"
        if 205 <= deg < 250:
            return "Back-Left (8 o'clock)"
        if 250 <= deg < 290:
            return "Left Side (9 o'clock)"
        return "Front-Left (10 o'clock)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "display_name": self.display_name,
            "azimuth_deg": self.azimuth_deg,
            "compass_direction": self.compass_direction,
            "primary_language": self.primary_language,
            "color_hex": self.color_hex,
            "badge_bg": self.badge_bg,
            "role": self.role,
            "total_speaking_time_s": round(self.total_speaking_time_s, 2),
            "turn_count": self.turn_count,
        }


@dataclass
class DiarizedTurn:
    """Speaker-attributed turn with overlap and spatial shift indicators."""
    turn_id: int
    speaker: SpeakerProfile
    start_s: float
    end_s: float
    is_overlapping: bool = False
    overlapping_with: Optional[str] = None
    spatial_shift_deg: int = 0  # Angular head-turn needed from previous speaker
    rapid_turn_switch: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "speaker": self.speaker.to_dict(),
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "is_overlapping": self.is_overlapping,
            "overlapping_with": self.overlapping_with,
            "spatial_shift_deg": self.spatial_shift_deg,
            "rapid_turn_switch": self.rapid_turn_switch,
        }


class GroupSpeakerTracker:
    """
    Manages multi-speaker enrollment, arrival-order Sortformer ID assignment,
    overlapping speech detection, and spatial look-direction cues.
    """

    def __init__(self, max_speakers: int = 6) -> None:
        self.max_speakers = max_speakers
        self.speakers: Dict[str, SpeakerProfile] = {}
        self.turns: List[DiarizedTurn] = []
        self._default_angles = [0, 295, 65, 250, 110, 180]

    def register_speaker(
        self,
        speaker_id: str,
        display_name: Optional[str] = None,
        azimuth_deg: Optional[int] = None,
        primary_language: str = "en",
        role: str = "Participant",
    ) -> SpeakerProfile:
        """Register or update a speaker in the group conversation."""
        if speaker_id in self.speakers:
            spk = self.speakers[speaker_id]
            if display_name:
                spk.display_name = display_name
            if azimuth_deg is not None:
                spk.azimuth_deg = azimuth_deg
            if primary_language:
                spk.primary_language = primary_language
            if role != "Participant":
                spk.role = role
            return spk

        idx = len(self.speakers)
        palette = HIGH_CONTRAST_SPEAKER_PALETTE[idx % len(HIGH_CONTRAST_SPEAKER_PALETTE)]
        angle = (
            azimuth_deg
            if azimuth_deg is not None
            else self._default_angles[idx % len(self._default_angles)]
        )
        profile = SpeakerProfile(
            speaker_id=speaker_id,
            display_name=display_name or f"Speaker {idx + 1}",
            azimuth_deg=angle,
            primary_language=primary_language,
            color_hex=palette["color"],
            badge_bg=palette["bg"],
            role=role,
        )
        self.speakers[speaker_id] = profile
        return profile

    def record_turn(
        self,
        speaker_id: str,
        start_s: float,
        end_s: float,
        display_name: Optional[str] = None,
        azimuth_deg: Optional[int] = None,
        language: str = "en",
        role: str = "Participant",
    ) -> DiarizedTurn:
        """
        Record a speaker turn, computing overlap against prior turns and angular
        gaze shift so the UI can show directional arrows for D/HoH participants.
        """
        speaker = self.register_speaker(
            speaker_id=speaker_id,
            display_name=display_name,
            azimuth_deg=azimuth_deg,
            primary_language=language,
            role=role,
        )

        is_overlapping = False
        overlapping_with: Optional[str] = None
        spatial_shift_deg = 0
        rapid_turn_switch = False

        if self.turns:
            prev = self.turns[-1]
            # Check if this turn starts before the previous speaker finished (overlap/crosstalk)
            if prev.speaker.speaker_id != speaker.speaker_id and start_s < (prev.end_s - 0.08):
                is_overlapping = True
                overlapping_with = prev.speaker.display_name

            # Compute shortest angular distance between previous speaker and current speaker
            raw_diff = abs(speaker.azimuth_deg - prev.speaker.azimuth_deg) % 360
            spatial_shift_deg = 360 - raw_diff if raw_diff > 180 else raw_diff

            # Flag rapid ping-pong speaker switches (< 0.45s gap between different speakers)
            gap = start_s - prev.end_s
            if prev.speaker.speaker_id != speaker.speaker_id and gap < 0.45:
                rapid_turn_switch = True

        duration = max(0.0, end_s - start_s)
        speaker.total_speaking_time_s += duration
        speaker.turn_count += 1

        turn = DiarizedTurn(
            turn_id=len(self.turns) + 1,
            speaker=speaker,
            start_s=start_s,
            end_s=end_s,
            is_overlapping=is_overlapping,
            overlapping_with=overlapping_with,
            spatial_shift_deg=spatial_shift_deg,
            rapid_turn_switch=rapid_turn_switch,
        )
        self.turns.append(turn)
        return turn

    def get_group_summary(self) -> Dict[str, Any]:
        """Return group conversation dynamics, speaker distribution, and overlap stats."""
        total_time = sum(s.total_speaking_time_s for s in self.speakers.values()) or 1.0
        overlap_count = sum(1 for t in self.turns if t.is_overlapping)
        speakers_list = []
        for spk in self.speakers.values():
            d = spk.to_dict()
            d["share_pct"] = round((spk.total_speaking_time_s / total_time) * 100.0, 1)
            speakers_list.append(d)

        return {
            "active_speakers_count": len(self.speakers),
            "total_turns": len(self.turns),
            "overlapping_turns_count": overlap_count,
            "speakers": speakers_list,
        }

    def reset(self) -> None:
        self.speakers.clear()
        self.turns.clear()
