"""
End-to-End Group Conversation Pipeline for ASLytics.
Unifies NVIDIA Parakeet-TDT on-device ASR, Sortformer group speaker tracking,
multilingual translation, ASL Gloss generation, and D/HoH visual/haptic alerts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from aslytics.accessibility.dhh_cues import DHHAccessibilityAnalyzer
from aslytics.config import ASLyticsConfig
from aslytics.diarization.group_tracker import GroupSpeakerTracker
from aslytics.models.parakeet_engine import ParakeetOnDeviceEngine
from aslytics.multilingual.translator import MultilingualBridge


BUILTIN_GROUP_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "multilingual_standup": {
        "id": "multilingual_standup",
        "title": "Multilingual Product Team Standup (EN / ES / FR / DE)",
        "setting": "Conference Table (5 Participants)",
        "description": (
            "Cross-functional engineering & accessibility design review where team members "
            "speak in English, Spanish, French, and German with rapid turn-taking and one overlap."
        ),
        "turns": [
            {
                "speaker_id": "spk_1",
                "display_name": "Maya (PM)",
                "role": "Product Lead",
                "azimuth_deg": 0,
                "language": "en",
                "start_s": 0.0,
                "duration_s": 3.4,
                "text": "Good morning everyone, let's check our progress on the on-device Parakeet release.",
                "translation_en": "Good morning everyone, let's check our progress on the on-device Parakeet release.",
                "acoustic_events": [],
            },
            {
                "speaker_id": "spk_2",
                "display_name": "Carlos (UX)",
                "role": "Accessibility Designer",
                "azimuth_deg": 295,
                "language": "es",
                "start_s": 3.7,
                "duration_s": 3.8,
                "text": "Hola equipo, podemos revisar el diseño de accesibilidad hoy?",
                "translation_en": "Hi team, can we review the accessibility design today?",
                "acoustic_events": [],
            },
            {
                "speaker_id": "spk_3",
                "display_name": "Chloé (ML)",
                "role": "Edge AI Engineer",
                "azimuth_deg": 65,
                "language": "fr",
                "start_s": 7.8,
                "duration_s": 4.2,
                "text": "Bonjour à tous, le modèle Parakeet fonctionne hors ligne avec une latence très faible.",
                "translation_en": "Hello everyone, the Parakeet model runs offline with very low latency.",
                "acoustic_events": [],
            },
            {
                "speaker_id": "spk_2",
                "display_name": "Carlos (UX)",
                "role": "Accessibility Designer",
                "azimuth_deg": 295,
                "language": "es",
                "start_s": 11.6,  # Slight overlap with Chloé finishing!
                "duration_s": 3.6,
                "text": "Claro, ya actualicé los subtítulos en tiempo real y el radar visual.",
                "translation_en": "Sure, I already updated the real-time captions and the visual radar.",
                "acoustic_events": ["OVERLAPPING SPEECH"],
            },
            {
                "speaker_id": "spk_4",
                "display_name": "Lukas (Audio)",
                "role": "DSP Architect",
                "azimuth_deg": 115,
                "language": "de",
                "start_s": 15.5,
                "duration_s": 4.3,
                "text": "Wir können jetzt gleichzeitig Deutsch, Spanisch und Französisch in der Gruppe verstehen.",
                "translation_en": "We can now understand German, Spanish, and French simultaneously in the group.",
                "acoustic_events": ["GROUP APPLAUSE"],
            },
            {
                "speaker_id": "spk_1",
                "display_name": "Maya (PM)",
                "role": "Product Lead",
                "azimuth_deg": 0,
                "language": "en",
                "start_s": 20.1,
                "duration_s": 3.2,
                "text": "Alex, do you have any question about the spatial speaker radar before we ship tomorrow?",
                "translation_en": "Alex, do you have any question about the spatial speaker radar before we ship tomorrow?",
                "acoustic_events": [],
            },
        ],
    },
    "family_dinner": {
        "id": "family_dinner",
        "title": "Multigenerational Family Dinner Table (EN / Punjabi / Hindi)",
        "setting": "Round Dining Table (4 Family Members)",
        "description": (
            "Lively family conversation switching naturally between Punjabi, Hindi, and English "
            "with laughter and direct questions to the user."
        ),
        "turns": [
            {
                "speaker_id": "spk_a",
                "display_name": "Dadi Ji (Grandma)",
                "role": "Family Elder",
                "azimuth_deg": 350,
                "language": "pa",
                "start_s": 0.0,
                "duration_s": 3.6,
                "text": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਜੀ, ਕੀ ਤੁਸੀਂ ਸਾਰੇ ਰਾਤ ਦੇ ਖਾਣੇ ਲਈ ਤਿਆਰ ਹੋ?",
                "translation_en": "Hello everyone, are you all ready for dinner?",
                "acoustic_events": [],
            },
            {
                "speaker_id": "spk_b",
                "display_name": "Rohan (Cousin)",
                "role": "Cousin",
                "azimuth_deg": 80,
                "language": "pa",
                "start_s": 3.8,
                "duration_s": 3.0,
                "text": "ਹਾਂਜੀ, ਅਸੀਂ ਬਸ ਪੰਜ ਮਿੰਟ ਵਿੱਚ ਆ ਰਹੇ ਹਾਂ!",
                "translation_en": "Yes, we are coming in just five minutes!",
                "acoustic_events": ["LAUGHTER"],
            },
            {
                "speaker_id": "spk_c",
                "display_name": "Priya (Aunt)",
                "role": "Aunt",
                "azimuth_deg": 265,
                "language": "hi",
                "start_s": 6.5,  # Overlap
                "duration_s": 3.7,
                "text": "नमस्ते, क्या हम अगले सप्ताह नई सुविधा लॉन्च कर रहे हैं?",
                "translation_en": "Hello, are we launching the new feature next week?",
                "acoustic_events": ["OVERLAPPING SPEECH"],
            },
            {
                "speaker_id": "spk_d",
                "display_name": "Simran (Sister)",
                "role": "Sister",
                "azimuth_deg": 40,
                "language": "en",
                "start_s": 10.5,
                "duration_s": 3.3,
                "text": "Alex, can you pass the water and tell Grandma about your UBC project today?",
                "translation_en": "Alex, can you pass the water and tell Grandma about your UBC project today?",
                "acoustic_events": ["TABLE TAP"],
            },
        ],
    },
}


class GroupConversationPipeline:
    """
    Coordinates Parakeet ASR, Sortformer speaker diarization, multilingual translation,
    ASL Gloss conversion, and D/HoH visual/haptic alerts.
    """

    def __init__(self, config: Optional[ASLyticsConfig] = None) -> None:
        self.config = config or ASLyticsConfig()
        self.asr_engine = ParakeetOnDeviceEngine(self.config)
        self.speaker_tracker = GroupSpeakerTracker(max_speakers=self.config.max_speakers)
        self.multilingual = MultilingualBridge()
        self.accessibility = DHHAccessibilityAnalyzer(watchwords=self.config.user_watchwords)
        self.history: List[Dict[str, Any]] = []

    def process_utterance(
        self,
        speaker_id: str,
        text: str,
        display_name: Optional[str] = None,
        role: str = "Participant",
        azimuth_deg: Optional[int] = None,
        start_s: Optional[float] = None,
        duration_s: Optional[float] = None,
        language_hint: Optional[str] = None,
        explicit_translation_en: Optional[str] = None,
        acoustic_events: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Process a single group conversation turn through all 4 stages."""
        if start_s is None:
            start_s = self.history[-1]["timing"]["end_s"] + 0.35 if self.history else 0.0

        # Stage 1: On-Device NVIDIA Parakeet-TDT Transcription & Word Durations
        segment = self.asr_engine.transcribe_utterance(
            text_hint=text,
            start_s=start_s,
            duration_s=duration_s,
            language_hint=language_hint,
        )

        # Stage 2: Group Speaker Diarization, Overlap & Spatial Radar
        diarized = self.speaker_tracker.record_turn(
            speaker_id=speaker_id,
            start_s=segment.start_s,
            end_s=segment.end_s,
            display_name=display_name,
            azimuth_deg=azimuth_deg,
            language=segment.language,
            role=role,
        )

        # Stage 3: Multilingual Translation & American Sign Language (ASL) Gloss
        translation = self.multilingual.translate(
            text=segment.text,
            source_lang=segment.language,
            target_lang=self.config.target_language,
            explicit_translation=explicit_translation_en
            if self.config.target_language == "en"
            else None,
        )
        english_basis = (
            explicit_translation_en
            or (
                translation["translated_text"]
                if self.config.target_language == "en"
                else self.multilingual.translate(segment.text, segment.language, "en")[
                    "translated_text"
                ]
            )
        )
        asl_gloss = self.multilingual.to_asl_gloss(english_basis)

        # Stage 4: Deaf / Hard-of-Hearing Accessibility Cues & Watchword Alerts
        dhh_cues = self.accessibility.analyze_turn(
            original_text=segment.text,
            translated_text=translation["translated_text"],
            words_per_minute=segment.words_per_minute,
            is_overlapping=diarized.is_overlapping,
            acoustic_events=acoustic_events,
        )

        turn_payload = {
            "turn_id": diarized.turn_id,
            "speaker": diarized.speaker.to_dict(),
            "diarization": {
                "is_overlapping": diarized.is_overlapping,
                "overlapping_with": diarized.overlapping_with,
                "spatial_shift_deg": diarized.spatial_shift_deg,
                "rapid_turn_switch": diarized.rapid_turn_switch,
            },
            "timing": {
                "start_s": round(segment.start_s, 2),
                "end_s": round(segment.end_s, 2),
                "duration_s": round(segment.duration_s, 2),
                "words_per_minute": segment.words_per_minute,
                "inference_latency_ms": round(segment.inference_latency_ms, 2),
            },
            "transcript": {
                "original_text": segment.text,
                "confidence": round(segment.confidence, 3),
                "words": [w.to_dict() for w in segment.words],
                "model_key": segment.model_key,
                "backend_used": segment.backend_used,
            },
            "multilingual": translation,
            "asl_gloss": asl_gloss.to_dict(),
            "accessibility": dhh_cues.to_dict(),
        }
        self.history.append(turn_payload)
        return turn_payload

    def run_scenario(self, scenario_id: str = "multilingual_standup") -> Dict[str, Any]:
        """Run a full multi-speaker multilingual group scenario and return structured output."""
        scenario = BUILTIN_GROUP_SCENARIOS.get(
            scenario_id, BUILTIN_GROUP_SCENARIOS["multilingual_standup"]
        )
        self.reset()
        processed_turns = []
        for raw_turn in scenario["turns"]:
            turn_out = self.process_utterance(
                speaker_id=raw_turn["speaker_id"],
                display_name=raw_turn["display_name"],
                role=raw_turn.get("role", "Participant"),
                azimuth_deg=raw_turn["azimuth_deg"],
                language_hint=raw_turn["language"],
                start_s=raw_turn["start_s"],
                duration_s=raw_turn["duration_s"],
                text=raw_turn["text"],
                explicit_translation_en=raw_turn.get("translation_en"),
                acoustic_events=raw_turn.get("acoustic_events", []),
            )
            processed_turns.append(turn_out)

        return {
            "scenario": {
                "id": scenario["id"],
                "title": scenario["title"],
                "setting": scenario["setting"],
                "description": scenario["description"],
            },
            "engine_status": self.asr_engine.get_status_summary(),
            "group_summary": self.speaker_tracker.get_group_summary(),
            "turns": processed_turns,
        }

    def export_webvtt(self) -> str:
        """Export current group transcript as accessible WebVTT with speaker & ASL Gloss cues."""
        lines = ["WEBVTT - ASLytics Accessible Multilingual Group Captions", ""]
        for item in self.history:
            start = self._format_vtt_time(item["timing"]["start_s"])
            end = self._format_vtt_time(item["timing"]["end_s"])
            spk = item["speaker"]["display_name"]
            dir_cue = item["speaker"]["compass_direction"]
            txt = item["multilingual"]["translated_text"]
            gloss = item["asl_gloss"]["gloss_text"]
            lines.append(f"{start} --> {end}")
            lines.append(f"<v {spk} [{dir_cue}]>{txt}")
            lines.append(f"NOTE ASL-GLOSS: {gloss}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_vtt_time(seconds: float) -> str:
        mins, secs = divmod(max(0.0, seconds), 60)
        hrs, mins = divmod(int(mins), 60)
        return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"

    def reset(self) -> None:
        self.speaker_tracker.reset()
        self.history.clear()
