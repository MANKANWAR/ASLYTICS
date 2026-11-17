#!/usr/bin/env python3
"""
Example: Streaming Multi-Speaker Group Conversation turns through ASLytics
with real-time ASL Gloss, Spatial Speaker Radar, and Watchword Alerts.
"""

from aslytics.config import ASLyticsConfig
from aslytics.pipeline import GroupConversationPipeline


def run_custom_group_session() -> None:
    config = ASLyticsConfig(
        asr_model="parakeet-tdt-0.6b-v3",
        target_language="en",
        user_watchwords=["Mankanwar", "everyone", "deadline"],
    )
    pipeline = GroupConversationPipeline(config)

    sample_turns = [
        {
            "speaker_id": "spk_1",
            "display_name": "Elena (Research)",
            "azimuth_deg": 35,
            "lang": "es",
            "text": "Hola equipo, podemos revisar el diseño de accesibilidad hoy?",
        },
        {
            "speaker_id": "spk_2",
            "display_name": "Marc (Systems)",
            "azimuth_deg": 280,
            "lang": "fr",
            "text": "Bonjour à tous, le modèle Parakeet fonctionne hors ligne avec une latence très faible.",
        },
        {
            "speaker_id": "spk_3",
            "display_name": "Harpreet (Lead)",
            "azimuth_deg": 120,
            "lang": "pa",
            "text": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਜੀ, ਕੀ ਤੁਸੀਂ ਸਾਰੇ ਰਾਤ ਦੇ ਖਾਣੇ ਲਈ ਤਿਆਰ ਹੋ?",
        },
        {
            "speaker_id": "spk_1",
            "display_name": "Elena (Research)",
            "azimuth_deg": 35,
            "lang": "en",
            "text": "Mankanwar, can we confirm our open source deadline tomorrow?",
        },
    ]

    for item in sample_turns:
        result = pipeline.process_utterance(
            speaker_id=item["speaker_id"],
            display_name=item["display_name"],
            azimuth_deg=item["azimuth_deg"],
            language_hint=item["lang"],
            text=item["text"],
        )
        spk = result["speaker"]
        ml = result["multilingual"]
        asl = result["asl_gloss"]
        acc = result["accessibility"]
        print(f"[{spk['display_name']} @ {spk['compass_direction']}] ({ml['source_lang_name']})")
        print(f"  Caption   : {ml['translated_text']}")
        print(f"  ASL Gloss : {asl['gloss_text']}  {asl['non_manual_markers']}")
        if acc["direct_address_to_user"]:
            print(f"  🔔 WATCHWORD ALERT: {acc['triggered_watchwords']}")
        print()


if __name__ == "__main__":
    run_custom_group_session()
