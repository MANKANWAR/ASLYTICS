"""
Unit tests for ASLytics On-Device Parakeet Pipeline, Group Speaker Diarization,
Multilingual Translation, ASL Gloss Bridge, and D/HoH Accessibility Alerts.
"""

import unittest

from aslytics.config import ASLyticsConfig, MODEL_REGISTRY
from aslytics.diarization.group_tracker import GroupSpeakerTracker
from aslytics.multilingual.translator import MultilingualBridge
from aslytics.pipeline import GroupConversationPipeline


class TestASLyticsPipeline(unittest.TestCase):
    def test_parakeet_models_registered(self) -> None:
        self.assertIn("parakeet-tdt-0.6b-v3", MODEL_REGISTRY)
        self.assertIn("parakeet-tdt-0.6b-v2", MODEL_REGISTRY)
        self.assertIn("canary-1b-v2", MODEL_REGISTRY)
        self.assertIn("sortformer-4spk-v1", MODEL_REGISTRY)
        spec = MODEL_REGISTRY["parakeet-tdt-0.6b-v3"]
        self.assertEqual(spec.nemo_id, "nvidia/parakeet-tdt-0.6b-v3")
        self.assertEqual(spec.languages, 25)

    def test_group_speaker_diarization_and_overlap(self) -> None:
        tracker = GroupSpeakerTracker()
        t1 = tracker.record_turn("s1", start_s=0.0, end_s=4.0, display_name="Maya", azimuth_deg=0)
        self.assertFalse(t1.is_overlapping)
        self.assertEqual(t1.speaker.compass_direction, "Across Table (12 o'clock)")

        # Overlapping turn starting at 3.5s before Maya finishes at 4.0s
        t2 = tracker.record_turn("s2", start_s=3.5, end_s=7.0, display_name="Carlos", azimuth_deg=90)
        self.assertTrue(t2.is_overlapping)
        self.assertEqual(t2.overlapping_with, "Maya")
        self.assertEqual(t2.spatial_shift_deg, 90)
        self.assertEqual(t2.speaker.compass_direction, "Right Side (3 o'clock)")

    def test_asl_gloss_conversion_and_nmm(self) -> None:
        bridge = MultilingualBridge()
        res = bridge.to_asl_gloss("Can we review the accessibility design today?")
        # "TODAY" should be fronted (Time-Topic-Comment), and YES-NO-Q NMM added
        self.assertIn("TODAY", res.time_tokens)
        self.assertTrue(res.gloss_text.startswith("TODAY /"))
        self.assertEqual(res.question_type, "YES-NO-Q")
        self.assertIn("BROW-RAISE (Yes/No Question)", res.non_manual_markers)

    def test_multilingual_standup_scenario_and_watchwords(self) -> None:
        cfg = ASLyticsConfig(target_language="en", user_watchwords=["Alex", "question"])
        pipeline = GroupConversationPipeline(cfg)
        output = pipeline.run_scenario("multilingual_standup")

        self.assertEqual(output["group_summary"]["active_speakers_count"], 4)
        self.assertGreaterEqual(output["group_summary"]["overlapping_turns_count"], 1)

        # Last turn addresses "Alex" with a "question"
        last_turn = output["turns"][-1]
        self.assertTrue(last_turn["accessibility"]["direct_address_to_user"])
        self.assertIn("alex", last_turn["accessibility"]["triggered_watchwords"])

        vtt = pipeline.export_webvtt()
        self.assertIn("WEBVTT", vtt)
        self.assertIn("NOTE ASL-GLOSS:", vtt)

    def test_family_dinner_punjabi_hindi_scenario(self) -> None:
        cfg = ASLyticsConfig(target_language="en", user_watchwords=["Alex"])
        pipeline = GroupConversationPipeline(cfg)
        output = pipeline.run_scenario("family_dinner")
        self.assertEqual(len(output["turns"]), 4)
        self.assertEqual(output["turns"][0]["multilingual"]["source_lang"], "pa")
        self.assertEqual(output["turns"][2]["multilingual"]["source_lang"], "hi")
        self.assertTrue(output["turns"][2]["diarization"]["is_overlapping"])


if __name__ == "__main__":
    unittest.main()
