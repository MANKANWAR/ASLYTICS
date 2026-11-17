"""
Command-Line Interface for ASLytics:
On-Device Multilingual Group Transcription for the Deaf & Hard-of-Hearing Community.
"""

from __future__ import annotations

import argparse
import json
from typing import List, Optional

from aslytics.config import ASLyticsConfig, MODEL_REGISTRY
from aslytics.pipeline import BUILTIN_GROUP_SCENARIOS, GroupConversationPipeline
from aslytics.server import run_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aslytics",
        description=(
            "ASLytics: Open-Source On-Device Multilingual Group Conversation "
            "Transcription & ASL Gloss Bridge powered by NVIDIA Parakeet AI."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    # `aslytics serve`
    serve_p = sub.add_parser("serve", help="Launch the local accessible Web UI & REST API")
    serve_p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    serve_p.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")

    # `aslytics demo`
    demo_p = sub.add_parser(
        "demo",
        help="Run a multi-speaker multilingual group scenario in the terminal",
    )
    demo_p.add_argument(
        "--scenario",
        choices=list(BUILTIN_GROUP_SCENARIOS.keys()),
        default="multilingual_standup",
        help="Group scenario to run",
    )
    demo_p.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="parakeet-tdt-0.6b-v3",
        help="NVIDIA on-device model to use",
    )
    demo_p.add_argument(
        "--target-lang",
        default="en",
        help="Target translation language code (default: en)",
    )
    demo_p.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON output",
    )

    # `aslytics models`
    sub.add_parser("models", help="List supported on-device NVIDIA Parakeet/Canary/Sortformer models")

    return parser


def print_terminal_scenario(result: dict) -> None:
    sc = result["scenario"]
    eng = result["engine_status"]
    print("=" * 78)
    print(f" ASLytics | {sc['title']}")
    print(f" Setting : {sc['setting']}")
    print(
        f" Model   : {eng['active_model']['name']} "
        f"[Backend: {eng['active_backend']} | Offline: {eng['offline_strict_mode']}]"
    )
    print("=" * 78)

    for turn in result["turns"]:
        spk = turn["speaker"]
        diar = turn["diarization"]
        tx = turn["transcript"]
        ml = turn["multilingual"]
        asl = turn["asl_gloss"]
        acc = turn["accessibility"]

        overlap_tag = (
            f" ⚠️ [OVERLAP with {diar['overlapping_with']}]" if diar["is_overlapping"] else ""
        )
        watch_tag = (
            f" 🔔 [ALERT: {', '.join(acc['triggered_watchwords']).upper()}]"
            if acc["triggered_watchwords"]
            else ""
        )
        print(
            f"\n[{turn['timing']['start_s']:5.1f}s -> {turn['timing']['end_s']:5.1f}s] "
            f"👤 {spk['display_name']} ({spk['compass_direction']}) "
            f"[{ml['source_lang_name']}]{overlap_tag}{watch_tag}"
        )
        print(f"  🗣️  Spoken     : {tx['original_text']}")
        if ml["is_translated"]:
            print(f"  🌐 Translated : {ml['translated_text']}")
        print(f"  🤟 ASL Gloss  : {asl['gloss_text']}")
        if asl["non_manual_markers"]:
            print(f"  👀 Facial NMM : {', '.join(asl['non_manual_markers'])}")
        if acc["acoustic_events"]:
            print(f"  🔊 Sound Cue  : {', '.join(acc['acoustic_events'])}")

    summary = result["group_summary"]
    print("\n" + "-" * 78)
    print(
        f" Group Summary: {summary['active_speakers_count']} speakers | "
        f"{summary['total_turns']} turns | "
        f"{summary['overlapping_turns_count']} overlapping speech events resolved"
    )
    print("-" * 78)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        run_server(host=args.host, port=args.port)
        return 0

    if args.command == "models":
        for spec in MODEL_REGISTRY.values():
            print(f"- {spec.key}: {spec.name} ({spec.parameters})")
            print(f"    Architecture : {spec.architecture}")
            print(f"    NeMo / HF ID : {spec.nemo_id}")
            print(f"    ONNX ID      : {spec.onnx_id}")
            if spec.mlx_id:
                print(f"    MLX ID       : {spec.mlx_id}")
            print(f"    Languages    : {spec.languages} | RAM: ~{spec.recommended_ram_gb} GB\n")
        return 0

    # Default to running the demo if `demo` or no subcommand given
    scenario_id = getattr(args, "scenario", "multilingual_standup")
    model_key = getattr(args, "model", "parakeet-tdt-0.6b-v3")
    target_lang = getattr(args, "target_lang", "en")
    as_json = getattr(args, "json", False)

    cfg = ASLyticsConfig(asr_model=model_key, target_language=target_lang)
    pipeline = GroupConversationPipeline(cfg)
    result = pipeline.run_scenario(scenario_id)

    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_terminal_scenario(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
