"""
Zero-Dependency Local HTTP & REST API Server for ASLytics.
Serves the accessible D/HoH Group Conversation Web UI and on-device Parakeet endpoints.
"""

from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from aslytics.config import ASLyticsConfig, MODEL_REGISTRY, SUPPORTED_LANGUAGES
from aslytics.pipeline import BUILTIN_GROUP_SCENARIOS, GroupConversationPipeline


WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
GLOBAL_PIPELINE = GroupConversationPipeline(ASLyticsConfig())


class ASLyticsHTTPRequestHandler(BaseHTTPRequestHandler):
    """Handles static web UI assets and JSON REST API calls."""

    def log_message(self, format: str, *args: Any) -> None:
        # Keep console logs clean
        return

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)

        if route == "/api/status":
            self._send_json(
                {
                    "status": "ok",
                    "engine": GLOBAL_PIPELINE.asr_engine.get_status_summary(),
                    "models": {
                        k: {
                            "key": v.key,
                            "name": v.name,
                            "architecture": v.architecture,
                            "parameters": v.parameters,
                            "nemo_id": v.nemo_id,
                            "onnx_id": v.onnx_id,
                            "mlx_id": v.mlx_id,
                            "languages": v.languages,
                            "recommended_ram_gb": v.recommended_ram_gb,
                            "description": v.description,
                        }
                        for k, v in MODEL_REGISTRY.items()
                    },
                    "supported_languages": SUPPORTED_LANGUAGES,
                    "scenarios": [
                        {
                            "id": s["id"],
                            "title": s["title"],
                            "setting": s["setting"],
                            "description": s["description"],
                            "turn_count": len(s["turns"]),
                        }
                        for s in BUILTIN_GROUP_SCENARIOS.values()
                    ],
                }
            )
            return

        if route == "/api/scenario":
            scenario_id = query.get("id", ["multilingual_standup"])[0]
            target_lang = query.get("target_lang", ["en"])[0]
            GLOBAL_PIPELINE.config.target_language = target_lang
            result = GLOBAL_PIPELINE.run_scenario(scenario_id)
            self._send_json(result)
            return

        if route == "/api/export.vtt":
            vtt_body = GLOBAL_PIPELINE.export_webvtt().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/vtt; charset=utf-8")
            self.send_header(
                "Content-Disposition", 'attachment; filename="aslytics_group_session.vtt"'
            )
            self.send_header("Content-Length", str(len(vtt_body)))
            self.end_headers()
            self.wfile.write(vtt_body)
            return

        # Serve static files from /web
        rel_path = "index.html" if route in ("/", "/index.html") else route.lstrip("/")
        file_path = (WEB_ROOT / rel_path).resolve()
        if not str(file_path).startswith(str(WEB_ROOT)) or not file_path.is_file():
            self._send_json({"error": "Not found"}, status=404)
            return

        mime, _ = mimetypes.guess_type(str(file_path))
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{mime or 'text/plain'}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=400)
            return

        if parsed.path == "/api/transcribe_turn":
            target_lang = data.get("target_language")
            if target_lang:
                GLOBAL_PIPELINE.config.target_language = target_lang
            watchwords = data.get("watchwords")
            if isinstance(watchwords, list):
                GLOBAL_PIPELINE.accessibility.watchwords = [w.lower() for w in watchwords]

            turn = GLOBAL_PIPELINE.process_utterance(
                speaker_id=data.get("speaker_id", "spk_live"),
                display_name=data.get("display_name", "Live Speaker"),
                role=data.get("role", "Participant"),
                azimuth_deg=int(data.get("azimuth_deg", 0)),
                text=data.get("text", ""),
                language_hint=data.get("language_hint"),
                acoustic_events=data.get("acoustic_events", []),
            )
            self._send_json(
                {
                    "turn": turn,
                    "group_summary": GLOBAL_PIPELINE.speaker_tracker.get_group_summary(),
                }
            )
            return

        if parsed.path == "/api/config":
            if "asr_model" in data and data["asr_model"] in MODEL_REGISTRY:
                GLOBAL_PIPELINE.config.asr_model = data["asr_model"]
                GLOBAL_PIPELINE.asr_engine.model_spec = MODEL_REGISTRY[data["asr_model"]]
            if "target_language" in data:
                GLOBAL_PIPELINE.config.target_language = data["target_language"]
            if "watchwords" in data and isinstance(data["watchwords"], list):
                GLOBAL_PIPELINE.config.user_watchwords = data["watchwords"]
                GLOBAL_PIPELINE.accessibility.watchwords = [
                    w.lower() for w in data["watchwords"]
                ]
            self._send_json(
                {
                    "status": "updated",
                    "engine": GLOBAL_PIPELINE.asr_engine.get_status_summary(),
                }
            )
            return

        self._send_json({"error": "Unknown endpoint"}, status=404)


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Start the local ASLytics web application and API server."""
    server = ThreadingHTTPServer((host, port), ASLyticsHTTPRequestHandler)
    print(f"ASLytics On-Device Server running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down ASLytics server.")
    finally:
        server.server_close()
