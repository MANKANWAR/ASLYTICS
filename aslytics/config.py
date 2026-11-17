"""
Configuration and Model Registry for ASLytics On-Device Pipeline.
Supports NVIDIA Parakeet-TDT, NVIDIA Canary, and NVIDIA Sortformer models
across NeMo (CUDA), ONNX Runtime (CPU/Edge), and MLX (Apple Silicon).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    """Metadata and runtime identifiers for an on-device speech model."""
    key: str
    name: str
    architecture: str
    parameters: str
    nemo_id: str
    onnx_id: str
    mlx_id: Optional[str]
    languages: int
    supports_word_timestamps: bool
    supports_punctuation: bool
    recommended_ram_gb: float
    description: str


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "parakeet-tdt-0.6b-v3": ModelSpec(
        key="parakeet-tdt-0.6b-v3",
        name="NVIDIA Parakeet TDT 0.6B v3 (Multilingual)",
        architecture="FastConformer + Token-and-Duration Transducer (TDT)",
        parameters="600M",
        nemo_id="nvidia/parakeet-tdt-0.6b-v3",
        onnx_id="istupakov/parakeet-tdt-0.6b-v3-onnx",
        mlx_id="mlx-community/parakeet-tdt-0.6b-v3",
        languages=25,
        supports_word_timestamps=True,
        supports_punctuation=True,
        recommended_ram_gb=2.2,
        description=(
            "Primary on-device multilingual ASR model. Delivers ultra-low latency "
            "transcription across 25 languages with native punctuation, capitalization, "
            "and word-level timestamps ideal for live group captioning."
        ),
    ),
    "parakeet-tdt-0.6b-v2": ModelSpec(
        key="parakeet-tdt-0.6b-v2",
        name="NVIDIA Parakeet TDT 0.6B v2 (English Ultra-Fast)",
        architecture="FastConformer + Token-and-Duration Transducer (TDT)",
        parameters="600M",
        nemo_id="nvidia/parakeet-tdt-0.6b-v2",
        onnx_id="istupakov/parakeet-tdt-0.6b-v2-onnx",
        mlx_id="mlx-community/parakeet-tdt-0.6b-v2",
        languages=1,
        supports_word_timestamps=True,
        supports_punctuation=True,
        recommended_ram_gb=1.8,
        description=(
            "High-throughput English ASR model with industry-leading WER and "
            "up to 3380x RTFx speed for resource-constrained edge devices."
        ),
    ),
    "canary-1b-v2": ModelSpec(
        key="canary-1b-v2",
        name="NVIDIA Canary 1B v2 (Multilingual ASR + Translation)",
        architecture="FastConformer Encoder + Transformer Decoder",
        parameters="1.0B",
        nemo_id="nvidia/canary-1b-v2",
        onnx_id="istupakov/canary-1b-v2-onnx",
        mlx_id=None,
        languages=25,
        supports_word_timestamps=True,
        supports_punctuation=True,
        recommended_ram_gb=3.8,
        description=(
            "Multi-task speech recognition and speech-to-text translation (AST) model "
            "for cross-lingual group conversations."
        ),
    ),
    "sortformer-4spk-v1": ModelSpec(
        key="sortformer-4spk-v1",
        name="NVIDIA Sortformer Diarizer v1 (Group Speaker Tracking)",
        architecture="Sortformer Arrival-Order Transformer Diarizer",
        parameters="118M",
        nemo_id="nvidia/diar_sortformer_4spk-v1",
        onnx_id="onnx-community/diar_sortformer_4spk-v1",
        mlx_id=None,
        languages=25,
        supports_word_timestamps=True,
        supports_punctuation=False,
        recommended_ram_gb=0.8,
        description=(
            "End-to-end neural speaker diarization model that resolves 'who spoke when' "
            "and detects overlapping speech in multi-person conversations."
        ),
    ),
}


SUPPORTED_LANGUAGES: Dict[str, Dict[str, str]] = {
    "en": {"name": "English", "native": "English", "family": "Germanic"},
    "es": {"name": "Spanish", "native": "Español", "family": "Romance"},
    "fr": {"name": "French", "native": "Français", "family": "Romance"},
    "de": {"name": "German", "native": "Deutsch", "family": "Germanic"},
    "it": {"name": "Italian", "native": "Italiano", "family": "Romance"},
    "pt": {"name": "Portuguese", "native": "Português", "family": "Romance"},
    "nl": {"name": "Dutch", "native": "Nederlands", "family": "Germanic"},
    "pl": {"name": "Polish", "native": "Polski", "family": "Slavic"},
    "uk": {"name": "Ukrainian", "native": "Українська", "family": "Slavic"},
    "ro": {"name": "Romanian", "native": "Română", "family": "Romance"},
    "cs": {"name": "Czech", "native": "Čeština", "family": "Slavic"},
    "sv": {"name": "Swedish", "native": "Svenska", "family": "Germanic"},
    "da": {"name": "Danish", "native": "Dansk", "family": "Germanic"},
    "fi": {"name": "Finnish", "native": "Suomi", "family": "Uralic"},
    "el": {"name": "Greek", "native": "Ελληνικά", "family": "Hellenic"},
    "hu": {"name": "Hungarian", "native": "Magyar", "family": "Uralic"},
    "bg": {"name": "Bulgarian", "native": "Български", "family": "Slavic"},
    "hr": {"name": "Croatian", "native": "Hrvatski", "family": "Slavic"},
    "sk": {"name": "Slovak", "native": "Slovenčina", "family": "Slavic"},
    "sl": {"name": "Slovenian", "native": "Slovenščina", "family": "Slavic"},
    "lt": {"name": "Lithuanian", "native": "Lietuvių", "family": "Baltic"},
    "lv": {"name": "Latvian", "native": "Latviešu", "family": "Baltic"},
    "et": {"name": "Estonian", "native": "Eesti", "family": "Uralic"},
    "mt": {"name": "Maltese", "native": "Malti", "family": "Semitic"},
    "ru": {"name": "Russian", "native": "Русский", "family": "Slavic"},
    # Community-extended bridge languages for translation / display
    "pa": {"name": "Punjabi", "native": "ਪੰਜਾਬੀ", "family": "Indo-Aryan"},
    "hi": {"name": "Hindi", "native": "हिन्दी", "family": "Indo-Aryan"},
    "zh": {"name": "Mandarin", "native": "中文", "family": "Sino-Tibetan"},
}


@dataclass
class ASLyticsConfig:
    """Runtime configuration for ASLytics on-device group transcription."""
    asr_model: str = "parakeet-tdt-0.6b-v3"
    diarizer_model: str = "sortformer-4spk-v1"
    backend: str = "auto"  # "auto", "onnx", "mlx", "nemo", or "simulation"
    quantization: str = "int8"  # "fp16", "int8", "fp32"
    target_language: str = "en"
    enable_asl_gloss: bool = True
    enable_overlap_alerts: bool = True
    enable_sound_event_cues: bool = True
    user_watchwords: List[str] = field(
        default_factory=lambda: ["Alex", "everyone", "question", "important", "urgent"]
    )
    max_speakers: int = 6
    chunk_duration_ms: int = 480
    offline_strict_mode: bool = True
