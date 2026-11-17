"""
ASLytics: On-Device Multilingual Group Conversation Transcription
for the Deaf and Hard of Hearing Community, powered by NVIDIA Parakeet AI.
"""

__version__ = "0.2.0"
__author__ = "Mankanwar Singh & ASLytics Contributors"

from aslytics.config import ASLyticsConfig, MODEL_REGISTRY, SUPPORTED_LANGUAGES
from aslytics.pipeline import GroupConversationPipeline

__all__ = [
    "__version__",
    "ASLyticsConfig",
    "MODEL_REGISTRY",
    "SUPPORTED_LANGUAGES",
    "GroupConversationPipeline",
]
