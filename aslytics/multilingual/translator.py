"""
Multilingual Translation & ASL Gloss Bridge for ASLytics.

Provides:
1. On-device cross-lingual normalization & translation for multilingual group settings
   (working alongside NVIDIA Parakeet-TDT 0.6B v3 & NVIDIA Canary 1B v2).
2. Rule-based & syntactic American Sign Language (ASL) Gloss transformation
   (Time-Topic-Comment syntax + Non-Manual Grammatical Markers) so Deaf signers
   can scan rapid group discussions in native visual-spatial grammar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aslytics.config import SUPPORTED_LANGUAGES


@dataclass
class ASLGlossResult:
    """Structured ASL Gloss output with Non-Manual Markers (NMM) and grammar breakdown."""
    gloss_text: str
    non_manual_markers: List[str] = field(default_factory=list)
    time_tokens: List[str] = field(default_factory=list)
    topic_comment_tokens: List[str] = field(default_factory=list)
    question_type: Optional[str] = None  # "WH-Q", "YES-NO-Q", or None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gloss_text": self.gloss_text,
            "non_manual_markers": self.non_manual_markers,
            "time_tokens": self.time_tokens,
            "topic_comment_tokens": self.topic_comment_tokens,
            "question_type": self.question_type,
        }


class MultilingualBridge:
    """
    Handles multilingual group conversation translation and ASL Gloss generation.
    """

    # Curated local phrasebook & lexical transfer for instant zero-cloud multilingual group turns
    CURATED_TRANSLATIONS: Dict[str, Dict[str, str]] = {
        "Hola equipo, podemos revisar el diseño de accesibilidad hoy?": {
            "en": "Hi team, can we review the accessibility design today?",
            "fr": "Salut l'équipe, peut-on revoir le design d'accessibilité aujourd'hui ?",
        },
        "Claro, ya actualicé los subtítulos en tiempo real y el radar visual.": {
            "en": "Sure, I already updated the real-time captions and the visual radar.",
            "fr": "Bien sûr, j'ai déjà mis à jour les sous-titres en temps réel et le radar visuel.",
        },
        "Bonjour à tous, le modèle Parakeet fonctionne hors ligne avec une latence très faible.": {
            "en": "Hello everyone, the Parakeet model runs offline with very low latency.",
            "es": "Hola a todos, el modelo Parakeet funciona sin conexión con muy baja latencia.",
        },
        "Wir können jetzt gleichzeitig Deutsch, Spanisch und Französisch in der Gruppe verstehen.": {
            "en": "We can now understand German, Spanish, and French simultaneously in the group.",
            "es": "Ahora podemos entender alemán, español y francés simultáneamente en el grupo.",
        },
        "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਜੀ, ਕੀ ਤੁਸੀਂ ਸਾਰੇ ਰਾਤ ਦੇ ਖਾਣੇ ਲਈ ਤਿਆਰ ਹੋ?": {
            "en": "Hello everyone, are you all ready for dinner?",
            "es": "Hola a todos, ¿están todos listos para la cena?",
        },
        "ਹਾਂਜੀ, ਅਸੀਂ ਬਸ ਪੰਜ ਮਿੰਟ ਵਿੱਚ ਆ ਰਹੇ ਹਾਂ!": {
            "en": "Yes, we are coming in just five minutes!",
            "es": "¡Sí, llegamos en solo cinco minutos!",
        },
        "नमस्ते, क्या हम अगले सप्ताह नई सुविधा लॉन्च कर रहे हैं?": {
            "en": "Hello, are we launching the new feature next week?",
            "es": "Hola, ¿lanzaremos la nueva función la próxima semana?",
        },
    }

    WORD_GLOSS_MAP: Dict[str, str] = {
        "going": "GO",
        "working": "WORK",
        "running": "RUN",
        "testing": "TEST",
        "launching": "LAUNCH",
        "updated": "FINISH-UPDATE",
        "reviewing": "REVIEW",
        "understand": "UNDERSTAND",
        "everyone": "ALL-YOU",
        "everybody": "ALL-YOU",
        "we": "WE",
        "us": "WE",
        "our": "OUR",
        "you": "IX-YOU",
        "your": "POSS-YOU",
        "i": "IX-ME",
        "me": "IX-ME",
        "my": "POSS-ME",
        "they": "IX-THEY",
        "he": "IX-HE",
        "she": "IX-SHE",
        "cannot": "CAN'T",
        "can't": "CAN'T",
        "don't": "NOT",
        "doesn't": "NOT",
        "didn't": "NOT",
        "won't": "REFUSE",
        "aren't": "NOT",
        "isn't": "NOT",
    }

    STOPWORDS_ASL = {
        "a", "an", "the", "am", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "in", "on", "at", "by", "do", "does", "did", "just", "very",
        "really", "that", "this",
    }

    TIME_WORDS = {
        "today", "tomorrow", "yesterday", "now", "later", "soon", "morning",
        "afternoon", "evening", "tonight", "next", "week", "already",
    }

    WH_WORDS = {"who", "what", "where", "when", "why", "how", "which"}

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str = "en",
         explicit_translation: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Translate an utterance from `source_lang` into `target_lang`.
        Returns original text, translated text, and language metadata.
        """
        src_meta = SUPPORTED_LANGUAGES.get(
            source_lang, {"name": source_lang.upper(), "native": source_lang, "family": "Other"}
        )
        tgt_meta = SUPPORTED_LANGUAGES.get(
            target_lang, {"name": target_lang.upper(), "native": target_lang, "family": "Other"}
        )

        if explicit_translation:
            translated = explicit_translation
        elif source_lang == target_lang:
            translated = text
        elif text.strip() in self.CURATED_TRANSLATIONS:
            entry = self.CURATED_TRANSLATIONS[text.strip()]
            translated = entry.get(target_lang, entry.get("en", text))
        else:
            translated = self._lexical_fallback_translate(text, source_lang, target_lang)

        return {
            "source_lang": source_lang,
            "source_lang_name": src_meta["name"],
            "source_lang_native": src_meta["native"],
            "target_lang": target_lang,
            "target_lang_name": tgt_meta["name"],
            "is_translated": source_lang != target_lang,
            "translated_text": translated,
        }

    @staticmethod
    def _lexical_fallback_translate(text: str, source_lang: str, target_lang: str) -> str:
        """Lightweight offline lexical translator for ad-hoc multilingual phrases."""
        if source_lang == target_lang:
            return text
        dictionary = {
            "hola": "hello",
            "equipo": "team",
            "gracias": "thank you",
            "reunión": "meeting",
            "proyecto": "project",
            "hoy": "today",
            "mañana": "tomorrow",
            "bonjour": "hello",
            "merci": "thank you",
            "équipe": "team",
            "aujourd'hui": "today",
            "oui": "yes",
            "non": "no",
            "hallo": "hello",
            "danke": "thank you",
            "heute": "today",
            "wir": "we",
            "können": "can",
        }
        tokens = text.split()
        out = []
        for tok in tokens:
            clean = re.sub(r"[^\w']", "", tok.lower())
            replacement = dictionary.get(clean)
            if replacement:
                out.append(replacement.capitalize() if tok[0].isupper() else replacement)
            else:
                out.append(tok)
        return " ".join(out)

    def to_asl_gloss(self, english_text: str) -> ASLGlossResult:
        """
        Convert an English utterance into structured American Sign Language (ASL) Gloss:
        - Extracts Time indicators to the front (Time-Topic-Comment syntax)
        - Removes English auxiliary verbs, copulas, and articles
        - Places WH-question words and negation markers in canonical clause-final / NMM positions
        - Annotates facial Non-Manual Markers (NMM)
        """
        raw = english_text.strip()
        is_question = raw.endswith("?")
        clean = re.sub(r"[^\w\s']", " ", raw)
        words = [w for w in clean.split() if w]

        time_tokens: List[str] = []
        content_tokens: List[str] = []
        wh_tokens: List[str] = []
        neg_tokens: List[str] = []
        nmm: List[str] = []

        for w in words:
            low = w.lower()
            if low in self.STOPWORDS_ASL:
                continue
            if low in self.TIME_WORDS:
                time_tokens.append(low.upper())
                continue
            if low in self.WH_WORDS:
                wh_tokens.append(low.upper())
                continue
            if low in ("not", "never", "no", "none", "don't", "doesn't", "didn't"):
                neg_tokens.append("NOT")
                continue

            mapped = self.WORD_GLOSS_MAP.get(low, low.upper())
            content_tokens.append(mapped)

        question_type: Optional[str] = None
        if is_question:
            if wh_tokens:
                question_type = "WH-Q"
                nmm.append("BROW-FURROW (WH-Question)")
            else:
                question_type = "YES-NO-Q"
                nmm.append("BROW-RAISE (Yes/No Question)")

        if neg_tokens or "CAN'T" in content_tokens or "REFUSE" in content_tokens:
            nmm.append("HEAD-SHAKE (Negation)")

        if not nmm and raw.endswith("!"):
            nmm.append("INTENSE-GAZE / NOD (Emphasis)")

        # Assemble canonical ASL order: [TIME] + [TOPIC/COMMENT] + [NEGATION] + [WH-WORD]
        ordered: List[str] = []
        if time_tokens:
            ordered.extend(time_tokens)
            ordered.append("/")  # Clause pause marker after TIME
        ordered.extend(content_tokens)
        ordered.extend(neg_tokens)
        ordered.extend(wh_tokens)

        gloss_str = " ".join(ordered).strip(" /")
        if is_question and question_type == "WH-Q":
            gloss_str = f"{gloss_str} [WH-Q]"
        elif is_question and question_type == "YES-NO-Q":
            gloss_str = f"{gloss_str} [Y/N-Q]"

        return ASLGlossResult(
            gloss_text=gloss_str or "—",
            non_manual_markers=nmm,
            time_tokens=time_tokens,
            topic_comment_tokens=content_tokens,
            question_type=question_type,
        )
