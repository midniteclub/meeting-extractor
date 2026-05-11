from typing import Callable, Dict, List, Optional, Tuple

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

_CHUNK = 4500  # Google Translate limit is ~5000 chars; use 4500 to be safe


class Translator:
    """Language detection + EN ↔ ZH translation."""

    def __init__(self, progress_cb: Optional[Callable[[str, int], None]] = None):
        self.cb = progress_cb or (lambda msg, pct: None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_language(self, text: str) -> str:
        if not LANGDETECT_AVAILABLE:
            return "en"
        try:
            code = detect(text[:2000])
            return self._normalize(code)
        except Exception:
            return "en"

    def translate_segments(
        self, segments: List[Dict], detected_lang: str
    ) -> Tuple[List[Dict], Optional[Dict]]:
        """Translate all segment texts; return (updated_segments, info_dict)."""
        if not TRANSLATOR_AVAILABLE:
            raise RuntimeError(
                "deep-translator not installed.\nRun: pip install deep-translator"
            )

        source, target, target_name = self._pick_direction(detected_lang)
        if source is None:
            self.cb(f"Language '{detected_lang}' not supported for translation — skipped.", 70)
            return segments, None

        total = len(segments)
        updated = []
        for i, seg in enumerate(segments):
            pct = 70 + int(i / max(total, 1) * 15)
            self.cb(f"Translating segment {i + 1}/{total}...", pct)
            copy = dict(seg)
            copy["translated_text"] = self._translate(seg["text"], source, target)
            copy["target_language"] = target_name
            updated.append(copy)

        info = {"source": source, "target": target, "target_name": target_name}
        return updated, info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalize(self, code: str) -> str:
        if code.lower().startswith("zh"):
            return "zh-CN"
        return code.lower()

    def _pick_direction(self, lang: str):
        if lang == "en":
            return "en", "zh-CN", "Chinese (Simplified)"
        if lang in ("zh-CN", "zh-TW", "zh"):
            return "zh-CN", "en", "English"
        return None, None, None

    def _translate(self, text: str, source: str, target: str) -> str:
        if not text.strip():
            return text
        try:
            if len(text) <= _CHUNK:
                return GoogleTranslator(source=source, target=target).translate(text) or text
            return self._translate_long(text, source, target)
        except Exception as e:
            return f"[Translation error: {e}]"

    def _translate_long(self, text: str, source: str, target: str) -> str:
        # Split on sentence boundaries, accumulate chunks below _CHUNK
        import re
        sentences = re.split(r"(?<=[.!?。！？])\s+", text)
        chunks, current = [], ""
        for s in sentences:
            if len(current) + len(s) > _CHUNK:
                if current:
                    chunks.append(current)
                current = s
            else:
                current += (" " if current else "") + s
        if current:
            chunks.append(current)
        parts = [
            GoogleTranslator(source=source, target=target).translate(c) or c
            for c in chunks
        ]
        return " ".join(parts)
