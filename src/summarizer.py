import re
from typing import Callable, Dict, List, Optional


class Summarizer:
    """Generates key-point outlines and summaries using Claude API or simple extraction."""

    def __init__(
        self,
        anthropic_api_key: str = "",
        progress_cb: Optional[Callable[[str, int], None]] = None,
    ):
        self.api_key = anthropic_api_key
        self.cb = progress_cb or (lambda msg, pct: None)
        self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, segments: List[Dict], language: str = "en") -> Dict[str, str]:
        self.cb("Generating summary...", 87)
        transcript = "\n".join(
            f"[{seg.get('speaker', 'Speaker')}] {seg['text']}"
            for seg in segments
            if seg.get("text")
        )
        client = self._get_client()
        if client:
            return self._summarize_with_claude(transcript, language, client)
        return self._summarize_simple(transcript, segments)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        if not self.api_key:
            return None
        if self._client is None:
            try:
                import anthropic
                import config
                kwargs: dict = {"api_key": self.api_key}
                if config.ANTHROPIC_BASE_URL:
                    kwargs["base_url"] = config.ANTHROPIC_BASE_URL
                self._client = anthropic.Anthropic(**kwargs)
            except ImportError:
                return None
        return self._client

    def _summarize_with_claude(self, transcript: str, language: str, client) -> Dict[str, str]:
        lang_name = "Chinese" if language.startswith("zh") else "English"
        prompt = f"""Analyze this meeting transcript and respond in {lang_name}.

Respond in EXACTLY this format (keep the ## headers):

## KEY POINTS OUTLINE
[numbered hierarchical outline of the main topics and sub-points]

## SUMMARY
[2–4 paragraph prose summary of the entire meeting]

## ACTION ITEMS / DECISIONS
[bullet list of concrete decisions or next steps, or write "None identified" if there are none]

TRANSCRIPT:
{transcript[:14000]}"""

        try:
            msg = client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return self._parse_response(msg.content[0].text)
        except Exception as e:
            self.cb(f"Claude API error: {e} — falling back to simple summarizer.", 88)
            return self._summarize_simple(transcript, [])

    def _parse_response(self, text: str) -> Dict[str, str]:
        result = {"key_points": "", "summary": "", "action_items": ""}
        patterns = {
            "key_points": r"## KEY POINTS OUTLINE\n(.*?)(?=\n## |\Z)",
            "summary": r"## SUMMARY\n(.*?)(?=\n## |\Z)",
            "action_items": r"## ACTION ITEMS / DECISIONS\n(.*?)(?=\n## |\Z)",
        }
        for key, pat in patterns.items():
            m = re.search(pat, text, re.DOTALL)
            if m:
                result[key] = m.group(1).strip()
        if not any(result.values()):
            result["summary"] = text
        return result

    def _summarize_simple(self, transcript: str, segments: List[Dict]) -> Dict[str, str]:
        sentences = [
            s.strip()
            for s in re.split(r"[.!?。！？\n]+", transcript)
            if len(s.strip()) > 25
        ]
        step = max(1, len(sentences) // 12)
        key_sentences = sentences[::step][:15]
        key_points = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(key_sentences))

        speakers = list(dict.fromkeys(
            seg.get("speaker", "Speaker") for seg in segments
        ))
        word_count = len(transcript.split())
        participant_str = f"{len(speakers)} participant(s): {', '.join(speakers)}" if speakers else "unknown participants"

        summary = (
            f"Meeting processed: ~{word_count} words with {participant_str}.\n\n"
            "For a detailed AI summary, add your ANTHROPIC_API_KEY to the .env file."
        )

        return {
            "key_points": key_points,
            "summary": summary,
            "action_items": "Set ANTHROPIC_API_KEY in .env for action-item extraction.",
        }
