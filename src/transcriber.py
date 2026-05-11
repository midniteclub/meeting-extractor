from pathlib import Path
from typing import Callable, List, Optional

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from pyannote.audio import Pipeline as DiarizePipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

Segment = dict  # keys: start, end, text, speaker, language


class Transcriber:
    """Whisper transcription with optional pyannote speaker diarization."""

    def __init__(
        self,
        model_name: str = "base",
        hf_token: str = "",
        progress_cb: Optional[Callable[[str, int], None]] = None,
    ):
        self.model_name = model_name
        self.hf_token = hf_token
        self.cb = progress_cb or (lambda msg, pct: None)
        self._whisper = None
        self._diarizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: Path,
        use_diarization: bool = True,
        num_speakers: Optional[int] = None,
    ) -> List[Segment]:
        self._ensure_whisper()
        self.cb("Transcribing audio with Whisper...", 20)

        result = self._whisper.transcribe(
            str(audio_path),
            word_timestamps=True,
            verbose=False,
        )

        segments: List[Segment] = [
            {
                "start": s["start"],
                "end": s["end"],
                "text": s["text"].strip(),
                "speaker": "Speaker",
                "language": result.get("language", "unknown"),
            }
            for s in result["segments"]
            if s["text"].strip()
        ]

        self.cb("Transcription complete.", 50)

        if use_diarization and self._ensure_diarizer():
            segments = self._apply_diarization(audio_path, segments, num_speakers)

        return segments

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_whisper(self):
        if not WHISPER_AVAILABLE:
            raise RuntimeError(
                "openai-whisper not installed.\nRun: pip install openai-whisper"
            )
        if self._whisper is None:
            self.cb(f"Loading Whisper '{self.model_name}' model...", 5)
            self._whisper = whisper.load_model(self.model_name)

    def _ensure_diarizer(self) -> bool:
        if not PYANNOTE_AVAILABLE:
            self.cb("pyannote.audio not installed — skipping speaker ID.", 50)
            return False
        if not self.hf_token:
            self.cb("No HF_TOKEN set — skipping speaker ID.", 50)
            return False
        if self._diarizer is None:
            self.cb("Loading speaker diarization model...", 52)
            try:
                self._diarizer = DiarizePipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_token,
                )
            except Exception as e:
                self.cb(f"Could not load diarizer: {e}", 52)
                return False
        return True

    def _apply_diarization(
        self,
        audio_path: Path,
        segments: List[Segment],
        num_speakers: Optional[int] = None,
    ) -> List[Segment]:
        hint = f" (expecting {num_speakers} speakers)" if num_speakers else ""
        self.cb(f"Running speaker diarization{hint}...", 55)
        try:
            kwargs: dict = {}
            if num_speakers:
                kwargs["num_speakers"] = num_speakers
            diarization = self._diarizer(str(audio_path), **kwargs)

            # Build a canonical speaker name map
            label_to_name: dict[str, str] = {}
            for _, _, spk in diarization.itertracks(yield_label=True):
                if spk not in label_to_name:
                    label_to_name[spk] = f"Speaker {len(label_to_name) + 1}"

            # Build a list of (turn, speaker_name) for fast lookup
            turns = [
                (turn, label_to_name[spk])
                for turn, _, spk in diarization.itertracks(yield_label=True)
            ]

            # Assign each segment the speaker with the most overlap;
            # fall back to nearest turn midpoint when there is no overlap
            for seg in segments:
                seg_mid = (seg["start"] + seg["end"]) / 2
                best_spk = "Speaker"
                best_overlap = 0.0
                best_dist = float("inf")

                for turn, spk_name in turns:
                    ov = max(0.0, min(seg["end"], turn.end) - max(seg["start"], turn.start))
                    if ov > best_overlap:
                        best_overlap = ov
                        best_spk = spk_name
                    if ov == 0:
                        dist = min(abs(seg_mid - turn.start), abs(seg_mid - turn.end))
                        if dist < best_dist:
                            best_dist = dist
                            if best_overlap == 0:
                                best_spk = spk_name

                seg["speaker"] = best_spk

            self.cb("Speaker diarization complete.", 65)
        except Exception as e:
            self.cb(f"Diarization failed ({e}) — using single-speaker mode.", 65)

        return segments
