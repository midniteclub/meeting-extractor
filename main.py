"""
Meeting Extractor — record, transcribe, translate, and summarize meetings.

Usage
-----
  python main.py                         Launch GUI (default)
  python main.py --record                Record from mic/speakers then process
  python main.py --process FILE          Process an existing audio/video file
  python main.py --model small           Override Whisper model size
  python main.py --no-diarization        Skip speaker identification
  python main.py --no-translate          Skip translation
  python main.py --no-video              Audio-only (no screen capture)
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import config
from src.transcriber import Transcriber
from src.translator import Translator
from src.summarizer import Summarizer
from src.report_generator import generate_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_lang(code: str) -> str:
    """Map a raw language code (Whisper/langdetect/user) to en / zh-CN."""
    code = (code or "").lower()
    if code.startswith("zh") or code in ("chinese", "mandarin"):
        return "zh-CN"
    if code.startswith("en") or code == "english":
        return "en"
    return code


def _translation_incomplete(segments: list, threshold: float = 0.25) -> bool:
    """True if enough segments lack a usable translation to warrant a fallback."""
    total = failed = 0
    for seg in segments:
        if not seg.get("text", "").strip():
            continue
        total += 1
        translated = seg.get("translated_text", "")
        if not translated.strip() or translated.startswith("[Translation error"):
            failed += 1
    return total > 0 and (failed / total) >= threshold


def _fill_english_from_whisper(segments: list, eng_segments: list) -> None:
    """Fill missing/failed English translations by aligning Whisper's
    translate-task output to each transcription segment by time overlap."""
    for seg in segments:
        translated = seg.get("translated_text", "")
        if translated.strip() and not translated.startswith("[Translation error"):
            continue  # keep good online translations
        parts = [
            (e["start"], e["text"])
            for e in eng_segments
            if min(seg["end"], e["end"]) - max(seg["start"], e["start"]) > 0
        ]
        if parts:
            parts.sort()
            seg["translated_text"] = " ".join(t for _, t in parts)
            seg["target_language"] = "English"


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    audio_path: Path,
    session_info: dict,
    model: str = None,
    use_diarization: bool = True,
    num_speakers: int = None,
    use_translation: bool = True,
    summary_language: str = "en",
    progress_cb=None,
) -> dict:
    """Transcribe → translate → summarize → report.  Returns result dict.

    `summary_language` is the language the AI summary / key points are written
    in — i.e. the language *you* read. It defaults to English so a
    Mandarin meeting still yields an English summary you can understand.
    """
    if progress_cb is None:
        progress_cb = lambda msg, pct: print(f"[{pct:3d}%] {msg}")

    model = model or config.WHISPER_MODEL

    # 1. Transcribe
    transcriber = Transcriber(
        model_name=model,
        hf_token=config.HF_TOKEN,
        progress_cb=progress_cb,
    )
    segments = transcriber.transcribe(
        audio_path,
        use_diarization=use_diarization,
        num_speakers=num_speakers,
    )

    translator = Translator(progress_cb=progress_cb)

    # Source language: trust Whisper's own detection first (it analyzed the
    # audio directly), and only fall back to text-based detection when Whisper
    # was unsure. This is more reliable than re-detecting the transcript.
    source_lang = _normalize_lang(segments[0].get("language", "")) if segments else ""
    if source_lang not in ("en", "zh-CN"):
        full_text = " ".join(s["text"] for s in segments)
        source_lang = translator.detect_language(full_text)
    progress_cb(f"Detected language: {source_lang}", 68)

    # 2. Translate
    translation_info = None
    if use_translation:
        if source_lang in ("en", "zh-CN", "zh-TW", "zh"):
            segments, translation_info = translator.translate_segments(segments, source_lang)

            # If we translated to English but the online translator failed
            # (rate limited / unreachable), fall back to Whisper's offline
            # translate task so the user still gets English.
            if (
                translation_info
                and translation_info.get("target") == "en"
                and _translation_incomplete(segments)
            ):
                progress_cb(
                    "Online translation unavailable — translating to English with Whisper...",
                    78,
                )
                try:
                    eng_segments = transcriber.translate_to_english(audio_path)
                    _fill_english_from_whisper(segments, eng_segments)
                except Exception as e:
                    progress_cb(f"Whisper translation fallback failed: {e}", 82)
        else:
            progress_cb(f"Language '{source_lang}' not in EN/ZH — translation skipped.", 68)

    # 3. Summarize — in the language the user reads (default English). When the
    # meeting was translated into that language, summarize from the translation
    # so the output is readable even without an LLM API key.
    out_lang = _normalize_lang(summary_language) or "en"
    text_field = "text"
    if (
        translation_info
        and translation_info.get("target") == out_lang
        and source_lang != out_lang
    ):
        text_field = "translated_text"
    summarizer = Summarizer(
        anthropic_api_key=config.ANTHROPIC_API_KEY,
        progress_cb=progress_cb,
    )
    summary = summarizer.generate(segments, out_lang, text_field=text_field)

    # 4. Report
    progress_cb("Writing report...", 93)
    output_files = generate_report(
        session_info=session_info,
        segments=segments,
        translation_info=translation_info,
        summary=summary,
        output_dir=config.OUTPUT_DIR,
    )
    progress_cb("Done!", 100)

    return {
        "segments": segments,
        "translation_info": translation_info,
        "summary": summary,
        "output_files": output_files,
    }


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _cli_record(args):
    from colorama import Fore, Style, init
    init(autoreset=True)
    from src.recorder import MeetingRecorder
    import time

    print(f"\n{Fore.CYAN}Meeting Extractor{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Press Ctrl+C to stop recording.{Style.RESET_ALL}\n")

    recorder = MeetingRecorder(config.OUTPUT_DIR, record_video=not args.no_video)
    session_name = recorder.start()
    print(f"{Fore.GREEN}Recording started — session: {session_name}{Style.RESET_ALL}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    print(f"\n{Fore.YELLOW}Stopping...{Style.RESET_ALL}")
    session_info = recorder.stop()
    print(f"{Fore.GREEN}Saved. Duration: {session_info['duration']:.0f}s{Style.RESET_ALL}")

    audio_path = session_info.get("audio_path")
    if not audio_path or not Path(audio_path).exists():
        print(f"{Fore.RED}No audio file found — aborting.{Style.RESET_ALL}")
        return

    results = run_pipeline(
        audio_path=Path(audio_path),
        session_info=session_info,
        model=args.model,
        use_diarization=not args.no_diarization,
        num_speakers=args.speakers,
        use_translation=not args.no_translate,
        summary_language=args.summary_language,
    )
    print(f"\n{Fore.GREEN}Reports saved:{Style.RESET_ALL}")
    for fmt, path in results["output_files"].items():
        print(f"  {fmt.upper()}: {path}")
    print(f"\n{Fore.CYAN}--- SUMMARY ---{Style.RESET_ALL}")
    print(results["summary"].get("summary", ""))


def _cli_process(args):
    audio_path = Path(args.process)
    if not audio_path.exists():
        print(f"File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    session_info = {
        "session_name": audio_path.stem,
        "audio_path": audio_path,
        "video_path": None,
        "duration": 0,
        "start_time": datetime.now(),
    }
    results = run_pipeline(
        audio_path=audio_path,
        session_info=session_info,
        model=args.model,
        use_diarization=not args.no_diarization,
        num_speakers=args.speakers,
        use_translation=not args.no_translate,
        summary_language=args.summary_language,
    )
    print("\nReports saved:")
    for fmt, path in results["output_files"].items():
        print(f"  {fmt.upper()}: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Meeting Extractor")
    parser.add_argument("--record", action="store_true", help="Start recording now (CLI mode)")
    parser.add_argument("--process", metavar="FILE", help="Process an existing audio/video file")
    parser.add_argument(
        "--model",
        default=None,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default from .env / 'base')",
    )
    parser.add_argument("--no-diarization", action="store_true", help="Skip speaker identification")
    parser.add_argument(
        "--speakers",
        type=int,
        default=None,
        metavar="N",
        help="Exact number of speakers (e.g. 2). Omit to let pyannote auto-detect.",
    )
    parser.add_argument("--no-translate", action="store_true", help="Skip translation")
    parser.add_argument("--no-video", action="store_true", help="Audio-only recording")
    parser.add_argument(
        "--summary-language",
        default="en",
        choices=["en", "zh"],
        help="Language for the AI summary / key points — the language you read (default: en)",
    )

    args = parser.parse_args()

    if args.record:
        _cli_record(args)
    elif args.process:
        _cli_process(args)
    else:
        from gui import MeetingExtractorGUI
        MeetingExtractorGUI().run()


if __name__ == "__main__":
    main()
