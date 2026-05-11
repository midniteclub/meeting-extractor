import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def generate_report(
    session_info: Dict,
    segments: List[Dict],
    translation_info: Optional[Dict],
    summary: Dict,
    output_dir: Path,
) -> Dict[str, Path]:
    """Write a human-readable .txt report and a machine-readable .json file."""

    name = session_info.get("session_name", datetime.now().strftime("%Y%m%d_%H%M%S"))
    start_time = session_info.get("start_time", datetime.now())
    duration = session_info.get("duration", 0)
    detected_lang = segments[0].get("language", "unknown") if segments else "unknown"
    speakers = list(dict.fromkeys(seg.get("speaker", "Speaker") for seg in segments))

    SEP = "=" * 70

    lines = [
        SEP,
        "MEETING ANALYSIS REPORT",
        SEP,
        f"Date       : {start_time.strftime('%Y-%m-%d %H:%M') if isinstance(start_time, datetime) else start_time}",
        f"Duration   : {_fmt(duration)}",
        f"Language   : {detected_lang}",
    ]

    if translation_info:
        lines.append(
            f"Translation: {translation_info.get('source')} → {translation_info.get('target')}"
        )

    lines += [
        f"Participants: {', '.join(speakers)}",
        "",
    ]

    # Key points
    lines += [SEP, "KEY POINTS OUTLINE", SEP, summary.get("key_points", ""), ""]

    # Summary
    lines += [SEP, "MEETING SUMMARY", SEP, summary.get("summary", ""), ""]

    # Action items
    action_items = summary.get("action_items", "").strip()
    if action_items:
        lines += [SEP, "ACTION ITEMS / DECISIONS", SEP, action_items, ""]

    # Full transcript
    lines += [SEP, "FULL TRANSCRIPT", SEP]
    for seg in segments:
        lines.append(f"[{_fmt(seg.get('start', 0))}] {seg.get('speaker', 'Speaker')}: {seg.get('text', '')}")
    lines.append("")

    # Translation
    if translation_info and any(seg.get("translated_text") for seg in segments):
        target_name = translation_info.get("target_name", "Translation")
        lines += [SEP, f"TRANSLATION ({target_name})", SEP]
        for seg in segments:
            txt = seg.get("translated_text", "")
            if txt:
                lines.append(
                    f"[{_fmt(seg.get('start', 0))}] {seg.get('speaker', 'Speaker')}: {txt}"
                )
        lines.append("")

    txt_path = output_dir / f"{name}_report.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    json_path = output_dir / f"{name}_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "session": session_info,
                "language": detected_lang,
                "translation": translation_info,
                "summary": summary,
                "segments": segments,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=_serializable,
        )

    return {"txt": txt_path, "json": json_path}
