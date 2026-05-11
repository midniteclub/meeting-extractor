import subprocess
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import config
from main import run_pipeline
from src.recorder import MeetingRecorder, list_monitors


class MeetingExtractorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Meeting Extractor")
        self.root.geometry("960x720")
        self.root.minsize(760, 580)

        self._recorder: MeetingRecorder | None = None
        self._rec_start: datetime | None = None
        self._timer_active = False

        # Populated in _build_ui; parallel list to the combobox values
        self._monitor_indices: list[int] = [1]

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────────────────
        hdr = ttk.Frame(self.root, padding=(12, 8))
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text="Meeting Extractor", font=("Segoe UI", 15, "bold")).pack(side=tk.LEFT)
        ttk.Label(
            hdr,
            text="  Record · Transcribe · Translate · Summarize",
            font=("Segoe UI", 9),
            foreground="#666",
        ).pack(side=tk.LEFT, pady=(4, 0))

        ttk.Separator(self.root).pack(fill=tk.X)

        # ── Controls row ────────────────────────────────────────────────────
        ctrl = ttk.Frame(self.root, padding=(10, 6))
        ctrl.pack(fill=tk.X)

        self.rec_btn = ttk.Button(
            ctrl, text="⏺  Start Recording", command=self._toggle_record, width=20
        )
        self.rec_btn.pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(ctrl, text="📂  Open File", command=self._open_file, width=14).pack(
            side=tk.LEFT, padx=(0, 12)
        )

        # Options
        opt = ttk.LabelFrame(ctrl, text="Options", padding=(6, 2))
        opt.pack(side=tk.LEFT, padx=(0, 10))

        self.var_video = tk.BooleanVar(value=config.RECORD_VIDEO)
        ttk.Checkbutton(opt, text="Record video", variable=self.var_video).pack(
            side=tk.LEFT, padx=4
        )
        self.var_diarize = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="Identify speakers", variable=self.var_diarize).pack(
            side=tk.LEFT, padx=4
        )
        self.speakers_var = tk.StringVar(value="Auto")
        ttk.Label(opt, text="# speakers:").pack(side=tk.LEFT, padx=(6, 1))
        ttk.Combobox(
            opt,
            textvariable=self.speakers_var,
            values=["Auto", "2", "3", "4", "5", "6"],
            width=5,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(0, 4))
        self.var_translate = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="Translate", variable=self.var_translate).pack(
            side=tk.LEFT, padx=4
        )

        # Model
        mdl = ttk.LabelFrame(ctrl, text="Whisper model", padding=(6, 2))
        mdl.pack(side=tk.LEFT, padx=(0, 10))
        self.model_var = tk.StringVar(value=config.WHISPER_MODEL)
        ttk.Combobox(
            mdl,
            textvariable=self.model_var,
            values=["tiny", "base", "small", "medium", "large"],
            width=7,
            state="readonly",
        ).pack()

        # Monitor selector
        mon_frame = ttk.LabelFrame(ctrl, text="Record monitor", padding=(6, 2))
        mon_frame.pack(side=tk.LEFT)
        monitors = list_monitors()
        self._monitor_indices = [idx for idx, _ in monitors]
        mon_labels = [label for _, label in monitors]
        self.monitor_var = tk.StringVar(value=mon_labels[0] if mon_labels else "Monitor 1")
        self.monitor_combo = ttk.Combobox(
            mon_frame,
            textvariable=self.monitor_var,
            values=mon_labels,
            width=22,
            state="readonly",
        )
        self.monitor_combo.current(0)
        self.monitor_combo.pack()

        ttk.Separator(self.root).pack(fill=tk.X)

        # ── Status bar ──────────────────────────────────────────────────────
        stat = ttk.Frame(self.root, padding=(10, 3))
        stat.pack(fill=tk.X)

        self.status_var = tk.StringVar(value="Ready — press Start Recording or open an audio file.")
        ttk.Label(stat, textvariable=self.status_var, font=("Segoe UI", 9)).pack(side=tk.LEFT)

        self.timer_var = tk.StringVar(value="")
        ttk.Label(
            stat, textvariable=self.timer_var, font=("Consolas", 10), foreground="#c0392b"
        ).pack(side=tk.RIGHT)

        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 4))

        # ── Notebook tabs ───────────────────────────────────────────────────
        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 4))

        self.transcript_box = self._make_tab(nb, "Transcript")
        self.translation_box = self._make_tab(nb, "Translation")
        self.summary_box = self._make_tab(nb, "Summary & Key Points")

        # ── Footer ──────────────────────────────────────────────────────────
        foot = ttk.Frame(self.root, padding=(10, 2, 10, 8))
        foot.pack(fill=tk.X)
        ttk.Label(
            foot,
            text="Reports are auto-saved to the 'outputs' folder.",
            foreground="#888",
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT)
        ttk.Button(foot, text="Open outputs folder", command=self._open_outputs).pack(
            side=tk.RIGHT
        )

    @staticmethod
    def _make_tab(nb: ttk.Notebook, title: str) -> scrolledtext.ScrolledText:
        frame = ttk.Frame(nb)
        nb.add(frame, text=title)
        box = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Segoe UI", 9))
        box.pack(fill=tk.BOTH, expand=True)
        return box

    # -----------------------------------------------------------------------
    # Recording
    # -----------------------------------------------------------------------

    def _toggle_record(self):
        if self._recorder and self._recorder.is_recording():
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        sel = self.monitor_combo.current()
        mon_idx = self._monitor_indices[sel] if sel >= 0 else 1
        try:
            self._recorder = MeetingRecorder(
                config.OUTPUT_DIR,
                record_video=self.var_video.get(),
                monitor_index=mon_idx,
            )
            session_name = self._recorder.start()
        except Exception as e:
            messagebox.showerror("Recording error", str(e))
            return

        self.rec_btn.config(text="⏹  Stop Recording")
        self._rec_start = datetime.now()
        self._timer_active = True
        self._tick_timer()
        self.status_var.set(f"Recording — session: {session_name}")
        self.progress["value"] = 0
        for box in (self.transcript_box, self.translation_box, self.summary_box):
            box.delete(1.0, tk.END)

    def _stop_recording(self):
        self._timer_active = False
        self.timer_var.set("")
        self.rec_btn.config(text="Processing…", state="disabled")
        self.status_var.set("Stopping recording…")

        session_info = self._recorder.stop()
        self._recorder = None

        audio_path = session_info.get("audio_path")
        if audio_path and Path(audio_path).exists():
            self._run_pipeline_async(Path(audio_path), session_info)
        else:
            self.status_var.set("Error: no audio captured.")
            self.rec_btn.config(text="⏺  Start Recording", state="normal")

    # -----------------------------------------------------------------------
    # File processing
    # -----------------------------------------------------------------------

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open audio / video file",
            filetypes=[
                (
                    "Audio/Video",
                    "*.wav *.mp3 *.mp4 *.m4a *.flac *.ogg *.mkv *.avi *.webm",
                ),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        audio_path = Path(path)
        session_info = {
            "session_name": audio_path.stem,
            "audio_path": audio_path,
            "video_path": None,
            "duration": 0,
            "start_time": datetime.now(),
        }
        self._run_pipeline_async(audio_path, session_info)

    # -----------------------------------------------------------------------
    # Pipeline execution
    # -----------------------------------------------------------------------

    def _run_pipeline_async(self, audio_path: Path, session_info: dict):
        self.rec_btn.config(state="disabled")
        self.progress["value"] = 0
        self.status_var.set("Starting pipeline…")
        threading.Thread(
            target=self._pipeline_thread,
            args=(audio_path, session_info),
            daemon=True,
        ).start()

    def _pipeline_thread(self, audio_path: Path, session_info: dict):
        def cb(msg: str, pct: int):
            self.root.after(0, lambda m=msg, p=pct: self._update_status(m, p))

        try:
            spk_val = self.speakers_var.get()
            num_speakers = int(spk_val) if spk_val != "Auto" else None
            results = run_pipeline(
                audio_path=audio_path,
                session_info=session_info,
                model=self.model_var.get(),
                use_diarization=self.var_diarize.get(),
                num_speakers=num_speakers,
                use_translation=self.var_translate.get(),
                progress_cb=cb,
            )
            self.root.after(0, lambda r=results: self._show_results(r))
        except Exception as e:
            self.root.after(0, lambda err=str(e): self._pipeline_error(err))

    def _update_status(self, msg: str, pct: int):
        self.status_var.set(msg)
        self.progress["value"] = pct

    def _show_results(self, results: dict):
        segs = results.get("segments", [])
        tinfo = results.get("translation_info")
        summ = results.get("summary", {})
        files = results.get("output_files", {})

        # Transcript tab
        self.transcript_box.delete(1.0, tk.END)
        for seg in segs:
            ts = self._fmt(seg.get("start", 0))
            self.transcript_box.insert(
                tk.END, f"[{ts}] {seg.get('speaker', 'Speaker')}: {seg.get('text', '')}\n"
            )

        # Translation tab
        self.translation_box.delete(1.0, tk.END)
        if tinfo:
            self.translation_box.insert(
                tk.END,
                f"Translation: {tinfo.get('source')} → {tinfo.get('target')}\n{'─'*50}\n\n",
            )
            for seg in segs:
                txt = seg.get("translated_text", "")
                if txt:
                    ts = self._fmt(seg.get("start", 0))
                    self.translation_box.insert(
                        tk.END,
                        f"[{ts}] {seg.get('speaker', 'Speaker')}: {txt}\n",
                    )
        else:
            self.translation_box.insert(tk.END, "No translation performed.\n")

        # Summary tab
        self.summary_box.delete(1.0, tk.END)
        self.summary_box.insert(tk.END, "KEY POINTS\n" + "─" * 50 + "\n")
        self.summary_box.insert(tk.END, summ.get("key_points", "") + "\n\n")
        self.summary_box.insert(tk.END, "SUMMARY\n" + "─" * 50 + "\n")
        self.summary_box.insert(tk.END, summ.get("summary", "") + "\n\n")
        ai = summ.get("action_items", "").strip()
        if ai:
            self.summary_box.insert(tk.END, "ACTION ITEMS / DECISIONS\n" + "─" * 50 + "\n")
            self.summary_box.insert(tk.END, ai + "\n")

        txt_path = files.get("txt", "")
        self.status_var.set(f"Done — report: {txt_path}")
        self.progress["value"] = 100
        self.rec_btn.config(text="⏺  Start Recording", state="normal")

    def _pipeline_error(self, error: str):
        messagebox.showerror("Pipeline error", f"Processing failed:\n\n{error}")
        self.status_var.set(f"Error: {error[:80]}")
        self.rec_btn.config(text="⏺  Start Recording", state="normal")

    # -----------------------------------------------------------------------
    # Timer
    # -----------------------------------------------------------------------

    def _tick_timer(self):
        if not self._timer_active:
            return
        elapsed = (datetime.now() - self._rec_start).total_seconds()
        m, s = divmod(int(elapsed), 60)
        self.timer_var.set(f"REC  {m:02d}:{s:02d}")
        self.root.after(1000, self._tick_timer)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _open_outputs(self):
        subprocess.Popen(f'explorer "{config.OUTPUT_DIR}"')

    @staticmethod
    def _fmt(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    def run(self):
        self.root.mainloop()
