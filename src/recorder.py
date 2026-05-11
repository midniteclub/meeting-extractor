import threading
import time
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import pyaudiowpatch as pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import mss
    import cv2
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False


class AudioRecorder:
    """Captures system audio via WASAPI loopback (what plays through speakers)."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self._recording = False
        self._thread = None
        self._frames = []
        self._sample_rate = 44100
        self._channels = 2

    def start(self):
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError(
                "PyAudioWPatch not installed.\n"
                "Run: pip install PyAudioWPatch"
            )
        if not SOUNDFILE_AVAILABLE:
            raise RuntimeError("soundfile not installed.\nRun: pip install soundfile")
        self._recording = True
        self._frames = []
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Path:
        self._recording = False
        if self._thread:
            self._thread.join(timeout=10)
        self._save()
        return self.output_path

    def is_recording(self) -> bool:
        return self._recording

    def _record_loop(self):
        p = pyaudio.PyAudio()
        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

            if not default_speakers.get("isLoopbackDevice", False):
                for loopback in p.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        default_speakers = loopback
                        break

            self._sample_rate = int(default_speakers["defaultSampleRate"])
            self._channels = default_speakers["maxInputChannels"]

            chunk = 1024
            stream = p.open(
                format=pyaudio.paInt16,
                channels=self._channels,
                rate=self._sample_rate,
                input=True,
                input_device_index=default_speakers["index"],
                frames_per_buffer=chunk,
            )
            while self._recording:
                data = stream.read(chunk, exception_on_overflow=False)
                self._frames.append(data)
            stream.stop_stream()
            stream.close()
        except Exception as e:
            self._record_error = str(e)
        finally:
            p.terminate()

    def _save(self):
        if not self._frames:
            return
        raw = np.frombuffer(b"".join(self._frames), dtype=np.int16)
        if self._channels > 1:
            raw = raw.reshape(-1, self._channels).mean(axis=1).astype(np.int16)
        audio_f32 = raw.astype(np.float32) / 32768.0
        sf.write(str(self.output_path), audio_f32, self._sample_rate)


def list_monitors() -> list[tuple[int, str]]:
    """Return [(mss_index, label), ...] for every available monitor.

    mss_index is 1-based and maps directly to sct.monitors[mss_index].
    Returns [(1, "Monitor 1")] as a safe fallback when mss is unavailable.
    """
    if not VIDEO_AVAILABLE:
        return [(1, "Monitor 1")]
    try:
        with mss.mss() as sct:
            return [
                (i, f"Monitor {i}  ({m['width']}×{m['height']})")
                for i, m in enumerate(sct.monitors[1:], start=1)
            ]
    except Exception:
        return [(1, "Monitor 1")]


class VideoRecorder:
    """Captures a chosen monitor using mss + OpenCV."""

    def __init__(self, output_path: Path, fps: int = 15, monitor_index: int = 1):
        self.output_path = output_path
        self.fps = fps
        self.monitor_index = monitor_index
        self._recording = False
        self._thread = None

    def start(self):
        if not VIDEO_AVAILABLE:
            raise RuntimeError(
                "mss and opencv-python required for video recording.\n"
                "Run: pip install mss opencv-python"
            )
        self._recording = True
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._recording = False
        if self._thread:
            self._thread.join(timeout=15)

    def is_recording(self) -> bool:
        return self._recording

    def _record_loop(self):
        with mss.mss() as sct:
            idx = min(self.monitor_index, len(sct.monitors) - 1)
            mon = sct.monitors[idx]
            w, h = mon["width"], mon["height"]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (w, h))
            frame_dur = 1.0 / self.fps
            while self._recording:
                t0 = time.time()
                img = np.array(sct.grab(mon))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                out.write(frame)
                sleep = frame_dur - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)
            out.release()


class MeetingRecorder:
    """Orchestrates audio + optional video recording."""

    def __init__(self, output_dir: Path, record_video: bool = True, monitor_index: int = 1):
        self.output_dir = output_dir
        self.record_video = record_video
        self.monitor_index = monitor_index
        self._audio_rec: AudioRecorder | None = None
        self._video_rec: VideoRecorder | None = None
        self._start_time: datetime | None = None
        self._session_name: str = ""

    def start(self) -> str:
        self._start_time = datetime.now()
        self._session_name = self._start_time.strftime("%Y%m%d_%H%M%S")

        audio_path = self.output_dir / f"{self._session_name}_audio.wav"
        self._audio_rec = AudioRecorder(audio_path)
        self._audio_rec.start()

        if self.record_video and VIDEO_AVAILABLE:
            video_path = self.output_dir / f"{self._session_name}_video.mp4"
            self._video_rec = VideoRecorder(video_path, monitor_index=self.monitor_index)
            self._video_rec.start()

        return self._session_name

    def stop(self) -> dict:
        audio_path = None
        video_path = None

        if self._audio_rec:
            audio_path = self._audio_rec.stop()

        if self._video_rec:
            self._video_rec.stop()
            video_path = self.output_dir / f"{self._session_name}_video.mp4"

        duration = (
            (datetime.now() - self._start_time).total_seconds()
            if self._start_time
            else 0
        )

        return {
            "session_name": self._session_name,
            "audio_path": audio_path,
            "video_path": video_path,
            "duration": duration,
            "start_time": self._start_time,
        }

    def is_recording(self) -> bool:
        return bool(self._audio_rec and self._audio_rec.is_recording())
