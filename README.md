# Meeting Extractor

Record any meeting — Zoom, browser, or any app — then automatically transcribe it with speaker labels, translate between English and Mandarin Chinese, and generate an AI-powered summary with a structured key-points outline.

Powered by [OpenAI Whisper](https://github.com/openai/whisper), [pyannote.audio](https://github.com/pyannote/pyannote-audio), and a configurable LLM (Claude or Qwen via DashScope). Includes both a GUI and a full CLI. Runs locally on Windows.

---

## Features

- **Screen + audio recording** — captures what plays through your speakers (WASAPI loopback), works with any app including Zoom, Teams, and browser-based meetings. Optional screen video capture.
- **Transcription** — local, offline transcription via OpenAI Whisper. No audio ever leaves your machine.
- **Speaker identification** — labels each line by who is speaking (Speaker 1, Speaker 2, etc.) via pyannote.audio. Optional — requires a free HuggingFace token.
- **Translation** — auto-detects English or Mandarin Chinese and translates to the other language. Both the original and translated transcripts are saved.
- **AI summary** — generates a structured key-points outline, a full prose summary, and a list of action items / decisions. Uses Claude or Qwen (DashScope) if an API key is configured, otherwise falls back to simple extractive summarization.
- **Reports** — saves a human-readable `.txt` report and a machine-readable `.json` file to the `outputs/` folder after every session.
- **Multi-monitor support** — choose which monitor to record from a dropdown in the GUI.
- **GUI + CLI** — use the desktop app for day-to-day use, or automate with the command line.

---

## Requirements

- **Python 3.10+** — [python.org](https://www.python.org/downloads/)
- **Windows 10/11** — audio capture uses the Windows WASAPI loopback API
- **~2 GB disk space** — for Python packages and the default Whisper model

---

## Installation

**1. Clone the repo**

```bat
git clone https://github.com/midniteclub/meeting_extractor.git
cd meeting_extractor
```

**2. Install dependencies**

Double-click `install.bat`, or run it from a terminal:

```bat
install.bat
```

This installs PyTorch (CPU build), Whisper, pyannote, and all other dependencies.

> If you have an NVIDIA GPU and want faster transcription, replace the PyTorch step in `install.bat` with the CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Configuration

Copy the example env file and fill in your keys:

```bat
copy .env.example .env
```

Then open `.env` in any text editor. All keys are optional — the app works without any of them, but features degrade gracefully:

```env
# LLM API key — used for AI-powered summaries
# For Claude:  https://console.anthropic.com
# For Qwen:    your DashScope API key
ANTHROPIC_API_KEY=your_api_key_here

# Only needed when using a non-Anthropic endpoint (e.g. DashScope)
ANTHROPIC_BASE_URL=https://dashscope.aliyuncs.com/apps/anthropic

# Model name — change this to match your provider
# Claude default:  claude-sonnet-4-6
# Qwen example:    qwen3.5-plus
ANTHROPIC_DEFAULT_SONNET_MODEL=claude-sonnet-4-6

# HuggingFace token — used for speaker identification
# 1. Create a free account at https://huggingface.co
# 2. Accept the model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
# 3. Generate a token at https://huggingface.co/settings/tokens
HF_TOKEN=your_token_here

# Whisper model size: tiny | base | small | medium | large  (default: base)
WHISPER_MODEL=base
```

| Key | Without it |
|---|---|
| `ANTHROPIC_API_KEY` | Falls back to simple extractive summarization |
| `HF_TOKEN` | Speaker identification is skipped; all speech labeled "Speaker" |

---

## Usage

### GUI (recommended)

```bat
python main.py
```

Opens the desktop app. Controls:

- **Start Recording** — begins capturing system audio (and screen if enabled). Click again to stop and automatically process.
- **Open File** — process an existing audio or video file without recording.
- **Options bar** — toggle video recording, speaker identification, and translation; set the number of speakers; choose the Whisper model size; select which monitor to record.
- **Tabs** — results appear in three tabs: *Transcript*, *Translation*, and *Summary & Key Points*.
- **Open outputs folder** — opens the folder where all reports are saved.

---

### CLI

**Launch the GUI** (default when no arguments are given):
```bat
python main.py
```

**Record a meeting and process it automatically:**
```bat
python main.py --record
```
Press `Ctrl+C` to stop recording. Processing starts immediately after.

**Process an existing audio or video file:**
```bat
python main.py --process path\to\meeting.wav
```

Supported formats: `.wav` `.mp3` `.mp4` `.m4a` `.flac` `.ogg` `.mkv` `.avi` `.webm`

---

### CLI flags

| Flag | Description |
|---|---|
| `--record` | Start recording immediately (no GUI) |
| `--process FILE` | Process an existing audio/video file |
| `--model SIZE` | Whisper model: `tiny` `base` `small` `medium` `large` (default: `base`) |
| `--speakers N` | Tell pyannote exactly how many speakers to find (e.g. `--speakers 2`). Greatly improves accuracy when you know the count. |
| `--no-diarization` | Skip speaker identification |
| `--no-translate` | Skip translation |
| `--no-video` | Record audio only, no screen capture |

**Examples:**

```bat
# Process a file with a higher-accuracy model, 2 known speakers, skip translation
python main.py --process meeting.mp4 --model small --speakers 2 --no-translate

# Record audio-only and skip speaker identification
python main.py --record --no-video --no-diarization

# Process two files simultaneously (open two terminals)
python main.py --process file1.wav --speakers 2
python main.py --process file2.wav --speakers 3
```

---

## Whisper model sizes

| Model | Download size | RAM needed | Speed | Accuracy |
|---|---|---|---|---|
| `tiny` | 75 MB | ~1 GB | Fastest | Lower — struggles with accents and Mandarin |
| `base` | 145 MB | ~1 GB | Fast | Good for clear audio *(default)* |
| `small` | 460 MB | ~2 GB | ~2x slower | Noticeably better EN/ZH accuracy |
| `medium` | 1.5 GB | ~5 GB | ~5x slower | Very good — handles accents and background noise |
| `large` | 3 GB | ~10 GB | ~10x slower | Best accuracy — benefits from a GPU |

**Recommendation:** use `small` for everyday recordings, `medium` for noisy or accented audio.

Models are downloaded automatically on first use and cached locally.

---

## Output files

Every processed session saves two files to the `outputs/` folder:

| File | Contents |
|---|---|
| `YYYYMMDD_HHMMSS_report.txt` | Human-readable report: key points outline, summary, action items, full transcript, and translation |
| `YYYYMMDD_HHMMSS_data.json` | Full structured data — all segments with timestamps, speaker labels, translated text, and summary |

**Example transcript output:**
```
[00:00] Speaker 1: Good morning everyone, let's get started.
[00:05] Speaker 2: Thanks for joining. Today we're reviewing...
```

---

## How audio capture works

The app uses **WASAPI loopback** to capture everything playing through your speakers or headphones — this includes Zoom calls, browser meetings, videos, and any other app audio. No virtual audio cable or additional software is required.

Your microphone is **not** captured by default. If you want to include your own voice, you will need to route your microphone through a virtual audio mixer before recording.

---

## Troubleshooting

**Everyone is labeled "Speaker" with no differentiation**
- Check that `HF_TOKEN` is set in your `.env`
- Make sure you accepted the model terms at [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- Try setting `--speakers N` to the exact number of people in the recording

**Translation errors or rate limiting**
- The app uses Google Translate's free tier. Very long recordings processed simultaneously may hit rate limits. Process files one at a time if this happens.

**Whisper is very slow**
- Switch to a smaller model (`tiny` or `base`) for faster results
- For large model performance, an NVIDIA GPU with the CUDA PyTorch build is recommended

**No audio captured in the recording**
- Make sure your meeting audio is playing through your default Windows audio output device
- Check that no other app has exclusive control of the audio device (Settings → Sound → Advanced)

**`PyAudioWPatch` install fails**
- Make sure you have the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) installed

---

## License

MIT — see [LICENSE](LICENSE) for details.
