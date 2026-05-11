import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

FRAME_RATE = int(os.getenv("FRAME_RATE", "15"))
AUDIO_SAMPLE_RATE = 16000

RECORD_VIDEO = os.getenv("RECORD_VIDEO", "true").lower() == "true"

ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "")
ANTHROPIC_MODEL = (
    os.getenv("ANTHROPIC_DEFAULT_SONNET_MODEL")
    or os.getenv("ANTHROPIC_MODEL")
    or "claude-sonnet-4-6"
)
