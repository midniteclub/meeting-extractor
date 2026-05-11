@echo off
setlocal
echo ============================================================
echo  Meeting Extractor — Dependency Installer
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

echo Step 1/5: Installing PyTorch (CPU build)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :error

echo.
echo Step 2/5: Installing Whisper...
pip install openai-whisper
if errorlevel 1 goto :error

echo.
echo Step 3/5: Installing speaker diarization (pyannote)...
pip install pyannote.audio
if errorlevel 1 goto :error

echo.
echo Step 4/5: Installing audio capture...
pip install PyAudioWPatch soundfile
if errorlevel 1 goto :error

echo.
echo Step 5/5: Installing remaining packages...
pip install deep-translator langdetect anthropic python-dotenv colorama tqdm mss opencv-python Pillow
if errorlevel 1 goto :error

echo.
echo ============================================================
echo  Installation complete!
echo ============================================================
echo.
echo Next steps:
echo  1. Copy .env.example to .env
echo     copy .env.example .env
echo.
echo  2. Edit .env and add your API keys:
echo     - ANTHROPIC_API_KEY  (for AI summaries — console.anthropic.com)
echo     - HF_TOKEN           (for speaker ID — huggingface.co/settings/tokens)
echo       Also accept terms at:
echo       https://huggingface.co/pyannote/speaker-diarization-3.1
echo.
echo  3. Run the app:
echo     python main.py
echo.
goto :end

:error
echo.
echo ERROR: Installation failed. Check the output above.

:end
pause
