@echo off
REM ---------------------------------------------------------------------------
REM premiere-agent bootstrap (Windows).
REM
REM Idempotent. Honors TORCH_INDEX env var so CPU-only / ROCm users can swap:
REM   set TORCH_INDEX=https://download.pytorch.org/whl/cpu
REM   install.bat
REM
REM Installs into a skill-local virtualenv (.venv) by default. Opt out and
REM install into the python on PATH instead with either of:
REM   install.bat --no-venv
REM   set PREMIERE_AGENT_NO_VENV=1
REM
REM Default is CUDA 12.1 wheels which match the cuDNN bundled with the
REM ONNX Runtime CUDA EP shipped in the onnxruntime-gpu>=1.22 wheel matrix.
REM ---------------------------------------------------------------------------
setlocal EnableExtensions EnableDelayedExpansion

REM Force UTF-8 so the CUDA device name (and Florence's tokenizer warnings)
REM render cleanly in the Windows console.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
chcp 65001 >nul

REM Resolve the script's own directory so the script works from any cwd.
pushd "%~dp0"

REM ---------------------------------------------------------------------------
REM Pick a python interpreter. Prefer `py -3` (the official launcher) which
REM resolves the highest installed 3.x; fall back to plain `python`.
REM ---------------------------------------------------------------------------
set "PYTHON="
where py >nul 2>nul && set "PYTHON=py -3"
if "%PYTHON%"=="" (
    where python >nul 2>nul && set "PYTHON=python"
)
if "%PYTHON%"=="" (
    echo ERROR: no python interpreter on PATH ^(install Python 3.10+ from python.org^).
    popd
    exit /b 1
)

echo [premiere-agent] python: %PYTHON%

REM ---------------------------------------------------------------------------
REM 0. Virtual environment (default on).
REM
REM    Everything installs into .venv so a stray global package (e.g. a CPU
REM    `onnxruntime` shadowing `onnxruntime-gpu`) can't silently demote a
REM    lane to CPU. helpers\_venv.py re-launches every helper under this
REM    interpreter, so the documented `python helpers\<script>.py` keeps
REM    working from any shell. Reused as-is when it already exists.
REM
REM    --no-venv / PREMIERE_AGENT_NO_VENV=1 skips this and installs into
REM    the interpreter picked above (the pre-venv behaviour).
REM ---------------------------------------------------------------------------
set "USE_VENV=1"
if /I "%~1"=="--no-venv" set "USE_VENV=0"
if "%PREMIERE_AGENT_NO_VENV%"=="1" set "USE_VENV=0"

if "%USE_VENV%"=="1" (
    if not exist ".venv\Scripts\python.exe" (
        echo [premiere-agent] creating virtualenv: %CD%\.venv
        %PYTHON% -m venv .venv || goto :err
    )
    REM Relative on purpose: we're pushd'd into the skill root, and %PYTHON%
    REM is expanded unquoted below, so an absolute path with spaces would split.
    set "PYTHON=.venv\Scripts\python.exe"
    echo [premiere-agent] using virtualenv: %CD%\.venv
) else (
    echo [premiere-agent] --no-venv: installing into %PYTHON%
)

REM ---------------------------------------------------------------------------
REM 1. Pip itself first.
REM ---------------------------------------------------------------------------
echo [premiere-agent] upgrading pip
%PYTHON% -m pip install --upgrade pip || goto :err

REM ---------------------------------------------------------------------------
REM 2. PyTorch from the right index.
REM ---------------------------------------------------------------------------
if "%TORCH_INDEX%"=="" set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
echo [premiere-agent] installing torch from %TORCH_INDEX%
%PYTHON% -m pip install torch torchvision torchaudio --index-url %TORCH_INDEX% || goto :err

REM ---------------------------------------------------------------------------
REM 3. The package itself. Base deps now include the full preprocess stack
REM    (Parakeet ONNX + transformers + spaCy + soxr + soundfile) AND the
REM    OpenTimelineIO FCPXML / xmeml adapters, since every real run needs
REM    both. Diarize / animations / flash / parakeet-NeMo stay opt-in
REM    (heavy, niche, or pre-gated).
REM ---------------------------------------------------------------------------
echo [premiere-agent] installing package
%PYTHON% -m pip install -e "." || goto :err

REM ---------------------------------------------------------------------------
REM 4. ffmpeg PATH check. Not fatal but warn loudly — most failures
REM    downstream trace back to a missing ffmpeg.
REM ---------------------------------------------------------------------------
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo.
    echo WARN: ffmpeg not found on PATH.
    echo       winget install Gyan.FFmpeg
    echo       or  choco install ffmpeg
    echo.
)

REM ---------------------------------------------------------------------------
REM 5. CUDA smoke test. Doesn't fail install if CUDA is missing — CPU
REM    fallback path exists. Just informs the user what they'll get.
REM ---------------------------------------------------------------------------
echo [premiere-agent] CUDA smoke test:
%PYTHON% -c "import torch; ok=torch.cuda.is_available(); name=torch.cuda.get_device_name(0) if ok else 'n/a'; tot=(torch.cuda.get_device_properties(0).total_memory/(1024**3)) if ok else 0.0; print(f'  cuda available : {ok}'); print(f'  device         : {name}'); print(f'  total VRAM     : {tot:.1f} GB')"

echo.
echo [premiere-agent] install complete.
echo   next: copy .env.example .env       ^(only needed for --diarize^)
echo         %PYTHON% helpers\preprocess_batch.py C:\path\to\your\videos
echo.

popd
endlocal
exit /b 0

:err
echo.
echo [premiere-agent] install FAILED.
popd
endlocal
exit /b 1
