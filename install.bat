@echo off
REM ---------------------------------------------------------------------------
REM video-use-premiere bootstrap (Windows).
REM
REM Idempotent. As of the Florence-2 ONNX port, the default [preprocess]
REM install is torch-FREE — both the speech lane (Parakeet ONNX) and the
REM visual lane (Florence-2 ONNX, helpers/florence_onnx.py) run on
REM ONNX Runtime via the bundled CUDA / TensorRT EPs.
REM
REM TORCH_INDEX is now LEGACY — only consulted if the user opts into the
REM [diarize] extra (which pulls pyannote.audio, the only remaining
REM torch-dependent component). For default installs it is ignored.
REM
REM Default ONNX Runtime CUDA EP target is CUDA 12.x via the cuDNN
REM bundled with onnxruntime-gpu>=1.22 — no separate torch CUDA install
REM is required for the speech / visual lanes.
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

echo [video-use-premiere] python: %PYTHON%

REM ---------------------------------------------------------------------------
REM 1. Pip itself first.
REM ---------------------------------------------------------------------------
echo [video-use-premiere] upgrading pip
%PYTHON% -m pip install --upgrade pip || goto :err

REM ---------------------------------------------------------------------------
REM 2. The package itself + heavy preprocess + fcpxml extras. Diarize is
REM    intentionally NOT pulled by default (pyannote is heavy AND the
REM    ONLY remaining torch-dependent component, so most users skip it).
REM
REM    The [preprocess] extra now installs:
REM      - onnx-asr[gpu,hub] + onnxruntime-gpu (speech + visual ONNX runtime)
REM      - tensorrt-cu12-libs (opt-in TensorRT EP for both lanes)
REM      - tokenizers + huggingface_hub (Florence-2 BART tokenizer + weights)
REM      - imageio-ffmpeg (frame extraction for visual_lane)
REM      - transformers (CLAP audio lane processor only — no torch needed
REM        for transformers' tokenizer/feature-extractor code paths)
REM      - soundfile + soxr (WAV I/O + fast resample)
REM ---------------------------------------------------------------------------
echo [video-use-premiere] installing package + preprocess + fcpxml extras
%PYTHON% -m pip install -e ".[preprocess,fcpxml]" || goto :err

REM ---------------------------------------------------------------------------
REM 3. Optional torch install for the [diarize] extra. Skipped by default.
REM    User opts in via INSTALL_DIARIZE=1 (then the script pulls torch from
REM    TORCH_INDEX and adds [diarize] to the pip install).
REM ---------------------------------------------------------------------------
if /I "%INSTALL_DIARIZE%"=="1" (
    if "%TORCH_INDEX%"=="" set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
    echo [video-use-premiere] [diarize] opt-in: installing torch from !TORCH_INDEX!
    %PYTHON% -m pip install torch torchvision torchaudio --index-url !TORCH_INDEX! || goto :err
    echo [video-use-premiere] installing [diarize] extra ^(pyannote.audio^)
    %PYTHON% -m pip install -e ".[diarize]" || goto :err
)

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
REM 5. ONNX Runtime providers smoke test. Doesn't fail install if no GPU
REM    EP is available — the CPU EP is a working fallback. Reports which
REM    EPs the freshly-installed onnxruntime-gpu wheel can actually load
REM    on this host. Useful for spotting cuDNN / CUDA version mismatches
REM    BEFORE the user kicks off a 30-min preprocess.
REM ---------------------------------------------------------------------------
echo [video-use-premiere] ONNX Runtime providers:
%PYTHON% -c "import onnxruntime as ort; eps = ort.get_available_providers(); print('  available EPs :', eps); has_cuda = 'CUDAExecutionProvider' in eps; has_trt = 'TensorrtExecutionProvider' in eps; has_dml = 'DmlExecutionProvider' in eps; print(f'  CUDA EP       : {has_cuda}'); print(f'  TensorRT EP   : {has_trt}'); print(f'  DirectML EP   : {has_dml}'); print(f'  ORT version   : {ort.__version__}')"

echo.
echo [video-use-premiere] install complete.
echo   next: copy .env.example .env       ^(only needed for --diarize^)
echo         %PYTHON% helpers\preprocess_batch.py C:\path\to\your\videos
echo.
echo   To opt into speaker diarization (adds torch + pyannote.audio):
echo         set INSTALL_DIARIZE=1 ^&^& install.bat
echo.

popd
endlocal
exit /b 0

:err
echo.
echo [video-use-premiere] install FAILED.
popd
endlocal
exit /b 1
