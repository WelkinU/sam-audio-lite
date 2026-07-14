@echo off
setlocal EnableDelayedExpansion

:: Change to the project root (one level up from this script's directory)
cd /d "%~dp0\.."

echo.
echo ============================================================
echo   sam-audio-lite -- Windows Setup
echo ============================================================
echo.

:: ── Step 1: Git ──────────────────────────────────────────────
echo [1/4] Checking for Git...
where git >nul 2>&1
if not errorlevel 1 (
    echo       Git already installed.
    goto :git_ok
)

:: Check the standard install location in case it's installed but not on PATH
if exist "C:\Program Files\Git\cmd\git.exe" (
    set "PATH=!PATH!;C:\Program Files\Git\cmd"
    echo       Git found at default location, added to PATH.
    goto :git_ok
)

echo       Git not found. Installing via winget...
where winget >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: winget is not available on this system.
    echo  Please install Git manually and re-run this script:
    echo    https://git-scm.com/download/win
    pause
    exit /b 1
)

winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo.
    echo  ERROR: Git installation failed. Install manually and re-run:
    echo    https://git-scm.com/download/win
    pause
    exit /b 1
)
set "PATH=!PATH!;C:\Program Files\Git\cmd"
echo       Git installed.

:git_ok
where git >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: git is still not accessible after installation.
    echo  Please restart this script from a new terminal, or add
    echo  Git to your PATH manually and re-run.
    pause
    exit /b 1
)

:: ── Step 2: uv ───────────────────────────────────────────────
echo [2/4] Checking for uv...
where uv >nul 2>&1
if not errorlevel 1 (
    echo       uv already installed.
    goto :uv_ok
)

:: uv may be installed but not on the current session's PATH
if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "PATH=!PATH!;%USERPROFILE%\.local\bin"
    echo       uv found at default location, added to PATH.
    goto :uv_ok
)

echo       uv not found. Installing...
:: Official uv install method (https://docs.astral.sh/uv/getting-started/installation/)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
if errorlevel 1 (
    echo.
    echo  ERROR: uv installation failed.
    echo  Install manually: https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)
set "PATH=!PATH!;%USERPROFILE%\.local\bin"
echo       uv installed.

:uv_ok
where uv >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: uv is still not accessible after installation.
    echo  Please restart this script from a new terminal.
    pause
    exit /b 1
)

:: ── Step 3: FFmpeg (Windows / TorchCodec) ───────────────────
echo [3/5] Checking for FFmpeg shared libraries...
powershell -ExecutionPolicy Bypass -File "%~dp0setup_ffmpeg.ps1"
if errorlevel 1 (
    echo.
    echo  WARNING: FFmpeg shared setup failed. Audio-only mode may still work;
    echo  visual / TV models need FFmpeg full-shared. Re-run:
    echo    powershell -ExecutionPolicy Bypass -File scripts\setup_ffmpeg.ps1
)

:: ── Step 4: uv sync ──────────────────────────────────────────
echo [4/5] Installing Python dependencies...
echo       (This may take a few minutes on first run)
echo.
uv sync
if errorlevel 1 (
    echo.
    echo  ERROR: uv sync failed. Check the output above for details.
    pause
    exit /b 1
)

:: ── Step 5: HuggingFace login ────────────────────────────────
echo.
echo [5/5] Setting up HuggingFace access...
echo.
uv run python scripts/setup_hf_token.py
if errorlevel 1 (
    echo.
    echo  Note: HuggingFace setup did not complete successfully.
    echo  You can re-run it at any time with:
    echo    uv run python scripts/setup_hf_token.py
)

echo.
echo ============================================================
echo   Setup complete!
echo.
echo   Start the app:
echo     uv run run_gradio.py
echo ============================================================
echo.
pause
