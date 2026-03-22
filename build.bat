@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  ArchRender AI -- EXE Builder
echo ============================================================
echo.

:: ── 1. Build the React frontend ─────────────────────────────────────────────
echo [1/3] Building React frontend...
cd /d "%~dp0frontend"

where npm >nul 2>&1
if errorlevel 1 (
    echo ERROR: npm not found. Install Node.js from https://nodejs.org
    pause & exit /b 1
)

call npm install
if errorlevel 1 ( echo npm install failed & pause & exit /b 1 )

call npm run build
if errorlevel 1 ( echo npm build failed & pause & exit /b 1 )

echo  -> frontend/dist created
echo.

:: ── 2. Install Python dependencies ──────────────────────────────────────────
echo [2/3] Installing Python dependencies...
cd /d "%~dp0backend"

python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt pyinstaller
if errorlevel 1 ( echo pip install failed & pause & exit /b 1 )

echo.

:: ── 3. Run PyInstaller ───────────────────────────────────────────────────────
echo [3/3] Building EXE with PyInstaller (this takes a few minutes)...
python -m PyInstaller archrender.spec --clean --noconfirm
if errorlevel 1 ( echo PyInstaller failed & pause & exit /b 1 )

echo.
echo ============================================================
echo  Done!  Your exe is at:
echo    %~dp0backend\dist\ArchRenderAI.exe
echo ============================================================
echo.
echo NOTE: Put your Replicate API token in a .env file next to
echo       the exe, or enter it in the app's token field.
echo.
pause
