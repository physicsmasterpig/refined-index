@echo off
REM build_app.bat — Build Refined Index Calculator v0.5 for Windows
REM
REM Usage:
REM   build_app.bat              # full build
REM   build_app.bat --clean      # clean previous artifacts, then rebuild
REM
REM Output:
REM   dist\ManifoldIndex.exe     — standalone executable (ready to distribute)
REM

setlocal enabledelayedexpansion

set SPEC=ManifoldIndex.spec
set EXE=dist\ManifoldIndex.exe
set APP_VERSION=1.0.6

REM ── Clean ─────────────────────────────────────────────────────────
if "%1"=="--clean" (
    echo 🧹  Cleaning previous build artifacts…
    rmdir /s /q build 2>nul
    rmdir /s /q dist 2>nul
)

REM ── Prefer project venv ───────────────────────────────────────────
set VENV_ROOT=%CD%\..\\.venv
if exist "%VENV_ROOT%\Scripts\activate.bat" (
    call "%VENV_ROOT%\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

REM ── Locate PyInstaller ────────────────────────────────────────────
for /f "delims=" %%i in ('where pyinstaller 2^>nul') do set PYINSTALLER=%%i
if "%PYINSTALLER%"=="" (
    if exist "%VENV_ROOT%\Scripts\pyinstaller.exe" (
        set PYINSTALLER=%VENV_ROOT%\Scripts\pyinstaller.exe
    ) else if exist ".venv\Scripts\pyinstaller.exe" (
        set PYINSTALLER=.venv\Scripts\pyinstaller.exe
    ) else (
        echo ❌  PyInstaller not found. Install with: pip install pyinstaller
        exit /b 1
    )
)

if not exist "%SPEC%" (
    echo ❌  Spec file not found: %SPEC%
    exit /b 1
)

REM ── Refresh package metadata so importlib.metadata.version() reflects
REM the current pyproject.toml (prevents a stale version ending up in the
REM window title after a version bump).
echo 🔄  Refreshing package metadata…
python -m pip install -e . --no-deps --quiet || exit /b 1

REM ── Preflight check ───────────────────────────────────────────────
echo 🔍  Checking entry point…
python -c "from manifold_index.app import launch_gui; print('  launch_gui importable')" || exit /b 1
python -c "from importlib.metadata import version; v = version('refined-index-calculator'); print('  package metadata: v' + v); assert v == '%APP_VERSION%', 'metadata ' + v + ' != APP_VERSION %APP_VERSION%'" || exit /b 1

REM ── Build ────────────────────────────────────────────────────────
echo.
echo 🔨  Building Refined Index Calculator v%APP_VERSION%…
echo     Spec:  %SPEC%
echo     Entry: launcher.py ^→ manifold_index.app:launch_gui
echo.

"%PYINSTALLER%" "%SPEC%" --noconfirm

echo.

REM ── Verify executable was created ────────────────────────────────
if not exist "%EXE%" (
    echo ❌  Build failed — %EXE% not found.
    exit /b 1
)

for /f %%A in ('dir /b "%EXE%" ^| find /c /v ""') do set SIZE=%%A
echo ✅  Built: %EXE%

echo.
echo Ready to distribute - just upload the .exe file:
echo     %EXE%
echo.
