@echo off
REM test_ci_local.bat
REM
REM GitHub Actions 워크플로우와 동일한 순서/환경으로 로컬에서 빌드 테스트.
REM
REM 목적:
REM   Mac → git push tag → GitHub Actions (windows-latest) 실행 전에
REM   이 스크립트로 Windows에서 미리 검증한다.
REM
REM 차이점:
REM   - GitHub Actions: setup-python으로 설치된 표준 pip Python
REM   - 이 스크립트: 시스템 Python 또는 지정 Python 사용
REM     (conda 환경이 아닌 표준 pip 환경으로 테스트하려면 아래 PYTHON 경로 수정)
REM
REM Usage:
REM   test_ci_local.bat [version]
REM   예: test_ci_local.bat v0.5.3
REM

setlocal enabledelayedexpansion

set VERSION=%~1
if "%VERSION%"=="" set VERSION=dev

REM ── Python 선택 ──────────────────────────────────────────────────
REM GitHub Actions는 표준 pip Python 사용.
REM conda 환경과의 차이를 검증하려면 conda가 아닌 Python을 지정.
REM 현재는 시스템 PATH의 python 사용 (conda base일 수 있음).
set PYTHON=python

echo.
echo ============================================================
echo  CI Local Test — ManifoldIndex Windows Build
echo  Version : %VERSION%
echo  Python  : %PYTHON%
echo ============================================================
echo.

cd /d "%~dp0"

REM ── Step 4: Install dependencies ────────────────────────────────
echo [Step 4/8] Installing dependencies...
%PYTHON% -m pip install --upgrade pip --quiet
%PYTHON% -m pip install snappy numpy scipy "PySide6>=6.6" pyinstaller Pillow --quiet
if errorlevel 1 (
    echo FAILED: pip install
    exit /b 1
)
echo   OK

REM ── Step 5: Generate .ico ────────────────────────────────────────
echo [Step 5/8] Generating icon...
%PYTHON% -c "from PIL import Image; img = Image.open('assets/ManifoldIndex_1024.png'); img.save('assets/ManifoldIndex.ico', format='ICO', sizes=[(256,256),(128,128),(64,64),(48,48),(32,32),(16,16)]); print('  OK')"
if errorlevel 1 (
    echo FAILED: icon generation
    exit /b 1
)

REM ── Step 6: Preflight check ──────────────────────────────────────
echo [Step 6/8] Checking entry point...
%PYTHON% -c "import sys; sys.path.insert(0,'src'); from manifold_index.app import launch_gui; print('  OK')"
if errorlevel 1 (
    echo FAILED: launch_gui not importable
    exit /b 1
)

REM ── Step 7: PyInstaller build ────────────────────────────────────
echo [Step 7/8] Building with PyInstaller...
pyinstaller ManifoldIndex_win.spec --noconfirm --clean
if errorlevel 1 (
    echo FAILED: pyinstaller build
    exit /b 1
)

REM ── Step 8: Verify output ────────────────────────────────────────
echo [Step 8/8] Verifying build output...
if not exist "dist\ManifoldIndex.exe" (
    echo FAILED: dist\ManifoldIndex.exe not found
    exit /b 1
)
echo   OK — dist\ManifoldIndex.exe

echo.
echo ============================================================
echo  BUILD SUCCESS
echo  Output: v0.5\dist\ManifoldIndex.exe
echo ============================================================
echo.
