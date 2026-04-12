# Windows 빌드 가이드 — Refined Index Calculator v0.5

## 왜 처음 빌드가 실패했는가

### 근본 원인: spec 파일이 macOS 전용으로만 작성되어 있었다

원본 `ManifoldIndex.spec`은 macOS 배포를 목표로 작성된 파일이다.  
Windows에서 그대로 실행하면 아래와 같은 문제들이 연쇄적으로 발생한다.

---

### 문제 1 — spec 구조가 macOS 전용

| 항목 | macOS spec 코드 | Windows에서 발생하는 문제 |
|---|---|---|
| `BUNDLE(...)` | macOS `.app` 번들 생성 블록 | Windows에 `BUNDLE` 개념 없음 → 빌드 오류 |
| `argv_emulation=True` | macOS에서 Finder 드래그 파일 처리용 | Windows에 해당 기능 없음 → 런타임 오류 |
| `icon="assets/ManifoldIndex.icns"` | macOS 전용 아이콘 포맷 | Windows는 `.ico`만 인식 |
| `multiprocessing.popen_spawn_posix` | Unix 프로세스 모델 | Windows에 존재하지 않는 모듈 → ImportError |
| `multiprocessing.popen_fork` | Unix fork 방식 | Windows에 존재하지 않는 모듈 → ImportError |

---

### 문제 2 — conda 환경 구조를 PyInstaller가 이해하지 못한다 (핵심)

이것이 `_sqlite3`, `_ssl` 등 연쇄 DLL 오류의 진짜 원인이다.

**일반 venv와 conda의 DLL 저장 위치 차이:**

```
일반 pip venv (PyInstaller가 잘 찾는 구조)
└── venv/
    ├── Lib/site-packages/   ← Python 패키지
    └── Scripts/             ← 실행 파일
    (DLL이 PATH에 있거나 site-packages 안에 있음)

conda 환경 (PyInstaller가 DLL을 못 찾는 구조)
└── anaconda3/envs/refined-index/
    ├── Lib/site-packages/   ← Python 패키지
    ├── DLLs/                ← .pyd 확장 모듈 (_sqlite3.pyd, _ssl.pyd, ...)
    └── Library/bin/         ← 시스템 DLL (sqlite3.dll, libssl-3-x64.dll, ...)
                                ↑ PyInstaller의 기본 탐색 경로에 포함되지 않음
```

PyInstaller는 `Analysis()` 단계에서 `.pyd` 파일들의 DLL 의존성을 추적할 때,
conda의 `Library\bin\` 경로를 알지 못한다.  
그 결과 빌드는 성공해도 실행 시 DLL을 찾지 못해 터진다.

**연쇄 오류 흐름:**

```
실행 → launcher.py
      → manifold_index/app/__main__.py
        → manifold_index/app/window.py
          → ...
            → manifold_index/core/data_packs.py
              → ssl.py (표준 라이브러리)
                → _ssl.pyd 로드 시도
                  → libssl-3-x64.dll 없음 → ImportError
```

ssl 이전에 sqlite3도 같은 이유로 실패한다.  
두 오류 모두 증상은 다르지만 원인은 동일하다: **`Library\bin`이 탐색 경로에 없음**.

**`Library\bin`에 있는 주요 DLL 목록:**

```
sqlite3.dll        ← _sqlite3.pyd 의존
libssl-3-x64.dll   ← _ssl.pyd 의존
libcrypto-3-x64.dll← _ssl.pyd 의존
liblzma.dll        ← _lzma.pyd 의존
libbz2.dll         ← _bz2.pyd 의존
ffi.dll            ← _ctypes.pyd 의존
libexpat.dll       ← pyexpat.pyd 의존
vcruntime140.dll   ← 거의 모든 C 확장 의존
msvcp140.dll       ← 거의 모든 C 확장 의존
...
```

---

### 잘못된 접근법: 오류가 날 때마다 DLL을 하나씩 추가하기

```python
# 이렇게 하면 안 된다 — 오류가 날 때마다 줄이 늘어난다
binaries += [("C:/Users/.../Library/bin/sqlite3.dll", ".")]
# 다음 실행 → ssl 오류
binaries += [("C:/Users/.../Library/bin/libssl-3-x64.dll", ".")]
# 다음 실행 → lzma 오류
binaries += [("C:/Users/.../Library/bin/liblzma.dll", ".")]
# ...끝이 없다
```

이 방식은 오류의 증상을 하나씩 치료하는 것이지, 원인을 해결하는 것이 아니다.

---

## 올바른 해결 방법

두 단계를 함께 적용해야 한다. 어느 하나만 하면 여전히 간헐적으로 실패한다.

### 단계 1 — Analysis() 전에 Library\bin을 PATH에 추가

PyInstaller의 DLL 의존성 탐색기는 `os.environ["PATH"]`를 기준으로 동작한다.  
`Analysis()` 객체가 생성되기 **전에** 경로를 주입하면, 탐색기가 모든 의존성을 자동으로 해소한다.

```python
# spec 파일 최상단 — Analysis() 호출보다 반드시 먼저
import os
from pathlib import Path

CONDA_ENV = Path(r"C:\Users\Minho\anaconda3\envs\refined-index")
CONDA_LIB_BIN = CONDA_ENV / "Library" / "bin"

os.environ["PATH"] = str(CONDA_LIB_BIN) + os.pathsep + os.environ.get("PATH", "")
```

이 한 줄이 핵심이다. 이후 `Analysis()`가 `.pyd` 파일들의 의존 DLL을 올바르게 추적한다.

### 단계 2 — Library\bin 전체를 binaries에 포함

단계 1만으로도 분석은 되지만, 탐색기가 간접 의존성을 놓치는 경우가 있다.  
glob으로 전체를 포함시켜 원천 차단한다.

```python
# 포함 제외 규칙
_EXCLUDE_PREFIXES = ("api-ms-win-",)     # Windows OS 내장 API 포워딩 스텁 — 번들 금지
_EXCLUDE_NAMES    = {"tcl86t.dll", "tk86t.dll"}  # tkinter 제외 시 불필요

binaries = []
if CONDA_LIB_BIN.exists():
    for _dll in sorted(CONDA_LIB_BIN.glob("*.dll")):
        name = _dll.name.lower()
        if name in _EXCLUDE_NAMES:
            continue
        if any(name.startswith(p) for p in _EXCLUDE_PREFIXES):
            continue
        binaries.append((str(_dll), "."))
```

**`api-ms-win-*`을 제외하는 이유:**  
이 파일들은 Windows API Set 포워딩 스텁으로, OS 자체의 일부다.  
번들에 포함하면 오히려 시스템 버전과 충돌해서 오류가 날 수 있다.

---

## macOS spec과 Windows spec의 차이점 요약

| 항목 | macOS (`ManifoldIndex.spec`) | Windows (`ManifoldIndex_win.spec`) |
|---|---|---|
| 번들 구조 | `EXE → COLLECT → BUNDLE` (.app) | `EXE → COLLECT` (폴더) |
| 아이콘 | `ManifoldIndex.icns` | `ManifoldIndex.ico` |
| `argv_emulation` | `True` | `False` |
| multiprocessing | `popen_spawn_posix`, `popen_fork` | `popen_spawn_win32` |
| DLL 경로 처리 | 불필요 (macOS는 dyld가 처리) | `Library\bin` PATH 주입 필수 |
| DLL 포함 | 해당 없음 | `Library\bin` glob 포함 |
| 배포 형태 | `.app` → `ditto` zip | 폴더 → `zipfile` zip |

---

## 빌드 환경 설정 (최초 1회)

conda 환경이 없다면 아래 순서로 생성한다.

```bash
# 1. 전용 conda 환경 생성
conda create -n refined-index python=3.11 -y

# 2. 의존성 설치
conda run -n refined-index pip install snappy numpy scipy PySide6 pyinstaller Pillow

# 3. 아이콘 생성 (Windows .ico 파일이 없을 때)
conda run -n refined-index python -c "
from PIL import Image
img = Image.open('assets/ManifoldIndex_1024.png')
img.save('assets/ManifoldIndex.ico', format='ICO',
         sizes=[(256,256),(128,128),(64,64),(48,48),(32,32),(16,16)])
"
```

---

## 빌드 실행

```bash
cd v0.5/

# 클린 빌드 (권장)
C:\Users\Minho\anaconda3\envs\refined-index\Scripts\pyinstaller.exe \
    ManifoldIndex_win.spec --noconfirm --clean

# 증분 빌드 (의존성 변경 없을 때)
C:\Users\Minho\anaconda3\envs\refined-index\Scripts\pyinstaller.exe \
    ManifoldIndex_win.spec --noconfirm
```

### 성공 기준

빌드 로그에서 아래 패턴이 없어야 한다:

```
# 이것들이 있으면 DLL 누락 → 런타임 오류
WARNING: Library not found: could not resolve 'sqlite3.dll'
WARNING: Library not found: could not resolve 'libssl*.dll'
```

---

## 배포 패키징

```python
# dist/ 디렉터리에서 실행
python -c "
import zipfile, pathlib
src = pathlib.Path('ManifoldIndex')
with zipfile.ZipFile('ManifoldIndex-v0.5.3-windows.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in src.rglob('*'):
        if f.is_file():
            zf.write(f, f)
"
```

> `Compress-Archive` (PowerShell)은 빌드 결과물의 `base_library.zip`에 대한 파일 잠금 충돌이 있어 실패한다. Python의 `zipfile` 모듈을 사용한다.

---

## 새 버전 빌드 시 체크리스트

1. `ManifoldIndex_win.spec`의 `APP_VERSION` 업데이트
2. `CONDA_ENV` 경로가 현재 머신에 맞는지 확인 (또는 `CONDA_ENV_ROOT` 환경변수 설정)
3. `--clean` 옵션으로 클린 빌드 실행
4. 빌드 로그에서 `WARNING: Library not found` 없음 확인
5. `dist/ManifoldIndex/ManifoldIndex.exe` 직접 실행하여 정상 동작 확인
6. zip 패키징 후 GitHub Release에 업로드
