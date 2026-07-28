"""Guard: every source file must be valid on the minimum supported Python.

pyproject declares ``requires-python = ">=3.10"`` and the Windows CI
builds with 3.11, while development happens on a much newer
interpreter.  Syntax accepted only by the newer one therefore compiles
fine locally, passes every test, and then makes PyInstaller drop the
module from the frozen bundle ("invalid module" in its warn file) — the
app dies at startup with ModuleNotFoundError on Windows only.

That shipped once already (a backslash inside an f-string expression,
legal from 3.12, a SyntaxError before it).  These tests fail fast on
the developer machine instead.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_MIN_VERSION = (3, 10)


def _sources() -> list[Path]:
    files = sorted(p for p in (_ROOT / "src").rglob("*.py"))
    files += [_ROOT / "launcher.py", _ROOT / "rthook_snappy.py"]
    return [p for p in files if p.is_file()]


def _illegal_fstrings(line: str) -> list[str]:
    """f-string literals whose *expression* part breaks before 3.12.

    PEP 701 (3.12) first allowed backslashes and reuse of the enclosing
    quote character inside the expression; both are SyntaxErrors on
    3.10/3.11.  ``ast.parse(feature_version=...)`` does not catch these
    because they are tokenizer-level, hence this textual check.
    """
    found: list[str] = []
    for m in re.finditer(r'\b(?:rf|fr|f)(["\'])((?:\\.|(?!\1).)*)\1', line):
        quote, body = m.group(1), m.group(2)
        # Doubled braces are literal text, not expressions.
        stripped = body.replace("{{", "\x00").replace("}}", "\x01")
        for expr in re.findall(r"\{([^{}]*)\}", stripped):
            if "\\" in expr or quote in expr:
                found.append(m.group(0))
                break
    return found


@pytest.mark.parametrize("path", _sources(), ids=lambda p: str(p.name))
def test_source_parses_on_minimum_python(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(path), feature_version=_MIN_VERSION)
    except SyntaxError as exc:  # pragma: no cover - failure path
        pytest.fail(
            f"{path.relative_to(_ROOT)}:{exc.lineno} is not valid on "
            f"Python {_MIN_VERSION[0]}.{_MIN_VERSION[1]}: {exc.msg}"
        )


@pytest.mark.parametrize("path", _sources(), ids=lambda p: str(p.name))
def test_no_pre_312_illegal_fstrings(path: Path) -> None:
    offenders: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for snippet in _illegal_fstrings(line):
            offenders.append(f"line {lineno}: {snippet}")
    assert not offenders, (
        f"{path.relative_to(_ROOT)} uses f-string expressions that are a "
        f"SyntaxError before Python 3.12 (PyInstaller would silently drop "
        f"this module from the Windows bundle):\n  " + "\n  ".join(offenders)
    )
