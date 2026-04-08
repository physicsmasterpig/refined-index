"""app/theme/icons.py — Unicode glyph registry.

BLUEPRINT §2.1: "Icons only where a glyph conveys information faster than a
word — and nowhere else."  All glyphs are plain Unicode characters embedded
in button labels or QLabel text.  No QIcon / resource files are used.
"""

# ---------------------------------------------------------------------------
# Card expand / collapse  (BLUEPRINT §2.6)
# ---------------------------------------------------------------------------
EXPAND   = "▾"   # downward-pointing triangle — expand card
COLLAPSE = "▴"   # upward-pointing triangle   — collapse card

# ---------------------------------------------------------------------------
# Status badge glyphs  (BLUEPRINT §2.4)
# ---------------------------------------------------------------------------
RUNNING  = "●"   # filled circle — running
DONE     = "✓"   # check mark   — done
WARNING  = "⚠"   # warning sign — warning
ERROR    = "✕"   # cross mark   — error
STALE    = "∿"   # tilde-like   — stale

# ---------------------------------------------------------------------------
# Data Hub: pause / resume  (BLUEPRINT §2.3 exception)
# ---------------------------------------------------------------------------
PLAY     = "▶"   # inline in "▶  Resume" button label
PAUSE    = "▐▐"  # inline in "▐▐  Pause" button label

# ---------------------------------------------------------------------------
# Advisory level tags  (displayed in small-caps label)
# ---------------------------------------------------------------------------
ADVISORY_INFO    = "ℹ"
ADVISORY_WARNING = "⚠"
ADVISORY_ERROR   = "✕"
ADVISORY_ACTION  = "◈"
