"""app/theme/colors.py — Palette colour constants.

All hex values taken directly from BLUEPRINT §2.2.
Import these names throughout the app/theme/style.py stylesheet and
any widget that needs to reference a colour programmatically.
"""

# ---------------------------------------------------------------------------
# Background / surface
# ---------------------------------------------------------------------------
BACKGROUND     = "#F9F9F8"   # warm off-white; eases long reading sessions
SURFACE        = "#FFFFFF"   # card surface
SURFACE_ALT    = "#F3F3F2"   # alternating table rows, collapsed card fill

# ---------------------------------------------------------------------------
# Borders
# ---------------------------------------------------------------------------
BORDER         = "#E2E2E0"
BORDER_STRONG  = "#C8C8C4"

# ---------------------------------------------------------------------------
# Accent (deep indigo)
# ---------------------------------------------------------------------------
ACCENT         = "#3B3B9A"   # card top border, focused inputs, primary btn bg
ACCENT_HOVER   = "#2E2E7A"
ACCENT_MUTED   = "#EBEBF5"   # advisory info background

# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
TEXT_PRIMARY   = "#1A1A1A"
TEXT_SECONDARY = "#5A5A5A"
TEXT_MUTED     = "#9A9A9A"
TEXT_ON_ACCENT = "#FFFFFF"

# ---------------------------------------------------------------------------
# Warning (amber)
# ---------------------------------------------------------------------------
WARNING_BG     = "#FFFBF0"
WARNING_BORDER = "#D4860A"
WARNING_TEXT   = "#7A4E00"

# ---------------------------------------------------------------------------
# Error (red)
# ---------------------------------------------------------------------------
ERROR_BG       = "#FFF5F5"
ERROR_BORDER   = "#C0392B"
ERROR_TEXT     = "#7D1E1E"

# ---------------------------------------------------------------------------
# Success (green)  — used for cache-hit indicators and "Done" badge
# ---------------------------------------------------------------------------
SUCCESS        = "#2E7D52"

# ---------------------------------------------------------------------------
# Advisory level colours  (border / background pairs)
# ---------------------------------------------------------------------------
ADVISORY_INFO_BORDER   = ACCENT          # "#3B3B9A"
ADVISORY_INFO_BG       = ACCENT_MUTED    # "#EBEBF5"

ADVISORY_WARNING_BORDER = WARNING_BORDER  # "#D4860A"
ADVISORY_WARNING_BG     = WARNING_BG      # "#FFFBF0"

ADVISORY_ERROR_BORDER  = ERROR_BORDER    # "#C0392B"
ADVISORY_ERROR_BG      = ERROR_BG        # "#FFF5F5"

ADVISORY_ACTION_BORDER = "#6A3B9A"       # purple
ADVISORY_ACTION_BG     = "#F5F0FF"

# ---------------------------------------------------------------------------
# Status badge colours  (BLUEPRINT §2.4)
# ---------------------------------------------------------------------------
STATUS_RUNNING_COLOR = ACCENT       # ● Running  — accent colour
STATUS_DONE_COLOR    = SUCCESS      # ✓ Done
STATUS_WARNING_COLOR = WARNING_BORDER
STATUS_ERROR_COLOR   = ERROR_BORDER
STATUS_LOCKED_COLOR  = TEXT_MUTED
STATUS_READY_COLOR   = TEXT_MUTED
STATUS_STALE_COLOR   = TEXT_MUTED

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
FONT_MONO = '"JetBrains Mono", "Fira Code", "Menlo", monospace'
FONT_UI   = 'system-ui, -apple-system, "Segoe UI", sans-serif'
# See BLUEPRINT §2.2 for full palette specification.
