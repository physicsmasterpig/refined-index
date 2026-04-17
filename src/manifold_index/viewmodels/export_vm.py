"""
viewmodels.export_vm
====================
ExportViewModel — display-ready data for the Export (Card ④) panel.

No Qt dependency.  Built from ``ExportService.available_data(session)``
and the user's format selections.

BLUEPRINT references
--------------------
§10.5  ExportCard (UI layout)
§8     ViewModel Layer (general principles)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manifold_index.viewmodels.advisory import Advisory


# ---------------------------------------------------------------------------
# ExportViewModel
# ---------------------------------------------------------------------------

@dataclass
class ExportAvailability:
    """Summary of what data is available in the current session.

    Each boolean flag controls whether the corresponding checkbox in the
    Export card is enabled and initially checked.
    """
    has_manifold: bool        # Card ① done
    n_index_queries: int      # number of computed I^ref entries
    has_weyl: bool            # Weyl check completed with valid a,b
    n_nc_cycles: int          # total NC cycles across all cusps
    n_fill_queries: int       # number of completed filling results

    @property
    def has_index(self) -> bool:
        return self.n_index_queries > 0

    @property
    def has_filling(self) -> bool:
        return self.n_fill_queries > 0

    @property
    def has_any(self) -> bool:
        return self.has_manifold or self.has_index or self.has_weyl or self.has_filling


@dataclass
class ExportFormatSelection:
    """User-selected output formats.

    Each flag maps to one ``ExportService`` call.
    """
    latex: bool          = True
    mathematica: bool    = True
    full_report: bool    = False
    json: bool           = False


@dataclass
class ExportViewModel:
    """Display-ready data for the Export card (Card ④).

    Attributes
    ----------
    availability : ExportAvailability
        What data is present in the session (drives checkbox state).
    format_selection : ExportFormatSelection
        Currently selected output formats.
    output_path : str
        User-chosen directory for file output (default: empty string).
    last_export_path : str | None
        Path of the most recently exported file (shown in the UI).
    clipboard_preview_latex : str
        Short preview of what will be in clipboard when "Copy LaTeX" is clicked.
    clipboard_preview_plain : str
        Short preview for "Copy Plain Text".
    advisories : list[Advisory]
        Card-level advisories (currently none defined in BLUEPRINT §12).
    is_unlocked : bool
        True when the card is unlocked (``session.has_any_results()``).
    """
    availability: ExportAvailability
    format_selection: ExportFormatSelection
    output_path: str = ""
    last_export_path: "str | None" = None
    clipboard_preview_latex: str = ""
    clipboard_preview_plain: str = ""
    advisories: list[Advisory] = field(default_factory=list)
    is_unlocked: bool = False


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_export_vm(
    available_data: dict,
    *,
    format_selection: "ExportFormatSelection | None" = None,
    output_path: str = "",
    last_export_path: "str | None" = None,
    clipboard_preview_latex: str = "",
    clipboard_preview_plain: str = "",
    weyl_valid: bool = False,
) -> ExportViewModel:
    """Construct an ``ExportViewModel`` from ``ExportService.available_data()`` output.

    Parameters
    ----------
    available_data : dict
        ``{"manifold": bool, "index_queries": int, "weyl": bool,
            "nc_cycles": int, "fill_queries": int}``
        — the dict returned by ``ExportService.available_data(session)``.
    format_selection : ExportFormatSelection | None
        Current format choices.  Defaults to ``ExportFormatSelection()`` if None.
    """
    avail = ExportAvailability(
        has_manifold=bool(available_data.get("manifold", False)),
        n_index_queries=int(available_data.get("index_queries", 0)),
        has_weyl=bool(available_data.get("weyl", False)) or weyl_valid,
        n_nc_cycles=int(available_data.get("nc_cycles", 0)),
        n_fill_queries=int(available_data.get("fill_queries", 0)),
    )

    if format_selection is None:
        format_selection = ExportFormatSelection()

    return ExportViewModel(
        availability=avail,
        format_selection=format_selection,
        output_path=output_path,
        last_export_path=last_export_path,
        clipboard_preview_latex=clipboard_preview_latex,
        clipboard_preview_plain=clipboard_preview_plain,
        advisories=[],
        is_unlocked=avail.has_any,
    )
