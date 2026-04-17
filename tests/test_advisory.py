"""
Phase 3 tests — Advisory system.

Covers: AdvisoryLevel, CardStatus, AdvisoryAction, Advisory,
        and every factory in the Advisories namespace.
"""
from __future__ import annotations

import pytest

from manifold_index.viewmodels.advisory import (
    Advisory,
    AdvisoryAction,
    AdvisoryLevel,
    Advisories,
    CardStatus,
)


# ---------------------------------------------------------------------------
# AdvisoryLevel
# ---------------------------------------------------------------------------

class TestAdvisoryLevel:
    def test_all_levels_exist(self):
        for name in ("INFO", "WARNING", "ERROR", "ACTION"):
            assert hasattr(AdvisoryLevel, name)

    def test_values_are_strings(self):
        assert AdvisoryLevel.INFO.value == "info"
        assert AdvisoryLevel.WARNING.value == "warning"
        assert AdvisoryLevel.ERROR.value == "error"
        assert AdvisoryLevel.ACTION.value == "action"


# ---------------------------------------------------------------------------
# CardStatus
# ---------------------------------------------------------------------------

class TestCardStatus:
    def test_all_statuses_exist(self):
        for name in ("LOCKED", "READY", "RUNNING", "DONE", "WARNING", "ERROR", "STALE"):
            assert hasattr(CardStatus, name)

    def test_values_are_strings(self):
        assert CardStatus.LOCKED.value == "locked"
        assert CardStatus.DONE.value == "done"
        assert CardStatus.STALE.value == "stale"


# ---------------------------------------------------------------------------
# AdvisoryAction
# ---------------------------------------------------------------------------

class TestAdvisoryAction:
    def test_label_stored(self):
        act = AdvisoryAction("Do it")
        assert act.label == "Do it"

    def test_callback_default_none(self):
        act = AdvisoryAction("Label")
        assert act.callback is None

    def test_callback_stored(self):
        called = []
        act = AdvisoryAction("Go", callback=lambda: called.append(1))
        act.callback()
        assert called == [1]


# ---------------------------------------------------------------------------
# Advisory dataclass and convenience constructors
# ---------------------------------------------------------------------------

class TestAdvisory:
    def test_info_constructor(self):
        adv = Advisory.info("X1", "Title", "Body")
        assert adv.advisory_id == "X1"
        assert adv.level == AdvisoryLevel.INFO
        assert adv.title == "Title"
        assert adv.body == "Body"
        assert adv.actions == []

    def test_warning_constructor(self):
        adv = Advisory.warning("X2", "Warn", "Details")
        assert adv.level == AdvisoryLevel.WARNING

    def test_error_constructor(self):
        adv = Advisory.error("X3", "Err", "Msg")
        assert adv.level == AdvisoryLevel.ERROR

    def test_action_constructor(self):
        acts = [AdvisoryAction("OK")]
        adv = Advisory.action("X4", "Choose", "Choose something", actions=acts)
        assert adv.level == AdvisoryLevel.ACTION
        assert len(adv.actions) == 1

    def test_is_blocking_error(self):
        adv = Advisory.error("E", "T", "B")
        assert adv.is_blocking is True

    def test_is_blocking_action(self):
        adv = Advisory.action("A", "T", "B")
        assert adv.is_blocking is True

    def test_is_not_blocking_info(self):
        adv = Advisory.info("I", "T", "B")
        assert adv.is_blocking is False

    def test_is_not_blocking_warning(self):
        adv = Advisory.warning("W", "T", "B")
        assert adv.is_blocking is False

    def test_show_when_collapsed_only_for_error(self):
        assert Advisory.error("E", "T", "B").show_when_collapsed is True
        assert Advisory.warning("W", "T", "B").show_when_collapsed is False
        assert Advisory.info("I", "T", "B").show_when_collapsed is False
        assert Advisory.action("A", "T", "B").show_when_collapsed is False


# ---------------------------------------------------------------------------
# Category A factories
# ---------------------------------------------------------------------------

class TestCategoryA:
    def test_A1_is_info(self):
        adv = Advisories.A1()
        assert adv.advisory_id == "A1"
        assert adv.level == AdvisoryLevel.INFO
        assert "easy" in adv.title.lower() or "η" not in adv.title

    def test_A2_is_error_with_name(self):
        adv = Advisories.A2("m999")
        assert adv.advisory_id == "A2"
        assert adv.level == AdvisoryLevel.ERROR
        assert "m999" in adv.body
        assert len(adv.actions) >= 1

    def test_A3_is_warning_with_action(self):
        adv = Advisories.A3()
        assert adv.advisory_id == "A3"
        assert adv.level == AdvisoryLevel.WARNING
        assert any("proceed" in a.label.lower() for a in adv.actions)


# ---------------------------------------------------------------------------
# Category B factories
# ---------------------------------------------------------------------------

class TestCategoryB:
    def test_B1_contains_qq(self):
        adv = Advisories.B1(20)
        assert adv.advisory_id == "B1"
        assert "20" in adv.body
        assert adv.level == AdvisoryLevel.INFO

    def test_B2_contains_both_qq(self):
        adv = Advisories.B2(cache_qq=12, req_qq=20)
        assert adv.advisory_id == "B2"
        assert adv.level == AdvisoryLevel.ACTION
        assert "12" in adv.body and "20" in adv.body
        assert len(adv.actions) == 2

    def test_B3_is_warning_no_actions(self):
        adv = Advisories.B3()
        assert adv.advisory_id == "B3"
        assert adv.level == AdvisoryLevel.WARNING
        assert adv.actions == []

    def test_B4_lists_edges(self):
        adv = Advisories.B4([0, 2])
        assert adv.advisory_id == "B4"
        assert "0" in adv.body and "2" in adv.body

    def test_B5_contains_value(self):
        adv = Advisories.B5(-0.75)
        assert adv.advisory_id == "B5"
        assert "-0.75" in adv.body or "-0.7500" in adv.body
        assert len(adv.actions) >= 1


# ---------------------------------------------------------------------------
# Category C factories
# ---------------------------------------------------------------------------

class TestCategoryC:
    def test_C1_is_action_with_three_buttons(self):
        adv = Advisories.C1()
        assert adv.advisory_id == "C1"
        assert adv.level == AdvisoryLevel.ACTION
        assert len(adv.actions) == 3

    def test_C2_is_warning_no_actions(self):
        adv = Advisories.C2()
        assert adv.advisory_id == "C2"
        assert adv.level == AdvisoryLevel.WARNING
        assert adv.actions == []

    def test_C3_contains_PQ_and_length(self):
        adv = Advisories.C3(3, 2, 2, 45.0)
        assert adv.advisory_id == "C3"
        assert "3" in adv.body and "2" in adv.body
        assert "45" in adv.body
        assert len(adv.actions) == 2

    def test_C4_contains_counts(self):
        adv = Advisories.C4(r=2, n=3, total=6)
        assert "6" in adv.body
        assert any("6" in a.label for a in adv.actions)

    def test_C5_is_error_with_slope(self):
        adv = Advisories.C5(4, 2)
        assert adv.advisory_id == "C5"
        assert adv.level == AdvisoryLevel.ERROR
        assert "4" in adv.body and "2" in adv.body

    def test_C6_is_info_with_actions(self):
        adv = Advisories.C6()
        assert adv.advisory_id == "C6"
        assert adv.level == AdvisoryLevel.INFO
        assert len(adv.actions) >= 1


# ---------------------------------------------------------------------------
# Category D factories
# ---------------------------------------------------------------------------

class TestCategoryD:
    def test_D1_is_action_with_name(self):
        adv = Advisories.D1("m004")
        assert adv.advisory_id == "D1"
        assert adv.level == AdvisoryLevel.ACTION
        assert "m004" in adv.body
        assert len(adv.actions) == 2

    def test_D2_contains_both_qq(self):
        adv = Advisories.D2(old_qq=20, new_qq=50)
        assert "20" in adv.body and "50" in adv.body
        assert len(adv.actions) == 2

    def test_D3_contains_both_qq(self):
        adv = Advisories.D3(saved_qq=12, now_qq=20)
        assert "12" in adv.body and "20" in adv.body
        assert len(adv.actions) == 2
