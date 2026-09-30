from __future__ import annotations

import threading

import pytest

from fabweave.core.timebox import TimeBox
from fabweave.kit.preparation import (
    KitBook,
    KitLine,
    StockUnit,
    chemical_shelf_decision,
)
from fabweave.workflows.control import prepare_kit


def _chemical(stock_id: str, revs, opened_at=0, shelf_hours=10, frozen=()) -> StockUnit:
    return StockUnit(
        stock_id,
        "chemical",
        frozenset(revs),
        opened_at,
        shelf_hours,
        tuple(frozen),
    )


def _part(stock_id: str, revs) -> StockUnit:
    return StockUnit(stock_id, "part", frozenset(revs))


def _book(*units: StockUnit) -> KitBook:
    return KitBook({unit.stock_id: unit for unit in units})


# --- opened shelf life, with freezing pausing the clock ---------------------

def test_chemical_shelf_life_uses_effective_age_with_freeze_pause():
    # opened at 0, 10h shelf, used through hour 12, frozen 8..12 (4h paused)
    # effective age at window end is 12 - 4 = 8, within the 10h shelf.
    chem = _chemical("CHEM-A", {2}, 0, 10, frozen=[TimeBox(8, 12)])
    line = KitLine("CHEM-A", 2, TimeBox(10, 12))
    assert chemical_shelf_decision(chem, line).allowed

    expired = _chemical("CHEM-B", {2}, 0, 5, frozen=[TimeBox(8, 12)])
    decision = chemical_shelf_decision(expired, KitLine("CHEM-B", 2, TimeBox(10, 12)))
    assert decision.reasons == ("shelf_life_expired:CHEM-B",)
    assert decision.evidence["remaining_shelf_hours"] == -3


def test_parts_and_unopened_chemicals_do_not_age():
    part = _part("PUMP-1", {1})
    assert chemical_shelf_decision(part, KitLine("PUMP-1", 1, TimeBox(0, 100))).allowed
    unopened = StockUnit("CHEM-X", "chemical", frozenset({1}))
    assert chemical_shelf_decision(unopened, KitLine("CHEM-X", 1, TimeBox(0, 100))).allowed


def test_frozen_storage_blocks_early_usage_until_thaw():
    book = _book(_chemical("CHEM-A", {2}, 0, 50))
    book.freeze("CHEM-A", 20)
    result = book.prepare("REQ-F", [KitLine("CHEM-A", 2, TimeBox(10, 15))])
    assert not result.allowed and result.reasons == ("stock_frozen:CHEM-A",)
    # A window starting at the thaw boundary (half-open) is fine.
    assert book.prepare("REQ-G", [KitLine("CHEM-A", 2, TimeBox(20, 25))]).allowed


# --- recipe qualification ---------------------------------------------------

def test_unqualified_recipe_revision_rejects_the_line():
    book = _book(_chemical("CHEM-A", {1, 2}))
    result = book.prepare("REQ-Q", [KitLine("CHEM-A", 9, TimeBox(0, 5))])
    assert not result.allowed and result.reasons == ("recipe_not_qualified:CHEM-A",)


# --- all-or-nothing: a late failure rolls back earlier lines -----------------

def test_kit_failure_rolls_back_every_line_even_the_valid_ones():
    book = _book(
        _chemical("CHEM-A", {2}, 0, 50),
        _part("PUMP-1", {2}),
        _chemical("CHEM-DEAD", {2}, 0, 5),  # expires at effective hour 5
    )
    # The dead chemical used at hour 10 fails; the two good lines must not stick.
    result = book.prepare(
        "REQ-BAD",
        [
            KitLine("CHEM-A", 2, TimeBox(0, 10)),
            KitLine("PUMP-1", 2, TimeBox(0, 10)),
            KitLine("CHEM-DEAD", 2, TimeBox(0, 10)),  # age 10 > shelf 5
        ],
    )
    assert not result.allowed
    assert "shelf_life_expired:CHEM-DEAD" in result.reasons
    assert result.reservations == ()
    # The two good stock units keep no occupancy from the failed kit.
    assert book.occupancies("CHEM-A") == ()
    assert book.occupancies("PUMP-1") == ()
    # And those good units are still bookable.
    assert book.prepare("REQ-RETRY", [KitLine("CHEM-A", 2, TimeBox(0, 2))]).allowed


def test_all_gates_are_reported_together_for_a_line():
    book = _book(_chemical("CHEM-A", {1}, 0, 5))  # wrong rev, expiring
    result = book.prepare(
        "REQ-MULTI",
        [
            KitLine("CHEM-A", 9, TimeBox(0, 20)),  # unqualified + expired
            KitLine("MISSING", 1, TimeBox(0, 20)),  # unknown stock
        ],
    )
    assert "recipe_not_qualified:CHEM-A" in result.reasons
    assert "shelf_life_expired:CHEM-A" in result.reasons
    assert "stock_missing:MISSING" in result.reasons
    assert book.occupancies("CHEM-A") == ()


def test_duplicate_line_in_one_kit_is_rejected_and_rolled_back():
    book = _book(_part("PUMP-1", {1}))
    result = book.prepare(
        "REQ-DUP",
        [KitLine("PUMP-1", 1, TimeBox(0, 5)), KitLine("PUMP-1", 1, TimeBox(6, 9))],
    )
    assert result.reasons == ("duplicate_line:PUMP-1",)
    assert book.occupancies("PUMP-1") == ()


# --- time conflicts between kits -------------------------------------------

def test_overlapping_window_on_same_stock_conflicts_but_touching_is_allowed():
    book = _book(_part("PUMP-1", {1}))
    assert book.prepare("REQ-1", [KitLine("PUMP-1", 1, TimeBox(0, 10))]).allowed
    conflict = book.prepare("REQ-2", [KitLine("PUMP-1", 1, TimeBox(5, 15))])
    assert not conflict.allowed and conflict.reasons == ("time_conflict:PUMP-1",)
    assert conflict.evidence["lines"]["PUMP-1"]["conflicting_requests"] == ("REQ-1",)
    # Half-open: a window starting exactly at 10 does not overlap.
    assert book.prepare("REQ-3", [KitLine("PUMP-1", 1, TimeBox(10, 20))]).allowed


def test_different_stock_units_never_conflict():
    book = _book(_part("PUMP-1", {1}), _part("PUMP-2", {1}))
    assert book.prepare("REQ-1", [KitLine("PUMP-1", 1, TimeBox(0, 10)), KitLine("PUMP-2", 1, TimeBox(0, 10))]).allowed


# --- concurrency: no double occupancy ---------------------------------------

def test_concurrent_requests_for_overlapping_stock_do_not_both_win():
    book = _book(_part("PUMP-1", {1}))
    barrier = threading.Barrier(2)
    outcomes: list[bool] = []
    lock = threading.Lock()

    def worker(request_id: str) -> None:
        result = book.prepare_concurrent(request_id, [KitLine("PUMP-1", 1, TimeBox(0, 10))], barrier)
        with lock:
            outcomes.append(result.allowed)

    threads = [threading.Thread(target=worker, args=(f"REQ-C{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == [False, True]
    assert len(book.occupancies("PUMP-1")) == 1


# --- idempotent replay ------------------------------------------------------

def test_replaying_the_same_request_returns_the_original_result():
    book = _book(_part("PUMP-1", {1}))
    lines = [KitLine("PUMP-1", 1, TimeBox(0, 10))]
    first = book.prepare("REQ-IDEM", lines)
    replay = book.prepare("REQ-IDEM", lines)
    assert first is replay
    assert replay.allowed and len(replay.reservations) == 1
    # Replay must not reserve stock a second time.
    assert len(book.occupancies("PUMP-1")) == 1


def test_replaying_a_failed_request_reproduces_the_failure_without_side_effects():
    book = _book(_chemical("CHEM-A", {2}, 0, 5))
    lines = [KitLine("CHEM-A", 2, TimeBox(0, 20))]
    first = book.prepare("REQ-FAIL", lines)
    replay = book.prepare("REQ-FAIL", lines)
    assert first is replay
    assert not replay.allowed and replay.reasons == ("shelf_life_expired:CHEM-A",)
    assert book.occupancies("CHEM-A") == ()


def test_same_request_id_with_different_kit_is_an_error():
    book = _book(_part("PUMP-1", {1}), _part("PUMP-2", {1}))
    book.prepare("REQ-X", [KitLine("PUMP-1", 1, TimeBox(0, 10))])
    with pytest.raises(ValueError):
        book.prepare("REQ-X", [KitLine("PUMP-2", 1, TimeBox(0, 10))])


# --- workflow composition ---------------------------------------------------

def test_prepare_kit_workflow_delegates_and_exposes_a_decision():
    book = _book(_part("PUMP-1", {1}))
    result = prepare_kit(book, "REQ-W", [KitLine("PUMP-1", 1, TimeBox(0, 5))])
    assert result.allowed and result.decision.allowed
    rejected = prepare_kit(book, "REQ-W2", [KitLine("GHOST", 1, TimeBox(0, 5))])
    assert not rejected.decision.allowed
    assert rejected.decision.reasons == ("stock_missing:GHOST",)
