from __future__ import annotations

import threading

import pytest

from fabweave.chemicals.kitting import CHEMICAL, PART, KitBook, KitItem, KitRequest
from fabweave.core.timebox import TimeBox
from fabweave.recipe.qualification import Qualification


def _book() -> KitBook:
    book = KitBook()
    # Chemical opened at 0 with 10h of opened shelf life; frozen 110..130 pauses aging.
    book.register(KitItem("CHEM-FRESH", CHEMICAL, opened_at=0, shelf_hours=100))
    book.register(
        KitItem(
            "CHEM-AGED",
            CHEMICAL,
            opened_at=0,
            shelf_hours=10,
            frozen_periods=(TimeBox(110, 130),),
        )
    )
    # Part qualified for recipe revision 2 until time 100; rev 3 was revoked at 40.
    book.register(
        KitItem(
            "PART-OK",
            PART,
            qualifications=(
                Qualification("TOOL", "PART-OK", 2, 0, 100),
                Qualification("TOOL", "PART-OK", 3, 0, 100, revoked_at=40),
            ),
        )
    )
    return book


def test_whole_kit_locks_chemicals_and_parts_together():
    book = _book()
    request = KitRequest("REQ-1", "JOB-A", TimeBox(50, 60), 2, ("CHEM-FRESH", "PART-OK"), 50)
    result = book.prepare(request)
    assert result.decision.allowed
    assert {line.item_id for line in result.lines} == {"CHEM-FRESH", "PART-OK"}
    assert len(book.reservations("CHEM-FRESH")) == 1
    assert len(book.reservations("PART-OK")) == 1


def test_shelf_life_respects_freeze_pause():
    book = _book()
    # At 120 raw age is 120 (>10) but 20 of those hours were frozen: effective age 100? No:
    # freeze 110..130 clipped to 110..120 = 10h, so effective age 110 > 10 -> expired.
    expired = KitRequest("REQ-X", "JOB-A", TimeBox(120, 125), 2, ("CHEM-AGED",), 120)
    assert not book.prepare(expired).decision.allowed
    # Earlier, at 105: effective age 105 > 10 still expired; use a fresh window at 5 -> age 5 ok.
    ok = KitRequest("REQ-Y", "JOB-B", TimeBox(5, 9), 2, ("CHEM-AGED",), 5)
    assert book.prepare(ok).decision.allowed


def test_freeze_pause_extends_an_otherwise_expired_chemical():
    book = KitBook()
    # Opened at 0, 9h shelf life; frozen 0..5, so at time 10 effective age is 5 -> usable.
    book.register(KitItem("CHEM-FZ", CHEMICAL, opened_at=0, shelf_hours=9, frozen_periods=(TimeBox(0, 5),)))
    result = book.prepare(KitRequest("REQ-F", "JOB-A", TimeBox(10, 12), 1, ("CHEM-FZ",), 10))
    assert result.decision.allowed


def test_recipe_qualification_rejects_expired_and_revoked_revisions():
    book = _book()
    # Revision 3 qualification was revoked at 40.
    revoked = KitRequest("REQ-R", "JOB-A", TimeBox(50, 60), 3, ("PART-OK",), 50)
    reasons = book.prepare(revoked).decision.reasons
    assert reasons == ("recipe_qualification_inactive:PART-OK",)
    # Revision 9 was never qualified for.
    missing = KitRequest("REQ-M", "JOB-A", TimeBox(50, 60), 9, ("PART-OK",), 50)
    assert book.prepare(missing).decision.reasons == ("recipe_qualification_missing:PART-OK",)


def test_time_conflict_blocks_overlapping_use_but_allows_touching_window():
    book = _book()
    book.prepare(KitRequest("REQ-1", "JOB-A", TimeBox(10, 20), 2, ("PART-OK",), 10))
    overlap = KitRequest("REQ-2", "JOB-B", TimeBox(15, 25), 2, ("PART-OK",), 15)
    assert book.prepare(overlap).decision.reasons == ("item_time_conflict:PART-OK",)
    # Half-open: a window starting exactly when the prior one ends does not overlap.
    touching = KitRequest("REQ-3", "JOB-B", TimeBox(20, 30), 2, ("PART-OK",), 20)
    assert book.prepare(touching).decision.allowed


def test_failed_kit_rolls_back_and_holds_nothing():
    book = _book()
    result = book.prepare(
        KitRequest("REQ-FAIL", "JOB-A", TimeBox(50, 60), 2, ("CHEM-FRESH", "PART-MISSING"), 50)
    )
    assert not result.decision.allowed and result.lines == ()
    assert book.held == ()
    # The good chemical in the rolled-back kit is still free for another job.
    retry = KitRequest("REQ-OK", "JOB-B", TimeBox(50, 60), 2, ("CHEM-FRESH",), 50)
    assert book.prepare(retry).decision.allowed


def test_all_independent_reasons_are_reported_together():
    book = _book()
    result = book.prepare(
        KitRequest("REQ-ALL", "JOB-A", TimeBox(120, 125), 9, ("CHEM-AGED", "PART-OK", "GHOST"), 120)
    )
    assert set(result.decision.reasons) == {
        "chemical_shelf_life_expired:CHEM-AGED",
        "recipe_qualification_missing:PART-OK",
        "item_unknown:GHOST",
    }
    assert book.held == ()


def test_replay_returns_the_original_result():
    book = _book()
    request = KitRequest("REQ-IDEM", "JOB-A", TimeBox(50, 60), 2, ("CHEM-FRESH", "PART-OK"), 50)
    first = book.prepare(request)
    replay = book.prepare(request)
    assert replay is first
    assert replay.decision == first.decision and replay.lines == first.lines
    assert len(book.reservations("CHEM-FRESH")) == 1

    # Replaying a previously rejected request returns the same rejection.
    bad = KitRequest("REQ-BAD", "JOB-A", TimeBox(50, 60), 9, ("GHOST",), 50)
    rejected = book.prepare(bad)
    assert book.prepare(bad) is rejected


def test_request_id_reuse_with_different_content_is_rejected():
    book = _book()
    book.prepare(KitRequest("REQ-DUP", "JOB-A", TimeBox(10, 20), 2, ("PART-OK",), 10))
    with pytest.raises(ValueError):
        book.prepare(KitRequest("REQ-DUP", "JOB-A", TimeBox(10, 20), 2, ("CHEM-FRESH",), 10))


def test_concurrent_requests_cannot_double_occupy_the_same_item():
    book = _book()
    outcomes: list[bool] = []

    def attempt(index: int) -> None:
        request = KitRequest(
            f"REQ-C{index}",
            f"JOB-{index}",
            TimeBox(10, 20),
            2,
            ("PART-OK", "CHEM-FRESH"),
            10,
        )
        outcomes.append(book.prepare(request).decision.allowed)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes.count(True) == 1
    assert len(book.reservations("PART-OK")) == 1
    assert len(book.reservations("CHEM-FRESH")) == 1


def test_empty_or_duplicate_lines_are_rejected():
    book = _book()
    with pytest.raises(ValueError):
        book.prepare(KitRequest("REQ-E", "JOB-A", TimeBox(10, 20), 2, (), 10))
    with pytest.raises(ValueError):
        book.prepare(KitRequest("REQ-D", "JOB-A", TimeBox(10, 20), 2, ("PART-OK", "PART-OK"), 10))
