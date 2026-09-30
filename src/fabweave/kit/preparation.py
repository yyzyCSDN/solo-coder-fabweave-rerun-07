"""Atomic kit preparation for chemicals and critical parts.

A *kit* is the set of chemical containers and critical parts that must be
prepared together for a single operation.  A kit is only usable when every
line passes four gates at once:

* **opened shelf life** – a chemical container is clocked from when it was
  opened, but frozen storage pauses that clock;
* **recipe qualification** – the container/part may only serve the recipe
  revision requested by the operation;
* **time conflict** – one physical stock unit cannot be promised to two kits
  whose usage windows overlap (half-open, so touching windows are allowed);
* **duplicate occupancy** – the same stock unit may not be listed twice in a
  single kit, and concurrent preparation requests must not both win.

Every gate is evaluated before anything is committed: a failure on any line
rolls the whole kit back.  A replayed request id returns the original result
without re-reserving stock.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from fabweave.chemicals.shelf_life import shelf_life_remaining
from fabweave.core.decision import Decision, allow, deny
from fabweave.core.ids import deterministic_id
from fabweave.core.timebox import TimeBox


@dataclass(frozen=True, slots=True)
class StockUnit:
    """A physical chemical container or critical part that can be occupied."""

    stock_id: str
    kind: str  # "chemical" or "part"
    recipe_revisions: frozenset[int]
    # Chemical aging fields; parts leave them at None and never age.
    opened_at: int | None = None
    shelf_hours: int | None = None
    frozen_periods: tuple[TimeBox, ...] = ()

    def qualified_for(self, recipe_revision: int) -> bool:
        return recipe_revision in self.recipe_revisions


@dataclass(frozen=True, slots=True)
class KitLine:
    """One requested stock unit plus the window it is needed for."""

    stock_id: str
    recipe_revision: int
    window: TimeBox


@dataclass(frozen=True, slots=True)
class KitReservation:
    """A committed occupancy of one stock unit for one kit request."""

    stock_id: str
    request_id: str
    window: TimeBox
    recipe_revision: int


@dataclass(frozen=True, slots=True)
class KitResult:
    """Outcome of preparing a kit, returned identically on replay."""

    request_id: str
    allowed: bool
    reasons: tuple[str, ...]
    reservations: tuple[KitReservation, ...]
    evidence: dict[str, object]

    @property
    def decision(self) -> Decision:
        return deny(*self.reasons, **self.evidence) if self.reasons else allow(**self.evidence)


def chemical_shelf_decision(unit: StockUnit, line: KitLine) -> Decision:
    """Opened-shelf-life gate evaluated at the end of the usage window.

    Frozen periods pause the clock, mirroring queue-time pause handling.
    Parts and unopened stock do not age.
    """
    if unit.kind != "chemical" or unit.opened_at is None or unit.shelf_hours is None:
        return allow(remaining_shelf_hours=None)
    try:
        remaining = shelf_life_remaining(unit.opened_at, unit.shelf_hours, line.window.end, list(unit.frozen_periods))
    except ValueError:
        return deny(f"shelf_life:inactive:{line.stock_id}", stock_id=line.stock_id)
    evidence = {"stock_id": line.stock_id, "remaining_shelf_hours": remaining}
    return deny(f"shelf_life_expired:{line.stock_id}", **evidence) if remaining < 0 else allow(**evidence)


def _line_decision(
    line: KitLine,
    stock: dict[str, StockUnit],
    frozen_until: dict[str, int],
    committed: list[KitReservation],
    seen: set[str],
) -> Decision:
    reasons: list[str] = []
    evidence: dict[str, object] = {"stock_id": line.stock_id}

    unit = stock.get(line.stock_id)
    if unit is None:
        return deny(f"stock_missing:{line.stock_id}", **evidence)
    if line.recipe_revision <= 0:
        reasons.append(f"recipe_revision_invalid:{line.stock_id}")
    elif not unit.qualified_for(line.recipe_revision):
        reasons.append(f"recipe_not_qualified:{line.stock_id}")

    # Frozen storage suspends a unit until the given instant; a window that
    # only touches the freeze boundary (half-open) is still available.
    thawed_at = frozen_until.get(line.stock_id)
    if thawed_at is not None and line.window.start < thawed_at:
        reasons.append(f"stock_frozen:{line.stock_id}")
        evidence["frozen_until"] = thawed_at

    shelf = chemical_shelf_decision(unit, line)
    reasons.extend(shelf.reasons)
    evidence.update(shelf.evidence)

    # Duplicate listing inside this kit, then overlap against every unit
    # already tentatively occupied (earlier lines plus other committed kits).
    if line.stock_id in seen:
        reasons.append(f"duplicate_line:{line.stock_id}")
    conflicts = [
        existing.request_id
        for existing in committed
        if existing.stock_id == line.stock_id
        and existing.request_id != ""  # intra-kit tentatives carry ""
        and existing.window.overlaps(line.window)
    ]
    if conflicts:
        reasons.append(f"time_conflict:{line.stock_id}")
        evidence["conflicting_requests"] = tuple(dict.fromkeys(conflicts))

    return deny(*reasons, **evidence) if reasons else allow(**evidence)


def _request_fingerprint(request_id: str, lines: tuple[KitLine, ...]) -> str:
    material = [(line.stock_id, line.recipe_revision, line.window.start, line.window.end) for line in lines]
    return deterministic_id("KITREQ", request_id, sorted(material))


class KitBook:
    """Holds committed kit occupancies and serializes preparation.

    A single re-entrant lock makes the validate-then-commit sequence atomic
    against concurrent preparation threads, preventing two requests from both
    observing free stock and then both reserving it.
    """

    def __init__(self, stock: dict[str, StockUnit] | None = None) -> None:
        self._stock = dict(stock or {})
        self._frozen_until: dict[str, int] = {}
        self._reservations: list[KitReservation] = []
        self._results: dict[str, KitResult] = {}
        self._fingerprints: dict[str, str] = {}
        self._lock = threading.RLock()

    def add_stock(self, unit: StockUnit) -> None:
        with self._lock:
            if unit.stock_id in self._stock:
                raise ValueError("stock unit already registered")
            self._stock[unit.stock_id] = unit

    def freeze(self, stock_id: str, until: int) -> None:
        with self._lock:
            if stock_id not in self._stock:
                raise ValueError("unknown stock unit")
            self._frozen_until[stock_id] = until

    def prepare(self, request_id: str, lines: list[KitLine]) -> KitResult:
        """Validate the whole kit, then commit every line or roll all back.

        Replaying the same ``request_id`` with the same lines returns the
        stored result; the same id with different lines is a client error.
        """
        if not lines:
            raise ValueError("empty kit")
        frozen_lines = tuple(lines)
        fingerprint = _request_fingerprint(request_id, frozen_lines)
        with self._lock:
            stored = self._results.get(request_id)
            if stored is not None:
                if self._fingerprints[request_id] != fingerprint:
                    raise ValueError("request id reused with a different kit")
                return stored
            return self._prepare_locked(request_id, frozen_lines, fingerprint)

    def _prepare_locked(
        self,
        request_id: str,
        lines: tuple[KitLine, ...],
        fingerprint: str,
    ) -> KitResult:
        reasons: list[str] = []
        line_evidence: dict[str, dict[str, object]] = {}
        tentative: list[KitReservation] = []
        seen: set[str] = set()

        # Validate against committed occupancies plus this kit's own earlier
        # lines, but do not mutate committed state until every line passes.
        for line in lines:
            decision = _line_decision(line, self._stock, self._frozen_until, [*self._reservations, *tentative], seen)
            seen.add(line.stock_id)
            reasons.extend(decision.reasons)
            line_evidence[line.stock_id] = dict(decision.evidence)
            tentative.append(KitReservation(line.stock_id, request_id, line.window, line.recipe_revision))

        if reasons:
            # Whole-kit rollback: nothing tentative is committed.
            result = KitResult(
                request_id,
                False,
                tuple(dict.fromkeys(reasons)),
                (),
                {"request_id": request_id, "lines": line_evidence},
            )
        else:
            self._reservations.extend(tentative)
            result = KitResult(
                request_id,
                True,
                (),
                tuple(tentative),
                {"request_id": request_id, "lines": line_evidence, "line_count": len(tentative)},
            )
        self._results[request_id] = result
        self._fingerprints[request_id] = fingerprint
        return result

    def occupancies(self, stock_id: str) -> tuple[KitReservation, ...]:
        with self._lock:
            return tuple(item for item in self._reservations if item.stock_id == stock_id)

    def prepare_concurrent(self, request_id: str, lines: list[KitLine], barrier: threading.Barrier) -> KitResult:
        """Prepare from threads that start validation at the same instant.

        Threads block on the barrier before taking the book lock, so the
        critical section is the only thing preventing double occupancy.
        """
        barrier.wait()
        return self.prepare(request_id, lines)
