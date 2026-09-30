from __future__ import annotations

import threading
from dataclasses import dataclass

from fabweave.chemicals.shelf_life import shelf_life_remaining
from fabweave.core.decision import Decision, allow, deny
from fabweave.core.timebox import TimeBox
from fabweave.recipe.qualification import Qualification

CHEMICAL = "chemical"
PART = "part"


@dataclass(frozen=True, slots=True)
class KitItem:
    """A chemical lot or critical part that may be drawn into a kit.

    Chemicals age from when they were opened; time spent frozen pauses that
    clock.  Parts are tracked against the recipe revisions they are qualified
    for.  Both kinds can only serve one job during any overlapping window.
    """

    item_id: str
    kind: str
    opened_at: int | None = None
    shelf_hours: int | None = None
    frozen_periods: tuple[TimeBox, ...] = ()
    qualifications: tuple[Qualification, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in (CHEMICAL, PART):
            raise ValueError("kit item must be a chemical or a part")
        if self.kind == CHEMICAL:
            if self.opened_at is None or self.shelf_hours is None:
                raise ValueError("chemical requires opening time and shelf life")
            if self.shelf_hours < 0:
                raise ValueError("negative shelf life")
        elif self.opened_at is not None:
            raise ValueError("parts do not age like chemicals")


@dataclass(frozen=True, slots=True)
class KitLine:
    """One reserved item of a kit, bound to the job that consumes it."""

    item_id: str
    job_id: str
    window: TimeBox
    recipe_revision: int


@dataclass(frozen=True, slots=True)
class KitRequest:
    request_id: str
    job_id: str
    window: TimeBox
    recipe_revision: int
    item_ids: tuple[str, ...]
    at: int

    def fingerprint(self) -> tuple[object, ...]:
        return (self.job_id, self.window, self.recipe_revision, self.item_ids, self.at)


@dataclass(frozen=True, slots=True)
class KitResult:
    request: KitRequest
    decision: Decision
    lines: tuple[KitLine, ...] = ()


def _line_reasons(line: KitLine, item: KitItem | None) -> tuple[str, ...]:
    if item is None:
        return (f"item_unknown:{line.item_id}",)
    reasons: list[str] = []
    if item.kind == CHEMICAL:
        remaining = shelf_life_remaining(
            item.opened_at,
            item.shelf_hours,
            line.window.start,
            list(item.frozen_periods),
        )
        if remaining < 0:
            reasons.append(f"chemical_shelf_life_expired:{line.item_id}")
    else:
        matching = [q for q in item.qualifications if q.recipe_revision == line.recipe_revision]
        if not matching:
            reasons.append(f"recipe_qualification_missing:{line.item_id}")
        elif not any(q.active_at(line.window.start) for q in matching):
            reasons.append(f"recipe_qualification_inactive:{line.item_id}")
    return tuple(reasons)


class KitBook:
    """Serializes whole-kit preparation so a kit is never locked halfway.

    Every requested item is validated against shelf life, freeze-paused age,
    recipe qualification and existing usage windows before any line is held.
    Requests are idempotent: replaying a request id returns the stored result,
    while reusing it with different content is rejected.
    """

    def __init__(self, registry: dict[str, KitItem] | None = None) -> None:
        self._registry: dict[str, KitItem] = dict(registry or {})
        self._held: list[KitLine] = []
        self._requests: dict[str, KitResult] = {}
        self._lock = threading.RLock()

    def register(self, item: KitItem) -> None:
        with self._lock:
            self._registry[item.item_id] = item

    def prepare(self, request: KitRequest) -> KitResult:
        with self._lock:
            previous = self._requests.get(request.request_id)
            if previous is not None:
                if previous.request.fingerprint() != request.fingerprint():
                    raise ValueError("request id reused with a different kit")
                return previous

            if not request.item_ids:
                raise ValueError("empty kit request")
            if len(request.item_ids) != len(set(request.item_ids)):
                raise ValueError("kit lists the same item twice")

            lines = [
                KitLine(item_id, request.job_id, request.window, request.recipe_revision)
                for item_id in request.item_ids
            ]

            reasons: list[str] = []
            candidate = [*self._held]
            for line in lines:
                item = self._registry.get(line.item_id)
                reasons.extend(_line_reasons(line, item))
                for existing in candidate:
                    if (
                        existing.item_id == line.item_id
                        and existing.job_id != line.job_id
                        and existing.window.overlaps(line.window)
                    ):
                        reasons.append(f"item_time_conflict:{line.item_id}")
                candidate.append(line)

            if reasons:
                # candidate was never published: the whole kit rolls back.
                decision = deny(*dict.fromkeys(reasons), request_id=request.request_id, at=request.at)
                result = KitResult(request, decision)
            else:
                self._held = candidate
                decision = allow(
                    request_id=request.request_id,
                    item_count=len(lines),
                    at=request.at,
                )
                result = KitResult(request, decision, tuple(lines))
            self._requests[request.request_id] = result
            return result

    def reservations(self, item_id: str) -> tuple[KitLine, ...]:
        with self._lock:
            return tuple(line for line in self._held if line.item_id == item_id)

    @property
    def held(self) -> tuple[KitLine, ...]:
        with self._lock:
            return tuple(self._held)
