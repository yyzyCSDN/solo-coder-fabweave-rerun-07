from __future__ import annotations

from dataclasses import dataclass, replace

from fabweave.chamber.matching import Tolerance, mismatch_reasons
from fabweave.chemicals.shelf_life import shelf_life_remaining
from fabweave.cleanroom.excursion import active_excursion
from fabweave.contamination.segregation import incompatible_pairs
from fabweave.core.decision import Decision, allow, deny
from fabweave.core.events import EventLog
from fabweave.core.timebox import TimeBox
from fabweave.dedication.tool import DedicationPolicy, policy_decision
from fabweave.dispatch.priority import DispatchCandidate, rank_candidates
from fabweave.energy.demand import demand_violations
from fabweave.genealogy.carrier import GenealogyGraph
from fabweave.holds.release import HoldState, release_decision
from fabweave.kit.preparation import KitBook, KitLine, KitResult
from fabweave.lot.merge import Lot, merge_lots
from fabweave.maintenance.window import MaintenancePolicy, assignment_decision
from fabweave.metrology.sampling import Wafer, deterministic_sample
from fabweave.queue_time.enforcer import QueueLimit, queue_time_decision
from fabweave.recipe.qualification import Qualification, qualification_decision
from fabweave.reticle.reservation import Reservation, ReservationBook
from fabweave.rework.authorization import ReworkPolicy, authorization_decision
from fabweave.shipment.commit import Allocation, CommitLedger
from fabweave.spc.control import western_electric_rules
from fabweave.takt.monitor import TaktResult, takt_result
from fabweave.yielding.rollup import StepYield, route_yield


def _combine(decisions: list[Decision], **evidence: object) -> Decision:
    reasons = tuple(reason for decision in decisions for reason in decision.reasons)
    merged = {key: value for decision in decisions for key, value in decision.evidence.items()}
    merged.update(evidence)
    return deny(*reasons, **merged) if reasons else allow(**merged)


def material_readiness(
    queue_limit: QueueLimit,
    entered_at: int,
    evaluated_at: int,
    queue_pauses: list[TimeBox],
    opened_at: int,
    shelf_hours: int,
    frozen_periods: list[TimeBox],
) -> Decision:
    queue = queue_time_decision(queue_limit, entered_at, evaluated_at, queue_pauses)
    remaining = shelf_life_remaining(opened_at, shelf_hours, evaluated_at, frozen_periods)
    chemical = deny("chemical_shelf_life_expired", remaining_shelf_hours=remaining) if remaining < 0 else allow(remaining_shelf_hours=remaining)
    return _combine([queue, chemical])


def rank_releasable(
    candidates: list[DispatchCandidate],
    now: int,
    holds: dict[str, HoldState],
    expected_versions: dict[str, int],
) -> list[DispatchCandidate]:
    evaluated: list[DispatchCandidate] = []
    for candidate in candidates:
        state = holds.get(candidate.lot_id)
        blocked = candidate.active_hold
        if state is not None:
            expected = expected_versions.get(candidate.lot_id, state.version)
            blocked = blocked or not release_decision(state, expected, now).allowed
        evaluated.append(replace(candidate, active_hold=blocked))
    return rank_candidates(evaluated, now)


def reserve_qualified(
    book: ReservationBook,
    request_id: str,
    requested: list[Reservation],
    assignments: dict[str, tuple[str, str]],
    qualifications: list[Qualification],
    at: int,
) -> Decision:
    decisions: list[Decision] = []
    for reservation in requested:
        assignment = assignments.get(reservation.job_id)
        if assignment is None:
            decisions.append(deny(f"assignment_missing:{reservation.job_id}"))
            continue
        tool_id, chamber_id = assignment
        decisions.append(qualification_decision(tool_id, chamber_id, reservation.recipe_revision, at, qualifications))
    combined = _combine(decisions, request_id=request_id)
    if not combined.allowed:
        return combined
    reserved = book.reserve_batch(request_id, requested)
    return allow(request_id=request_id, reservation_count=len(reserved))


def prepare_kit(book: KitBook, request_id: str, lines: list[KitLine]) -> KitResult:
    """Prepare a chemical + critical-part kit atomically.

    Delegates to the kit book, which evaluates opened shelf life (freeze
    pauses honored), recipe qualification, time conflicts and duplicate
    occupancy before committing, rolls the whole kit back on any failure and
    returns the stored result when the same request is replayed.
    """
    return book.prepare(request_id, lines)


def qualified_assignment(
    policy: MaintenancePolicy,
    last_pm_at: float,
    accumulated_run_hours: float,
    now: int,
    job_window: TimeBox,
    planned_maintenance: list[TimeBox],
    tool_id: str,
    chamber_id: str,
    recipe_revision: int,
    qualifications: list[Qualification],
) -> Decision:
    maintenance = assignment_decision(policy, last_pm_at, accumulated_run_hours, now, job_window, planned_maintenance)
    qualification = qualification_decision(tool_id, chamber_id, recipe_revision, job_window.start, qualifications)
    return _combine([maintenance, qualification])


@dataclass(frozen=True, slots=True)
class SampleEvaluation:
    wafer_ids: tuple[str, ...]
    rules: tuple[str, ...]


def evaluate_sample(
    wafers: list[Wafer],
    sample_count: int,
    required_zones: set[str],
    measurements: dict[str, float],
    center: float,
    sigma: float,
) -> SampleEvaluation:
    selected = deterministic_sample(wafers, sample_count, required_zones)
    missing = sorted(wafer.wafer_id for wafer in selected if wafer.wafer_id not in measurements)
    if missing:
        raise ValueError(f"sample measurements missing: {', '.join(missing)}")
    values = [measurements[wafer.wafer_id] for wafer in selected]
    return SampleEvaluation(tuple(wafer.wafer_id for wafer in selected), western_electric_rules(values, center, sigma))


def chamber_sample_decision(
    reference: dict[str, float],
    candidate: dict[str, float],
    tolerances: dict[str, Tolerance],
    wafers: list[Wafer],
    sample_count: int,
    required_zones: set[str],
) -> Decision:
    reasons = list(mismatch_reasons(reference, candidate, tolerances))
    try:
        sample = deterministic_sample(wafers, sample_count, required_zones)
    except ValueError as error:
        reasons.append(f"sample:{error}")
        sample = ()
    evidence = {"sample_wafer_ids": tuple(wafer.wafer_id for wafer in sample)}
    return deny(*reasons, **evidence) if reasons else allow(**evidence)


def environmental_containment(
    readings: list[tuple[int, float]],
    open_limit: float,
    close_limit: float,
    required_open: int,
    required_close: int,
    material_classes: set[str],
    contamination_matrix: dict[str, set[str]],
) -> Decision:
    excursion = active_excursion(readings, open_limit, close_limit, required_open, required_close)
    pairs = incompatible_pairs(material_classes, contamination_matrix)
    reasons = [*(("cleanroom_excursion",) if excursion else ()), *(f"incompatible:{left}:{right}" for left, right in pairs)]
    return deny(*reasons, active_excursion=excursion, incompatible_pairs=pairs) if reasons else allow(active_excursion=False, incompatible_pairs=())


def containment_scope(graph: GenealogyGraph, seeds: list[str], max_depth: int) -> tuple[str, ...]:
    impacted = set(seeds)
    for seed in seeds:
        impacted.update(graph.related(seed, "upstream", max_depth))
        impacted.update(graph.related(seed, "downstream", max_depth))
    return tuple(sorted(impacted))


@dataclass(frozen=True, slots=True)
class ContainmentRecord:
    scope: tuple[str, ...]
    event_sequence: int


def record_containment(
    log: EventLog,
    request_id: str,
    graph: GenealogyGraph,
    seeds: list[str],
    max_depth: int,
    at: int,
) -> ContainmentRecord:
    scope = containment_scope(graph, seeds, max_depth)
    event = log.append(
        f"containment:{request_id}",
        0,
        request_id,
        at,
        "containment_planned",
        {"scope": list(scope)},
    )
    return ContainmentRecord(tuple(event.data["scope"]), event.sequence)


def powered_assignment(
    loads: list[tuple[int, float]],
    demand_windows: list[tuple[int, float]],
    maintenance_policy: MaintenancePolicy,
    last_pm_at: float,
    run_hours: float,
    now: int,
    job_window: TimeBox,
    maintenance_windows: list[TimeBox],
) -> Decision:
    widths = demand_violations(loads, demand_windows)
    demand = deny(*(f"demand_limit_exceeded:{width}" for width in widths), violated_windows=widths) if widths else allow(violated_windows=())
    maintenance = assignment_decision(maintenance_policy, last_pm_at, run_hours, now, job_window, maintenance_windows)
    return _combine([demand, maintenance])


def rework_release_decision(
    policy: ReworkPolicy,
    operation: str,
    prior_reworks: int,
    defect_code: str,
    engineer_approval: str | None,
    hold: HoldState,
    expected_hold_version: int,
    at: int,
) -> Decision:
    authorization = authorization_decision(policy, operation, prior_reworks, defect_code, engineer_approval)
    release = release_decision(hold, expected_hold_version, at)
    return _combine([authorization, release])


def merge_and_commit(request_id: str, lots: list[Lot], ledger: CommitLedger, order_id: str) -> tuple[Lot, Allocation]:
    merged = merge_lots(request_id, lots)
    allocation = ledger.allocate(request_id, order_id, merged.quantity)
    return merged, allocation


def dedicated_qualification(
    product: str,
    tool_id: str,
    chamber_id: str,
    recipe_revision: int,
    at: int,
    dedication_policies: list[DedicationPolicy],
    qualifications: list[Qualification],
) -> Decision:
    dedication = policy_decision(product, tool_id, at, dedication_policies)
    qualification = qualification_decision(tool_id, chamber_id, recipe_revision, at, qualifications)
    return _combine([dedication, qualification])


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    takt: TaktResult
    route_yield: float


def performance_summary(
    completions: list[int],
    window_start: int,
    window_end: int,
    steps: list[StepYield],
    expected_steps: list[str],
) -> PerformanceSummary:
    return PerformanceSummary(takt_result(completions, window_start, window_end), route_yield(steps, expected_steps))
