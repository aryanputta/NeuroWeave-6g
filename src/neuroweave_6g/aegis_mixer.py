from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .types import CellDecision, CellDemand, ResourceBudget, SliceDecision, SliceDemand

CandidateScoreFn = Callable[["ActionCandidate", CellDemand, ResourceBudget, int], float]


@dataclass(slots=True)
class ActionCandidate:
    slice_id: str
    action_name: str
    allocated_prbs: float
    allocated_gpu: float
    inspected: bool = False
    isolated: bool = False
    throttled: bool = False
    degraded: bool = False
    controller_cost: int = 0
    coarse_score: float = 0.0
    ranking_score: float = 0.0
    note: str = ""


def build_aegis_mixer_decision(
    *,
    cell: CellDemand,
    budget: ResourceBudget,
    controller_queue: int,
    score_fn: CandidateScoreFn | None = None,
    policy_name: str = "aegis_mixer",
) -> CellDecision:
    attack_pressure = max((slice_demand.anomaly_score for slice_demand in cell.slices), default=0.0)
    overload = controller_queue > 70 or cell.control_events > 90
    candidates = build_candidates_for_cell(
        cell=cell,
        budget=budget,
        controller_queue=controller_queue,
    )

    shortlist_size = min(len(candidates), max(len(cell.slices) * 2, 8))
    shortlist = sorted(candidates, key=lambda candidate: candidate.coarse_score, reverse=True)[:shortlist_size]
    scorer = score_fn or heuristic_candidate_score
    for candidate in shortlist:
        candidate.ranking_score = scorer(candidate, cell, budget, controller_queue)

    slice_map = {slice_demand.slice_id: slice_demand for slice_demand in cell.slices}
    remaining_prbs = budget.prbs_per_cell
    remaining_gpu = budget.gpu_per_cell
    decisions: dict[str, SliceDecision] = {}
    controller_actions_used = 8

    ranked_candidates = sorted(shortlist, key=lambda item: item.ranking_score, reverse=True)
    if overload or attack_pressure > 0.86:
        phase_specs = [
            ("suspicious", remaining_prbs, remaining_gpu),
            (
                "mission_critical",
                budget.prbs_per_cell * (0.46 if attack_pressure > 0.90 else 0.40),
                budget.gpu_per_cell * (0.52 if overload else 0.46),
            ),
            (
                "ai_edge",
                budget.prbs_per_cell * (0.30 if overload else 0.24),
                budget.gpu_per_cell * (0.32 if overload else 0.26),
            ),
            ("other", remaining_prbs, remaining_gpu),
        ]
    else:
        phase_specs = [
            ("mission_critical", budget.prbs_per_cell * 0.38, budget.gpu_per_cell * 0.40),
            ("ai_edge", budget.prbs_per_cell * 0.28, budget.gpu_per_cell * 0.34),
            ("suspicious", remaining_prbs, remaining_gpu),
            ("other", remaining_prbs, remaining_gpu),
        ]

    for phase_name, phase_prbs, phase_gpu in phase_specs:
        decisions, remaining_prbs, remaining_gpu, controller_actions_used = _select_phase(
            ranked_candidates=ranked_candidates,
            slice_map=slice_map,
            phase_name=phase_name,
            phase_prbs=min(phase_prbs, remaining_prbs) if phase_name != "suspicious" else remaining_prbs,
            phase_gpu=min(phase_gpu, remaining_gpu) if phase_name != "suspicious" else remaining_gpu,
            decisions=decisions,
            remaining_prbs=remaining_prbs,
            remaining_gpu=remaining_gpu,
            controller_actions_used=controller_actions_used,
        )

    for slice_demand in cell.slices:
        if slice_demand.slice_id in decisions:
            continue
        fallback = build_fallback_candidate(
            slice_demand=slice_demand,
            overload=overload,
            attack_pressure=attack_pressure,
            remaining_prbs=remaining_prbs,
            remaining_gpu=remaining_gpu,
        )
        decisions[slice_demand.slice_id] = candidate_to_slice_decision(fallback)
        remaining_prbs = max(0.0, remaining_prbs - fallback.allocated_prbs)
        remaining_gpu = max(0.0, remaining_gpu - fallback.allocated_gpu)
        controller_actions_used += fallback.controller_cost

    inspection_actions = sum(2 if decision.inspected else 0 for decision in decisions.values())
    controller_actions_used += inspection_actions
    return CellDecision(
        policy_name=policy_name,
        slice_decisions=decisions,
        inspection_actions=inspection_actions,
        controller_actions_used=controller_actions_used,
        note="Retrieval-then-ranking controller that prioritizes high-value actions under latency and compute pressure.",
    )


def build_candidates_for_cell(
    *,
    cell: CellDemand,
    budget: ResourceBudget,
    controller_queue: int,
) -> list[ActionCandidate]:
    attack_pressure = max((slice_demand.anomaly_score for slice_demand in cell.slices), default=0.0)
    overload = controller_queue > 70 or cell.control_events > 90
    candidates: list[ActionCandidate] = []
    for slice_demand in cell.slices:
        candidates.extend(
            _build_candidates_for_slice(
                slice_demand=slice_demand,
                controller_queue=controller_queue,
                overload=overload,
                attack_pressure=attack_pressure,
            )
        )
    return candidates


def heuristic_candidate_score(
    candidate: ActionCandidate,
    cell: CellDemand,
    budget: ResourceBudget,
    controller_queue: int,
) -> float:
    attack_pressure = max((slice_demand.anomaly_score for slice_demand in cell.slices), default=0.0)
    overload = controller_queue > 70 or cell.control_events > 90
    total_requested_prbs = sum(slice_demand.requested_prbs for slice_demand in cell.slices)
    total_requested_gpu = sum(slice_demand.requested_gpu for slice_demand in cell.slices)
    slice_demand = next(slice_demand for slice_demand in cell.slices if slice_demand.slice_id == candidate.slice_id)
    prb_efficiency = candidate.allocated_prbs / max(total_requested_prbs, 1e-6)
    gpu_efficiency = candidate.allocated_gpu / max(total_requested_gpu, 1e-6) if total_requested_gpu > 0 else 0.0
    budget_pressure = (total_requested_prbs / max(budget.prbs_per_cell, 1e-6)) + (
        total_requested_gpu / max(budget.gpu_per_cell, 1e-6)
    )
    queue_penalty = (controller_queue / 220.0) * candidate.controller_cost * 0.22

    mission_bonus = 4.2 if slice_demand.mission_critical and not candidate.isolated else 0.0
    ai_bonus = 2.4 if slice_demand.kind == "ai_edge" and not candidate.isolated else 0.0
    attack_bonus = (
        3.4 * slice_demand.anomaly_score
        if candidate.isolated
        else 1.2 * slice_demand.anomaly_score
        if candidate.inspected
        else -0.5 * slice_demand.anomaly_score
    )
    false_positive_penalty = 1.8 if candidate.isolated and slice_demand.anomaly_score < 0.86 else 0.0
    degrade_bonus = 1.2 if candidate.degraded and overload and slice_demand.kind in {"ai_edge", "background"} else 0.0
    background_penalty = 1.55 if overload and slice_demand.kind == "background" and candidate.action_name == "serve_full" else 0.0
    resource_penalty = (prb_efficiency * 2.0) + (gpu_efficiency * 1.7)
    overload_bonus = 0.45 if overload and candidate.controller_cost <= 4 else 0.0
    throughput_bonus = 0.35 * slice_demand.utility if candidate.action_name in {"serve_full", "protect_full"} else 0.0
    attack_mode_bonus = 1.0 if attack_pressure > 0.90 and candidate.inspected else 0.0
    ai_floor_bonus = (
        1.3
        if overload and slice_demand.kind == "ai_edge" and candidate.action_name in {"serve_degraded", "serve_floor"}
        else 0.0
    )
    mission_floor_bonus = (
        1.2
        if overload and slice_demand.mission_critical and candidate.action_name in {"protect_full", "protect_floor"}
        else 0.0
    )

    return (
        candidate.coarse_score
        + mission_bonus
        + ai_bonus
        + attack_bonus
        + degrade_bonus
        + overload_bonus
        + throughput_bonus
        + attack_mode_bonus
        + ai_floor_bonus
        + mission_floor_bonus
        - false_positive_penalty
        - background_penalty
        - queue_penalty
        - resource_penalty
        - max(0.0, budget_pressure - 2.0) * 0.55
    )


def candidate_to_slice_decision(candidate: ActionCandidate) -> SliceDecision:
    return SliceDecision(
        slice_id=candidate.slice_id,
        allocated_prbs=candidate.allocated_prbs,
        allocated_gpu=candidate.allocated_gpu,
        inspected=candidate.inspected,
        isolated=candidate.isolated,
        throttled=candidate.throttled,
        degraded=candidate.degraded,
        note=candidate.note,
    )


def candidate_feature_row(
    *,
    candidate: ActionCandidate,
    cell: CellDemand,
    budget: ResourceBudget,
    controller_queue: int,
) -> dict[str, float]:
    slice_demand = next(slice_demand for slice_demand in cell.slices if slice_demand.slice_id == candidate.slice_id)
    total_requested_prbs = sum(item.requested_prbs for item in cell.slices)
    total_requested_gpu = sum(item.requested_gpu for item in cell.slices)
    attack_pressure = max((item.anomaly_score for item in cell.slices), default=0.0)
    overload = 1.0 if controller_queue > 70 or cell.control_events > 90 else 0.0
    kind_flags = {
        "kind_mission_critical": 1.0 if slice_demand.kind == "mission_critical" else 0.0,
        "kind_ai_edge": 1.0 if slice_demand.kind == "ai_edge" else 0.0,
        "kind_broadband": 1.0 if slice_demand.kind == "broadband" else 0.0,
        "kind_background": 1.0 if slice_demand.kind == "background" else 0.0,
        "kind_suspicious": 1.0 if slice_demand.kind == "suspicious" else 0.0,
    }
    action_flags = {
        "action_protect_full": 1.0 if candidate.action_name == "protect_full" else 0.0,
        "action_protect_floor": 1.0 if candidate.action_name == "protect_floor" else 0.0,
        "action_serve_full": 1.0 if candidate.action_name == "serve_full" else 0.0,
        "action_serve_degraded": 1.0 if candidate.action_name == "serve_degraded" else 0.0,
        "action_serve_floor": 1.0 if candidate.action_name == "serve_floor" else 0.0,
        "action_isolate": 1.0 if candidate.action_name == "isolate" else 0.0,
        "action_inspect_limited": 1.0 if candidate.action_name == "inspect_limited" else 0.0,
        "action_serve_scaled": 1.0 if candidate.action_name == "serve_scaled" else 0.0,
        "action_fallback_floor": 1.0 if candidate.action_name == "fallback_floor" else 0.0,
        "action_fallback_isolate": 1.0 if candidate.action_name == "fallback_isolate" else 0.0,
    }
    return {
        "controller_queue": float(controller_queue),
        "control_events": float(cell.control_events),
        "backhaul_pressure": cell.backhaul_pressure,
        "mobility_pressure": cell.mobility_pressure,
        "slice_requested_prbs": slice_demand.requested_prbs,
        "slice_requested_gpu": slice_demand.requested_gpu,
        "slice_latency_budget_ms": slice_demand.latency_budget_ms,
        "slice_anomaly_score": slice_demand.anomaly_score,
        "slice_utility": slice_demand.utility,
        "slice_encrypted": 1.0 if slice_demand.encrypted else 0.0,
        "slice_mission_critical": 1.0 if slice_demand.mission_critical else 0.0,
        "slice_ai_enabled": 1.0 if slice_demand.ai_enabled else 0.0,
        "slice_suspicious": 1.0 if slice_demand.suspicious else 0.0,
        "cell_total_requested_prbs": total_requested_prbs,
        "cell_total_requested_gpu": total_requested_gpu,
        "budget_prbs_per_cell": budget.prbs_per_cell,
        "budget_gpu_per_cell": budget.gpu_per_cell,
        "budget_controller_actions_per_step": float(budget.controller_actions_per_step),
        "candidate_allocated_prbs": candidate.allocated_prbs,
        "candidate_allocated_gpu": candidate.allocated_gpu,
        "candidate_inspected": 1.0 if candidate.inspected else 0.0,
        "candidate_isolated": 1.0 if candidate.isolated else 0.0,
        "candidate_throttled": 1.0 if candidate.throttled else 0.0,
        "candidate_degraded": 1.0 if candidate.degraded else 0.0,
        "candidate_controller_cost": float(candidate.controller_cost),
        "candidate_coarse_score": candidate.coarse_score,
        "attack_pressure": attack_pressure,
        "overload": overload,
        **kind_flags,
        **action_flags,
    }


def build_fallback_candidate(
    *,
    slice_demand: SliceDemand,
    overload: bool,
    attack_pressure: float,
    remaining_prbs: float,
    remaining_gpu: float,
) -> ActionCandidate:
    if slice_demand.suspicious and (overload or attack_pressure > 0.88):
        return ActionCandidate(
            slice_id=slice_demand.slice_id,
            action_name="fallback_isolate",
            allocated_prbs=0.0,
            allocated_gpu=0.0,
            inspected=True,
            isolated=True,
            controller_cost=4,
            note="Fallback isolation after shortlist contention.",
        )

    prb_ratio = (
        0.88
        if slice_demand.mission_critical
        else 0.78
        if slice_demand.kind == "ai_edge"
        else 0.45
        if slice_demand.kind == "broadband"
        else 0.25
    )
    gpu_ratio = 0.90 if slice_demand.mission_critical else 0.76 if slice_demand.kind == "ai_edge" else 0.20
    allocated_prbs = min(slice_demand.requested_prbs * prb_ratio, remaining_prbs)
    allocated_gpu = min(slice_demand.requested_gpu * gpu_ratio, remaining_gpu)
    return ActionCandidate(
        slice_id=slice_demand.slice_id,
        action_name="fallback_floor",
        allocated_prbs=allocated_prbs,
        allocated_gpu=allocated_gpu,
        inspected=slice_demand.anomaly_score > 0.45,
        degraded=True,
        throttled=overload and not slice_demand.mission_critical,
        controller_cost=2,
        note="Fallback floor allocation after higher-ranked actions consumed budget.",
    )


def _build_candidates_for_slice(
    *,
    slice_demand: SliceDemand,
    controller_queue: int,
    overload: bool,
    attack_pressure: float,
) -> list[ActionCandidate]:
    base_prbs = slice_demand.requested_prbs
    base_gpu = slice_demand.requested_gpu
    queue_pressure = min(1.0, controller_queue / 160.0)

    candidates: list[ActionCandidate] = []
    if slice_demand.mission_critical:
        candidates.append(
            ActionCandidate(
                slice_id=slice_demand.slice_id,
                action_name="protect_full",
                allocated_prbs=base_prbs,
                allocated_gpu=base_gpu,
                inspected=slice_demand.anomaly_score > 0.30,
                controller_cost=6,
                coarse_score=5.2 + slice_demand.utility + queue_pressure,
                note="Protect mission-critical slice with full reservation.",
            )
        )
        candidates.append(
            ActionCandidate(
                slice_id=slice_demand.slice_id,
                action_name="protect_floor",
                allocated_prbs=base_prbs * 0.86,
                allocated_gpu=base_gpu * 0.88,
                inspected=slice_demand.anomaly_score > 0.24,
                degraded=True,
                controller_cost=4,
                coarse_score=4.7 + slice_demand.utility + queue_pressure,
                note="Maintain mission-critical floor while saving controller budget.",
            )
        )
        return candidates

    if slice_demand.kind == "ai_edge":
        candidates.extend(
            [
                ActionCandidate(
                    slice_id=slice_demand.slice_id,
                    action_name="serve_full",
                    allocated_prbs=base_prbs,
                    allocated_gpu=base_gpu,
                    inspected=slice_demand.anomaly_score > 0.35,
                    controller_cost=5,
                    coarse_score=3.8 + slice_demand.utility + (0.35 if overload else 0.0),
                    note="Preserve edge-AI quality when the job is still viable.",
                ),
                ActionCandidate(
                    slice_id=slice_demand.slice_id,
                    action_name="serve_degraded",
                    allocated_prbs=base_prbs * 0.82,
                    allocated_gpu=base_gpu * 0.74,
                    inspected=slice_demand.anomaly_score > 0.40,
                    degraded=True,
                    throttled=overload,
                    controller_cost=4,
                    coarse_score=3.5 + slice_demand.utility + (0.55 if overload else 0.15),
                    note="Degrade edge-AI workload instead of dropping it outright.",
                ),
                ActionCandidate(
                    slice_id=slice_demand.slice_id,
                    action_name="serve_floor",
                    allocated_prbs=base_prbs * 0.76,
                    allocated_gpu=base_gpu * 0.78,
                    inspected=slice_demand.anomaly_score > 0.44,
                    degraded=True,
                    throttled=True,
                    controller_cost=3,
                    coarse_score=3.1 + slice_demand.utility + (0.75 if overload else 0.0),
                    note="Keep an edge-AI floor under heavy queue pressure.",
                ),
            ]
        )
        return candidates

    if slice_demand.suspicious:
        candidates.extend(
            [
                ActionCandidate(
                    slice_id=slice_demand.slice_id,
                    action_name="isolate",
                    allocated_prbs=0.0,
                    allocated_gpu=0.0,
                    inspected=True,
                    isolated=True,
                    controller_cost=6,
                    coarse_score=4.4 + attack_pressure + queue_pressure + slice_demand.anomaly_score,
                    note="Hard isolate suspicious burst.",
                ),
                ActionCandidate(
                    slice_id=slice_demand.slice_id,
                    action_name="inspect_limited",
                    allocated_prbs=base_prbs * 0.10,
                    allocated_gpu=base_gpu * 0.05,
                    inspected=True,
                    throttled=True,
                    degraded=True,
                    controller_cost=4,
                    coarse_score=3.5 + slice_demand.anomaly_score + (0.25 if attack_pressure < 0.93 else 0.0),
                    note="Throttle and inspect before full isolation.",
                ),
            ]
        )
        return candidates

    service_full_bias = 0.18 if slice_demand.kind == "broadband" else 0.0
    throttle_bias = 0.55 if overload and slice_demand.kind == "background" else 0.18 if overload else 0.0
    candidates.extend(
        [
            ActionCandidate(
                slice_id=slice_demand.slice_id,
                action_name="serve_full",
                allocated_prbs=base_prbs,
                allocated_gpu=base_gpu,
                controller_cost=3,
                coarse_score=2.0 + slice_demand.utility + service_full_bias,
                note="Admit benign demand at requested level.",
            ),
            ActionCandidate(
                slice_id=slice_demand.slice_id,
                action_name="serve_scaled",
                allocated_prbs=base_prbs * (0.80 if slice_demand.kind == "broadband" else 0.70),
                allocated_gpu=base_gpu * (0.80 if slice_demand.kind == "background" else 1.0),
                degraded=slice_demand.kind == "background",
                throttled=overload,
                controller_cost=2,
                coarse_score=1.9 + slice_demand.utility + throttle_bias,
                note="Scale benign demand to free budget for higher-value slices.",
            ),
        ]
    )
    return candidates


def _select_phase(
    *,
    ranked_candidates: list[ActionCandidate],
    slice_map: dict[str, SliceDemand],
    phase_name: str,
    phase_prbs: float,
    phase_gpu: float,
    decisions: dict[str, SliceDecision],
    remaining_prbs: float,
    remaining_gpu: float,
    controller_actions_used: int,
) -> tuple[dict[str, SliceDecision], float, float, int]:
    phase_remaining_prbs = phase_prbs
    phase_remaining_gpu = phase_gpu
    for candidate in ranked_candidates:
        if candidate.slice_id in decisions:
            continue
        slice_demand = slice_map[candidate.slice_id]
        if not _matches_phase(slice_demand=slice_demand, phase_name=phase_name):
            continue
        fits_prb = candidate.allocated_prbs <= remaining_prbs + 1e-6 and candidate.allocated_prbs <= phase_remaining_prbs + 1e-6
        fits_gpu = candidate.allocated_gpu <= remaining_gpu + 1e-6 and candidate.allocated_gpu <= phase_remaining_gpu + 1e-6
        if not candidate.isolated and (not fits_prb or not fits_gpu):
            continue
        decisions[candidate.slice_id] = candidate_to_slice_decision(candidate)
        remaining_prbs = max(0.0, remaining_prbs - candidate.allocated_prbs)
        remaining_gpu = max(0.0, remaining_gpu - candidate.allocated_gpu)
        phase_remaining_prbs = max(0.0, phase_remaining_prbs - candidate.allocated_prbs)
        phase_remaining_gpu = max(0.0, phase_remaining_gpu - candidate.allocated_gpu)
        controller_actions_used += candidate.controller_cost
    return decisions, remaining_prbs, remaining_gpu, controller_actions_used


def _matches_phase(*, slice_demand: SliceDemand, phase_name: str) -> bool:
    if phase_name == "mission_critical":
        return slice_demand.mission_critical
    if phase_name == "ai_edge":
        return slice_demand.kind == "ai_edge"
    if phase_name == "suspicious":
        return slice_demand.suspicious
    return slice_demand.kind in {"broadband", "background"}
