from __future__ import annotations

from dataclasses import dataclass

from .aegis_mixer import build_aegis_mixer_decision
from .types import CellDecision, CellDemand, ResourceBudget, SliceDecision, SliceDemand


@dataclass(slots=True)
class Policy:
    name: str

    def decide(self, cell: CellDemand, budget: ResourceBudget, controller_queue: int) -> CellDecision:
        raise NotImplementedError


def available_policies() -> list[str]:
    return ["static_qos", "throughput_first", "security_only", "failure_aware", "aegis_mixer"]


def build_policy(name: str) -> Policy:
    mapping = {
        "static_qos": StaticQoSPolicy,
        "throughput_first": ThroughputFirstPolicy,
        "security_only": SecurityOnlyPolicy,
        "failure_aware": FailureAwarePolicy,
        "aegis_mixer": AegisMixerPolicy,
    }
    if name not in mapping:
        raise ValueError(f"Unknown policy {name!r}. Available: {', '.join(available_policies())}")
    return mapping[name]()


def _normalize_weights(weighted: list[tuple[SliceDemand, float]], total_budget: float) -> dict[str, float]:
    positive = [(slice_demand, max(weight, 0.0)) for slice_demand, weight in weighted]
    weight_sum = sum(weight for _, weight in positive)
    if weight_sum <= 0:
        return {slice_demand.slice_id: 0.0 for slice_demand, _ in positive}
    return {
        slice_demand.slice_id: total_budget * (weight / weight_sum)
        for slice_demand, weight in positive
    }


class StaticQoSPolicy(Policy):
    def __init__(self) -> None:
        super().__init__(name="static_qos")

    def decide(self, cell: CellDemand, budget: ResourceBudget, controller_queue: int) -> CellDecision:
        prb_weights = {
            "mission_critical": 0.36,
            "ai_edge": 0.28,
            "broadband": 0.24,
            "background": 0.12,
            "suspicious": 0.08,
        }
        gpu_weights = {
            "mission_critical": 0.45,
            "ai_edge": 0.45,
            "broadband": 0.00,
            "background": 0.10,
            "suspicious": 0.05,
        }
        prb_alloc = _normalize_weights([(slice_demand, prb_weights[slice_demand.kind]) for slice_demand in cell.slices], budget.prbs_per_cell)
        gpu_alloc = _normalize_weights([(slice_demand, gpu_weights[slice_demand.kind]) for slice_demand in cell.slices], budget.gpu_per_cell)
        return CellDecision(
            policy_name=self.name,
            slice_decisions={
                slice_demand.slice_id: SliceDecision(
                    slice_id=slice_demand.slice_id,
                    allocated_prbs=min(slice_demand.requested_prbs, prb_alloc[slice_demand.slice_id]),
                    allocated_gpu=min(slice_demand.requested_gpu, gpu_alloc[slice_demand.slice_id]),
                    inspected=slice_demand.suspicious,
                )
                for slice_demand in cell.slices
            },
            inspection_actions=sum(1 for slice_demand in cell.slices if slice_demand.suspicious),
            controller_actions_used=18 + len(cell.slices),
            note="Fixed QoS shares, no explicit overload or attack protection.",
        )


class ThroughputFirstPolicy(Policy):
    def __init__(self) -> None:
        super().__init__(name="throughput_first")

    def decide(self, cell: CellDemand, budget: ResourceBudget, controller_queue: int) -> CellDecision:
        prb_alloc = _normalize_weights(
            [(slice_demand, slice_demand.requested_prbs * slice_demand.utility) for slice_demand in cell.slices],
            budget.prbs_per_cell,
        )
        gpu_alloc = _normalize_weights(
            [(slice_demand, (slice_demand.requested_gpu + 1.0) * slice_demand.utility) for slice_demand in cell.slices],
            budget.gpu_per_cell,
        )
        decisions: dict[str, SliceDecision] = {}
        for slice_demand in cell.slices:
            decisions[slice_demand.slice_id] = SliceDecision(
                slice_id=slice_demand.slice_id,
                allocated_prbs=min(slice_demand.requested_prbs, prb_alloc[slice_demand.slice_id]),
                allocated_gpu=min(slice_demand.requested_gpu, gpu_alloc[slice_demand.slice_id]),
                inspected=slice_demand.suspicious and slice_demand.anomaly_score > 0.95,
                note="Optimizes for offered demand and utility.",
            )
        return CellDecision(
            policy_name=self.name,
            slice_decisions=decisions,
            inspection_actions=sum(1 for slice_demand in cell.slices if slice_demand.suspicious and slice_demand.anomaly_score > 0.95),
            controller_actions_used=24 + len(cell.slices),
            note="High utilization policy; tends to over-admit suspicious and low-priority demand.",
        )


class SecurityOnlyPolicy(Policy):
    def __init__(self) -> None:
        super().__init__(name="security_only")

    def decide(self, cell: CellDemand, budget: ResourceBudget, controller_queue: int) -> CellDecision:
        benign_slices = [slice_demand for slice_demand in cell.slices if slice_demand.anomaly_score < 0.72]
        suspicious_slices = [slice_demand for slice_demand in cell.slices if slice_demand.anomaly_score >= 0.72]
        prb_alloc = _normalize_weights(
            [(slice_demand, slice_demand.utility + (0.4 if slice_demand.mission_critical else 0.0)) for slice_demand in benign_slices],
            budget.prbs_per_cell,
        )
        gpu_alloc = _normalize_weights(
            [(slice_demand, slice_demand.requested_gpu + (8.0 if slice_demand.mission_critical else 2.0)) for slice_demand in benign_slices],
            budget.gpu_per_cell,
        )
        decisions: dict[str, SliceDecision] = {}
        for slice_demand in benign_slices:
            decisions[slice_demand.slice_id] = SliceDecision(
                slice_id=slice_demand.slice_id,
                allocated_prbs=min(slice_demand.requested_prbs, prb_alloc[slice_demand.slice_id]),
                allocated_gpu=min(slice_demand.requested_gpu, gpu_alloc[slice_demand.slice_id]),
                inspected=False,
                note="Benign slice passed security gate.",
            )
        for slice_demand in suspicious_slices:
            decisions[slice_demand.slice_id] = SliceDecision(
                slice_id=slice_demand.slice_id,
                allocated_prbs=0.0,
                allocated_gpu=0.0,
                isolated=True,
                inspected=True,
                note="Isolated due to anomaly score.",
            )
        return CellDecision(
            policy_name=self.name,
            slice_decisions=decisions,
            inspection_actions=max(4, len(suspicious_slices) * 6),
            controller_actions_used=32 + len(cell.slices) + len(suspicious_slices) * 4,
            note="Aggressive isolation; can protect control plane but may over-react and starve AI demand.",
        )


class FailureAwarePolicy(Policy):
    def __init__(self) -> None:
        super().__init__(name="failure_aware")

    def decide(self, cell: CellDemand, budget: ResourceBudget, controller_queue: int) -> CellDecision:
        attack_pressure = max((slice_demand.anomaly_score for slice_demand in cell.slices), default=0.0)
        overload = controller_queue > 70 or cell.control_events > 90
        decisions: dict[str, SliceDecision] = {}

        mission_slices = [slice_demand for slice_demand in cell.slices if slice_demand.mission_critical]
        ai_slices = [slice_demand for slice_demand in cell.slices if slice_demand.kind == "ai_edge"]
        remaining = [slice_demand for slice_demand in cell.slices if slice_demand not in mission_slices + ai_slices]

        reserved_prbs = budget.prbs_per_cell * (0.46 if overload or attack_pressure > 0.85 else 0.34)
        reserved_gpu = budget.gpu_per_cell * (0.50 if overload else 0.38)

        mission_prbs = _normalize_weights([(slice_demand, 1.0 + slice_demand.utility) for slice_demand in mission_slices], reserved_prbs)
        mission_gpu = _normalize_weights([(slice_demand, 8.0 + slice_demand.requested_gpu) for slice_demand in mission_slices], reserved_gpu)

        used_prbs = 0.0
        used_gpu = 0.0
        for slice_demand in mission_slices:
            prbs = min(slice_demand.requested_prbs, mission_prbs.get(slice_demand.slice_id, 0.0))
            gpu = min(slice_demand.requested_gpu, mission_gpu.get(slice_demand.slice_id, 0.0))
            decisions[slice_demand.slice_id] = SliceDecision(
                slice_id=slice_demand.slice_id,
                allocated_prbs=prbs,
                allocated_gpu=gpu,
                inspected=slice_demand.anomaly_score > 0.35,
                note="Mission-critical reserve.",
            )
            used_prbs += prbs
            used_gpu += gpu

        remaining_prbs = max(0.0, budget.prbs_per_cell - used_prbs)
        remaining_gpu = max(0.0, budget.gpu_per_cell - used_gpu)

        filtered_remaining: list[SliceDemand] = []
        for slice_demand in ai_slices + remaining:
            if slice_demand.suspicious and (overload or slice_demand.anomaly_score > 0.88):
                decisions[slice_demand.slice_id] = SliceDecision(
                    slice_id=slice_demand.slice_id,
                    allocated_prbs=0.0,
                    allocated_gpu=0.0,
                    isolated=True,
                    inspected=True,
                    note="Suspicious slice isolated under control-plane stress.",
                )
                continue
            filtered_remaining.append(slice_demand)

        prb_weights: list[tuple[SliceDemand, float]] = []
        gpu_weights: list[tuple[SliceDemand, float]] = []
        for slice_demand in filtered_remaining:
            prb_weight = slice_demand.utility
            gpu_weight = slice_demand.requested_gpu + 1.0
            if slice_demand.kind == "ai_edge":
                prb_weight += 0.45
                gpu_weight += 4.5
            if slice_demand.kind == "background" and overload:
                prb_weight *= 0.35
                gpu_weight *= 0.25
            if slice_demand.kind == "broadband" and attack_pressure > 0.80:
                prb_weight *= 0.75
            prb_weights.append((slice_demand, prb_weight))
            gpu_weights.append((slice_demand, gpu_weight))

        prb_alloc = _normalize_weights(prb_weights, remaining_prbs)
        gpu_alloc = _normalize_weights(gpu_weights, remaining_gpu)
        for slice_demand in filtered_remaining:
            allocated_prbs = min(slice_demand.requested_prbs, prb_alloc.get(slice_demand.slice_id, 0.0))
            allocated_gpu = min(slice_demand.requested_gpu, gpu_alloc.get(slice_demand.slice_id, 0.0))
            degraded = False
            throttled = False
            if overload and slice_demand.kind == "background":
                allocated_prbs *= 0.45
                allocated_gpu *= 0.25
                degraded = True
                throttled = True
            if attack_pressure > 0.80 and slice_demand.kind == "broadband":
                allocated_prbs *= 0.82
                throttled = True
            decisions[slice_demand.slice_id] = SliceDecision(
                slice_id=slice_demand.slice_id,
                allocated_prbs=allocated_prbs,
                allocated_gpu=allocated_gpu,
                inspected=slice_demand.anomaly_score > 0.48,
                degraded=degraded,
                throttled=throttled,
                note="Adaptive allocation with control-plane protection.",
            )

        inspection_actions = sum(2 if decision.inspected else 0 for decision in decisions.values())
        controller_actions_used = 12 + len(cell.slices) + inspection_actions + (3 if overload else 0)
        return CellDecision(
            policy_name=self.name,
            slice_decisions=decisions,
            inspection_actions=inspection_actions,
            controller_actions_used=controller_actions_used,
            note="Co-optimizes mission-critical continuity, attack isolation, and edge AI admission.",
        )


class AegisMixerPolicy(Policy):
    def __init__(self) -> None:
        super().__init__(name="aegis_mixer")

    def decide(self, cell: CellDemand, budget: ResourceBudget, controller_queue: int) -> CellDecision:
        return build_aegis_mixer_decision(
            cell=cell,
            budget=budget,
            controller_queue=controller_queue,
        )
