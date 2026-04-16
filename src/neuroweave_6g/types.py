from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SliceKind = Literal["mission_critical", "ai_edge", "broadband", "background", "suspicious"]


@dataclass(slots=True)
class SliceDemand:
    slice_id: str
    kind: SliceKind
    requested_prbs: float
    requested_gpu: float
    latency_budget_ms: float
    anomaly_score: float
    utility: float
    encrypted: bool = False

    @property
    def mission_critical(self) -> bool:
        return self.kind == "mission_critical"

    @property
    def ai_enabled(self) -> bool:
        return self.kind in {"mission_critical", "ai_edge"}

    @property
    def suspicious(self) -> bool:
        return self.kind == "suspicious" or self.anomaly_score >= 0.8


@dataclass(slots=True)
class CellDemand:
    cell_id: str
    control_events: int
    backhaul_pressure: float
    mobility_pressure: float
    slices: list[SliceDemand] = field(default_factory=list)


@dataclass(slots=True)
class ScenarioStep:
    step_idx: int
    cells: list[CellDemand]


@dataclass(slots=True)
class Scenario:
    name: str
    description: str
    steps: list[ScenarioStep]
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ResourceBudget:
    prbs_per_cell: float = 100.0
    gpu_per_cell: float = 70.0
    controller_actions_per_step: int = 220


@dataclass(slots=True)
class SliceDecision:
    slice_id: str
    allocated_prbs: float
    allocated_gpu: float
    isolated: bool = False
    throttled: bool = False
    inspected: bool = False
    degraded: bool = False
    note: str = ""


@dataclass(slots=True)
class CellDecision:
    policy_name: str
    slice_decisions: dict[str, SliceDecision]
    inspection_actions: int
    controller_actions_used: int
    note: str = ""


@dataclass(slots=True)
class SliceObservation:
    step_idx: int
    scenario_name: str
    policy_name: str
    cell_id: str
    slice_id: str
    kind: SliceKind
    latency_ms: float
    service_ratio: float
    sla_met: bool
    isolated: bool
    suspicious: bool
    mission_critical: bool
    ai_enabled: bool
    deadline_missed: bool


@dataclass(slots=True)
class StepSummary:
    step_idx: int
    scenario_name: str
    policy_name: str
    controller_queue: int
    controller_latency_ms: float
    attack_leakage_rate: float
    critical_slice_survival_rate: float
    ai_deadline_miss_rate: float
    overall_sla_rate: float


@dataclass(slots=True)
class ScenarioResult:
    scenario_name: str
    policy_name: str
    description: str
    notes: list[str]
    step_summaries: list[StepSummary]
    slice_observations: list[SliceObservation]
    summary_metrics: dict[str, float]
