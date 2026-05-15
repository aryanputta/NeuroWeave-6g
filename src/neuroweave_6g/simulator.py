from __future__ import annotations

import math
from statistics import mean
from typing import Callable
from collections import deque
from dataclasses import replace

from .policies import build_policy
from .scenario import build_scenario
from .types import CellDecision, CellDemand, ResourceBudget, ScenarioResult, SimulationConfig, SliceObservation, StepSummary

DecisionBuilder = Callable[[CellDemand, ResourceBudget, int], CellDecision]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((percentile / 100.0) * len(ordered)) - 1))
    return ordered[index]


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def evaluate_cell_decisions(
    *,
    scenario_name: str,
    policy_name: str,
    step_idx: int,
    cell: CellDemand,
    controller_latency_ms: float,
    decision_map,
) -> list[SliceObservation]:
    observations: list[SliceObservation] = []
    for slice_demand in cell.slices:
        decision = decision_map[slice_demand.slice_id]
        prb_ratio = min(1.0, decision.allocated_prbs / max(slice_demand.requested_prbs, 1e-6))
        gpu_ratio = 1.0
        if slice_demand.ai_enabled:
            gpu_ratio = min(1.0, decision.allocated_gpu / max(slice_demand.requested_gpu, 1e-6))

        service_ratio = 0.0 if decision.isolated else min(prb_ratio, gpu_ratio)
        isolation_penalty = 18.0 if decision.isolated else 0.0
        deficit_penalty = (1.0 - service_ratio) * (4.5 if slice_demand.mission_critical else 8.5)
        anomaly_penalty = max(0.0, slice_demand.anomaly_score - (0.74 if decision.inspected else 0.36)) * 6.5
        congestion_penalty = cell.backhaul_pressure * 6.0 + cell.mobility_pressure * 4.0
        latency_ms = (
            1.4
            + controller_latency_ms * 0.16
            + congestion_penalty
            + deficit_penalty
            + anomaly_penalty
            + isolation_penalty
        )
        sla_met = (not decision.isolated) and service_ratio >= (0.84 if slice_demand.mission_critical else 0.64) and latency_ms <= slice_demand.latency_budget_ms
        deadline_missed = slice_demand.ai_enabled and (latency_ms > slice_demand.latency_budget_ms or service_ratio < 0.75)
        observations.append(
            SliceObservation(
                step_idx=step_idx,
                scenario_name=scenario_name,
                policy_name=policy_name,
                cell_id=cell.cell_id,
                slice_id=slice_demand.slice_id,
                kind=slice_demand.kind,
                latency_ms=round(latency_ms, 4),
                service_ratio=round(service_ratio, 4),
                sla_met=sla_met,
                isolated=decision.isolated,
                suspicious=slice_demand.suspicious,
                mission_critical=slice_demand.mission_critical,
                ai_enabled=slice_demand.ai_enabled,
                deadline_missed=deadline_missed,
            )
        )
    return observations


def summarize_observations(observations: list[SliceObservation]) -> dict[str, float]:
    critical_obs = [obs for obs in observations if obs.mission_critical]
    ai_obs = [obs for obs in observations if obs.ai_enabled]
    suspicious_obs = [obs for obs in observations if obs.suspicious]
    isolated_benign = [obs for obs in observations if obs.isolated and not obs.suspicious]
    return {
        "critical_slice_survival_rate": round(sum(1 for obs in critical_obs if obs.sla_met) / max(1, len(critical_obs)), 4),
        "overall_sla_rate": round(sum(1 for obs in observations if obs.sla_met) / max(1, len(observations)), 4),
        "ai_deadline_miss_rate": round(sum(1 for obs in ai_obs if obs.deadline_missed) / max(1, len(ai_obs)), 4),
        "attack_leakage_rate": round(
            sum(1 for obs in suspicious_obs if (not obs.isolated and obs.service_ratio > 0.10)) / max(1, len(suspicious_obs)),
            4,
        ),
        "false_positive_isolation_rate": round(len(isolated_benign) / max(1, len(observations) - len(suspicious_obs)), 4),
        "mean_service_ratio": round(_safe_mean([obs.service_ratio for obs in observations]), 4),
    }


def project_controller_latency_ms(
    *,
    controller_queue: int,
    net_incoming_events: int,
) -> float:
    return 2.8 + (controller_queue * 0.028) + (net_incoming_events * 0.012)


def simulate_policy_on_scenario(
    *,
    scenario_name: str,
    policy_name: str,
    steps: int = 18,
    seed: int = 7,
    budget: ResourceBudget | None = None,
) -> ScenarioResult:
    scenario = build_scenario(scenario_name, steps=steps, seed=seed)
    policy = build_policy(policy_name)
    return simulate_decision_builder_on_scenario(
        scenario_name=scenario_name,
        policy_name=policy.name,
        decision_builder=lambda cell, effective_budget, controller_queue: policy.decide(cell, effective_budget, controller_queue),
        steps=steps,
        seed=seed,
        budget=budget,
        scenario=scenario,
        config=SimulationConfig(),
    )


def simulate_decision_builder_on_scenario(
    *,
    scenario_name: str,
    policy_name: str,
    decision_builder: DecisionBuilder,
    steps: int = 18,
    seed: int = 7,
    budget: ResourceBudget | None = None,
    scenario=None,
    config: SimulationConfig | None = None,
) -> ScenarioResult:
    effective_scenario = scenario or build_scenario(scenario_name, steps=steps, seed=seed)
    effective_budget = budget or ResourceBudget()
    effective_config = config or SimulationConfig()

    controller_queue = 0
    step_summaries: list[StepSummary] = []
    slice_observations: list[SliceObservation] = []
    controller_latencies: list[float] = []
    mitigation_queue = deque([0 for _ in range(effective_config.mitigation_delay_steps + 1)], maxlen=effective_config.mitigation_delay_steps + 1)
    history_steps: list[list[CellDemand]] = []

    for step in effective_scenario.steps:
        total_incoming_events = sum(cell.control_events for cell in step.cells)
        cell_observations: list[SliceObservation] = []
        controller_actions_spent = 0
        attack_leakage = 0
        suspicious_total = 0
        mitigation_credit = mitigation_queue.popleft() if effective_config.mitigation_delay_steps > 0 else 0
        pending_mitigation_credit = 0

        per_cell_decisions = []
        observed_cells = _build_observed_cells(
            live_cells=step.cells,
            history_steps=history_steps,
            stale_telemetry_steps=effective_config.stale_telemetry_steps,
        )
        for live_cell, observed_cell in zip(step.cells, observed_cells):
            cell_decision = decision_builder(observed_cell, effective_budget, controller_queue)
            controller_actions_spent += cell_decision.controller_actions_used
            per_cell_decisions.append((live_cell, cell_decision))
            for slice_demand in live_cell.slices:
                decision = cell_decision.slice_decisions[slice_demand.slice_id]
                if slice_demand.suspicious and decision.isolated:
                    pending_credit = 18
                elif slice_demand.suspicious and decision.inspected:
                    pending_credit = 8
                else:
                    pending_credit = 0
                if effective_config.mitigation_delay_steps > 0:
                    pending_mitigation_credit += pending_credit
                else:
                    mitigation_credit += pending_credit

        net_incoming_events = max(0, total_incoming_events - mitigation_credit)
        effective_controller_capacity = max(
            50,
            effective_budget.controller_actions_per_step - max(0, controller_actions_spent - 48),
        )
        controller_queue = max(0, controller_queue + net_incoming_events - effective_controller_capacity)
        controller_latency_ms = project_controller_latency_ms(
            controller_queue=controller_queue,
            net_incoming_events=net_incoming_events,
        )
        controller_latencies.append(controller_latency_ms)

        for cell, cell_decision in per_cell_decisions:
            decision_map = cell_decision.slice_decisions
            observations = evaluate_cell_decisions(
                scenario_name=effective_scenario.name,
                policy_name=policy_name,
                step_idx=step.step_idx,
                cell=cell,
                controller_latency_ms=controller_latency_ms,
                decision_map=decision_map,
            )
            cell_observations.extend(observations)
            for observation in observations:
                if observation.suspicious:
                    suspicious_total += 1
                    if not observation.isolated and observation.service_ratio > 0.10:
                        attack_leakage += 1

        slice_observations.extend(cell_observations)
        critical = [obs for obs in cell_observations if obs.mission_critical]
        ai_slices = [obs for obs in cell_observations if obs.ai_enabled]
        step_summaries.append(
            StepSummary(
                step_idx=step.step_idx,
                scenario_name=effective_scenario.name,
                policy_name=policy_name,
                controller_queue=controller_queue,
                controller_latency_ms=round(controller_latency_ms, 4),
                attack_leakage_rate=round(attack_leakage / max(1, suspicious_total), 4),
                critical_slice_survival_rate=round(sum(1 for obs in critical if obs.sla_met) / max(1, len(critical)), 4),
                ai_deadline_miss_rate=round(sum(1 for obs in ai_slices if obs.deadline_missed) / max(1, len(ai_slices)), 4),
                overall_sla_rate=round(sum(1 for obs in cell_observations if obs.sla_met) / max(1, len(cell_observations)), 4),
            )
        )
        if effective_config.mitigation_delay_steps > 0:
            mitigation_queue.append(pending_mitigation_credit)
        history_steps.append([replace(cell, slices=list(cell.slices)) for cell in step.cells])

    summary_metrics = summarize_observations(slice_observations)
    summary_metrics.update(
        {
            "controller_p95_latency_ms": round(_percentile(controller_latencies, 95), 4),
            "controller_queue_peak": float(max(summary.controller_queue for summary in step_summaries)),
        }
    )
    return ScenarioResult(
        scenario_name=effective_scenario.name,
        policy_name=policy_name,
        description=effective_scenario.description,
        notes=effective_scenario.notes,
        step_summaries=step_summaries,
        slice_observations=slice_observations,
        summary_metrics=summary_metrics,
    )


def _build_observed_cells(
    *,
    live_cells: list[CellDemand],
    history_steps: list[list[CellDemand]],
    stale_telemetry_steps: int,
) -> list[CellDemand]:
    if stale_telemetry_steps <= 0 or len(history_steps) < stale_telemetry_steps:
        return live_cells
    stale_cells = history_steps[-stale_telemetry_steps]
    stale_map = {cell.cell_id: cell for cell in stale_cells}
    observed_cells: list[CellDemand] = []
    for live_cell in live_cells:
        prior = stale_map.get(live_cell.cell_id)
        if prior is None:
            observed_cells.append(live_cell)
            continue
        observed_cells.append(
            replace(
                live_cell,
                control_events=prior.control_events,
                backhaul_pressure=prior.backhaul_pressure,
                mobility_pressure=prior.mobility_pressure,
                slices=list(prior.slices),
            )
        )
    return observed_cells
