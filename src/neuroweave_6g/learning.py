from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .aegis_mixer import (
    ActionCandidate,
    build_aegis_mixer_decision,
    build_candidates_for_cell,
    build_fallback_candidate,
    candidate_feature_row,
    candidate_to_slice_decision,
)
from .policies import available_policies, build_policy
from .scenario import build_scenario, list_scenarios
from .simulator import (
    evaluate_cell_decisions,
    project_controller_latency_ms,
    simulate_decision_builder_on_scenario,
    summarize_observations,
)
from .types import CellDemand, ResourceBudget, ScenarioResult, SliceDecision


@dataclass(slots=True)
class LinearRewardModel:
    feature_names: list[str]
    means: list[float]
    stds: list[float]
    weights: list[float]
    bias: float
    metadata: dict[str, object]

    def predict_row(self, row: dict[str, float]) -> float:
        score = self.bias
        for index, feature_name in enumerate(self.feature_names):
            value = row[feature_name]
            normalized = (value - self.means[index]) / self.stds[index]
            score += self.weights[index] * normalized
        return score

    def scorer(self):
        def _score(candidate: ActionCandidate, cell: CellDemand, budget: ResourceBudget, controller_queue: int) -> float:
            return self.predict_row(
                candidate_feature_row(
                    candidate=candidate,
                    cell=cell,
                    budget=budget,
                    controller_queue=controller_queue,
                )
            )

        return _score

    def to_json_dict(self) -> dict[str, object]:
        return {
            "feature_names": self.feature_names,
            "means": self.means,
            "stds": self.stds,
            "weights": self.weights,
            "bias": self.bias,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> "LinearRewardModel":
        return cls(
            feature_names=[str(item) for item in payload["feature_names"]],
            means=[float(item) for item in payload["means"]],
            stds=[float(item) for item in payload["stds"]],
            weights=[float(item) for item in payload["weights"]],
            bias=float(payload["bias"]),
            metadata=dict(payload.get("metadata", {})),
        )


def export_learning_traces(
    *,
    output_path: str | Path,
    steps: int = 18,
    seeds: list[int] | None = None,
    scenario_names: list[str] | None = None,
    source_policy_names: list[str] | None = None,
    budget: ResourceBudget | None = None,
) -> Path:
    effective_budget = budget or ResourceBudget()
    seed_values = seeds or [7, 11, 19]
    scenarios = scenario_names or list_scenarios()
    source_policies = source_policy_names or available_policies()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for seed in seed_values:
        for scenario_name in scenarios:
            scenario = build_scenario(scenario_name, steps=steps, seed=seed)
            for source_policy_name in source_policies:
                source_policy = build_policy(source_policy_name)
                controller_queue = 0
                for step in scenario.steps:
                    total_incoming_events = sum(cell.control_events for cell in step.cells)
                    controller_actions_spent = 0
                    mitigation_credit = 0

                    for cell in step.cells:
                        for candidate in build_candidates_for_cell(
                            cell=cell,
                            budget=effective_budget,
                            controller_queue=controller_queue,
                        ):
                            reward_payload = score_candidate_counterfactual(
                                scenario_name=scenario_name,
                                cell=cell,
                                candidate=candidate,
                                budget=effective_budget,
                                controller_queue=controller_queue,
                            )
                            rows.append(
                                {
                                    "scenario": scenario_name,
                                    "seed": seed,
                                    "source_policy": source_policy_name,
                                    "step_idx": step.step_idx,
                                    "cell_id": cell.cell_id,
                                    "slice_id": candidate.slice_id,
                                    "action_name": candidate.action_name,
                                    **candidate_feature_row(
                                        candidate=candidate,
                                        cell=cell,
                                        budget=effective_budget,
                                        controller_queue=controller_queue,
                                    ),
                                    **reward_payload,
                                }
                            )

                        source_decision = source_policy.decide(cell, effective_budget, controller_queue)
                        controller_actions_spent += source_decision.controller_actions_used
                        for slice_demand in cell.slices:
                            decision = source_decision.slice_decisions[slice_demand.slice_id]
                            if slice_demand.suspicious and decision.isolated:
                                mitigation_credit += 18
                            elif slice_demand.suspicious and decision.inspected:
                                mitigation_credit += 8

                    net_incoming_events = max(0, total_incoming_events - mitigation_credit)
                    effective_controller_capacity = max(
                        50,
                        effective_budget.controller_actions_per_step - max(0, controller_actions_spent - 48),
                    )
                    controller_queue = max(0, controller_queue + net_incoming_events - effective_controller_capacity)

    if not rows:
        output.write_text("", encoding="utf-8")
        return output

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output


def score_candidate_counterfactual(
    *,
    scenario_name: str,
    cell: CellDemand,
    candidate: ActionCandidate,
    budget: ResourceBudget,
    controller_queue: int,
) -> dict[str, float]:
    attack_pressure = max((slice_demand.anomaly_score for slice_demand in cell.slices), default=0.0)
    overload = controller_queue > 70 or cell.control_events > 90
    decision_map = _build_counterfactual_decision_map(
        cell=cell,
        candidate=candidate,
        overload=overload,
        attack_pressure=attack_pressure,
        budget=budget,
    )
    controller_actions_used = 8 + candidate.controller_cost + sum(
        2 if decision.inspected else 0 for decision in decision_map.values()
    )
    mitigation_credit = 0
    for slice_demand in cell.slices:
        decision = decision_map[slice_demand.slice_id]
        if slice_demand.suspicious and decision.isolated:
            mitigation_credit += 18
        elif slice_demand.suspicious and decision.inspected:
            mitigation_credit += 8
    net_incoming_events = max(0, cell.control_events - mitigation_credit)
    effective_controller_capacity = max(
        50,
        budget.controller_actions_per_step - max(0, controller_actions_used - 48),
    )
    projected_queue = max(0, controller_queue + net_incoming_events - effective_controller_capacity)
    controller_latency_ms = project_controller_latency_ms(
        controller_queue=projected_queue,
        net_incoming_events=net_incoming_events,
    )
    observations = evaluate_cell_decisions(
        scenario_name=scenario_name,
        policy_name="counterfactual",
        step_idx=0,
        cell=cell,
        controller_latency_ms=controller_latency_ms,
        decision_map=decision_map,
    )
    metrics = summarize_observations(observations)
    reward = (
        4.0 * metrics["critical_slice_survival_rate"]
        + 2.0 * metrics["overall_sla_rate"]
        - 1.5 * metrics["ai_deadline_miss_rate"]
        - 2.5 * metrics["attack_leakage_rate"]
        - 0.02 * controller_latency_ms
        - 0.002 * projected_queue
        + 0.5 * metrics["mean_service_ratio"]
    )
    return {
        **metrics,
        "projected_controller_queue": float(projected_queue),
        "projected_controller_latency_ms": round(controller_latency_ms, 4),
        "reward": round(reward, 6),
    }


def train_linear_reward_model(
    *,
    trace_path: str | Path,
    model_output_path: str | Path,
    epochs: int = 220,
    learning_rate: float = 0.03,
) -> tuple[Path, dict[str, float]]:
    trace_rows = _read_trace_rows(Path(trace_path))
    if not trace_rows:
        raise ValueError(f"No trace rows found in {trace_path}")

    excluded = {
        "scenario",
        "seed",
        "source_policy",
        "step_idx",
        "cell_id",
        "slice_id",
        "action_name",
        "critical_slice_survival_rate",
        "overall_sla_rate",
        "ai_deadline_miss_rate",
        "attack_leakage_rate",
        "false_positive_isolation_rate",
        "mean_service_ratio",
        "projected_controller_queue",
        "projected_controller_latency_ms",
        "reward",
    }
    feature_names = [key for key in trace_rows[0].keys() if key not in excluded]
    means, stds = _fit_normalization(trace_rows, feature_names)
    weights = [0.0 for _ in feature_names]
    bias = 0.0
    targets = [float(row["reward"]) for row in trace_rows]

    for _ in range(epochs):
        grad_w = [0.0 for _ in feature_names]
        grad_b = 0.0
        sample_count = len(trace_rows)
        for row, target in zip(trace_rows, targets):
            prediction = bias
            normalized_values = []
            for index, feature_name in enumerate(feature_names):
                normalized = (float(row[feature_name]) - means[index]) / stds[index]
                normalized_values.append(normalized)
                prediction += weights[index] * normalized
            error = prediction - target
            grad_b += error
            for index, normalized in enumerate(normalized_values):
                grad_w[index] += error * normalized
        scale = 2.0 / max(1, len(trace_rows))
        bias -= learning_rate * scale * grad_b
        for index in range(len(weights)):
            weights[index] -= learning_rate * scale * grad_w[index]

    model = LinearRewardModel(
        feature_names=feature_names,
        means=means,
        stds=stds,
        weights=weights,
        bias=bias,
        metadata={
            "epochs": epochs,
            "learning_rate": learning_rate,
            "trace_rows": len(trace_rows),
        },
    )
    metrics = _evaluate_model(model, trace_rows)
    output = Path(model_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model.to_json_dict(), indent=2), encoding="utf-8")
    return output, metrics


def load_linear_reward_model(model_path: str | Path) -> LinearRewardModel:
    payload = json.loads(Path(model_path).read_text(encoding="utf-8"))
    return LinearRewardModel.from_json_dict(payload)


def simulate_learned_aegis_mixer_on_scenario(
    *,
    scenario_name: str,
    model_path: str | Path,
    steps: int = 18,
    seed: int = 7,
    budget: ResourceBudget | None = None,
) -> ScenarioResult:
    model = load_linear_reward_model(model_path)
    return simulate_decision_builder_on_scenario(
        scenario_name=scenario_name,
        policy_name="aegis_mixer_learned",
        decision_builder=lambda cell, effective_budget, controller_queue: build_aegis_mixer_decision(
            cell=cell,
            budget=effective_budget,
            controller_queue=controller_queue,
            score_fn=model.scorer(),
            policy_name="aegis_mixer_learned",
        ),
        steps=steps,
        seed=seed,
        budget=budget,
    )


def benchmark_with_learned_policy(
    *,
    model_path: str | Path,
    steps: int = 18,
    seed: int = 7,
    scenario_names: list[str] | None = None,
    budget: ResourceBudget | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scenarios = scenario_names or list_scenarios()
    base_policies = ["static_qos", "throughput_first", "security_only", "failure_aware", "aegis_mixer"]
    for scenario_name in scenarios:
        for policy_name in base_policies:
            result = simulate_decision_builder_on_scenario(
                scenario_name=scenario_name,
                policy_name=policy_name,
                decision_builder=lambda cell, effective_budget, controller_queue, policy=build_policy(policy_name): policy.decide(
                    cell, effective_budget, controller_queue
                ),
                steps=steps,
                seed=seed,
                budget=budget,
            )
            rows.append({"scenario": scenario_name, "policy": policy_name, **result.summary_metrics})
        learned = simulate_learned_aegis_mixer_on_scenario(
            scenario_name=scenario_name,
            model_path=model_path,
            steps=steps,
            seed=seed,
            budget=budget,
        )
        rows.append({"scenario": scenario_name, "policy": learned.policy_name, **learned.summary_metrics})
    return rows


def _build_counterfactual_decision_map(
    *,
    cell: CellDemand,
    candidate: ActionCandidate,
    overload: bool,
    attack_pressure: float,
    budget: ResourceBudget,
) -> dict[str, SliceDecision]:
    remaining_prbs = max(0.0, budget.prbs_per_cell - candidate.allocated_prbs)
    remaining_gpu = max(0.0, budget.gpu_per_cell - candidate.allocated_gpu)
    decisions: dict[str, SliceDecision] = {candidate.slice_id: candidate_to_slice_decision(candidate)}
    ordered_slices = sorted(
        [slice_demand for slice_demand in cell.slices if slice_demand.slice_id != candidate.slice_id],
        key=lambda slice_demand: (
            0 if slice_demand.mission_critical else 1 if slice_demand.kind == "ai_edge" else 2 if slice_demand.suspicious else 3
        ),
    )
    for slice_demand in ordered_slices:
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
    return decisions


def _read_trace_rows(path: Path) -> list[dict[str, float | str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _fit_normalization(rows: list[dict[str, float | str]], feature_names: list[str]) -> tuple[list[float], list[float]]:
    means: list[float] = []
    stds: list[float] = []
    for feature_name in feature_names:
        values = [float(row[feature_name]) for row in rows]
        mean_value = sum(values) / max(1, len(values))
        variance = sum((value - mean_value) ** 2 for value in values) / max(1, len(values))
        means.append(mean_value)
        stds.append(max(variance**0.5, 1e-6))
    return means, stds


def _evaluate_model(model: LinearRewardModel, rows: list[dict[str, float | str]]) -> dict[str, float]:
    errors = []
    predictions = []
    targets = []
    for row in rows:
        target = float(row["reward"])
        prediction = model.predict_row({key: float(value) for key, value in row.items() if key in model.feature_names})
        errors.append((prediction - target) ** 2)
        predictions.append(prediction)
        targets.append(target)
    mse = sum(errors) / max(1, len(errors))
    target_mean = sum(targets) / max(1, len(targets))
    total_var = sum((target - target_mean) ** 2 for target in targets)
    r2 = 1.0 - (sum(errors) / total_var) if total_var > 0 else 0.0
    return {
        "mse": round(mse, 6),
        "r2": round(r2, 6),
        "row_count": float(len(rows)),
    }
