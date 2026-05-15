from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Protocol
from collections import deque

from .aegis_mixer import (
    ActionCandidate,
    build_aegis_mixer_decision,
    build_candidates_for_cell,
    build_fallback_candidate,
    candidate_feature_row,
    candidate_to_slice_decision,
)
from .policies import available_policies, build_policy
from .scenario import build_scenario, list_scenarios, scale_scenario
from .simulator import (
    evaluate_cell_decisions,
    project_controller_latency_ms,
    simulate_decision_builder_on_scenario,
    summarize_observations,
)
from .types import CellDemand, ResourceBudget, ScenarioResult, SimulationConfig, SliceDecision


class RewardModel(Protocol):
    feature_names: list[str]
    metadata: dict[str, object]

    def predict_row(self, row: dict[str, float]) -> float:
        ...

    def scorer(self):
        ...

    def to_json_dict(self) -> dict[str, object]:
        ...


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
            normalized = (row[feature_name] - self.means[index]) / self.stds[index]
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
            "model_type": "linear",
            "feature_names": self.feature_names,
            "means": self.means,
            "stds": self.stds,
            "weights": self.weights,
            "bias": self.bias,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class MLPRewardModel:
    feature_names: list[str]
    means: list[float]
    stds: list[float]
    hidden_weights: list[list[float]]
    hidden_biases: list[float]
    output_weights: list[float]
    output_bias: float
    metadata: dict[str, object]

    def predict_row(self, row: dict[str, float]) -> float:
        normalized = [(row[name] - self.means[index]) / self.stds[index] for index, name in enumerate(self.feature_names)]
        hidden_values: list[float] = []
        for neuron_weights, neuron_bias in zip(self.hidden_weights, self.hidden_biases):
            activation = neuron_bias
            for weight, value in zip(neuron_weights, normalized):
                activation += weight * value
            hidden_values.append(math.tanh(activation))
        return self.output_bias + sum(weight * value for weight, value in zip(self.output_weights, hidden_values))

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
            "model_type": "mlp",
            "feature_names": self.feature_names,
            "means": self.means,
            "stds": self.stds,
            "hidden_weights": self.hidden_weights,
            "hidden_biases": self.hidden_biases,
            "output_weights": self.output_weights,
            "output_bias": self.output_bias,
            "metadata": self.metadata,
        }


def export_learning_traces(
    *,
    output_path: str | Path,
    steps: int = 18,
    seeds: list[int] | None = None,
    scenario_names: list[str] | None = None,
    source_policy_names: list[str] | None = None,
    budget: ResourceBudget | None = None,
    config: SimulationConfig | None = None,
) -> Path:
    effective_budget = budget or ResourceBudget()
    effective_config = config or SimulationConfig()
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
                mitigation_delay = deque([0 for _ in range(effective_config.mitigation_delay_steps + 1)], maxlen=effective_config.mitigation_delay_steps + 1)
                history_steps: list[list[CellDemand]] = []
                for step in scenario.steps:
                    total_incoming_events = sum(cell.control_events for cell in step.cells)
                    controller_actions_spent = 0
                    mitigation_credit = mitigation_delay.popleft() if effective_config.mitigation_delay_steps > 0 else 0
                    pending_mitigation_credit = 0
                    live_cells = step.cells
                    observed_cells = _build_trace_observed_cells(
                        live_cells=live_cells,
                        history_steps=history_steps,
                        stale_telemetry_steps=effective_config.stale_telemetry_steps,
                    )

                    for live_cell, observed_cell in zip(live_cells, observed_cells):
                        for candidate in build_candidates_for_cell(
                            cell=observed_cell,
                            budget=effective_budget,
                            controller_queue=controller_queue,
                        ):
                            reward_payload = score_candidate_counterfactual(
                                scenario_name=scenario_name,
                                live_cell=live_cell,
                                observed_cell=observed_cell,
                                candidate=candidate,
                                budget=effective_budget,
                                controller_queue=controller_queue,
                                config=effective_config,
                            )
                            rows.append(
                                {
                                    "scenario": scenario_name,
                                    "seed": seed,
                                    "source_policy": source_policy_name,
                                    "step_idx": step.step_idx,
                                    "cell_id": live_cell.cell_id,
                                    "slice_id": candidate.slice_id,
                                    "action_name": candidate.action_name,
                                    "stale_telemetry_steps": float(effective_config.stale_telemetry_steps),
                                    "mitigation_delay_steps": float(effective_config.mitigation_delay_steps),
                                    **candidate_feature_row(
                                        candidate=candidate,
                                        cell=observed_cell,
                                        budget=effective_budget,
                                        controller_queue=controller_queue,
                                    ),
                                    **reward_payload,
                                }
                            )

                        source_decision = source_policy.decide(observed_cell, effective_budget, controller_queue)
                        controller_actions_spent += source_decision.controller_actions_used
                        pending_credit = 0
                        for slice_demand in live_cell.slices:
                            decision = source_decision.slice_decisions[slice_demand.slice_id]
                            if slice_demand.suspicious and decision.isolated:
                                pending_credit += 18
                            elif slice_demand.suspicious and decision.inspected:
                                pending_credit += 8
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
                    if effective_config.mitigation_delay_steps > 0:
                        mitigation_delay.append(pending_mitigation_credit)
                    history_steps.append(live_cells)

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
    live_cell: CellDemand,
    observed_cell: CellDemand,
    candidate: ActionCandidate,
    budget: ResourceBudget,
    controller_queue: int,
    config: SimulationConfig | None = None,
) -> dict[str, float]:
    effective_config = config or SimulationConfig()
    attack_pressure = max((slice_demand.anomaly_score for slice_demand in observed_cell.slices), default=0.0)
    overload = controller_queue > 70 or observed_cell.control_events > 90
    decision_map = _build_counterfactual_decision_map(
        live_cell=live_cell,
        observed_cell=observed_cell,
        candidate=candidate,
        overload=overload,
        attack_pressure=attack_pressure,
        budget=budget,
    )
    controller_actions_used = 8 + candidate.controller_cost + sum(2 if decision.inspected else 0 for decision in decision_map.values())
    mitigation_credit = 0 if effective_config.mitigation_delay_steps > 0 else _pending_credit_from_decisions(live_cell, decision_map)
    net_incoming_events = max(0, live_cell.control_events - mitigation_credit)
    effective_controller_capacity = max(50, budget.controller_actions_per_step - max(0, controller_actions_used - 48))
    projected_queue = max(0, controller_queue + net_incoming_events - effective_controller_capacity)
    controller_latency_ms = project_controller_latency_ms(
        controller_queue=projected_queue,
        net_incoming_events=net_incoming_events,
    )
    observations = evaluate_cell_decisions(
        scenario_name=scenario_name,
        policy_name="counterfactual",
        step_idx=0,
        cell=live_cell,
        controller_latency_ms=controller_latency_ms,
        decision_map=decision_map,
    )
    metrics = summarize_observations(observations)
    reward = compute_reward(metrics, controller_latency_ms=controller_latency_ms, projected_queue=projected_queue)
    return {
        **metrics,
        "projected_controller_queue": float(projected_queue),
        "projected_controller_latency_ms": round(controller_latency_ms, 4),
        "reward": round(reward, 6),
    }


def compute_reward(metrics: dict[str, float], *, controller_latency_ms: float, projected_queue: int) -> float:
    return (
        4.0 * metrics["critical_slice_survival_rate"]
        + 2.0 * metrics["overall_sla_rate"]
        - 1.5 * metrics["ai_deadline_miss_rate"]
        - 2.5 * metrics["attack_leakage_rate"]
        - 0.02 * controller_latency_ms
        - 0.002 * projected_queue
        + 0.5 * metrics["mean_service_ratio"]
    )


def train_reward_model(
    *,
    trace_path: str | Path,
    model_output_path: str | Path,
    model_type: str = "linear",
    epochs: int = 220,
    learning_rate: float = 0.03,
    hidden_dim: int = 12,
    train_seeds: list[int] | None = None,
    eval_seeds: list[int] | None = None,
) -> tuple[Path, dict[str, float]]:
    trace_rows = _read_trace_rows(Path(trace_path))
    if not trace_rows:
        raise ValueError(f"No trace rows found in {trace_path}")
    train_rows, eval_rows = split_trace_rows_by_seed(trace_rows, train_seeds=train_seeds, eval_seeds=eval_seeds)
    feature_names = _feature_names_from_rows(train_rows)
    means, stds = _fit_normalization(train_rows, feature_names)
    normalized_train = _normalized_dataset(train_rows, feature_names, means, stds)
    normalized_eval = _normalized_dataset(eval_rows or train_rows, feature_names, means, stds)

    if model_type == "linear":
        model = _train_linear_model(
            feature_names=feature_names,
            means=means,
            stds=stds,
            dataset=normalized_train,
            epochs=epochs,
            learning_rate=learning_rate,
        )
    elif model_type == "mlp":
        model = _train_mlp_model(
            feature_names=feature_names,
            means=means,
            stds=stds,
            dataset=normalized_train,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
        )
    else:
        raise ValueError(f"Unknown model_type {model_type!r}")

    metrics = {}
    metrics.update({f"train_{key}": value for key, value in _evaluate_model(model, normalized_train).items()})
    metrics.update({f"eval_{key}": value for key, value in _evaluate_model(model, normalized_eval).items()})
    metrics["train_seed_count"] = float(len({int(row["seed"]) for row in train_rows}))
    metrics["eval_seed_count"] = float(len({int(row["seed"]) for row in (eval_rows or train_rows)}))
    output = Path(model_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model.to_json_dict(), indent=2), encoding="utf-8")
    return output, metrics


def load_reward_model(model_path: str | Path) -> RewardModel:
    payload = json.loads(Path(model_path).read_text(encoding="utf-8"))
    model_type = payload.get("model_type", "linear")
    if model_type == "linear":
        return LinearRewardModel(
            feature_names=[str(item) for item in payload["feature_names"]],
            means=[float(item) for item in payload["means"]],
            stds=[float(item) for item in payload["stds"]],
            weights=[float(item) for item in payload["weights"]],
            bias=float(payload["bias"]),
            metadata=dict(payload.get("metadata", {})),
        )
    if model_type == "mlp":
        return MLPRewardModel(
            feature_names=[str(item) for item in payload["feature_names"]],
            means=[float(item) for item in payload["means"]],
            stds=[float(item) for item in payload["stds"]],
            hidden_weights=[[float(value) for value in row] for row in payload["hidden_weights"]],
            hidden_biases=[float(value) for value in payload["hidden_biases"]],
            output_weights=[float(value) for value in payload["output_weights"]],
            output_bias=float(payload["output_bias"]),
            metadata=dict(payload.get("metadata", {})),
        )
    raise ValueError(f"Unknown serialized model type {model_type!r}")


def simulate_learned_aegis_mixer_on_scenario(
    *,
    scenario_name: str,
    model_path: str | Path,
    steps: int = 18,
    seed: int = 7,
    budget: ResourceBudget | None = None,
    scenario=None,
    config: SimulationConfig | None = None,
    shortlist_size: int | None = None,
) -> ScenarioResult:
    model = load_reward_model(model_path)
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
        scenario=scenario,
        config=config or SimulationConfig(),
    )


def benchmark_with_learned_policy(
    *,
    model_path: str | Path,
    steps: int = 18,
    seed: int = 7,
    scenario_names: list[str] | None = None,
    budget: ResourceBudget | None = None,
    config: SimulationConfig | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scenarios = scenario_names or list_scenarios()
    base_policies = ["static_qos", "throughput_first", "security_only", "failure_aware", "aegis_mixer"]
    effective_config = config or SimulationConfig()
    for scenario_name in scenarios:
        for policy_name in base_policies:
            result = simulate_decision_builder_on_scenario(
                scenario_name=scenario_name,
                policy_name=policy_name,
                decision_builder=lambda cell, effective_budget, controller_queue, policy=build_policy(policy_name): policy.decide(cell, effective_budget, controller_queue),
                steps=steps,
                seed=seed,
                budget=budget,
                config=effective_config,
            )
            rows.append({"scenario": scenario_name, "policy": policy_name, **result.summary_metrics})
        learned = simulate_learned_aegis_mixer_on_scenario(
            scenario_name=scenario_name,
            model_path=model_path,
            steps=steps,
            seed=seed,
            budget=budget,
            config=effective_config,
        )
        rows.append({"scenario": scenario_name, "policy": learned.policy_name, **learned.summary_metrics})
    return rows


def run_ablation_sweep(
    *,
    model_path: str | Path,
    output_path: str | Path,
    steps: int = 8,
    seed: int = 7,
    scenario_names: list[str] | None = None,
    controller_scales: list[float] | None = None,
    attack_scales: list[float] | None = None,
    ai_scales: list[float] | None = None,
    stale_options: list[int] | None = None,
    mitigation_options: list[int] | None = None,
) -> Path:
    rows: list[dict[str, object]] = []
    scenarios = scenario_names or ["ai_spike", "encrypted_ddos", "mixed_failure"]
    controller_scale_values = controller_scales or [0.7, 1.0, 1.3]
    attack_scale_values = attack_scales or [0.8, 1.0, 1.3]
    ai_scale_values = ai_scales or [0.8, 1.0, 1.3]
    stale_values = stale_options or [0, 1]
    mitigation_values = mitigation_options or [0, 1]
    for scenario_name in scenarios:
        for controller_scale in controller_scale_values:
            for attack_scale in attack_scale_values:
                for ai_scale in ai_scale_values:
                    base_scenario = build_scenario(scenario_name, steps=steps, seed=seed)
                    scaled = scale_scenario(
                        base_scenario,
                        attack_scale=attack_scale,
                        ai_scale=ai_scale,
                        control_scale=1.0,
                    )
                    budget = ResourceBudget(
                        prbs_per_cell=100.0,
                        gpu_per_cell=70.0,
                        controller_actions_per_step=int(220 * controller_scale),
                    )
                    for stale_steps in stale_values:
                        for mitigation_delay in mitigation_values:
                            config = SimulationConfig(
                                stale_telemetry_steps=stale_steps,
                                mitigation_delay_steps=mitigation_delay,
                            )
                            for policy_name in ["failure_aware", "aegis_mixer", "aegis_mixer_learned"]:
                                if policy_name == "aegis_mixer_learned":
                                    result = simulate_learned_aegis_mixer_on_scenario(
                                        scenario_name=scaled.name,
                                        model_path=model_path,
                                        steps=steps,
                                        seed=seed,
                                        budget=budget,
                                        scenario=scaled,
                                        config=config,
                                    )
                                else:
                                    result = simulate_decision_builder_on_scenario(
                                        scenario_name=scaled.name,
                                        policy_name=policy_name,
                                        decision_builder=lambda cell, effective_budget, controller_queue, policy=build_policy(policy_name): policy.decide(cell, effective_budget, controller_queue),
                                        steps=steps,
                                        seed=seed,
                                        budget=budget,
                                        scenario=scaled,
                                        config=config,
                                    )
                                rows.append(
                                    {
                                        "scenario": scenario_name,
                                        "policy": result.policy_name,
                                        "controller_scale": controller_scale,
                                        "attack_scale": attack_scale,
                                        "ai_scale": ai_scale,
                                        "stale_telemetry_steps": stale_steps,
                                        "mitigation_delay_steps": mitigation_delay,
                                        **result.summary_metrics,
                                    }
                                )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return output


def analyze_oracle_regret(
    *,
    model_path: str | Path | None = None,
    scenario_name: str = "mixed_failure",
    steps: int = 8,
    seed: int = 7,
    budget: ResourceBudget | None = None,
    config: SimulationConfig | None = None,
) -> list[dict[str, object]]:
    effective_budget = budget or ResourceBudget()
    effective_config = config or SimulationConfig()
    scenario = build_scenario(scenario_name, steps=steps, seed=seed)
    learned_model = load_reward_model(model_path) if model_path is not None else None
    policies = {
        "failure_aware": lambda cell, queue: build_policy("failure_aware").decide(cell, effective_budget, queue),
        "aegis_mixer": lambda cell, queue: build_aegis_mixer_decision(cell=cell, budget=effective_budget, controller_queue=queue),
    }
    if learned_model is not None:
        policies["aegis_mixer_learned"] = lambda cell, queue: build_aegis_mixer_decision(
            cell=cell,
            budget=effective_budget,
            controller_queue=queue,
            score_fn=learned_model.scorer(),
            policy_name="aegis_mixer_learned",
        )

    rows: list[dict[str, object]] = []
    controller_queue = 0
    history_steps: list[list[CellDemand]] = []
    mitigation_delay = [0 for _ in range(effective_config.mitigation_delay_steps)]
    for step in scenario.steps:
        live_cells = step.cells
        observed_cells = _build_trace_observed_cells(
            live_cells=live_cells,
            history_steps=history_steps,
            stale_telemetry_steps=effective_config.stale_telemetry_steps,
        )
        total_incoming_events = sum(cell.control_events for cell in live_cells)
        mitigation_credit = mitigation_delay.pop(0) if effective_config.mitigation_delay_steps > 0 else 0
        pending_policy_credit = 0
        for live_cell, observed_cell in zip(live_cells, observed_cells):
            oracle = compute_cell_oracle(
                scenario_name=scenario_name,
                live_cell=live_cell,
                observed_cell=observed_cell,
                budget=effective_budget,
                controller_queue=controller_queue,
                config=effective_config,
            )
            for policy_name, builder in policies.items():
                decision = builder(observed_cell, controller_queue)
                row = evaluate_policy_regret(
                    scenario_name=scenario_name,
                    live_cell=live_cell,
                    observed_cell=observed_cell,
                    controller_queue=controller_queue,
                    budget=effective_budget,
                    config=effective_config,
                    oracle=oracle,
                    policy_name=policy_name,
                    decision_map=decision.slice_decisions,
                )
                rows.append({"step_idx": step.step_idx, "cell_id": live_cell.cell_id, **row})
                if policy_name == "failure_aware":
                    pending_policy_credit += _pending_credit_from_decisions(live_cell, decision.slice_decisions)
        if effective_config.mitigation_delay_steps > 0:
            mitigation_delay.append(pending_policy_credit)
        else:
            mitigation_credit += pending_policy_credit
        net_incoming_events = max(0, total_incoming_events - mitigation_credit)
        effective_controller_capacity = effective_budget.controller_actions_per_step
        controller_queue = max(0, controller_queue + net_incoming_events - effective_controller_capacity)
        history_steps.append(live_cells)
    return rows


def compute_cell_oracle(
    *,
    scenario_name: str,
    live_cell: CellDemand,
    observed_cell: CellDemand,
    budget: ResourceBudget,
    controller_queue: int,
    config: SimulationConfig | None = None,
) -> dict[str, object]:
    grouped = _group_candidates_by_slice(
        build_candidates_for_cell(
            cell=observed_cell,
            budget=budget,
            controller_queue=controller_queue,
        )
    )
    observed_map = {slice_demand.slice_id: slice_demand for slice_demand in observed_cell.slices}
    for slice_id, slice_demand in observed_map.items():
        grouped.setdefault(slice_id, []).append(
            ActionCandidate(
                slice_id=slice_id,
                action_name="oracle_drop",
                allocated_prbs=0.0,
                allocated_gpu=0.0,
                isolated=slice_demand.suspicious,
                inspected=slice_demand.suspicious,
                degraded=True,
                controller_cost=1,
                coarse_score=-1.0,
                note="Oracle fallback keep-feasible action.",
            )
        )
    best: dict[str, object] | None = None
    for combo in product(*grouped.values()):
        total_prbs = sum(candidate.allocated_prbs for candidate in combo)
        total_gpu = sum(candidate.allocated_gpu for candidate in combo)
        if total_prbs > budget.prbs_per_cell + 1e-6 or total_gpu > budget.gpu_per_cell + 1e-6:
            continue
        decision_map = {candidate.slice_id: candidate_to_slice_decision(candidate) for candidate in combo}
        reward_payload = _score_decision_map(
            scenario_name=scenario_name,
            live_cell=live_cell,
            decision_map=decision_map,
            budget=budget,
            controller_queue=controller_queue,
            config=config or SimulationConfig(),
        )
        candidate_payload = {
            "decision_map": decision_map,
            **reward_payload,
        }
        if best is None or float(candidate_payload["reward"]) > float(best["reward"]):
            best = candidate_payload
    if best is None:
        raise RuntimeError("Oracle search found no feasible candidate combination")
    return best


def evaluate_policy_regret(
    *,
    scenario_name: str,
    live_cell: CellDemand,
    observed_cell: CellDemand,
    controller_queue: int,
    budget: ResourceBudget,
    config: SimulationConfig,
    oracle: dict[str, object],
    policy_name: str,
    decision_map: dict[str, SliceDecision],
) -> dict[str, object]:
    policy_payload = _score_decision_map(
        scenario_name=scenario_name,
        live_cell=live_cell,
        decision_map=decision_map,
        budget=budget,
        controller_queue=controller_queue,
        config=config,
    )
    return {
        "scenario": scenario_name,
        "policy": policy_name,
        "oracle_reward": oracle["reward"],
        "policy_reward": policy_payload["reward"],
        "reward_regret": round(float(oracle["reward"]) - float(policy_payload["reward"]), 6),
        "oracle_critical_slice_survival_rate": oracle["critical_slice_survival_rate"],
        "policy_critical_slice_survival_rate": policy_payload["critical_slice_survival_rate"],
        "oracle_overall_sla_rate": oracle["overall_sla_rate"],
        "policy_overall_sla_rate": policy_payload["overall_sla_rate"],
    }


def _score_decision_map(
    *,
    scenario_name: str,
    live_cell: CellDemand,
    decision_map: dict[str, SliceDecision],
    budget: ResourceBudget,
    controller_queue: int,
    config: SimulationConfig,
) -> dict[str, float]:
    controller_actions_used = 8 + sum(2 if decision.inspected else 0 for decision in decision_map.values())
    mitigation_credit = 0 if config.mitigation_delay_steps > 0 else _pending_credit_from_decisions(live_cell, decision_map)
    net_incoming_events = max(0, live_cell.control_events - mitigation_credit)
    effective_controller_capacity = max(50, budget.controller_actions_per_step - max(0, controller_actions_used - 48))
    projected_queue = max(0, controller_queue + net_incoming_events - effective_controller_capacity)
    controller_latency_ms = project_controller_latency_ms(
        controller_queue=projected_queue,
        net_incoming_events=net_incoming_events,
    )
    observations = evaluate_cell_decisions(
        scenario_name=scenario_name,
        policy_name="oracle_eval",
        step_idx=0,
        cell=live_cell,
        controller_latency_ms=controller_latency_ms,
        decision_map=decision_map,
    )
    metrics = summarize_observations(observations)
    metrics["projected_controller_queue"] = float(projected_queue)
    metrics["projected_controller_latency_ms"] = round(controller_latency_ms, 4)
    metrics["reward"] = round(compute_reward(metrics, controller_latency_ms=controller_latency_ms, projected_queue=projected_queue), 6)
    return metrics


def split_trace_rows_by_seed(
    rows: list[dict[str, float | str]],
    *,
    train_seeds: list[int] | None,
    eval_seeds: list[int] | None,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    if train_seeds is None and eval_seeds is None:
        seeds = sorted({int(row["seed"]) for row in rows})
        pivot = max(1, int(len(seeds) * 0.67))
        train_seeds = seeds[:pivot]
        eval_seeds = seeds[pivot:] or seeds[:pivot]
    train_seed_set = {int(seed) for seed in (train_seeds or [])}
    eval_seed_set = {int(seed) for seed in (eval_seeds or [])}
    train_rows = [row for row in rows if int(row["seed"]) in train_seed_set]
    eval_rows = [row for row in rows if int(row["seed"]) in eval_seed_set]
    return train_rows or rows, eval_rows


def _train_linear_model(
    *,
    feature_names: list[str],
    means: list[float],
    stds: list[float],
    dataset: list[tuple[list[float], float]],
    epochs: int,
    learning_rate: float,
) -> LinearRewardModel:
    weights = [0.0 for _ in feature_names]
    bias = 0.0
    for _ in range(epochs):
        grad_w = [0.0 for _ in feature_names]
        grad_b = 0.0
        for features, target in dataset:
            prediction = bias + sum(weight * value for weight, value in zip(weights, features))
            error = prediction - target
            grad_b += error
            for index, value in enumerate(features):
                grad_w[index] += error * value
        scale = 2.0 / max(1, len(dataset))
        bias -= learning_rate * scale * grad_b
        for index in range(len(weights)):
            weights[index] -= learning_rate * scale * grad_w[index]
    return LinearRewardModel(
        feature_names=feature_names,
        means=means,
        stds=stds,
        weights=weights,
        bias=bias,
        metadata={"model_type": "linear", "epochs": epochs, "learning_rate": learning_rate, "trace_rows": len(dataset)},
    )


def _train_mlp_model(
    *,
    feature_names: list[str],
    means: list[float],
    stds: list[float],
    dataset: list[tuple[list[float], float]],
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
) -> MLPRewardModel:
    input_dim = len(feature_names)
    hidden_weights = [[0.01 * (((hidden_idx + 1) * (feature_idx + 2)) % 7 - 3) for feature_idx in range(input_dim)] for hidden_idx in range(hidden_dim)]
    hidden_biases = [0.0 for _ in range(hidden_dim)]
    output_weights = [0.01 * (idx - hidden_dim / 2.0) for idx in range(hidden_dim)]
    output_bias = 0.0

    for _ in range(epochs):
        grad_hidden_weights = [[0.0 for _ in range(input_dim)] for _ in range(hidden_dim)]
        grad_hidden_biases = [0.0 for _ in range(hidden_dim)]
        grad_output_weights = [0.0 for _ in range(hidden_dim)]
        grad_output_bias = 0.0

        for features, target in dataset:
            hidden_linear = []
            hidden_activations = []
            for neuron_weights, neuron_bias in zip(hidden_weights, hidden_biases):
                value = neuron_bias + sum(weight * feature for weight, feature in zip(neuron_weights, features))
                hidden_linear.append(value)
                hidden_activations.append(math.tanh(value))
            prediction = output_bias + sum(weight * activation for weight, activation in zip(output_weights, hidden_activations))
            error = prediction - target
            grad_output_bias += error
            for hidden_index, activation in enumerate(hidden_activations):
                grad_output_weights[hidden_index] += error * activation
            for hidden_index in range(hidden_dim):
                backprop = error * output_weights[hidden_index] * (1.0 - hidden_activations[hidden_index] ** 2)
                grad_hidden_biases[hidden_index] += backprop
                for feature_index, feature in enumerate(features):
                    grad_hidden_weights[hidden_index][feature_index] += backprop * feature

        scale = 2.0 / max(1, len(dataset))
        output_bias -= learning_rate * scale * grad_output_bias
        for hidden_index in range(hidden_dim):
            output_weights[hidden_index] -= learning_rate * scale * grad_output_weights[hidden_index]
            hidden_biases[hidden_index] -= learning_rate * scale * grad_hidden_biases[hidden_index]
            for feature_index in range(input_dim):
                hidden_weights[hidden_index][feature_index] -= learning_rate * scale * grad_hidden_weights[hidden_index][feature_index]

    return MLPRewardModel(
        feature_names=feature_names,
        means=means,
        stds=stds,
        hidden_weights=hidden_weights,
        hidden_biases=hidden_biases,
        output_weights=output_weights,
        output_bias=output_bias,
        metadata={
            "model_type": "mlp",
            "epochs": epochs,
            "learning_rate": learning_rate,
            "trace_rows": len(dataset),
            "hidden_dim": hidden_dim,
        },
    )


def _feature_names_from_rows(rows: list[dict[str, float | str]]) -> list[str]:
    excluded = {
        "scenario",
        "seed",
        "source_policy",
        "step_idx",
        "cell_id",
        "slice_id",
        "action_name",
        "stale_telemetry_steps",
        "mitigation_delay_steps",
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
    return [key for key in rows[0].keys() if key not in excluded]


def _normalized_dataset(
    rows: list[dict[str, float | str]],
    feature_names: list[str],
    means: list[float],
    stds: list[float],
) -> list[tuple[list[float], float]]:
    dataset: list[tuple[list[float], float]] = []
    for row in rows:
        features = [(float(row[name]) - means[index]) / stds[index] for index, name in enumerate(feature_names)]
        dataset.append((features, float(row["reward"])))
    return dataset


def _build_counterfactual_decision_map(
    *,
    live_cell: CellDemand,
    observed_cell: CellDemand,
    candidate: ActionCandidate,
    overload: bool,
    attack_pressure: float,
    budget: ResourceBudget,
) -> dict[str, SliceDecision]:
    remaining_prbs = max(0.0, budget.prbs_per_cell - candidate.allocated_prbs)
    remaining_gpu = max(0.0, budget.gpu_per_cell - candidate.allocated_gpu)
    decisions: dict[str, SliceDecision] = {candidate.slice_id: candidate_to_slice_decision(candidate)}
    observed_map = {slice_demand.slice_id: slice_demand for slice_demand in observed_cell.slices}
    live_map = {slice_demand.slice_id: slice_demand for slice_demand in live_cell.slices}
    ordered_ids = sorted(
        [slice_id for slice_id in live_map if slice_id != candidate.slice_id],
        key=lambda slice_id: (
            0 if live_map[slice_id].mission_critical else 1 if live_map[slice_id].kind == "ai_edge" else 2 if live_map[slice_id].suspicious else 3
        ),
    )
    for slice_id in ordered_ids:
        observed_slice = observed_map.get(slice_id, live_map[slice_id])
        fallback = build_fallback_candidate(
            slice_demand=observed_slice,
            overload=overload,
            attack_pressure=attack_pressure,
            remaining_prbs=remaining_prbs,
            remaining_gpu=remaining_gpu,
        )
        fallback.slice_id = slice_id
        decisions[slice_id] = candidate_to_slice_decision(fallback)
        remaining_prbs = max(0.0, remaining_prbs - fallback.allocated_prbs)
        remaining_gpu = max(0.0, remaining_gpu - fallback.allocated_gpu)
    return decisions


def _build_trace_observed_cells(
    *,
    live_cells: list[CellDemand],
    history_steps: list[list[CellDemand]],
    stale_telemetry_steps: int,
) -> list[CellDemand]:
    if stale_telemetry_steps <= 0 or len(history_steps) < stale_telemetry_steps:
        return live_cells
    stale_cells = history_steps[-stale_telemetry_steps]
    stale_map = {cell.cell_id: cell for cell in stale_cells}
    observed: list[CellDemand] = []
    for live_cell in live_cells:
        observed.append(stale_map.get(live_cell.cell_id, live_cell))
    return observed


def _pending_credit_from_decisions(live_cell: CellDemand, decision_map: dict[str, SliceDecision]) -> int:
    credit = 0
    for slice_demand in live_cell.slices:
        decision = decision_map[slice_demand.slice_id]
        if slice_demand.suspicious and decision.isolated:
            credit += 18
        elif slice_demand.suspicious and decision.inspected:
            credit += 8
    return credit


def _group_candidates_by_slice(candidates: list[ActionCandidate]) -> dict[str, list[ActionCandidate]]:
    grouped: dict[str, list[ActionCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.slice_id, []).append(candidate)
    return grouped


def _read_trace_rows(path: Path) -> list[dict[str, float | str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


def _evaluate_model(model: RewardModel, dataset: list[tuple[list[float], float]]) -> dict[str, float]:
    errors = []
    predictions = []
    targets = []
    for features, target in dataset:
        row = {model.feature_names[index]: features[index] * 1.0 for index in range(len(model.feature_names))}
        if isinstance(model, LinearRewardModel):
            prediction = model.bias + sum(weight * value for weight, value in zip(model.weights, features))
        elif isinstance(model, MLPRewardModel):
            hidden_values = []
            for neuron_weights, neuron_bias in zip(model.hidden_weights, model.hidden_biases):
                activation = neuron_bias + sum(weight * value for weight, value in zip(neuron_weights, features))
                hidden_values.append(math.tanh(activation))
            prediction = model.output_bias + sum(weight * value for weight, value in zip(model.output_weights, hidden_values))
        else:
            prediction = model.predict_row(row)
        errors.append((prediction - target) ** 2)
        predictions.append(prediction)
        targets.append(target)
    mse = sum(errors) / max(1, len(errors))
    target_mean = sum(targets) / max(1, len(targets))
    total_var = sum((target - target_mean) ** 2 for target in targets)
    r2 = 1.0 - (sum(errors) / total_var) if total_var > 0 else 0.0
    return {"mse": round(mse, 6), "r2": round(r2, 6), "row_count": float(len(dataset))}
