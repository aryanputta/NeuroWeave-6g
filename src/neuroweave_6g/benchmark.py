from __future__ import annotations

import csv
from pathlib import Path

from .policies import available_policies
from .scenario import list_scenarios
from .simulator import simulate_policy_on_scenario


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary_rows: list[dict[str, object]]) -> None:
    lines = ["# NeuroWeave-6g Benchmark Report", ""]
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in summary_rows:
        grouped.setdefault(str(row["scenario"]), []).append(row)

    for scenario_name, rows in grouped.items():
        ordered_rows = sorted(
            rows,
            key=lambda row: (
                float(row["critical_slice_survival_rate"]),
                float(row["overall_sla_rate"]),
                -float(row["attack_leakage_rate"]),
                -float(row["controller_p95_latency_ms"]),
            ),
            reverse=True,
        )
        best = max(
            rows,
            key=lambda row: (
                float(row["critical_slice_survival_rate"]),
                -float(row["attack_leakage_rate"]),
                -float(row["controller_p95_latency_ms"]),
            ),
        )
        baselines = {str(row["policy"]): row for row in rows}
        static_row = baselines.get("static_qos")
        failure_aware_row = baselines.get("failure_aware")
        lines.extend(
            [
                f"## {scenario_name}",
                "",
                f"- Best policy: `{best['policy']}`",
                f"- Critical slice survival: `{best['critical_slice_survival_rate']}`",
                f"- Controller p95 latency: `{best['controller_p95_latency_ms']}` ms",
                f"- Attack leakage: `{best['attack_leakage_rate']}`",
                f"- AI deadline miss rate: `{best['ai_deadline_miss_rate']}`",
                "",
                "| Policy | Critical Survival | Overall SLA | Controller p95 ms | AI Miss Rate | Attack Leakage | False Positive Isolation |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in ordered_rows:
            lines.append(
                "| "
                + f"{row['policy']} | {row['critical_slice_survival_rate']} | {row['overall_sla_rate']} | "
                + f"{row['controller_p95_latency_ms']} | {row['ai_deadline_miss_rate']} | "
                + f"{row['attack_leakage_rate']} | {row['false_positive_isolation_rate']} |"
            )
        lines.append("")
        if static_row is not None:
            lines.append(
                "- Winner delta vs `static_qos`: "
                + f"critical survival `+{round(float(best['critical_slice_survival_rate']) - float(static_row['critical_slice_survival_rate']), 4)}`"
                + ", "
                + f"controller p95 `{round(float(best['controller_p95_latency_ms']) - float(static_row['controller_p95_latency_ms']), 4)}` ms"
                + ", "
                + f"AI miss `{round(float(best['ai_deadline_miss_rate']) - float(static_row['ai_deadline_miss_rate']), 4)}`"
                + ", "
                + f"attack leakage `{round(float(best['attack_leakage_rate']) - float(static_row['attack_leakage_rate']), 4)}`"
            )
        if failure_aware_row is not None and best["policy"] != "failure_aware":
            lines.append(
                "- Winner delta vs `failure_aware`: "
                + f"critical survival `+{round(float(best['critical_slice_survival_rate']) - float(failure_aware_row['critical_slice_survival_rate']), 4)}`"
                + ", "
                + f"controller p95 `{round(float(best['controller_p95_latency_ms']) - float(failure_aware_row['controller_p95_latency_ms']), 4)}` ms"
                + ", "
                + f"AI miss `{round(float(best['ai_deadline_miss_rate']) - float(failure_aware_row['ai_deadline_miss_rate']), 4)}`"
                + ", "
                + f"attack leakage `{round(float(best['attack_leakage_rate']) - float(failure_aware_row['attack_leakage_rate']), 4)}`"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark_suite(
    *,
    output_dir: str | Path = "results",
    steps: int = 18,
    seed: int = 7,
    scenario_names: list[str] | None = None,
    policy_names: list[str] | None = None,
) -> dict[str, Path]:
    output_root = Path(output_dir)
    scenarios = scenario_names or list_scenarios()
    policies = policy_names or available_policies()

    summary_rows: list[dict[str, object]] = []
    step_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []

    for scenario_name in scenarios:
        for policy_name in policies:
            result = simulate_policy_on_scenario(
                scenario_name=scenario_name,
                policy_name=policy_name,
                steps=steps,
                seed=seed,
            )
            summary_rows.append(
                {
                    "scenario": result.scenario_name,
                    "policy": result.policy_name,
                    **result.summary_metrics,
                }
            )
            for step_summary in result.step_summaries:
                step_rows.append(
                    {
                        "scenario": step_summary.scenario_name,
                        "policy": step_summary.policy_name,
                        "step_idx": step_summary.step_idx,
                        "controller_queue": step_summary.controller_queue,
                        "controller_latency_ms": step_summary.controller_latency_ms,
                        "attack_leakage_rate": step_summary.attack_leakage_rate,
                        "critical_slice_survival_rate": step_summary.critical_slice_survival_rate,
                        "ai_deadline_miss_rate": step_summary.ai_deadline_miss_rate,
                        "overall_sla_rate": step_summary.overall_sla_rate,
                    }
                )
            for observation in result.slice_observations:
                slice_rows.append(
                    {
                        "scenario": observation.scenario_name,
                        "policy": observation.policy_name,
                        "step_idx": observation.step_idx,
                        "cell_id": observation.cell_id,
                        "slice_id": observation.slice_id,
                        "kind": observation.kind,
                        "latency_ms": observation.latency_ms,
                        "service_ratio": observation.service_ratio,
                        "sla_met": observation.sla_met,
                        "isolated": observation.isolated,
                        "suspicious": observation.suspicious,
                        "mission_critical": observation.mission_critical,
                        "ai_enabled": observation.ai_enabled,
                        "deadline_missed": observation.deadline_missed,
                    }
                )

    outputs = {
        "summary_metrics": output_root / "raw" / "summary_metrics.csv",
        "step_metrics": output_root / "raw" / "step_metrics.csv",
        "slice_metrics": output_root / "raw" / "slice_metrics.csv",
        "report": output_root / "reports" / "benchmark_report.md",
    }
    _write_csv(outputs["summary_metrics"], summary_rows)
    _write_csv(outputs["step_metrics"], step_rows)
    _write_csv(outputs["slice_metrics"], slice_rows)
    _write_report(outputs["report"], summary_rows)
    return outputs
