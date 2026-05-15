from __future__ import annotations

from dataclasses import dataclass

from .policies import available_policies
from .simulator import simulate_policy_on_scenario


METRIC_LABELS = {
    "critical_slice_survival_rate": "Critical survival",
    "overall_sla_rate": "Overall SLA",
    "controller_p95_latency_ms": "Controller p95 ms",
    "ai_deadline_miss_rate": "AI miss rate",
    "attack_leakage_rate": "Attack leakage",
}


@dataclass(slots=True)
class DemoSnapshot:
    scenario_name: str
    winner: str
    rows: list[dict[str, float | str]]
    takeaways: list[str]


def build_demo_snapshot(
    *,
    scenario_name: str,
    steps: int = 18,
    seed: int = 7,
    policy_names: list[str] | None = None,
) -> DemoSnapshot:
    chosen_policies = policy_names or available_policies()
    rows: list[dict[str, float | str]] = []
    for policy_name in chosen_policies:
        result = simulate_policy_on_scenario(
            scenario_name=scenario_name,
            policy_name=policy_name,
            steps=steps,
            seed=seed,
        )
        rows.append(
            {
                "policy": policy_name,
                **result.summary_metrics,
            }
        )

    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["critical_slice_survival_rate"]),
            float(row["overall_sla_rate"]),
            -float(row["attack_leakage_rate"]),
            -float(row["controller_p95_latency_ms"]),
        ),
        reverse=True,
    )
    winner = str(ordered[0]["policy"])
    return DemoSnapshot(
        scenario_name=scenario_name,
        winner=winner,
        rows=ordered,
        takeaways=_build_takeaways(scenario_name=scenario_name, ordered_rows=ordered),
    )


def render_demo_text(
    *,
    scenario_name: str,
    steps: int = 18,
    seed: int = 7,
    policy_names: list[str] | None = None,
) -> str:
    snapshot = build_demo_snapshot(
        scenario_name=scenario_name,
        steps=steps,
        seed=seed,
        policy_names=policy_names,
    )
    lines = [
        "NeuroWeave-6g Demo",
        f"Scenario: {snapshot.scenario_name}",
        f"Winner: {snapshot.winner}",
        "",
        "Policy scoreboard",
        "policy            critical  sla      p95_ms   ai_miss  atk_leak",
        "---------------  --------  -------  -------  -------  --------",
    ]
    for row in snapshot.rows:
        lines.append(
            f"{str(row['policy']):15}  "
            f"{float(row['critical_slice_survival_rate']):8.4f}  "
            f"{float(row['overall_sla_rate']):7.4f}  "
            f"{float(row['controller_p95_latency_ms']):7.3f}  "
            f"{float(row['ai_deadline_miss_rate']):7.4f}  "
            f"{float(row['attack_leakage_rate']):8.4f}"
        )

    lines.extend(["", "Interview takeaways"])
    for takeaway in snapshot.takeaways:
        lines.append(f"- {takeaway}")

    lines.extend(["", "Showcase framing"])
    lines.extend(_build_showcase_lines(snapshot=snapshot))
    return "\n".join(lines)


def _build_takeaways(*, scenario_name: str, ordered_rows: list[dict[str, float | str]]) -> list[str]:
    winner = ordered_rows[0]
    static = _find_row(ordered_rows, "static_qos")
    failure_aware = _find_row(ordered_rows, "failure_aware")
    aegis_mixer = _find_row(ordered_rows, "aegis_mixer")
    takeaways: list[str] = []

    if winner["policy"] == "aegis_mixer" and static is not None:
        takeaways.append(
            "AegisMixer wins by ranking interventions under a fixed controller budget instead of applying one static slice allocation rule."
        )
        takeaways.append(
            "Against static_qos it improves overall SLA by "
            + f"{_delta(aegis_mixer, static, 'overall_sla_rate')} and reduces AI miss rate by {_inverse_delta(aegis_mixer, static, 'ai_deadline_miss_rate')}."
        )
    elif winner["policy"] == "failure_aware" and static is not None:
        takeaways.append(
            "FailureAware wins because the operating regime rewards survivability-first isolation and mission-critical reservation over broad service preservation."
        )
        takeaways.append(
            "Against static_qos it improves critical survival by "
            + f"{_delta(failure_aware, static, 'critical_slice_survival_rate')} and cuts attack leakage by {_inverse_delta(failure_aware, static, 'attack_leakage_rate')}."
        )

    if aegis_mixer is not None and failure_aware is not None:
        if float(aegis_mixer["overall_sla_rate"]) > float(failure_aware["overall_sla_rate"]):
            takeaways.append(
                "AegisMixer preserves more total service, which shows the value of recommendation-style action ranking in broad-SLA regimes."
            )
        if float(failure_aware["critical_slice_survival_rate"]) > float(aegis_mixer["critical_slice_survival_rate"]):
            takeaways.append(
                "FailureAware protects critical slices better, which exposes the main tradeoff: broad SLA preservation versus safety-first survivability."
            )

    if scenario_name == "ai_spike":
        takeaways.append(
            "This is the cleanest demo of cross-domain transfer: a recommender-style controller beats fixed telecom heuristics under compute surge."
        )
    if scenario_name in {"encrypted_ddos", "mixed_failure"}:
        takeaways.append(
            "This regime is useful for showing that one policy is not globally best. The benchmark surfaces different winners for different operator objectives."
        )
    return takeaways


def _build_showcase_lines(*, snapshot: DemoSnapshot) -> list[str]:
    lines = [
        f"- Command: `python3 -m src.main --mode demo --scenario {snapshot.scenario_name}`",
        "- Skill signal: systems benchmarking, multi-objective tradeoff analysis, and architecture transfer from recommendation systems into AI-RAN control.",
    ]
    if snapshot.winner == "aegis_mixer":
        lines.append(
            "- Story: explain how candidate generation, retrieval, and reranking let the controller spend limited decision budget on the highest-value actions."
        )
    else:
        lines.append(
            "- Story: explain why a safety-first controller still wins when encrypted attack pressure dominates and the network must preserve mission-critical continuity first."
        )
    return lines


def _find_row(rows: list[dict[str, float | str]], policy_name: str) -> dict[str, float | str] | None:
    for row in rows:
        if row["policy"] == policy_name:
            return row
    return None


def _delta(a: dict[str, float | str] | None, b: dict[str, float | str] | None, metric: str) -> str:
    if a is None or b is None:
        return "n/a"
    return f"{float(a[metric]) - float(b[metric]):+.4f}"


def _inverse_delta(a: dict[str, float | str] | None, b: dict[str, float | str] | None, metric: str) -> str:
    if a is None or b is None:
        return "n/a"
    return f"{float(b[metric]) - float(a[metric]):+.4f}"
