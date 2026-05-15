from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neuroweave_6g.benchmark import run_benchmark_suite
from neuroweave_6g.demo import render_demo_text
from neuroweave_6g.learning import (
    benchmark_with_learned_policy,
    export_learning_traces,
    simulate_learned_aegis_mixer_on_scenario,
    train_linear_reward_model,
)
from neuroweave_6g.policies import available_policies
from neuroweave_6g.scenario import list_scenarios
from neuroweave_6g.simulator import simulate_policy_on_scenario


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NeuroWeave-6g control-plane resilience benchmark")
    parser.add_argument(
        "--mode",
        choices=["simulate", "benchmark", "demo", "export_traces", "train_reranker", "simulate_learned", "benchmark_learned"],
        default="benchmark",
    )
    parser.add_argument("--scenario", choices=list_scenarios(), default="mixed_failure")
    parser.add_argument("--policy", choices=available_policies(), default="failure_aware")
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--trace-path", default="results/learning/candidate_traces.csv")
    parser.add_argument("--model-path", default="artifacts/learned_aegis_mixer.json")
    parser.add_argument("--train-epochs", type=int, default=220)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "simulate":
        result = simulate_policy_on_scenario(
            scenario_name=args.scenario,
            policy_name=args.policy,
            steps=args.steps,
            seed=args.seed,
        )
        print(f"scenario={result.scenario_name}")
        print(f"policy={result.policy_name}")
        for key, value in result.summary_metrics.items():
            print(f"{key}={value}")
        return 0

    if args.mode == "demo":
        print(
            render_demo_text(
                scenario_name=args.scenario,
                steps=args.steps,
                seed=args.seed,
            )
        )
        return 0

    if args.mode == "export_traces":
        trace_path = export_learning_traces(
            output_path=args.trace_path,
            steps=args.steps,
            seeds=[args.seed, args.seed + 4, args.seed + 12],
        )
        print(f"trace_path={trace_path}")
        return 0

    if args.mode == "train_reranker":
        trace_path = export_learning_traces(
            output_path=args.trace_path,
            steps=args.steps,
            seeds=[args.seed, args.seed + 4, args.seed + 12],
        )
        model_path, metrics = train_linear_reward_model(
            trace_path=trace_path,
            model_output_path=args.model_path,
            epochs=args.train_epochs,
        )
        print(f"trace_path={trace_path}")
        print(f"model_path={model_path}")
        for key, value in metrics.items():
            print(f"{key}={value}")
        return 0

    if args.mode == "simulate_learned":
        result = simulate_learned_aegis_mixer_on_scenario(
            scenario_name=args.scenario,
            model_path=args.model_path,
            steps=args.steps,
            seed=args.seed,
        )
        print(f"scenario={result.scenario_name}")
        print(f"policy={result.policy_name}")
        for key, value in result.summary_metrics.items():
            print(f"{key}={value}")
        return 0

    if args.mode == "benchmark_learned":
        rows = benchmark_with_learned_policy(
            model_path=args.model_path,
            steps=args.steps,
            seed=args.seed,
        )
        for row in rows:
            print(
                f"scenario={row['scenario']} policy={row['policy']} "
                f"critical={row['critical_slice_survival_rate']} "
                f"sla={row['overall_sla_rate']} "
                f"p95_ms={row['controller_p95_latency_ms']} "
                f"ai_miss={row['ai_deadline_miss_rate']} "
                f"attack_leak={row['attack_leakage_rate']}"
            )
        return 0

    outputs = run_benchmark_suite(
        output_dir=args.output_dir,
        steps=args.steps,
        seed=args.seed,
    )
    for label, path in outputs.items():
        print(f"{label}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
