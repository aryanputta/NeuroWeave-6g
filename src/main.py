from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neuroweave_6g.benchmark import run_benchmark_suite
from neuroweave_6g.demo import render_demo_text
from neuroweave_6g.policies import available_policies
from neuroweave_6g.scenario import list_scenarios
from neuroweave_6g.simulator import simulate_policy_on_scenario


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NeuroWeave-6g control-plane resilience benchmark")
    parser.add_argument("--mode", choices=["simulate", "benchmark", "demo"], default="benchmark")
    parser.add_argument("--scenario", choices=list_scenarios(), default="mixed_failure")
    parser.add_argument("--policy", choices=available_policies(), default="failure_aware")
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="results")
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
