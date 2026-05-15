"""NeuroWeave-6g package."""

from .benchmark import run_benchmark_suite
from .demo import build_demo_snapshot, render_demo_text
from .learning import export_learning_traces, simulate_learned_aegis_mixer_on_scenario, train_linear_reward_model
from .scenario import list_scenarios
from .simulator import simulate_policy_on_scenario

__all__ = [
    "build_demo_snapshot",
    "export_learning_traces",
    "list_scenarios",
    "render_demo_text",
    "run_benchmark_suite",
    "simulate_learned_aegis_mixer_on_scenario",
    "simulate_policy_on_scenario",
    "train_linear_reward_model",
]
