from neuroweave_6g.benchmark import run_benchmark_suite
from neuroweave_6g.simulator import simulate_policy_on_scenario


def test_failure_aware_beats_static_under_mixed_failure() -> None:
    static = simulate_policy_on_scenario(scenario_name="mixed_failure", policy_name="static_qos", steps=12, seed=7)
    failure_aware = simulate_policy_on_scenario(
        scenario_name="mixed_failure",
        policy_name="failure_aware",
        steps=12,
        seed=7,
    )

    assert failure_aware.summary_metrics["critical_slice_survival_rate"] > static.summary_metrics["critical_slice_survival_rate"]
    assert failure_aware.summary_metrics["attack_leakage_rate"] < static.summary_metrics["attack_leakage_rate"]


def test_benchmark_suite_writes_outputs(tmp_path) -> None:
    outputs = run_benchmark_suite(output_dir=tmp_path, steps=8, seed=7)
    assert outputs["summary_metrics"].exists()
    assert outputs["step_metrics"].exists()
    assert outputs["slice_metrics"].exists()
    assert outputs["report"].exists()

