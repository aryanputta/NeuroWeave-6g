from neuroweave_6g.demo import build_demo_snapshot, render_demo_text
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


def test_aegis_mixer_beats_static_under_ai_spike() -> None:
    static = simulate_policy_on_scenario(scenario_name="ai_spike", policy_name="static_qos", steps=12, seed=7)
    aegis_mixer = simulate_policy_on_scenario(
        scenario_name="ai_spike",
        policy_name="aegis_mixer",
        steps=12,
        seed=7,
    )

    assert aegis_mixer.summary_metrics["overall_sla_rate"] > static.summary_metrics["overall_sla_rate"]
    assert aegis_mixer.summary_metrics["ai_deadline_miss_rate"] < static.summary_metrics["ai_deadline_miss_rate"]


def test_demo_snapshot_picks_aegis_mixer_for_ai_spike() -> None:
    snapshot = build_demo_snapshot(scenario_name="ai_spike", steps=12, seed=7)
    assert snapshot.winner == "aegis_mixer"
    assert any("AegisMixer wins" in takeaway for takeaway in snapshot.takeaways)


def test_demo_text_includes_showcase_sections() -> None:
    demo_text = render_demo_text(scenario_name="mixed_failure", steps=8, seed=7)
    assert "Policy scoreboard" in demo_text
    assert "Interview takeaways" in demo_text
    assert "Showcase framing" in demo_text
    assert "--mode demo --scenario mixed_failure" in demo_text


def test_benchmark_suite_writes_outputs(tmp_path) -> None:
    outputs = run_benchmark_suite(output_dir=tmp_path, steps=8, seed=7)
    assert outputs["summary_metrics"].exists()
    assert outputs["step_metrics"].exists()
    assert outputs["slice_metrics"].exists()
    assert outputs["report"].exists()
