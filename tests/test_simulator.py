from neuroweave_6g.demo import build_demo_snapshot, render_demo_text
from neuroweave_6g.benchmark import run_benchmark_suite
from neuroweave_6g.learning import (
    analyze_oracle_regret,
    benchmark_with_learned_policy,
    export_learning_traces,
    run_ablation_sweep,
    simulate_learned_aegis_mixer_on_scenario,
    train_reward_model,
)
from neuroweave_6g.simulator import simulate_policy_on_scenario
from neuroweave_6g.types import SimulationConfig


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


def test_learning_pipeline_exports_and_trains(tmp_path) -> None:
    trace_path = export_learning_traces(
        output_path=tmp_path / "candidate_traces.csv",
        steps=4,
        seeds=[7],
        scenario_names=["ai_spike"],
        source_policy_names=["static_qos", "failure_aware"],
    )
    model_path, metrics = train_reward_model(
        trace_path=trace_path,
        model_output_path=tmp_path / "learned_model.json",
        epochs=40,
        learning_rate=0.04,
        train_seeds=[7],
        eval_seeds=[7],
    )

    assert trace_path.exists()
    assert model_path.exists()
    assert metrics["train_row_count"] > 0


def test_learned_policy_simulates_after_training(tmp_path) -> None:
    trace_path = export_learning_traces(
        output_path=tmp_path / "candidate_traces.csv",
        steps=4,
        seeds=[7],
        scenario_names=["ai_spike"],
        source_policy_names=["static_qos", "failure_aware"],
    )
    model_path, _ = train_reward_model(
        trace_path=trace_path,
        model_output_path=tmp_path / "learned_model.json",
        model_type="mlp",
        epochs=40,
        learning_rate=0.04,
        hidden_dim=8,
        train_seeds=[7],
        eval_seeds=[7],
    )
    result = simulate_learned_aegis_mixer_on_scenario(
        scenario_name="ai_spike",
        model_path=model_path,
        steps=4,
        seed=7,
        config=SimulationConfig(stale_telemetry_steps=1, mitigation_delay_steps=1),
    )

    assert result.policy_name == "aegis_mixer_learned"
    assert result.summary_metrics["critical_slice_survival_rate"] >= 0.5


def test_ablation_and_oracle_outputs(tmp_path) -> None:
    trace_path = export_learning_traces(
        output_path=tmp_path / "candidate_traces.csv",
        steps=4,
        seeds=[7],
        scenario_names=["ai_spike"],
        source_policy_names=["static_qos", "failure_aware"],
    )
    model_path, _ = train_reward_model(
        trace_path=trace_path,
        model_output_path=tmp_path / "learned_model.json",
        model_type="linear",
        epochs=30,
        learning_rate=0.04,
        train_seeds=[7],
        eval_seeds=[7],
    )
    ablation_output = run_ablation_sweep(
        model_path=model_path,
        output_path=tmp_path / "ablation.csv",
        steps=4,
        seed=7,
        scenario_names=["ai_spike"],
        controller_scales=[1.0],
        attack_scales=[1.0],
        ai_scales=[1.0],
        stale_options=[0, 1],
        mitigation_options=[0, 1],
    )
    benchmark_rows = benchmark_with_learned_policy(
        model_path=model_path,
        steps=4,
        seed=7,
        config=SimulationConfig(stale_telemetry_steps=1, mitigation_delay_steps=1),
    )
    regret_rows = analyze_oracle_regret(
        model_path=model_path,
        scenario_name="mixed_failure",
        steps=4,
        seed=7,
        config=SimulationConfig(stale_telemetry_steps=1, mitigation_delay_steps=1),
    )

    assert ablation_output.exists()
    assert any(row["policy"] == "aegis_mixer_learned" for row in benchmark_rows)
    assert len(regret_rows) > 0


def test_benchmark_suite_writes_outputs(tmp_path) -> None:
    outputs = run_benchmark_suite(output_dir=tmp_path, steps=8, seed=7)
    assert outputs["summary_metrics"].exists()
    assert outputs["step_metrics"].exists()
    assert outputs["slice_metrics"].exists()
    assert outputs["report"].exists()
