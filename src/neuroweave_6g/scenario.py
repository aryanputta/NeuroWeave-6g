from __future__ import annotations

from dataclasses import replace
import random

from .types import CellDemand, Scenario, ScenarioStep, SliceDemand

CELL_IDS = ["cell-west", "cell-central", "cell-east"]


def list_scenarios() -> list[str]:
    return ["normal", "ai_spike", "encrypted_ddos", "mixed_failure"]


def build_scenario(name: str, *, steps: int = 18, seed: int = 7) -> Scenario:
    builders = {
        "normal": _build_normal,
        "ai_spike": _build_ai_spike,
        "encrypted_ddos": _build_encrypted_ddos,
        "mixed_failure": _build_mixed_failure,
    }
    if name not in builders:
        raise ValueError(f"Unknown scenario {name!r}. Available: {', '.join(list_scenarios())}")
    return builders[name](steps=steps, seed=seed)


def scale_scenario(
    scenario: Scenario,
    *,
    attack_scale: float = 1.0,
    ai_scale: float = 1.0,
    control_scale: float = 1.0,
) -> Scenario:
    scaled_steps: list[ScenarioStep] = []
    for step in scenario.steps:
        scaled_cells: list[CellDemand] = []
        for cell in step.cells:
            scaled_slices: list[SliceDemand] = []
            scaled_control_events = int(cell.control_events * control_scale)
            for slice_demand in cell.slices:
                requested_prbs = slice_demand.requested_prbs
                requested_gpu = slice_demand.requested_gpu
                anomaly_score = slice_demand.anomaly_score
                if slice_demand.kind in {"suspicious"}:
                    requested_prbs *= attack_scale
                    requested_gpu *= attack_scale
                    anomaly_score = min(1.0, anomaly_score * (0.72 + 0.28 * attack_scale))
                    scaled_control_events += int(18 * max(0.0, attack_scale - 1.0))
                if slice_demand.kind in {"ai_edge", "mission_critical"}:
                    requested_gpu *= ai_scale
                    requested_prbs *= 0.92 + 0.08 * ai_scale
                    scaled_control_events += int(6 * max(0.0, ai_scale - 1.0))
                scaled_slices.append(
                    replace(
                        slice_demand,
                        requested_prbs=requested_prbs,
                        requested_gpu=requested_gpu,
                        anomaly_score=anomaly_score,
                    )
                )
            scaled_cells.append(
                replace(
                    cell,
                    control_events=scaled_control_events,
                    slices=scaled_slices,
                )
            )
        scaled_steps.append(replace(step, cells=scaled_cells))
    suffix = []
    if attack_scale != 1.0:
        suffix.append(f"attackx{attack_scale}")
    if ai_scale != 1.0:
        suffix.append(f"aix{ai_scale}")
    if control_scale != 1.0:
        suffix.append(f"ctrlx{control_scale}")
    scaled_name = scenario.name if not suffix else f"{scenario.name}__{'__'.join(suffix)}"
    scaled_notes = list(scenario.notes) + [
        f"Scaled scenario with attack_scale={attack_scale}, ai_scale={ai_scale}, control_scale={control_scale}."
    ]
    return replace(scenario, name=scaled_name, steps=scaled_steps, notes=scaled_notes)


def _rand(rng: random.Random, center: float, spread: float) -> float:
    return max(0.0, center + rng.uniform(-spread, spread))


def _base_cell(
    *,
    rng: random.Random,
    cell_id: str,
    hotspot: float,
    ai_multiplier: float,
    attack_multiplier: float,
    backhaul_pressure: float,
    mobility_pressure: float,
) -> CellDemand:
    mission_critical = SliceDemand(
        slice_id=f"{cell_id}:mission",
        kind="mission_critical",
        requested_prbs=_rand(rng, 18.0 * hotspot, 3.5),
        requested_gpu=_rand(rng, 16.0 * ai_multiplier, 3.0),
        latency_budget_ms=8.0,
        anomaly_score=max(0.01, _rand(rng, 0.08, 0.03)),
        utility=1.0,
        encrypted=True,
    )
    ai_edge = SliceDemand(
        slice_id=f"{cell_id}:ai-edge",
        kind="ai_edge",
        requested_prbs=_rand(rng, 20.0 * hotspot, 5.0),
        requested_gpu=_rand(rng, 24.0 * ai_multiplier, 6.0),
        latency_budget_ms=14.0,
        anomaly_score=max(0.02, _rand(rng, 0.10, 0.05)),
        utility=0.92,
        encrypted=True,
    )
    broadband = SliceDemand(
        slice_id=f"{cell_id}:broadband",
        kind="broadband",
        requested_prbs=_rand(rng, 32.0 * hotspot, 7.0),
        requested_gpu=0.0,
        latency_budget_ms=25.0,
        anomaly_score=max(0.02, _rand(rng, 0.14, 0.05)),
        utility=0.60,
        encrypted=False,
    )
    background = SliceDemand(
        slice_id=f"{cell_id}:background",
        kind="background",
        requested_prbs=_rand(rng, 14.0 * hotspot, 5.0),
        requested_gpu=_rand(rng, 3.0, 1.0),
        latency_budget_ms=55.0,
        anomaly_score=max(0.02, _rand(rng, 0.12, 0.04)),
        utility=0.32,
        encrypted=False,
    )
    slices = [mission_critical, ai_edge, broadband, background]

    control_events = int(28 * hotspot + ai_edge.requested_gpu * 0.55 + broadband.requested_prbs * 0.25)
    if attack_multiplier > 0:
        slices.append(
            SliceDemand(
                slice_id=f"{cell_id}:encrypted-burst",
                kind="suspicious",
                requested_prbs=_rand(rng, 36.0 * attack_multiplier, 9.0),
                requested_gpu=_rand(rng, 8.0 * attack_multiplier, 2.5),
                latency_budget_ms=40.0,
                anomaly_score=min(1.0, _rand(rng, 0.92, 0.05)),
                utility=0.08,
                encrypted=True,
            )
        )
        control_events += int(45 * attack_multiplier)

    return CellDemand(
        cell_id=cell_id,
        control_events=control_events,
        backhaul_pressure=backhaul_pressure,
        mobility_pressure=mobility_pressure,
        slices=slices,
    )


def _scenario_template(
    *,
    name: str,
    description: str,
    steps: int,
    seed: int,
    ai_multiplier: float,
    attack_multiplier: float,
    hotspot_cell: str | None,
    backhaul_pressure: float,
    mobility_pressure: float,
    notes: list[str],
) -> Scenario:
    rng = random.Random(seed)
    built_steps: list[ScenarioStep] = []
    for step_idx in range(steps):
        cells: list[CellDemand] = []
        for cell_id in CELL_IDS:
            phase_hotspot = 1.0
            if hotspot_cell == cell_id:
                phase_hotspot = 1.35 if step_idx >= steps // 3 else 1.15
            step_ai_multiplier = ai_multiplier * (1.20 if step_idx >= steps // 2 else 1.0)
            step_attack_multiplier = attack_multiplier * (1.10 if step_idx % 4 in {1, 2} else 0.85)
            cells.append(
                _base_cell(
                    rng=rng,
                    cell_id=cell_id,
                    hotspot=phase_hotspot,
                    ai_multiplier=step_ai_multiplier,
                    attack_multiplier=step_attack_multiplier,
                    backhaul_pressure=backhaul_pressure + (0.08 if cell_id == hotspot_cell else 0.0),
                    mobility_pressure=mobility_pressure + (0.03 if step_idx % 5 == 0 else 0.0),
                )
            )
        built_steps.append(ScenarioStep(step_idx=step_idx, cells=cells))
    return Scenario(name=name, description=description, steps=built_steps, notes=notes)


def _build_normal(*, steps: int, seed: int) -> Scenario:
    return _scenario_template(
        name="normal",
        description="Balanced AI-RAN traffic with moderate edge inference load and low anomaly pressure.",
        steps=steps,
        seed=seed,
        ai_multiplier=1.0,
        attack_multiplier=0.0,
        hotspot_cell=None,
        backhaul_pressure=0.10,
        mobility_pressure=0.08,
        notes=[
            "Paper grounding: shared AI+RAN infrastructure from AI-RAN in Action.",
            "Used as the low-stress baseline before overload and attack scenarios.",
        ],
    )


def _build_ai_spike(*, steps: int, seed: int) -> Scenario:
    return _scenario_template(
        name="ai_spike",
        description="Shared infrastructure stress where edge AI demand surges faster than the controller adapts.",
        steps=steps,
        seed=seed,
        ai_multiplier=1.7,
        attack_multiplier=0.0,
        hotspot_cell="cell-central",
        backhaul_pressure=0.18,
        mobility_pressure=0.10,
        notes=[
            "Grounded in AI-RAN shared-infrastructure and edge-hosted AI concepts.",
            "Designed to pressure GPU admission and controller queueing before security events begin.",
        ],
    )


def _build_encrypted_ddos(*, steps: int, seed: int) -> Scenario:
    return _scenario_template(
        name="encrypted_ddos",
        description="JA4-style encrypted traffic burst that overloads control decisions and threatens slice isolation.",
        steps=steps,
        seed=seed,
        ai_multiplier=1.0,
        attack_multiplier=1.3,
        hotspot_cell="cell-central",
        backhaul_pressure=0.20,
        mobility_pressure=0.14,
        notes=[
            "Grounded in the SD5GJA4 paper's controller-risk and encrypted-traffic framing.",
            "Suspicious slices use high anomaly scores but still compete for PRBs and inspection actions.",
        ],
    )


def _build_mixed_failure(*, steps: int, seed: int) -> Scenario:
    return _scenario_template(
        name="mixed_failure",
        description="Hotspot cell with concurrent AI spike, encrypted attack burst, and elevated backhaul pressure.",
        steps=steps,
        seed=seed,
        ai_multiplier=1.75,
        attack_multiplier=1.25,
        hotspot_cell="cell-central",
        backhaul_pressure=0.28,
        mobility_pressure=0.18,
        notes=[
            "Combines AI-RAN shared GPU pressure with SDN control-plane attack pressure.",
            "Optional non-terrestrial and heterogeneous backhaul stress from the 6G space paper is approximated through sustained backhaul penalties.",
        ],
    )
