# NeuroWeave-6g

Networking-first AI-RAN resilience benchmark for 6G edge systems.

`NeuroWeave-6g` implements **AegisRAN**, a failure-aware AI-RAN control-plane simulator for a hard systems problem surfaced by the papers in your Brain: what happens when AI-native RAN control, edge inference demand, and encrypted attack traffic all compete for the same shared infrastructure.

## Why This Project Exists

Your local 6G paper set points to three converging realities:

- `AI-RAN in Action: Turning 5G Infrastructure into an Intelligent Growth Platform` frames AI-RAN as shared edge compute plus connectivity, with mission-critical RAN functions competing with edge AI workloads.
- `6G network architecture – a proposal for early alignment` argues that 6G needs interface-driven, cloud-native, automated network functions with service assurance and intent-based management from the start.
- `Comparative Detection of DDoS Attacks on Software-Defined 5G Network Data Using Deep Neural Networks and Machine Learning Methods` shows that SDN-based 5G control planes become vulnerable under encrypted attack traffic and that JA4-style features matter for real-time mitigation.

Most student telecom projects stop at slicing or congestion prediction.
This repo focuses on the harder failure mode:

1. slice demand spikes
2. edge AI jobs saturate shared GPU budget
3. encrypted suspicious traffic floods the control loop
4. the controller itself starts missing timing and wrong slices survive

## What NeuroWeave-6g Simulates

- multi-cell AI-RAN slice demand across mission-critical, AI-edge, broadband, and background traffic
- shared radio PRB budget and shared edge GPU budget per cell
- controller backlog and decision-latency growth under stress
- encrypted suspicious traffic modeled with JA4-style anomaly pressure
- backhaul and mobility penalties to reflect 6G heterogeneity and service assurance stress

## Policies Compared

- `static_qos`: fixed class shares, no real resilience logic
- `throughput_first`: high utilization, weak attack awareness
- `security_only`: aggressive isolation without compute awareness
- `failure_aware`: protects critical slices, isolates suspicious bursts, and degrades low-priority demand when the control plane is close to collapse

## Output Metrics

- critical slice survival rate
- overall SLA rate
- controller p95 latency
- control queue peak
- AI deadline miss rate
- attack leakage rate
- false positive isolation rate

## Proof Snapshot

From the committed benchmark run in `results/raw/summary_metrics.csv`:

- In `encrypted_ddos`, `failure_aware` raised critical slice survival from `0.4259` to `0.6296` compared with `static_qos`.
- In `encrypted_ddos`, `failure_aware` cut controller p95 latency from `60.816 ms` to `43.824 ms`.
- In `encrypted_ddos`, `failure_aware` reduced attack leakage from `1.0` to `0.0`.
- In `mixed_failure`, `failure_aware` raised critical slice survival from `0.2778` to `0.4074` compared with `static_qos`.
- In `mixed_failure`, `failure_aware` cut controller queue peak from `2379` to `1785`.
- In `mixed_failure`, `failure_aware` reduced AI deadline misses from `0.7593` to `0.5278`.

## Repo Layout

- `src/neuroweave_6g/` core simulator, scenario generator, policies, benchmark runner
- `docs/architecture.md` system design
- `docs/paper_map.md` direct mapping from your paper folder to implemented concepts
- `docs/benchmark_plan.md` evaluation design
- `results/` benchmark artifacts

## Quick Start

```bash
python3 -m pytest -q
python3 -m src.main --mode benchmark --steps 18 --seed 7
python3 -m src.main --mode simulate --scenario mixed_failure --policy failure_aware
```

## Recruiter Angle

This is not presented as "I built a 6G dashboard."
It is presented as:

> I built a failure-aware AI-RAN control-plane benchmark that models how mission-critical slices, edge inference, and encrypted attack traffic compete for shared radio, compute, and controller budget, then measured which control policies keep the network alive under overload.

That is a much stronger systems story for telecom, networking, distributed systems, and edge AI roles.

See also:

- `results/reports/benchmark_report.md`
- `docs/paper_map.md`
- `docs/architecture.md`
