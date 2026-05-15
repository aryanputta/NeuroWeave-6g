# NeuroWeave-6g

Networking-first AI-RAN resilience benchmark for 6G edge systems.

`NeuroWeave-6g` implements **AegisRAN**, a failure-aware AI-RAN control-plane simulator for a hard systems problem surfaced by the papers in your Brain: what happens when AI-native RAN control, edge inference demand, and encrypted attack traffic all compete for the same shared infrastructure.

The repo now also includes a research-oriented next step beyond hand-tuned heuristics:

- rollout trace export for candidate actions
- counterfactual reward labeling from simulator outcomes
- a learned linear reranker for `aegis_mixer`
- learned-policy benchmarking against the fixed baselines

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
- `aegis_mixer`: retrieval-then-ranking controller derived from `x-algorithm` style candidate generation and reranking, aimed at action prioritization under shared control budget
- `aegis_mixer_learned`: same candidate pipeline as `aegis_mixer`, but reranked by a learned reward model trained on exported simulator traces

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

- In `ai_spike`, `aegis_mixer` improved overall SLA from `0.8056` to `0.9722` compared with `static_qos`.
- In `ai_spike`, `aegis_mixer` reduced AI deadline misses from `0.3333` to `0.1389`.
- In `encrypted_ddos`, `failure_aware` raised critical slice survival from `0.4259` to `0.6296` compared with `static_qos`.
- In `encrypted_ddos`, `failure_aware` cut controller p95 latency from `60.816 ms` to `43.824 ms`.
- In `encrypted_ddos`, `failure_aware` reduced attack leakage from `1.0` to `0.0`.
- In `mixed_failure`, `aegis_mixer` improved overall SLA from `0.3926` to `0.5963` while also reducing attack leakage from `1.0` to `0.0`.
- In `mixed_failure`, `failure_aware` still remains the survivability winner, raising critical slice survival from `0.2778` to `0.4074`.

## AegisMixer Takeaway

`aegis_mixer` is not a replacement for `failure_aware`.
It exposes a different control objective:

- `aegis_mixer` is best when the system needs to rank many feasible interventions quickly and preserve broader service quality during compute-heavy surges.
- `failure_aware` is best when survivability of mission-critical slices under attack is the primary objective.

That split is useful rather than embarrassing. It demonstrates that control-policy quality depends on the operating regime, which is exactly the systems point this project is trying to make.

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
python3 -m src.main --mode demo --scenario ai_spike
python3 -m src.main --mode train_reranker --steps 8 --seed 7
python3 -m src.main --mode benchmark_learned --steps 8 --seed 7
```

## Demo Flow

Use the demo mode when you want a fast, interview-friendly walkthrough instead of raw CSV output.

Recommended live demo commands:

```bash
python3 -m src.main --mode demo --scenario ai_spike
python3 -m src.main --mode demo --scenario mixed_failure
python3 -m src.main --mode demo --scenario encrypted_ddos
```

What each shows:

- `ai_spike`: `aegis_mixer` as the ranking-first winner under edge compute surge
- `mixed_failure`: broad SLA preservation versus survivability-first tradeoff
- `encrypted_ddos`: why `failure_aware` still wins when attack-heavy continuity matters most

## Learned Reranker Workflow

This is the research-heavy path in the repo now.

1. Export candidate traces with counterfactual rewards
```bash
python3 -m src.main --mode export_traces --steps 8 --seed 7
```

2. Train the reranker
```bash
python3 -m src.main --mode train_reranker --steps 8 --seed 7
```

3. Evaluate the learned controller
```bash
python3 -m src.main --mode simulate_learned --scenario ai_spike --steps 8 --seed 7
python3 -m src.main --mode benchmark_learned --steps 8 --seed 7
```

What this buys you:

- no longer just a hand-tuned scoring function
- reproducible state-action-outcome traces
- a real learning-based comparison against the fixed heuristics
- a path toward ablations, regret analysis, and more paper-like results

## Recruiter Angle

This is not presented as "I built a 6G dashboard."
It is presented as:

> I built a failure-aware AI-RAN control-plane benchmark that models how mission-critical slices, edge inference, and encrypted attack traffic compete for shared radio, compute, and controller budget, then measured which control policies keep the network alive under overload.

With `aegis_mixer`, the story gets stronger:

> I also derived a retrieval-plus-ranking controller from a production recommendation architecture and tested when that decision layer beats static telecom heuristics and when it still loses to a safety-first resilience policy.

With the learned reranker, the story is stronger again:

> I exported counterfactual action traces from the simulator, trained a reward model to rerank controller interventions, and compared the learned controller against fixed heuristics under compute-surge and attack-heavy regimes.

That is a stronger systems story for telecom, networking, distributed systems, edge AI, and infrastructure roles because it combines:

- recommendation-system architecture transfer
- control-plane benchmarking
- explicit multi-objective tradeoff analysis
- honest failure-case reporting
- a runnable CLI demo that explains the winner by operating regime

See also:

- `results/reports/benchmark_report.md`
- `docs/paper_map.md`
- `docs/architecture.md`
