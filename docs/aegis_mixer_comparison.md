# AegisMixer Comparison

## What AegisMixer Is

`AegisMixer` is the `x-algorithm` derivative inside this project.
It takes the recommendation-system pattern:

- generate candidates
- retrieve a shortlist
- rerank with richer context
- spend budget only on the highest-value actions

and applies it to AI-RAN controller actions instead of content ranking.

## Benchmark Summary

Using the local benchmark run in `results/raw/summary_metrics.csv`:

### Where AegisMixer Wins

- `ai_spike`
  - best overall SLA: `0.9722`
  - best critical slice survival among top policies: `1.0`
  - lower controller p95 latency than `static_qos`: `5.248 ms` vs `5.304 ms`
  - lower AI deadline miss rate than `static_qos`: `0.1389` vs `0.3333`

- `mixed_failure`
  - best overall SLA: `0.5963`
  - zero attack leakage
  - lower controller p95 latency than `static_qos`: `63.328 ms` vs `73.768 ms`

- `encrypted_ddos`
  - best overall SLA: `0.6815`
  - zero attack leakage
  - much lower controller p95 latency than `static_qos`: `51.944 ms` vs `60.816 ms`

### Where FailureAware Still Wins

- `encrypted_ddos`
  - best critical slice survival: `0.6296`
  - best AI deadline miss rate: `0.1852`
  - best controller p95 latency: `43.824 ms`

- `mixed_failure`
  - best critical slice survival: `0.4074`
  - best controller p95 latency: `56.776 ms`
  - best attack-first survivability profile

## Interpretation

`AegisMixer` is better when the system objective is:

- preserve broader service quality
- keep more slices partially alive
- prioritize among many competing interventions quickly
- stay robust during compute-heavy surges

`failure_aware` is better when the system objective is:

- protect mission-critical slices first
- accept collateral degradation elsewhere
- minimize worst-case control-plane collapse under attack

This is the useful result.
The project does not collapse to "one best policy."
It shows that a recommender-style controller changes the operating point of the network.

## Demo Incorporation

To show this skill live, use the built-in CLI demo:

```bash
python3 -m src.main --mode demo --scenario ai_spike
python3 -m src.main --mode demo --scenario mixed_failure
python3 -m src.main --mode demo --scenario encrypted_ddos
```

What this demonstrates:

- you can explain recommendation-style candidate retrieval and reranking in a non-recommendation domain
- you can compare policies under different operating regimes rather than cherry-picking one best metric
- you can talk through the benchmark like a systems engineer, not just show a README chart

## Why This Is Attractive

### For recruiters

- It is a transferable systems idea, not a clone of a social feed.
- It shows architectural reasoning across domains.
- It produces measurable benchmark tradeoffs instead of a vague architecture diagram.

### For telecom and AI infrastructure teams

- It demonstrates that action ranking is a first-class bottleneck in AI-native control loops.
- It makes the control objective explicit: survivability vs broader SLA preservation.
- It provides a clean path to future learned retrieval and reranking policies.

### For your portfolio

- `failure_aware` tells the resilience story.
- `aegis_mixer` tells the algorithm-transfer story.
- Together they make the project deeper than a standard scheduler benchmark.
- the CLI demo gives you a direct showcase artifact for interviews, hackathons, and project videos

## Research-Heavy Next Step Now Implemented

The repo no longer stops at a hand-tuned ranker.
It now supports:

- candidate trace export
- counterfactual reward labeling
- learned reranker training
- learned-policy simulation and benchmarking

Core commands:

```bash
python3 -m src.main --mode export_traces --steps 8 --seed 7
python3 -m src.main --mode train_reranker --steps 8 --seed 7
python3 -m src.main --mode benchmark_learned --steps 8 --seed 7
```

This matters because it moves the project closer to a real systems-paper pattern:

- define an action space
- generate offline traces
- learn a ranking model from counterfactual outcomes
- compare learned and rule-based controllers by operating regime
