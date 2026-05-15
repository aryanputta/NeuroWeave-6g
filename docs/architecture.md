# Architecture

## Goal

Model the control-plane failure modes of an AI-native 6G network where radio scheduling, edge AI admission, and encrypted traffic defense share the same limited infrastructure.

## Core Components

### Scenario Generator
- creates deterministic multi-cell traffic traces
- supports `normal`, `ai_spike`, `encrypted_ddos`, and `mixed_failure`
- produces per-cell demand for slices, controller events, backhaul pressure, and mobility pressure

### Policy Layer
- each policy sees the same per-cell demand and current controller backlog
- outputs slice-level decisions:
  - PRB allocation
  - GPU allocation
  - isolation
  - throttling
  - inspection

### AegisMixer Decision Layer
- generates multiple feasible interventions per slice instead of one fixed rule outcome
- retrieves a shortlist of high-value actions using coarse urgency features
- reranks that shortlist using richer context:
  - mission-critical continuity value
  - edge-AI deadline pressure
  - anomaly and attack pressure
  - controller action cost
  - shared PRB and GPU budget pressure
- selects actions in phases:
  - suspicious isolation
  - mission-critical protection
  - edge-AI preservation
  - remaining benign traffic

### Simulator
- applies policy decisions to each cell
- updates controller queue based on incoming events and actions spent
- computes end-to-end latency and SLA outcomes per slice
- aggregates step metrics and scenario metrics

## Design Choices

### Why deterministic scenarios

For a portfolio project, deterministic replay matters more than flashy randomness.
You want benchmark deltas that can be reproduced exactly.

### Why controller backlog is central

The networking papers support AI-native orchestration and attack-aware operation, but the real systems gap is when the controller itself becomes the bottleneck.
That is why controller queue and controller p95 latency are first-class outputs.

### Why retrieval plus ranking belongs here

The control plane often has more possible interventions than it can evaluate deeply within one decision interval.
That is structurally similar to large-scale recommendation:

- too many candidates to score exhaustively
- state evolves step to step
- multiple objectives conflict
- latency budget is strict

`AegisMixer` tests whether a recommender-style control layer can improve action quality without collapsing controller timing.

### Why suspicious traffic is modeled as slices

This keeps the simulator unified:
- benign traffic competes for service quality
- suspicious traffic competes for the same radio and control budget
- policies must decide whether to isolate, inspect, or tolerate

## Next Extensions

- O-RAN xApp and rApp control-loop structure instead of a single policy interface
- real O-RAN or SDN telemetry traces
- explicit NTN path simulation for satellite or aerial backhaul
- reinforcement learning or intent-based policy optimization
- admission cost model for model placement and offload latency
- learned retrieval and reranking models instead of hand-tuned scoring formulas
