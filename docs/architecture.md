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

