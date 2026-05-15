# Benchmark Plan

## Main Question

Which control policy best preserves mission-critical service continuity when AI-RAN networks face simultaneous compute pressure, slice contention, and encrypted attack traffic?

## Scenarios

- `normal`
- `ai_spike`
- `encrypted_ddos`
- `mixed_failure`

## Policies

- `static_qos`
- `throughput_first`
- `security_only`
- `failure_aware`
- `aegis_mixer`

## Required Metrics

- critical slice survival rate
- overall SLA rate
- controller p95 latency
- controller queue peak
- AI deadline miss rate
- attack leakage rate
- false positive isolation rate

## Success Condition

The project only claims a policy is better if it improves critical continuity and attack containment without hiding a controller-latency collapse somewhere else.

## Expected Tradeoff Surfaces

- `failure_aware` should dominate when attack containment and mission-critical continuity matter most.
- `aegis_mixer` should be strongest when action prioritization under compute surge matters more than worst-case survivability.
- Any claim that a policy is "best" must specify the regime and the objective, not just one metric.
