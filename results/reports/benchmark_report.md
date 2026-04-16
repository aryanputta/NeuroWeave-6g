# NeuroWeave-6g Benchmark Report

## normal

- Best policy: `static_qos`
- Critical slice survival: `1.0`
- Controller p95 latency: `4.696` ms
- Attack leakage: `0.0`
- AI deadline miss rate: `0.0`

## ai_spike

- Best policy: `static_qos`
- Critical slice survival: `1.0`
- Controller p95 latency: `5.304` ms
- Attack leakage: `0.0`
- AI deadline miss rate: `0.3333`

## encrypted_ddos

- Best policy: `failure_aware`
- Critical slice survival: `0.6296`
- Controller p95 latency: `43.824` ms
- Attack leakage: `0.0`
- AI deadline miss rate: `0.1852`

## mixed_failure

- Best policy: `failure_aware`
- Critical slice survival: `0.4074`
- Controller p95 latency: `56.776` ms
- Attack leakage: `0.0`
- AI deadline miss rate: `0.5278`
