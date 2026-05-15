# NeuroWeave-6g Benchmark Report

## normal

- Best policy: `static_qos`
- Critical slice survival: `1.0`
- Controller p95 latency: `4.696` ms
- Attack leakage: `0.0`
- AI deadline miss rate: `0.0`

| Policy | Critical Survival | Overall SLA | Controller p95 ms | AI Miss Rate | Attack Leakage | False Positive Isolation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aegis_mixer | 1.0 | 1.0 | 4.696 | 0.287 | 0.0 | 0.0 |
| static_qos | 1.0 | 0.9583 | 4.696 | 0.0 | 0.0 | 0.0 |
| failure_aware | 1.0 | 0.8704 | 4.696 | 0.0 | 0.0 | 0.0 |
| throughput_first | 1.0 | 0.7546 | 4.696 | 0.0 | 0.0 | 0.0 |
| security_only | 1.0 | 0.7454 | 4.696 | 0.0 | 0.0 | 0.0 |

- Winner delta vs `static_qos`: critical survival `+0.0`, controller p95 `0.0` ms, AI miss `0.0`, attack leakage `0.0`
- Winner delta vs `failure_aware`: critical survival `+0.0`, controller p95 `0.0` ms, AI miss `0.0`, attack leakage `0.0`

## ai_spike

- Best policy: `aegis_mixer`
- Critical slice survival: `1.0`
- Controller p95 latency: `5.248` ms
- Attack leakage: `0.0`
- AI deadline miss rate: `0.1389`

| Policy | Critical Survival | Overall SLA | Controller p95 ms | AI Miss Rate | Attack Leakage | False Positive Isolation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aegis_mixer | 1.0 | 0.9722 | 5.248 | 0.1389 | 0.0 | 0.0 |
| static_qos | 1.0 | 0.8056 | 5.304 | 0.3333 | 0.0 | 0.0 |
| throughput_first | 0.9815 | 0.75 | 9.3 | 0.0093 | 0.0 | 0.0 |
| security_only | 0.9815 | 0.713 | 20.22 | 0.1481 | 0.0 | 0.0 |
| failure_aware | 0.6852 | 0.7222 | 5.248 | 0.037 | 0.0 | 0.0 |

- Winner delta vs `static_qos`: critical survival `+0.0`, controller p95 `-0.056` ms, AI miss `-0.1944`, attack leakage `0.0`
- Winner delta vs `failure_aware`: critical survival `+0.3148`, controller p95 `0.0` ms, AI miss `0.1019`, attack leakage `0.0`

## encrypted_ddos

- Best policy: `failure_aware`
- Critical slice survival: `0.6296`
- Controller p95 latency: `43.824` ms
- Attack leakage: `0.0`
- AI deadline miss rate: `0.1852`

| Policy | Critical Survival | Overall SLA | Controller p95 ms | AI Miss Rate | Attack Leakage | False Positive Isolation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| failure_aware | 0.6296 | 0.3259 | 43.824 | 0.1852 | 0.0 | 0.0 |
| static_qos | 0.4259 | 0.5259 | 60.816 | 0.2963 | 1.0 | 0.0 |
| aegis_mixer | 0.4074 | 0.6815 | 51.944 | 0.3981 | 0.0 | 0.0 |
| throughput_first | 0.3148 | 0.4259 | 78.912 | 0.4352 | 0.8333 | 0.0 |
| security_only | 0.3148 | 0.3889 | 72.552 | 0.3889 | 0.0 | 0.0 |

- Winner delta vs `static_qos`: critical survival `+0.2037`, controller p95 `-16.992` ms, AI miss `-0.1111`, attack leakage `-1.0`

## mixed_failure

- Best policy: `failure_aware`
- Critical slice survival: `0.4074`
- Controller p95 latency: `56.776` ms
- Attack leakage: `0.0`
- AI deadline miss rate: `0.5278`

| Policy | Critical Survival | Overall SLA | Controller p95 ms | AI Miss Rate | Attack Leakage | False Positive Isolation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| failure_aware | 0.4074 | 0.2259 | 56.776 | 0.5278 | 0.0 | 0.0 |
| static_qos | 0.2778 | 0.3926 | 73.768 | 0.7593 | 1.0 | 0.0 |
| aegis_mixer | 0.2593 | 0.5963 | 63.328 | 0.5648 | 0.0 | 0.0 |
| security_only | 0.2593 | 0.3 | 85.504 | 0.6204 | 0.0 | 0.0 |
| throughput_first | 0.2037 | 0.3444 | 91.864 | 0.6389 | 0.8333 | 0.0 |

- Winner delta vs `static_qos`: critical survival `+0.1296`, controller p95 `-16.992` ms, AI miss `-0.2315`, attack leakage `-1.0`
