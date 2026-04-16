# Paper Map

This repo is intentionally grounded in the papers and notes already present in your Brain.

## AI-RAN in Action: Turning 5G Infrastructure into an Intelligent Growth Platform

Used for:
- shared AI and RAN infrastructure framing
- mission-critical RAN functions receiving priority access to compute
- edge-hosted AI workloads competing with telecom functions
- autonomous network operations and agentic optimization

Implemented as:
- shared PRB and GPU budget per cell
- mission-critical reserve path in the `failure_aware` policy
- AI spike and mixed failure scenarios

## 6G network architecture – a proposal for early alignment

Used for:
- cloud-native and AI-native architecture direction
- interface and network-function centric design
- service assurance, predictability, resilience, and automation

Implemented as:
- controller backlog and p95 decision latency as first-class metrics
- multi-cell, network-function style simulator rather than a single heuristic script
- benchmark focus on service assurance outcomes, not raw throughput only

## Comparative Detection of DDoS Attacks on Software-Defined 5G Network Data Using Deep Neural Networks and Machine Learning Methods

Used for:
- controller vulnerability in SDN-style 5G systems
- JA4-style encrypted traffic analysis
- realistic threat framing around control-plane overload

Implemented as:
- suspicious encrypted burst slices
- anomaly-score driven isolation and inspection logic
- attack leakage rate and false positive isolation rate metrics

## Emerging Space Communication and Network Technologies for Sixth-Generation Ubiquitous Connectivity

Used for:
- heterogeneity of future 6G systems
- terrestrial plus non-terrestrial operational complexity

Implemented as:
- sustained backhaul penalties in mixed-failure scenarios
- explicit note that future versions should model NTN paths separately instead of only approximating them

