# Reproducible Benchmark Summary

Scenario input SHA-256: `75e9365b297da4833adb072a79029a8c3244e58fb47f4463dbcc9a52449d0c33`

These are synthetic first-order estimates for architecture comparison, not measured product claims.

| Policy | Feasible | Mean decode (ms) | Throughput (token/s) | Remote read (GiB) | Bottleneck |
|---|---:|---:|---:|---:|---|
| HBM only | no | 0.00 | 0.00 | 0.00 | capacity |
| CXL sliding window 1024 | yes | 171.29 | 93.41 | 899.94 | remote_transfer |
| Near-memory cold KV 1024 | yes | 33.28 | 480.79 | 89.99 | remote_transfer |
| Near-memory stress 0.1 TOPS | yes | 410.50 | 38.98 | 89.99 | near_memory_compute |

## Decision

- HBM-only is rejected because model weights plus full long-context KV exceed capacity.
- Naive tiering restores feasibility but exposes remote-transfer latency.
- The near-memory proxy reduces transferred cold-KV bytes and wins in this scenario.
- The slow-compute stress case reverses that win and exposes near-memory compute.
- The conclusion is conditional: change workload or bandwidth inputs and regenerate the evidence.
