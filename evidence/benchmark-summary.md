# Reproducible Benchmark Summary

Scenario-set SHA-256: `6c494ca01417e6b51ee6aec7359b084d57e17c1cca8e8b9b48347cded529e302`
Model-source SHA-256: `b2ff2bfb644c3aca0a1307731b046add648dd46a6b8c5f3b1c44c06bae429a7c`
Generator SHA-256: `dd9d905137eba4496e3865caa8dbc167703c286f165fd95f2a4a38aafb6cbfc8`

All policy results below use bundled synthetic hardware knobs. They are deterministic first-order estimates, not measurements or product claims.

| Policy | Feasible | Mean decode (ms) | Throughput (token/s) | Remote media read (GiB) | Link read (GiB) | Page read amp. | Bottleneck |
|---|---:|---:|---:|---:|---:|---:|---|
| HBM only | no | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000× | capacity |
| Remote sliding window 1024 | yes | 254.30 | 62.92 | 900.88 | 945.92 | 1.0010× | remote_link |
| Near-memory cold KV 1024 | yes | 133.67 | 119.70 | 900.88 | 0.53 | 1.0010× | remote_memory |
| Near-memory stress 0.1 TOPS | yes | 678.96 | 23.57 | 900.88 | 0.53 | 1.0010× | near_memory_compute |

## Decision

- HBM-only is rejected after subtracting the explicit runtime reserve and page-allocating the full KV cache.
- Sliding-window tiering is feasible, but page-rounded cold KV traverses both remote media and the bandwidth-limited interconnect.
- Near-memory attention still scans the same remote pages; it sends queries and returns `(output, max, sum)` partial state instead of returning cold KV.
- The base near-memory point wins, while the committed slow-compute stress point loses. This is a model counterexample, not a hardware observation.

## Deterministic sensitivity

Across the listed feasible one-at-a-time multipliers, near-memory speedup spans 1.045× to 3.625×.
The computed counterexample lowers synthetic near-memory peak throughput from 12.000 TOPS to 0.142202 TOPS; the winner changes from near_memory to sliding_window.

| Peak link bandwidth (GB/s) | Near-memory break-even peak TOPS | Status |
|---:|---:|---|
| 32 | 0.142202 | within_bounds |
| 64 | 0.284405 | within_bounds |
| 128 | 0.568810 | within_bounds |
| 256 | 0.597251 | within_bounds |

The envelope is not a confidence interval: it covers only declared deterministic perturbations. Correlated uncertainty, queueing, kernels, topology, and hardware behavior remain outside the model.

## Measurement boundary

The embedded Apple M4 MPS artifacts contain median/p95 aggregate summaries, not raw iterations. They check a device-copy equation and one-layer SDPA prediction shape; they do not validate HBM, CXL, remote media, near-memory/PIM, CUDA, or end-to-end serving. No measurement value calibrates the synthetic scenarios.
