# Validation Strategy

## Claims the test suite protects

1. GQA KV sizing uses `kv_heads`, not `attention_heads`.
2. Weight and KV precision change capacity in the expected direction.
3. HBM-only fails clearly when weights plus KV exceed capacity.
4. Tiering can restore capacity when remote capacity is sufficient.
5. More remote bandwidth or overlap cannot increase modeled tiering latency.
6. A larger HBM window reduces remote traffic but consumes more HBM.
7. Near-memory reduction lowers transferred cold-KV bytes by the configured ratio.
8. Slow near-memory compute can erase the transfer benefit and change the winner.
9. The same scenario produces byte-for-byte stable JSON evidence.
10. Bundled systems are labeled synthetic to block accidental product claims.
11. Transfer-model fitting uses disjoint calibration and validation sizes.
12. Stored error metrics are recomputed from raw measurement samples during verification.
13. Attention calibration contexts are disjoint from validation contexts.
14. Copy bandwidth, GEMM throughput, modeled FLOPs, and attention errors are recomputed.

Serialized evidence rounds floating-point metrics to ten decimal places so Python and operating-system differences below the reporting precision do not create false changes.

## Verification layers

| Layer | Check |
|---|---|
| Domain | invalid dimensions, precisions, head relationships, and policy values fail fast |
| Equations | closed-form weight and KV examples |
| Properties | monotonic bandwidth, overlap, and window behavior |
| Failure states | HBM, weight, and remote-capacity rejection |
| Interfaces | CLI summaries, detailed traces, API validation, dashboard delivery |
| Evidence | generated JSON and Markdown must match committed artifacts |
| Measurement | GPU backend metadata, raw samples, fitted parameters, and held-out error |

## Remaining validation gaps

The committed PyTorch runs validate a local device-copy equation and a single fused SDPA layer on Apple unified memory. They do not calibrate HBM, CXL, remote-memory service, KV paging, multi-layer decoding, or end-to-end serving. Those require CUDA/server traces, hardware counters, and a trusted memory simulator, with topology and power-state metadata versioned alongside the samples.
