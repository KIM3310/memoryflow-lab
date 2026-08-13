# PyTorch GPU Aggregate Measurement Summary

## Environment

- Device: `Apple M4 (10-core GPU)`
- Backend: `mps`
- PyTorch: `2.8.0`
- Dtype: `float16`
- Attention run: `2026-08-05T09:26:05.436535+00:00`
- Copy run: `2026-08-05T09:25:55.262960+00:00`

## Decode attention measurement

PyTorch SDPA shape: batch 1, 8 heads, head dimension 64, one query token.

| Split | Context | Median (ms) | p95 (ms) | Modeled bytes (MiB) |
|---|---:|---:|---:|---:|
| calibration | 256 | 0.2146 | 0.2577 | 0.50 |
| validation | 512 | 0.2407 | 0.3014 | 1.00 |
| calibration | 1,024 | 0.2594 | 0.2815 | 2.00 |
| validation | 2,048 | 0.2764 | 0.3073 | 4.00 |
| calibration | 4,096 | 0.3388 | 0.5768 | 8.00 |
| validation | 8,192 | 0.4250 | 0.5219 | 16.00 |

Calibration contexts are disjoint from validation contexts.

| Model | Validation MAPE | Max error |
|---|---:|---:|
| independent copy/GEMM roofline | 37.46% | 51.30% |
| attention-calibrated affine | 4.72% | 9.66% |

| Validation context | Measured (ms) | Roofline (ms) | Calibrated (ms) |
|---:|---:|---:|---:|
| 512 | 0.2407 | 0.3044 | 0.2319 |
| 2,048 | 0.2764 | 0.3721 | 0.2787 |
| 8,192 | 0.4250 | 0.6430 | 0.4660 |

## Supporting device-copy measurement

Calibration uses 4, 16, 64 MiB transfers. Validation uses separate 1, 8, 32 MiB transfers.

| Split | Size (MiB) | Median (ms) | p95 (ms) | Observed GB/s |
|---|---:|---:|---:|---:|
| validation | 1 | 0.2833 | 0.3217 | 3.70 |
| calibration | 4 | 0.3609 | 0.3919 | 11.62 |
| validation | 8 | 0.4497 | 0.4942 | 18.66 |
| calibration | 16 | 0.7056 | 0.7648 | 23.78 |
| validation | 32 | 1.0160 | 1.1096 | 33.03 |
| calibration | 64 | 1.7007 | 1.7916 | 39.46 |

| Transfer model | Validation MAPE | Max error |
|---|---:|---:|
| bytes / bandwidth | 53.21% | 90.62% |
| base latency + bytes / bandwidth | 8.52% | 17.12% |

## Boundary

The Apple M4 (10-core GPU) MPS artifacts contain median/p95 aggregate summaries, not raw iterations. The attention run measures one fused PyTorch SDPA layer with preallocated KV tensors; its declared exclusions are: model weights, multi-layer execution, KV allocation/paging, HBM, CXL, remote memory, near-memory/PIM, or end-to-end serving. The copy run checks the fixed-latency transfer equation shape on MPS; its declared exclusions are: HBM, CXL, remote memory, near-memory/PIM, LLM end-to-end latency, or vendor products. These artifacts do not provide measurements for CUDA or other backends. Neither artifact calibrates the synthetic scenarios or supports a named-product claim.
