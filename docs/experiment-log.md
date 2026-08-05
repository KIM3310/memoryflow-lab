# Experiment Log

## Question

For a long-context, batched 7B GQA decode workload on a capacity-constrained accelerator, which KV placement policy is worth carrying forward?

## Hypothesis 1: HBM-only is simplest and therefore best

**Prediction:** keeping all KV in HBM avoids remote latency.

**Test:** account for FP16 weights and the full KV footprint before calculating speed.

**Result:** rejected. The layout is not feasible under the bundled capacity constraint. Performance is irrelevant after a capacity failure.

## Hypothesis 2: Remote tiering solves the problem

**Prediction:** keeping the newest 1,024 tokens in HBM and spilling cold KV restores feasibility.

**Test:** model all cold-KV bytes read during each attention step and expose the non-overlapped transfer time.

**Result:** capacity is restored, but remote movement becomes the dominant component. The idea solves one constraint while creating another.

## Hypothesis 3: Move the operation, not all the data

**Prediction:** a near-memory proxy that reduces cold-KV transfer by 90% beats naive tiering in this case.

**Test:** keep capacity and workload inputs identical; change only the transferred fraction.

**Result:** accepted provisionally. The proxy reduces remote traffic and latency in this scenario.

## Falsification: near-memory compute is too slow

**Prediction:** lowering near-memory throughput far enough must erase the movement benefit.

**Test:** reduce only `near_memory_tops` from 12 to 0.1 while preserving workload, capacity, bandwidth, placement window, and reduction ratio.

**Result:** the winner reverses and `near_memory_compute` becomes the bottleneck. The simulator can therefore reject the favored architecture instead of encoding a fixed conclusion.

## Measurement check: does an independent roofline predict decode attention?

**Prediction:** copy bandwidth and GEMM throughput alone will not fully explain fused attention latency because kernel dispatch, softmax, and kernel-specific utilization are absent from those microbenchmarks.

**Test:** run PyTorch SDPA with one query token, batch 1, 8 heads, head dimension 64, and FP16 KV. Fit no attention parameters for the independent roofline. Separately fit an affine attention model on contexts 256, 1,024, and 4,096, then validate both models on 512, 2,048, and 8,192 tokens.

**Result:** the independent roofline produced 37.46% validation MAPE. The attention-calibrated model produced 4.72% MAPE with 9.66% maximum error. Kernel-specific calibration is therefore required before using the analytical latency as a numerical prediction.

## Measurement check: does bandwidth alone explain transfer time?

**Prediction:** a `bytes / bandwidth` equation fitted at a large transfer will under-predict smaller synchronized transfers because dispatch and synchronization introduce a fixed term.

**Test:** on Apple M4 MPS, use PyTorch `copy_` for 1, 4, 8, 16, 32, and 64 MiB device tensors. Fit on 4, 16, and 64 MiB, then validate on the held-out 1, 8, and 32 MiB sizes.

**Result:** bandwidth-only validation MAPE was 53.21%. An affine `base latency + bytes / bandwidth` fit reduced MAPE to 8.52%. The run supports the equation shape used for remote service, but does not calibrate the synthetic system profile.

## Why the conclusion is provisional

The near-memory path includes a throughput parameter, but not bank conflicts, compiler lowering, synchronization detail, thermal behavior, or hardware counters. The SDPA run is one layer with preallocated KV; it does not include model weights, KV paging, multi-layer execution, or serving. The next validation step is a CUDA/HBM end-to-end decode trace and memory-system calibration, not another policy or interface.

## Decision record

- Carry near-memory placement forward as the candidate architecture.
- Keep naive tiering as the operational fallback and comparison baseline.
- Reject HBM-only for this capacity point.
- Re-run the sweep whenever workload shape or memory-system assumptions change.
