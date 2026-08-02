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

## Why the conclusion is provisional

The model does not include near-memory compute throughput, bank conflicts, compiler support, synchronization, thermal behavior, or hardware counters. The next serious validation step is trace- and measurement-based calibration, not adding more UI.

## Decision record

- Carry near-memory placement forward as the candidate architecture.
- Keep naive tiering as the operational fallback and comparison baseline.
- Reject HBM-only for this capacity point.
- Re-run the sweep whenever workload shape or memory-system assumptions change.
