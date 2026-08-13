# Validation and Evidence Boundary

## Evidence tiers

MemoryFlow keeps three evidence classes separate:

| Tier | Contents | Permitted claim |
|---|---|---|
| Synthetic scenarios | Explicit capacities, peak rates, efficiencies, latency, energy coefficients, and policies | Conditional behavior of the published equations |
| Apple M4/MPS aggregates | Median/p95 summaries for PyTorch device copy and one-layer SDPA, plus held-out prediction errors | Limited equation-shape checks on that recorded MPS environment |
| Future system validation | HBM/CXL/remote/PIM counters, traces, power, and end-to-end serving | Not present in this repository |

The Apple artifacts do not calibrate the bundled system profile. No committed CUDA measurement exists. Documentation of a CUDA procedure is not evidence that the procedure ran.

## Protected claims

Automated tests protect the following mathematical and provenance properties:

- GQA KV bytes use `kv_heads`; attention-query/partial-state bytes use `attention_heads`.
- Hot and cold capacity is independently page-rounded at the final post-write length.
- Cold useful bytes never exceed physical remote-media scan bytes.
- HBM reserve is unavailable to weights and KV.
- Every peak bandwidth/TOPS input is multiplied by its named efficiency.
- Remote-media time and link time use distinct bytes and effective rates.
- Ordinary tiering returns page KV across the link; near-memory returns `(output, max, sum)` state and still scans remote pages.
- Near-memory compute uses the page-rounded cold extent; accelerator compute removes only useful cold attention work.
- Increasing a non-limiting rate can leave latency unchanged, while increasing a limiting rate cannot make its modeled policy slower.
- A deliberately slow near-memory design reverses the base policy ordering.
- Simulation, optimization, analysis, and evidence generation are deterministic.

These are internal-consistency tests, not hardware accuracy guarantees.

## Exact schema and provenance checks

Scenario schema `2.0` requires exact root and nested key sets. All workload, system, policy, efficiency, energy, page, reserve, and provenance fields must be explicit. Unknown/missing fields, duplicate JSON keys, `NaN`/infinity, booleans used as numbers, unsupported versions, invalid dimensions, and unbounded input sizes are rejected.

Simulation result schema `2.0`, analysis schema `1.0`, measurement schema `2.0`, and evidence schema `2.0` are recorded in outputs. `scripts/build_evidence.py` commits:

- each scenario path, SHA-256, schema, hardware-profile kind, and measurement scope;
- each measurement path, SHA-256, backend, aggregation level, and raw-iteration flag;
- each model-source path/SHA-256 and a deterministic source-set digest;
- the evidence-generator script path/SHA-256; and
- a deterministic scenario-set digest.

The validator recomputes these values and requires embedded measurement objects to exactly equal their source files.

Measurement validators require exact keys at every stored level, disjoint calibration/validation inputs, consistent protocol membership, valid median/p95 summaries, recomputed modeled bytes/FLOPs/bandwidth, recomputed fitted parameters and held-out errors, matching copy/attention environments, `summary_statistics` aggregation, and `raw_iterations_included: false`. The committed pair must identify Apple M4 with the MPS backend and explicitly exclude HBM, CXL, remote memory, near-memory/PIM, and end-to-end validation.

## Measurement results and limits

The MPS attention aggregate contains one query token, batch 1, 8 heads, head dimension 64, and preallocated FP16 K/V for one fused PyTorch SDPA call. The independent copy/GEMM roofline has 37.46% held-out MAPE; an attention-affine fit using separate calibration contexts has 4.72% held-out MAPE. The copy affine equation has 8.52% held-out MAPE.

Those errors quantify only the listed held-out points from one recorded run. Aggregate medians/p95 values do not expose iteration distributions, drift, thermal effects, or run-to-run uncertainty. The attention-affine fit is kernel-specific, not a universal hardware bandwidth. The measurements exclude weights, multiple layers, allocator/paging behavior, remote devices, serving, and power.

## Deterministic uncertainty framing

The generated report varies listed inputs one at a time and reports the resulting speedup envelope, grid winners, break-even points, and a slow-compute counterexample. This is scenario sensitivity, not probabilistic uncertainty quantification. It does not account for correlated changes or model-form error. A robust engineering conclusion requires measured ranges, joint/corner analysis appropriate to those ranges, and independent system traces.

## Required external validation

Before using MemoryFlow for procurement or product claims, collect at minimum:

1. end-to-end decode traces across batch/context/output distributions;
2. allocator and page-fault/eviction traces at the actual page/sector granularity;
3. achieved HBM, interconnect, and remote-media bandwidth in both directions;
4. transaction counts and fixed latency by layer under concurrency;
5. near-memory kernel correctness, partial-state precision, throughput, and media traffic;
6. power/energy measurements with idle and infrastructure boundaries;
7. repeated runs across thermal and software states on the target platform.

Then replace synthetic knobs with cited measurements, declare their uncertainty and provenance, and validate on held-out workloads. Until then, the dashboard supports architecture reasoning only.
