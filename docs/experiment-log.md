# Experiment Log

## Question

Can remote KV placement make a synthetic long-context decode configuration feasible, and under what modeled conditions does near-memory partial-state attention beat returning cold KV over the link?

All four bundled profiles explicitly declare `hardware_profile: synthetic` and `measurement_scope: none`. Their knobs are not inferred from Apple M4 data or vendor specifications.

## H1 — HBM-only capacity

**Input:** 7B FP16 weights, FP16 GQA KV, 8,192 prompt tokens, 64 generated tokens, 16 synchronized sequences, 24 GiB HBM, 2 GiB reserve, 16-token KV pages.

**Test:** page-allocate the post-final-write KV at length 8,256 and admit reserve, weights, and hot KV.

**Result:** rejected. The runtime reserve plus weights and full page-allocated KV exceed HBM.

**Decision:** any performance number for this placement would be misleading, so the result remains an explicit capacity failure.

## H2 — Remote sliding window

**Change:** keep only 1,024 tokens per sequence in HBM; place older pages in the remote tier.

**Result:** feasible. Across decode, the cold scan reads about 900.88 GiB from remote media and about 945.92 GiB from the interconnect after the 5% protocol factor. The modeled link is the most frequent bottleneck.

**Decision:** capacity relief does not imply adequate latency; media and link service must be evaluated separately.

## H3 — Near-memory partial-state attention

**Change:** preserve the same placement and media pages, but send per-layer queries to remote compute and return `(output, row_max, row_sum)` per query head.

**Result:** feasible. Remote-media reads remain about 900.88 GiB, while remote-to-accelerator partial-state link reads are about 0.53 GiB. With the synthetic 12 TOPS peak / 60% efficiency knob, remote media becomes limiting and mean decode latency is lower than ordinary tiering.

**Decision:** near-memory reduces **link** movement, not remote-media scanning. The win is conditional on compute, media, link, fixed latency, overlap, page size, and precision.

## H4 — Counterexample

**Change:** reduce only peak near-memory throughput to 0.1 TOPS in the committed stress scenario. Workload, pages, capacities, link/media rates, efficiencies other than near-memory compute, and placement remain fixed.

**Result:** feasible but slower than ordinary tiering; near-memory compute is the bottleneck.

**Decision:** reject the universal claim “near-memory always wins.” The analysis also computes the conditional break-even and constructs a deterministic point below it.

## Sensitivity and break-even

The generated analysis evaluates four peak link bandwidths, seven peak near-memory throughputs, twelve one-at-a-time input axes, and five declared multipliers. A monotone geometric bisection locates the peak near-memory TOPS where the two policies tie. The exact thresholds are regenerated in [`evidence/benchmark-summary.md`](../evidence/benchmark-summary.md).

This sweep is not sampling and its min/max is not a confidence interval. It answers “what does this equation do under these declared perturbations?” rather than “how likely is this outcome on hardware?”

## Separate MPS equation checks

Committed Apple M4 MPS artifacts contain median/p95 aggregate summaries, not timing iterations. Held-out sizes/contexts check transfer and one-layer attention equation shapes. They do not exercise the scenario's HBM reserve, page allocator, CXL/remote paths, PIM, multi-layer decode, CUDA, energy, or end-to-end serving. No MPS fitted value is used above.

## Reproduction

```bash
make verify
.venv/bin/memoryflow simulate scenarios/7b-long-context-tiered.json --steps
.venv/bin/memoryflow analyze scenarios/7b-long-context-tiered.json
```

`make verify` confirms exact input/source hashes, deterministic generated files, schema boundaries, static analysis, and test coverage.
