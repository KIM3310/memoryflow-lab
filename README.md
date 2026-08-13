# MemoryFlow Lab

[![CI](https://github.com/KIM3310/memoryflow-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/KIM3310/memoryflow-lab/actions/workflows/ci.yml)
[![Live results](https://img.shields.io/badge/live-results-0d6447)](https://kim3310.github.io/memoryflow-lab/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab)](https://www.python.org/)

MemoryFlow is a deterministic, first-order model for one decision:

> When weights and a long-context KV cache do not fit in usable HBM, which placement policies remain feasible, and which remote service becomes limiting?

It compares HBM-only placement, a page-aware hot HBM window with remote cold KV, and near-memory attention that scans cold pages in place and exchanges query-dependent partial state. The model separates capacity, useful bytes, physical remote-media bytes, interconnect bytes, compute, and fixed service latency. It is not cycle-accurate and does not predict a named product.

## What changed in 0.3

- KV allocations and cold scans are rounded to `kv_page_tokens`; useful and physical bytes remain separate.
- An explicit HBM runtime reserve is subtracted before weights and resident KV are admitted.
- Peak bandwidth/TOPS inputs are derated by independent effective-efficiency factors.
- Remote-media service and interconnect service have separate bandwidth, traffic, energy, and bottleneck accounting.
- Near-memory mode sends each layer's query and returns `(output, max, sum)` state per query head; it still pays page-rounded media reads and compute.
- `memoryflow analyze` and `/v1/analyses` produce deterministic grids, break-even points, one-at-a-time sensitivity, and a bounded slow-compute counterexample search. The search explicitly reports when no losing point is reachable within valid rates; analysis applies the same page alignment, numerical, and HBM capacity gates as simulation.
- Scenario, result, analysis, measurement, and generated-evidence schemas are versioned and validated with exact key sets and SHA-256 provenance for inputs, model sources, and the generator.

## Reproduce

```bash
make install
make verify
make run
```

Open `http://127.0.0.1:8000`, or inspect:

- [`evidence/benchmark-summary.md`](evidence/benchmark-summary.md)
- [`evidence/measurement-summary.md`](evidence/measurement-summary.md)
- [`site/results.json`](site/results.json)

`make verify` runs Ruff formatting/lint, strict mypy, tests with at least 90% coverage, deterministic evidence regeneration, and exact measurement-artifact checks.

## Model at a glance

```text
versioned scenario + explicit synthetic provenance
        |
        +--> HBM reserve + weights + page-allocated hot KV capacity
        +--> page-allocated remote cold-KV capacity
        +--> effective local compute/HBM roofline
        +--> remote-media read/write service
        +--> interconnect payload/protocol service
        +--> optional page-rounded near-memory attention
        +--> declared overlap
        |
        v
feasibility + latency + throughput + traffic + energy proxy + bottleneck
```

Peak inputs are not silently treated as achieved rates:

```text
effective_rate = peak_rate * efficiency
```

Remote service is:

```text
remote_service = layers * base_latency
               + max(link_bytes / effective_link_bandwidth,
                     media_bytes / effective_media_bandwidth,
                     near_memory_flops / effective_near_memory_tops)
```

See [`docs/model.md`](docs/model.md) for the complete contract, byte directions, page rounding, and omissions.

## Bundled results

The committed scenarios use an explicitly **synthetic** 24 GiB design point, a 2 GiB HBM reserve, FP16 7B GQA weights/KV, an 8,192-token context, and 16 synchronized sequences. The bandwidth, TOPS, efficiencies, latency, and energy coefficients are illustrative design knobs, not vendor specifications and not values derived from the Apple measurement artifacts.

- HBM-only fails capacity after reserve and page allocation.
- A 1,024-token HBM window restores feasibility; its critical remote service is the interconnect.
- Near-memory mode reads the same cold pages from remote media but returns about 0.53 GiB of partial-state link traffic instead of about 946 GiB of page KV traffic over the 64 generated steps.
- The base 12 TOPS near-memory knob wins in this synthetic case. A separate 0.1 TOPS stress scenario loses, so offload is not asserted as a universal win.

The generated analysis computes break-even peak near-memory throughput for four link rates and perturbs declared inputs one at a time. Those ranges are deterministic scenario sensitivity, **not confidence intervals**. They do not cover correlated errors or establish probabilities. Perturbations that cross a capacity or safe-rate boundary remain explicit infeasible records with a reason/headroom and are excluded from the speedup envelope rather than aborting or becoming JSON `null` numeric corruption.

## CLI and API

```bash
# One result; add --steps for per-token records
.venv/bin/memoryflow simulate scenarios/7b-long-context-tiered.json

# Page-aligned HBM-window sweep and latency/energy Pareto front
.venv/bin/memoryflow optimize scenarios/7b-long-context-tiered.json   --windows 128,256,512,1024,2048,4096

# Sensitivity, link/compute grid, break-even, and counterexample
.venv/bin/memoryflow analyze scenarios/7b-long-context-tiered.json
```

The FastAPI service exposes `POST /v1/simulations`, `POST /v1/analyses`, `GET /health`, and the static dashboard. Scenario inputs reject missing or unknown fields, duplicate JSON keys, non-standard numbers, invalid provenance, and unsupported schema versions.

## Measurement boundary

The two committed Apple M4/MPS JSON files contain **aggregate summaries** (median and p95 per input), environment/protocol metadata, and derived validation summaries. Raw timing iterations are not committed.

- The copy artifact checks the shape of `base_latency + bytes / bandwidth` on held-out sizes.
- The one-layer PyTorch SDPA artifact compares an independent copy/GEMM roofline and a calibration-context affine fit on held-out context lengths.
- The artifacts do **not** measure or validate HBM, CXL, remote memory, KV paging, near-memory/PIM, multi-layer decode, or end-to-end serving.
- Apple unified memory and MPS are not substitutes for a discrete HBM/CXL system.
- No CUDA measurement is committed or claimed. [`docs/cuda-validation.md`](docs/cuda-validation.md) is a procedure for a future local CUDA run, not evidence that one occurred.
- Measurement values are not bound to the bundled synthetic hardware profile.

Exact artifact key sets, aggregate-only metadata, derived values, environment agreement, scope exclusions, and source SHA-256 values are checked during `make verify`.

## Documentation

| Document | Purpose |
|---|---|
| [`docs/model.md`](docs/model.md) | equations, service composition, and falsification conditions |
| [`docs/validation.md`](docs/validation.md) | protected claims, uncertainty boundary, and evidence tiers |
| [`docs/experiment-log.md`](docs/experiment-log.md) | hypothesis-by-hypothesis decision record |
| [`docs/architecture.md`](docs/architecture.md) | module/data-flow boundaries |
| [`docs/cuda-validation.md`](docs/cuda-validation.md) | unexecuted local CUDA profiling procedure |
| [`docs/container.md`](docs/container.md) | hardened container workflow |

## Repository map

| Path | Purpose |
|---|---|
| `src/memoryflow/domain.py` | validated workload/system/policy/provenance/result contracts |
| `src/memoryflow/simulator.py` | page-aware capacity, traffic, latency, energy, and bottlenecks |
| `src/memoryflow/analysis.py` | deterministic sensitivity, break-even, and counterexample analysis |
| `src/memoryflow/optimizer.py` | page-aligned window sweep and Pareto filtering |
| `src/memoryflow/measurement.py` | transfer/attention fits over aggregate samples |
| `scripts/build_evidence.py` | deterministic results plus exact input/source provenance |
| `scenarios/` | versioned synthetic inputs with all knobs explicit |
| `evidence/measurements/` | committed MPS aggregate summaries; no raw iterations |
| `site/` | static dashboard generated from the same evidence payload |

## Limits

The simulator omits prefill, queueing, allocator dynamics within a page, kernel launch and fusion details, topology contention, bank conflicts, coherence, failure/retry behavior, thermals, power states, and production scheduling. Energy values are coefficient-based estimates. Fixed latency assumes one coalesced batch transaction per active layer. Read/write bytes share each modeled service budget and the remote media/link/compute paths pipeline ideally inside `max(...)`. These assumptions must be replaced or calibrated before any hardware decision.

## License

MIT
