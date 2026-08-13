# Architecture

MemoryFlow is intentionally small: immutable contracts feed pure deterministic calculations, while file/API/site layers only validate, serialize, and present them.

## Modules

- `domain.py` defines validated workload, memory-system, policy, scenario-provenance, step, and result contracts. It also centralizes units, limits, effective-rate properties, and stable serialization.
- `io.py` enforces scenario schema `2.0`, exact nested key sets, duplicate-key rejection, finite standard JSON, file-size limits, and deterministic JSON writing.
- `simulator.py` implements post-write page-aware capacity gates and per-token useful/physical/link traffic, local/remote service, energy, and bottleneck accounting.
- `optimizer.py` validates page-aligned window sweeps and computes the latency/energy Pareto front.
- `analysis.py` compares ordinary and near-memory policies on a deterministic grid, uses monotone geometric bisection for compute break-even, performs one-at-a-time sensitivity, and constructs a counterexample.
- `measurement.py` fits transfer and attention equations from aggregate samples; it has no dependency on scenario hardware knobs.
- `api.py` exposes simulation/analysis and mounts the generated static site.

## Evidence flow

```text
scenarios/*.json --exact schema--> SimulationRequest
       |                                |
       |                                +--> simulate / optimize / analyze
       |                                             |
       +--SHA-256 manifest---------------------------+
                                                     v
Apple M4 MPS aggregate JSON --strict validation--> build_evidence.py
                |                                    |
                +--separate evidence tier-------------+
                                                     v
                           site/results.json + benchmark-summary.md
                                                     |
                                                     v
                                      static dashboard / FastAPI
```

`build_evidence.py` records each input, model-source, and generator-script SHA-256 and validates its own exact evidence schema before writing. It does not copy MPS-derived rates into synthetic scenarios. The static site makes both evidence tiers visible but labels their boundary.

## Trust boundaries

1. **Untrusted scenario input:** exact schemas, domain validation, size limits, and finite-number checks run before simulation.
2. **Synthetic computation:** pure dataclass inputs produce deterministic outputs; no network, clock, random source, or device probe is used.
3. **Committed measurement summaries:** exact aggregate schemas and all derived fields are recomputed; raw iterations are absent.
4. **Presentation:** the site reads only generated `results.json`; it does not recompute model equations in JavaScript.

This design keeps mathematical changes reviewable: changing capacity, traffic, or service equations requires corresponding domain, tests, evidence, and documentation changes.
