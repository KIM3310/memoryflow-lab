# Architecture

## Boundaries

- `domain.py` owns validated units and contracts.
- `simulator.py` owns deterministic equations and failure states.
- `optimizer.py` owns experiment enumeration and Pareto selection.
- `io.py` owns JSON translation only.
- `cli.py` and `api.py` are adapters; neither contains simulation rules.
- `measurement.py` fits and validates transfer and attention latency models from raw samples.
- `scripts/build_evidence.py` regenerates derived artifacts from committed scenarios and measurements.

## Data flow

```text
scenario JSON
  -> typed request
  -> capacity checks
  -> per-token placement and traffic accounting
  -> roofline latency and energy proxy
  -> summary + trace
  -> CLI/API/evidence builder
```

The measurement path is separate from simulation:

```text
PyTorch GPU copy + SDPA samples
  -> disjoint calibration/validation inputs
  -> independent roofline + kernel-calibrated fits
  -> held-out error report
  -> committed JSON + generated Markdown/dashboard summary
```

## Why not an LLM-generated recommendation?

The architecture decision is deterministic and inspectable. An LLM could explain the result, but it must not invent the result. Keeping model narration outside the numerical core makes review, testing, and disagreement easier.
