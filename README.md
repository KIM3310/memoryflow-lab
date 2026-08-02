# MemoryFlow Lab

[![CI](https://github.com/KIM3310/memoryflow-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/KIM3310/memoryflow-lab/actions/workflows/ci.yml)
[![Live evidence](https://img.shields.io/badge/live-evidence-0d6447)](https://kim3310.github.io/memoryflow-lab/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab)](https://www.python.org/)

A reproducible co-design lab for one concrete question:

> When HBM cannot hold an LLM's weights and long-context KV cache, when does tiering KV to remote memory help, and when does data movement erase the gain?

MemoryFlow models autoregressive decode across three placement policies: HBM-only, a hot-window plus remote-memory tier, and a near-memory proxy that reduces cold-KV transfer. It reports feasibility, latency floors, throughput, traffic, capacity, energy proxies, and bottlenecks from versioned synthetic inputs.

**This is an educational first-order model, not a claim about any SK hynix or vendor product.** Hardware profiles are synthetic by design.

## Three-minute evaluation

1. Open the [live evidence dashboard](https://kim3310.github.io/memoryflow-lab/).
2. Read the three hypotheses in [`docs/experiment-log.md`](docs/experiment-log.md).
3. Inspect the equations and limits in [`docs/model.md`](docs/model.md).
4. Run `make verify` to reproduce the checked-in evidence.
5. Change CXL bandwidth or the HBM window in a scenario and see whether the conclusion survives.

## Why this project exists

SK hynix describes Software Solution work as analyzing AI-model behavior, designing KV-cache storage and data-movement paths, and simulating how new memory technology affects AI performance. Its 2026 hiring process also replaces the conventional essay with AI-usage and semiconductor-job evidence, followed by a half-day deep interview focused on applied AI, domain expertise, and fundamental reasoning.

This project is deliberately shaped as inspectable interview evidence:

| Hiring signal | Repository evidence |
|---|---|
| AI workload understanding | GQA-aware KV sizing and prefill/decode separation boundary |
| Memory architecture | HBM capacity, bandwidth, remote tier, movement, and near-memory trade-offs |
| Fundamental reasoning | explicit hypotheses, equations, rejected configuration, sensitivity sweep |
| AI-assisted engineering | documented human decisions and machine-assisted implementation boundary |
| Communication | concise dashboard, model notes, limitations, and Korean interview guide |

Official role and hiring references: [System Architecture & Software Solution](https://news.skhynix.co.kr/ambassador-job-report-ep3/), [2026 half-day deep interview](https://news.skhynix.co.kr/ai-talent-recruit-2026-02/).

## Baseline finding

The bundled case uses a synthetic 24 GiB HBM accelerator, a 7B FP16 GQA model, 8,192-token context, and 16 concurrent sequences.

- **HBM-only:** rejected because weights plus full KV exceed capacity.
- **Naive tiering:** feasible, but cold-KV transfer becomes the dominant cost.
- **Near-memory proxy:** feasible and transfers less cold-KV data in this scenario.
- **Slow near-memory stress:** reverses the win when remote compute cannot keep up.

The third result is not universal. The decision changes with model precision, batch size, context length, HBM window, remote bandwidth, overlap, and the assumed reduction ratio. That conditionality is the point of the lab.

## Architecture

```text
Versioned scenario JSON
        |
        v
Workload model -> capacity gate -> token-step simulator -> metrics
                                             |
                                             +-> policy sweep / Pareto filter
                                             +-> CLI JSON evidence
                                             +-> FastAPI
                                             +-> static review dashboard
```

The numerical core is dependency-light Python. FastAPI is only an adapter; the simulator can run from tests or the CLI without a server.

## Quick start

```bash
make install
make verify
make run
```

Then open `http://127.0.0.1:8000` or the API docs at `http://127.0.0.1:8000/docs`.

Run one scenario:

```bash
.venv/bin/memoryflow simulate scenarios/7b-long-context-tiered.json --steps
```

Sweep placement windows and inspect the Pareto front:

```bash
.venv/bin/memoryflow optimize scenarios/7b-long-context-tiered.json
```

## Repository map

| Path | Purpose |
|---|---|
| `src/memoryflow/domain.py` | validated workload, memory-system, policy, and result contracts |
| `src/memoryflow/simulator.py` | capacity gate, token-step roofline model, traffic and energy accounting |
| `src/memoryflow/optimizer.py` | HBM-window sweep and latency/energy Pareto filtering |
| `scenarios/` | versioned synthetic experiment inputs |
| `tests/` | equations, monotonic properties, failure states, API, CLI, and reproducibility |
| `evidence/` | regenerated benchmark decision record |
| `site/` | static recruiter/reviewer surface generated from the same evidence |
| `docs/` | architecture, model, validation, experiment log, and interview defense |

## Evidence boundary

- No proprietary fab, customer, model-serving, or product data is used.
- Results are analytical estimates, not cycle-accurate simulation or hardware measurements.
- `near_memory` is a parameterized reduction proxy, not an AiM/AiMX implementation.
- A production claim would require traces, hardware counters, calibrated transfer overlap, and validation against a trusted simulator or platform.

## Background bridge

The repository turns material from the **AI Semiconductor Architecture Design and Performance Optimization** course (Seoul ICT Innovation Square / KAIT, July 2026) into a falsifiable software artifact. The course is context; the equations, tests, decisions, and limitations are the evidence.

## License

MIT
