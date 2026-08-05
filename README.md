# MemoryFlow Lab

[![CI](https://github.com/KIM3310/memoryflow-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/KIM3310/memoryflow-lab/actions/workflows/ci.yml)
[![Live results](https://img.shields.io/badge/live-results-0d6447)](https://kim3310.github.io/memoryflow-lab/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab)](https://www.python.org/)

MemoryFlow is a first-order model for one question:

> When model weights and a long-context KV cache do not fit in HBM, which placement policies remain feasible, and when does remote data movement dominate decode latency?

It compares three policies under versioned synthetic inputs:

- HBM-only placement;
- a hot HBM window with a remote cold tier;
- a near-memory proxy that reduces cold-KV transfer but adds remote compute.

The simulator reports capacity failures, per-step latency components, throughput, traffic, energy proxies, and bottlenecks. It is not cycle-accurate and does not model a named product.

## Reproduce

```bash
make install
make verify
make run
```

Open `http://127.0.0.1:8000`, or inspect the generated files directly:

- [`evidence/benchmark-summary.md`](evidence/benchmark-summary.md)
- [`evidence/measurement-summary.md`](evidence/measurement-summary.md)
- [`site/results.json`](site/results.json)

`make verify` runs formatting, static analysis, unit/property tests, scenario regeneration, and measurement-data validation.

## Analytical model

```text
scenario JSON
      |
      v
workload + memory system + placement policy
      |
      +--> weight/KV capacity gates
      +--> decode FLOPs and HBM traffic
      +--> remote transfer and near-memory service
      +--> compute/memory overlap
      |
      v
latency + throughput + traffic + energy proxy + bottleneck
```

The main equations and omissions are defined in [`docs/model.md`](docs/model.md). Key properties are tested rather than inferred from fixed examples: increasing remote bandwidth or overlap cannot increase modeled tiering latency, increasing the HBM window reduces remote traffic, and insufficient near-memory throughput can reverse the selected policy.

## PyTorch GPU validation

Two measurements test different parts of the analytical model. Both use disjoint calibration and validation inputs.

### Decode attention

The main measurement runs PyTorch `scaled_dot_product_attention` with one query token and preallocated KV tensors. The committed Apple M4 MPS run uses batch 1, 8 heads, head dimension 64, and FP16.

- Calibration contexts: 256, 1,024, and 4,096 tokens
- Validation contexts: 512, 2,048, and 8,192 tokens
- Timing: 15 warmups and 60 synchronized measurements per context

| Analytical model | Held-out MAPE | Maximum error |
|---|---:|---:|
| Independent copy/GEMM roofline | 37.46% | 51.30% |
| Attention-calibrated affine model | 4.72% | 9.66% |

The first model uses separately measured copy bandwidth, fixed latency, and GEMM throughput. Its error shows that independent peak-style microbenchmarks do not directly predict a fused attention kernel. Fitting only the calibration contexts reduces the error on the three unseen context lengths.

### Device-copy support measurement

The second measurement checks the remote-transfer equation shape:

```text
transfer_ms = base_latency_us / 1000 + bytes / bandwidth
```

| Transfer model | Held-out MAPE | Maximum error |
|---|---:|---:|
| `bytes / bandwidth` | 53.21% | 90.62% |
| `base latency + bytes / bandwidth` | 8.52% | 17.12% |

Raw samples, environment metadata, fitted parameters, and per-point prediction errors are committed in:

- [`evidence/measurements/apple-m4-mps-attention.json`](evidence/measurements/apple-m4-mps-attention.json)
- [`evidence/measurements/apple-m4-mps-copy.json`](evidence/measurements/apple-m4-mps-copy.json)

All derived metrics are recomputed from the raw samples during `make verify`.

To repeat both measurements on a CUDA or MPS device:

```bash
make install-measure
make measure
```

Local runs write ignored `local-*.json` files and cannot overwrite committed references accidentally. Both scripts accept an explicit device label, output path, warmup count, repeat count, and calibration/validation sets.

## Scenario results

The bundled case uses a synthetic 24 GiB accelerator profile, a 7B FP16 GQA workload, an 8,192-token context, and 16 concurrent sequences.

- HBM-only is rejected because weights plus the full KV cache exceed capacity.
- Windowed tiering restores capacity but exposes remote-transfer latency.
- The near-memory proxy reduces transfer in the base scenario.
- Reducing only near-memory throughput reverses that result.

These results are conditional on model precision, batch size, context length, HBM window, bandwidth, overlap, and the proxy reduction ratio.

## CLI

Run one scenario:

```bash
.venv/bin/memoryflow simulate scenarios/7b-long-context-tiered.json --steps
```

Sweep HBM windows and emit the latency-energy Pareto front:

```bash
.venv/bin/memoryflow optimize scenarios/7b-long-context-tiered.json
```

## Container

```bash
make docker-verify
```

The production image runs as UID `10001`, uses a read-only filesystem during the smoke test, drops Linux capabilities, and verifies the live API and static dashboard. See [`docs/container.md`](docs/container.md).

## Repository map

| Path | Purpose |
|---|---|
| `src/memoryflow/domain.py` | workload, memory-system, policy, and result contracts |
| `src/memoryflow/simulator.py` | capacity gates, token-step latency, traffic, and energy accounting |
| `src/memoryflow/optimizer.py` | HBM-window sweep and Pareto filtering |
| `src/memoryflow/measurement.py` | transfer and attention model fitting with held-out comparison |
| `scripts/measure_torch.py` | synchronized PyTorch GPU copy measurement |
| `scripts/measure_attention.py` | synchronized PyTorch SDPA/KV attention measurement |
| `scenarios/` | versioned synthetic experiment inputs |
| `evidence/` | generated scenario summaries and committed raw measurements |
| `tests/` | equations, properties, failure states, interfaces, and measurement fitting |
| `site/` | static results dashboard generated from the same inputs |

## Limits

- Synthetic scenarios are analytical estimates, not measured product performance.
- The measured attention path is one fused layer with preallocated KV, not end-to-end decoding.
- Apple unified memory and MPS are not substitutes for HBM, CXL, or a CUDA server.
- The near-memory policy is a parameterized proxy, not an implementation.
- Kernel launch, paging, bank conflicts, compiler lowering, synchronization topology, power states, and thermal effects are outside the simulator.
- End-to-end validation requires framework traces, hardware counters, and a trusted memory simulator or server platform.

## License

MIT
