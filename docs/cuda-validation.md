# CUDA Validation Runbook

This runbook produces local CUDA measurements and a PyTorch Profiler trace without
overwriting the committed Apple M4 MPS references.

## Requirements

- NVIDIA GPU visible in `nvidia-smi`
- CUDA-enabled PyTorch installed through `make install-measure`
- Enough device memory for the selected GEMM size and attention contexts

The commands fail instead of falling back to CPU when CUDA is unavailable.

## Measurement

```bash
.measure-venv/bin/python -m scripts.measure_torch \
  --device cuda \
  --device-label "<exact nvidia-smi model>" \
  --warmup 15 \
  --repeats 60 \
  --output evidence/measurements/local-cuda-copy.json

.measure-venv/bin/python -m scripts.measure_attention \
  --device cuda \
  --device-label "<exact nvidia-smi model>" \
  --warmup 15 \
  --repeats 60 \
  --gemm-repeats 30 \
  --output evidence/measurements/local-cuda-attention.json

.measure-venv/bin/python -m scripts.build_measurement_summary \
  --copy-input evidence/measurements/local-cuda-copy.json \
  --attention-input evidence/measurements/local-cuda-attention.json \
  --output evidence/measurements/local-cuda-summary.md
```

## Kernel profile

```bash
make profile-cuda VENV=.measure-venv
```

This writes ignored files under `evidence/profiles/local-cuda-attention/`:

- `summary.json`: device metadata, protocol, and top operators by self device time
- `trace.json`: Chrome/PyTorch Profiler trace

Open the trace in `chrome://tracing` or TensorBoard's profiler view. Record the dominant
SDPA kernel, host-side gaps, memory allocation events, and unexpected synchronizations.

## Review gate

Before treating a CUDA run as reference evidence:

1. confirm the device name, compute capability, CUDA, cuDNN, PyTorch, dtype, and protocol;
2. repeat the run in at least three fresh processes and compare medians;
3. keep calibration and validation inputs disjoint;
4. inspect the trace for compilation, allocation, or synchronization inside the timed region;
5. retain raw JSON and the exact commands used;
6. state that one fused SDPA layer is not end-to-end LLM serving;
7. do not label a CUDA device-copy result as CXL or remote-memory bandwidth.

Replacing the committed reference files is a separate review decision. A local run should
remain `local-*` until its environment and repeatability have been checked.
