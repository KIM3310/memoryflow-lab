# Model Contract

## Scope

MemoryFlow is a first-order analytical model for comparing placement policies during batched autoregressive LLM decode. It is useful for rejecting impossible configurations and exposing dominant data-movement trade-offs. It is not cycle-accurate and does not predict a named product.

## Equations

### Model weights

```text
weight_bytes = parameter_count * weight_bits / 8
```

Weights are assumed to remain in HBM and to be read once per synchronized batch step.

### KV cache

For grouped-query attention:

```text
kv_bytes_per_token_per_sequence
  = 2 * layers * kv_heads * head_dim * kv_bits / 8
```

The factor `2` represents key and value. Using `kv_heads` rather than all attention heads is essential for GQA/MQA.

### Capacity gate

```text
hbm_available_for_kv = hbm_capacity - weight_bytes
resident_kv <= hbm_available_for_kv
cold_kv <= remote_capacity
```

Infeasible layouts return a reason instead of a misleading zero-cost result.

### Decode work

```text
parameter_flops = 2 * parameters
attention_flops = 4 * layers * hidden_size * sequence_length
decode_flops = (parameter_flops + attention_flops) * batch
```

This intentionally omits kernel launch, normalization, activation, routing, quantization overhead, and device utilization loss.

### Roofline latency floor

```text
compute_ms = decode_flops / accelerator_tops
hbm_ms = (weight_bytes + resident_kv_read) / hbm_bandwidth
local_floor_ms = max(compute_ms, hbm_ms)
latency_ms = local_floor_ms + remote_ms * (1 - overlap_ratio)
```

The use of `max` expresses overlap between local compute and HBM movement. Remote overlap is exposed as a separate, configurable assumption.

### Near-memory proxy

```text
transferred_cold_kv = cold_kv * (1 - reduction_ratio)
remote_attention_flops = 4 * layers * hidden_size * cold_tokens * batch
remote_service_ms = max(reduced_transfer_ms, remote_attention_flops / near_memory_tops)
```

The accelerator no longer performs the cold-token portion of attention, while the remote tier must complete that work. This creates an explicit counterexample: if `near_memory_tops` is too low, saved transfer bytes do not make the policy faster. The model still omits a PIM instruction set, bank conflicts, compiler lowering, synchronization detail, and thermal limits. Those omissions prevent the result from being presented as product performance.

## What would falsify the current result?

- HBM capacity large enough to retain the full KV cache.
- Quantized weights or KV that remove the capacity pressure.
- Remote bandwidth or overlap high enough that naive tiering is no longer dominant.
- A near-memory implementation whose compute/coordination cost exceeds saved transfer time.
- A workload whose KV traffic is small relative to compute or weight movement.
