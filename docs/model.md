# Model Contract

## Scope and units

MemoryFlow models synchronized batched autoregressive **decode** with an existing KV cache. Prefill and initial KV placement are excluded. Bytes use exact byte counts internally; capacity/reporting uses GiB (`2^30` bytes), while bandwidth inputs use decimal GB/s (`10^9` bytes/s). TOPS means `10^12` operations/s.

Bundled system values are synthetic design knobs. A result is a deterministic first-order estimate, not a measurement, confidence interval, queueing model, or named-product prediction.

For decode step `i`:

```text
L_i = context_tokens + i                 # KV read extent
L_after = L_i + 1                        # allocation after writing the new KV
B = concurrent_sequences
P = kv_page_tokens
```

The capacity gate uses `context_tokens + generated_tokens`, because the last modeled step writes one more token than it reads. The schema also enforces finite, documented implementation bounds on context, output length, batch, architecture dimensions, page/window sizes, and system coefficients so pathological inputs cannot overflow first-order arithmetic.

## Workload bytes and FLOPs

```text
weight_bytes = parameter_count_b * 10^9 * weight_bits / 8

kv_bytes_per_layer_token_sequence
  = 2 * kv_heads * head_dim * kv_bits / 8
kv_bytes_per_token_sequence
  = layers * kv_bytes_per_layer_token_sequence
```

The factor 2 is key plus value. GQA/MQA uses `kv_heads`, not `attention_heads`. Inputs must satisfy:

```text
attention_heads % kv_heads == 0
attention_heads * head_dim == hidden_size
```

First-order full decode work is:

```text
parameter_flops = 2 * parameter_count
attention_flops = 4 * layers * hidden_size * L_i
decode_flops = (parameter_flops + attention_flops) * B
```

The parameter work scales with batch, while one weight read is reused across the synchronized batch.

## Page-aware placement and capacity

For HBM-only, all `L` tokens are hot. For either remote policy:

```text
hot_tokens  = min(L, hbm_window_tokens)
cold_tokens = L - hot_tokens
allocated(x) = ceil(x / P) * P            # zero maps to zero

hot_allocated_bytes  = allocated(hot_tokens)  * B * kv_bytes_per_token_sequence
cold_allocated_bytes = allocated(cold_tokens) * B * kv_bytes_per_token_sequence
```

`hbm_window_tokens` must be divisible by `P`, so the steady hot window does not straddle a page. Hot and cold tiers are rounded independently; their unused slots are reported as fragmentation.

```text
usable_hbm = hbm_capacity - hbm_reserved
weights <= usable_hbm
weights + hot_allocated_bytes <= usable_hbm
cold_allocated_bytes <= remote_capacity
```

The HBM reserve represents runtime/workspace capacity that the KV allocator cannot consume. A negative capacity headroom produces an explicit infeasible result.

### Capacity versus traffic

Capacity, useful traffic, and physical scan traffic are deliberately different:

- HBM reads use useful hot KV bytes because resident vectors are directly addressable in this first-order model.
- A cold attention scan reads every occupied cold page, so remote-media and ordinary-tier link reads use `cold_allocated_bytes` at that step.
- The newly produced KV is written to HBM. Once the hot window is full, one token per sequence is logically evicted to remote media and crosses the link. Streaming writes transmit the useful token bytes; unwritten padding reserves capacity but is not fabricated as write traffic.
- A later cold scan reads the whole final page, including masked/padded slots. This creates `remote_memory_read_amplification = physical cold reads / useful cold bytes`.

This is a page-granularity transfer model, not a dynamic allocator simulation. A real system that fetches smaller sectors or writes full pages needs different equations.

## Peak and effective rates

All bandwidth and TOPS fields are peak inputs. Each critical path uses a separate effective factor:

```text
effective_hbm_bandwidth   = hbm_bandwidth * hbm_bandwidth_efficiency
effective_link_bandwidth  = remote_bandwidth * remote_bandwidth_efficiency
effective_media_bandwidth = remote_memory_bandwidth
                            * remote_memory_bandwidth_efficiency
effective_accelerator     = accelerator_tops * accelerator_compute_efficiency
effective_near_memory     = near_memory_tops * near_memory_compute_efficiency
```

Efficiencies are explicit values in `(0, 1]`; they are not fitted from the committed MPS artifacts. Every derived bandwidth and compute rate must be at least `1e-9` in its declared GB/s or TOPS unit (one byte/s or 1,000 FLOP/s respectively). This intentionally broad numerical-safety floor rejects subnormal products before division and result serialization.

## Local service

For ordinary tiering, all decode FLOPs execute on the accelerator. Near-memory mode subtracts the **useful** cold attention FLOPs from accelerator work.

```text
compute_ms = accelerator_flops / effective_accelerator
hbm_bytes  = weight_bytes + useful_hot_kv_read + new_kv_write
hbm_ms     = hbm_bytes / effective_hbm_bandwidth
local_floor_ms = max(compute_ms, hbm_ms)
```

The maximum assumes ideal overlap between accelerator compute and HBM traffic at model granularity. Activations, normalization, launch overhead, and temporary workspaces are omitted from traffic and service, while the HBM reserve protects their capacity only coarsely.

## Distinct remote media and interconnect service

Directions are named from the accelerator:

- `interconnect_read`: remote to accelerator;
- `interconnect_write`: accelerator to remote;
- `remote_memory_read/write`: physical remote-media service, independent of link direction.

A protocol multiplier applies only to link payload:

```text
wire_bytes = payload_bytes * (1 + remote_protocol_overhead_ratio)
link_ms  = (interconnect_read + interconnect_write) / effective_link_bandwidth
media_ms = (remote_memory_read + remote_memory_write) / effective_media_bandwidth
```

Reads and writes share each budget. Link and media are not added serially; the model assumes they pipeline and uses their maximum. Because transformer layers are sequential, one coalesced batch transaction per active layer contributes fixed latency:

```text
fixed_ms = layers * remote_base_latency_us / 1000
remote_service_ms = fixed_ms + max(link_ms, media_ms, near_memory_compute_ms)
exposed_remote_ms = remote_service_ms * (1 - transfer_overlap_ratio)
latency_ms = local_floor_ms + exposed_remote_ms
```

The overlap ratio is a declared fractional assumption, not an inferred scheduler result.

## Near-memory partial-state attention

Near-memory mode does **not** apply an arbitrary reduction ratio. It scans page-rounded cold K/V on remote media and exchanges sufficient query-dependent state to combine cold and hot softmax attention.

For every layer, sequence, and query head:

```text
query_payload_bytes
  = layers * B * attention_heads * head_dim * activation_bits / 8

partial_state_payload_bytes
  = layers * B * attention_heads * (head_dim + 2)
    * near_memory_accumulator_bits / 8
```

The returned state is `(output_vector, row_max, row_sum)`; the two scalars support numerically stable combination with the hot attention segment. Query payload is an interconnect write and partial state is an interconnect read. KV eviction remains an interconnect/media write.

Near-memory work includes padded page slots (with masking):

```text
physical_cold_attention_flops
  = 4 * layers * hidden_size * allocated(cold_tokens) * B
near_memory_compute_ms
  = physical_cold_attention_flops / effective_near_memory_tops
```

Cold media read bytes are therefore identical between ordinary tiering and near-memory for the same placement. The link bytes differ; this distinction prevents the model from pretending that near-memory eliminates remote-media service.

## Energy proxy

```text
energy = accelerator_flops * accelerator_compute_energy
       + physical_cold_flops * near_memory_compute_energy
       + hbm_read_write_bytes * hbm_energy_per_byte
       + remote_media_read_write_bytes * remote_memory_energy_per_byte
       + interconnect_wire_bytes * remote_link_energy_per_byte
```

Coefficients are synthetic inputs. Static power, time-dependent power, cooling, and conversion losses are excluded, so this is a comparison proxy rather than a power prediction.

## Reported tails and bottlenecks

`p95_step_latency_ms` is the nearest-rank 95th percentile across deterministic generated-token steps in one run. It is not request-level production p95. The summary bottleneck is the most frequent step label. Ties are deterministic but do not imply physical exclusivity.

## Sensitivity, break-even, and counterexample

`analyze_design_space` provides:

1. a fixed link-bandwidth × near-memory-TOPS grid;
2. a monotone geometric-bisection threshold where near-memory latency first matches ordinary tiering;
3. one-at-a-time multipliers for peak rates, efficiencies, fixed latency, protocol overhead, HBM reserve, and overlap; and
4. a bounded low-near-memory-compute search that either constructs an ordinary-tiering win or reports `not_reachable_within_bounds` (for example when no cold KV exists or the remote link is already at its safe-rate floor).

The one-at-a-time minimum/maximum is not a joint bound or statistical confidence interval. It excludes correlations and any unlisted model discrepancy. Capacity-only perturbations such as HBM reserve can leave latency unchanged while still changing headroom. If a perturbation crosses a capacity or safe-rate boundary, the report retains that point as `feasible: false` with its rejection reason and available capacity headroom; latency, speedup, and winner are `null`, and the point is excluded from the finite speedup envelope. A break-even value is conditional on every other scenario input.

## Falsification conditions

Replace or reject this model when measurements show that:

- cache allocation or transfer granularity differs materially from `kv_page_tokens`;
- HBM reserve varies with context/batch enough to alter feasibility;
- achieved rates cannot be represented by stable efficiency factors;
- media, link, and compute do not pipeline as `max(...)`;
- per-layer fixed latency is hidden, multiplied by more transactions, or dominated by queueing;
- near-memory hardware cannot emit sufficient stable `(output, max, sum)` state;
- attention work over padded slots differs from the masked physical extent;
- framework/kernel overhead or topology contention changes the selected policy.

The committed MPS aggregate summaries test limited equation shapes only; they do not falsify or validate the HBM/remote/near-memory system model.
