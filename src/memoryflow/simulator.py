from __future__ import annotations

from collections import Counter
from statistics import fmean

from memoryflow.domain import GIB, SimulationRequest, SimulationResult, StepMetrics

ASSUMPTIONS = (
    "Autoregressive decode is modeled as a synchronized batch of concurrent sequences.",
    "Weights are read once per decode step and reused across the batch.",
    "Latency is a first-order roofline estimate, not a product-performance claim.",
    "Near-memory mode is a configurable proxy that reduces cold-KV transfer bytes.",
    "All bundled hardware profiles are synthetic and intentionally avoid vendor specifications.",
)


def _empty_result(request: SimulationRequest, reason: str) -> SimulationResult:
    return SimulationResult(
        workload_name=request.workload.name,
        system_name=request.system.name,
        policy_name=request.policy.name,
        feasible=False,
        rejection_reason=reason,
        mean_decode_latency_ms=0.0,
        p95_decode_latency_ms=0.0,
        throughput_tokens_s=0.0,
        total_hbm_read_gib=0.0,
        total_remote_read_gib=0.0,
        estimated_energy_j=0.0,
        peak_hbm_kv_gib=0.0,
        peak_remote_kv_gib=0.0,
        bottleneck="capacity",
        assumptions=ASSUMPTIONS,
        steps=(),
    )


def simulate(request: SimulationRequest) -> SimulationResult:
    request.validate()
    workload = request.workload
    system = request.system
    policy = request.policy

    hbm_available = system.hbm_capacity_bytes - workload.weight_bytes
    if hbm_available <= 0:
        return _empty_result(request, "model weights exceed HBM capacity")

    steps: list[StepMetrics] = []
    total_energy_j = 0.0

    for generated_index in range(workload.generated_tokens):
        sequence_length = workload.context_tokens + generated_index
        total_kv_tokens = sequence_length * workload.concurrent_sequences

        if policy.kind == "hbm_only":
            hbm_tokens_per_sequence = sequence_length
        else:
            hbm_tokens_per_sequence = min(sequence_length, policy.hbm_window_tokens)

        hbm_kv_bytes = (
            hbm_tokens_per_sequence
            * workload.concurrent_sequences
            * workload.kv_bytes_per_token_per_sequence
        )
        remote_tokens = total_kv_tokens - hbm_tokens_per_sequence * workload.concurrent_sequences
        remote_kv_bytes = remote_tokens * workload.kv_bytes_per_token_per_sequence

        if hbm_kv_bytes > hbm_available:
            return _empty_result(request, "weights plus resident KV cache exceed HBM capacity")
        if remote_kv_bytes > system.remote_capacity_bytes:
            return _empty_result(request, "cold KV cache exceeds remote-memory capacity")

        full_decode_flops = workload.decode_flops(sequence_length)
        cold_tokens_per_sequence = sequence_length - hbm_tokens_per_sequence
        remote_attention_flops = 0.0
        if policy.kind == "near_memory":
            remote_attention_flops = (
                4
                * workload.layers
                * workload.hidden_size
                * cold_tokens_per_sequence
                * workload.concurrent_sequences
            )
        accelerator_flops = full_decode_flops - remote_attention_flops
        compute_ms = accelerator_flops / (system.accelerator_tops * 1_000_000_000_000) * 1000

        hbm_read_bytes = workload.weight_bytes + hbm_kv_bytes
        hbm_ms = hbm_read_bytes / (system.hbm_bandwidth_gbps * 1_000_000_000) * 1000

        if policy.kind == "near_memory":
            transferred_remote_bytes = remote_kv_bytes * (1 - policy.near_memory_reduction_ratio)
        else:
            transferred_remote_bytes = remote_kv_bytes

        remote_transfer_ms = (
            transferred_remote_bytes / (system.remote_bandwidth_gbps * 1_000_000_000) * 1000
            if transferred_remote_bytes
            else 0.0
        )
        near_memory_compute_ms = (
            remote_attention_flops / (system.near_memory_tops * 1_000_000_000_000) * 1000
        )
        remote_service_ms = max(remote_transfer_ms, near_memory_compute_ms)
        if remote_service_ms:
            remote_service_ms += system.remote_base_latency_us / 1000

        local_floor_ms = max(compute_ms, hbm_ms)
        exposed_remote_ms = remote_service_ms * (1 - policy.transfer_overlap_ratio)
        latency_ms = local_floor_ms + exposed_remote_ms

        components = {
            "compute": compute_ms,
            "hbm_bandwidth": hbm_ms,
            "remote_transfer": remote_transfer_ms,
            "near_memory_compute": near_memory_compute_ms,
        }
        bottleneck = max(components, key=components.__getitem__)

        total_energy_j += (
            full_decode_flops * system.compute_energy_pj_per_flop
            + hbm_read_bytes * system.hbm_energy_pj_per_byte
            + transferred_remote_bytes * system.remote_energy_pj_per_byte
        ) * 1e-12

        steps.append(
            StepMetrics(
                token_index=generated_index,
                sequence_length=sequence_length,
                compute_ms=compute_ms,
                hbm_ms=hbm_ms,
                remote_transfer_ms=remote_transfer_ms,
                near_memory_compute_ms=near_memory_compute_ms,
                latency_ms=latency_ms,
                hbm_kv_gib=hbm_kv_bytes / GIB,
                remote_kv_gib=remote_kv_bytes / GIB,
                hbm_read_gib=hbm_read_bytes / GIB,
                remote_read_gib=transferred_remote_bytes / GIB,
                bottleneck=bottleneck,
            )
        )

    mean_latency = fmean(step.latency_ms for step in steps)
    sorted_latency = sorted(step.latency_ms for step in steps)
    p95_index = max(0, int(len(sorted_latency) * 0.95 + 0.999999) - 1)
    p95_latency = sorted_latency[p95_index]
    bottleneck = Counter(step.bottleneck for step in steps).most_common(1)[0][0]

    return SimulationResult(
        workload_name=workload.name,
        system_name=system.name,
        policy_name=policy.name,
        feasible=True,
        rejection_reason=None,
        mean_decode_latency_ms=mean_latency,
        p95_decode_latency_ms=p95_latency,
        throughput_tokens_s=workload.concurrent_sequences * 1000 / mean_latency,
        total_hbm_read_gib=sum(step.hbm_read_gib for step in steps),
        total_remote_read_gib=sum(step.remote_read_gib for step in steps),
        estimated_energy_j=total_energy_j,
        peak_hbm_kv_gib=max(step.hbm_kv_gib for step in steps),
        peak_remote_kv_gib=max(step.remote_kv_gib for step in steps),
        bottleneck=bottleneck,
        assumptions=ASSUMPTIONS,
        steps=tuple(steps),
    )
