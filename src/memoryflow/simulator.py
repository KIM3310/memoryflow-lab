from __future__ import annotations

import math
from collections import Counter
from statistics import fmean

from memoryflow.domain import GIB, SimulationRequest, SimulationResult, StepMetrics

ASSUMPTIONS = (
    "Autoregressive decode is a synchronized batch; prefill and initial KV placement are excluded.",
    "Weights are resident in HBM, read once per decode step, and reused across the batch.",
    "Peak rates are derated by explicit efficiency inputs before roofline calculations.",
    "Remote KV scans are page-rounded; useful bytes, media bytes, and link bytes are separate.",
    "Remote reads and writes share one modeled link/media budget and can pipeline with each other.",
    "Remote fixed latency assumes one coalesced batch transaction per active layer.",
    "Near-memory attention sends each query and returns (output, max, sum) state per query head.",
    "Near-memory compute includes masked work across the page-rounded cold-token extent.",
    "All results are deterministic first-order estimates, not confidence intervals "
    "or product claims.",
)


def _round_up_tokens(tokens: int, page_tokens: int) -> int:
    if tokens <= 0:
        return 0
    return math.ceil(tokens / page_tokens) * page_tokens


def _placement(
    request: SimulationRequest, sequence_length: int
) -> tuple[int, int, float, float, float, float]:
    workload = request.workload
    policy = request.policy
    if policy.kind == "hbm_only":
        hot_tokens = sequence_length
    else:
        hot_tokens = min(sequence_length, policy.hbm_window_tokens)
    cold_tokens = sequence_length - hot_tokens
    hot_allocated_tokens = _round_up_tokens(hot_tokens, policy.kv_page_tokens)
    cold_allocated_tokens = _round_up_tokens(cold_tokens, policy.kv_page_tokens)
    bytes_per_token = workload.kv_bytes_per_token_per_sequence
    batch = workload.concurrent_sequences
    return (
        hot_tokens,
        cold_tokens,
        hot_tokens * batch * bytes_per_token,
        cold_tokens * batch * bytes_per_token,
        hot_allocated_tokens * batch * bytes_per_token,
        cold_allocated_tokens * batch * bytes_per_token,
    )


def _finite_result(result: SimulationResult) -> SimulationResult:
    result.assert_finite()
    return result


def _empty_result(
    request: SimulationRequest,
    reason: str,
    peak_hbm_kv_bytes: float,
    peak_remote_kv_bytes: float,
    fragmentation_pct: float,
    hbm_headroom_bytes: float,
    remote_headroom_bytes: float,
) -> SimulationResult:
    return _finite_result(
        SimulationResult(
            workload_name=request.workload.name,
            system_name=request.system.name,
            policy_name=request.policy.name,
            input_hardware_profile=request.provenance.hardware_profile,
            feasible=False,
            rejection_reason=reason,
            mean_decode_latency_ms=0.0,
            p95_step_latency_ms=0.0,
            throughput_tokens_s=0.0,
            total_hbm_read_gib=0.0,
            total_hbm_write_gib=0.0,
            total_cold_kv_useful_gib=0.0,
            total_remote_memory_read_gib=0.0,
            total_remote_memory_write_gib=0.0,
            total_interconnect_read_gib=0.0,
            total_interconnect_write_gib=0.0,
            remote_memory_read_amplification=0.0,
            interconnect_read_per_remote_byte=0.0,
            mean_remote_service_ms=0.0,
            mean_exposed_remote_ms=0.0,
            estimated_energy_j=0.0,
            peak_hbm_kv_allocated_gib=peak_hbm_kv_bytes / GIB,
            peak_remote_kv_allocated_gib=peak_remote_kv_bytes / GIB,
            peak_kv_fragmentation_pct=fragmentation_pct,
            hbm_capacity_headroom_gib=hbm_headroom_bytes / GIB,
            remote_capacity_headroom_gib=remote_headroom_bytes / GIB,
            bottleneck="capacity",
            assumptions=ASSUMPTIONS,
            steps=(),
        )
    )


def simulate(request: SimulationRequest) -> SimulationResult:
    request.validate()
    workload = request.workload
    system = request.system
    policy = request.policy

    # The final decode step writes one more KV token than it reads.
    peak_sequence_length = workload.context_tokens + workload.generated_tokens
    _, _, _, _, peak_hbm_kv_bytes, peak_remote_kv_bytes = _placement(request, peak_sequence_length)
    exact_peak_kv_bytes = (
        peak_sequence_length
        * workload.concurrent_sequences
        * workload.kv_bytes_per_token_per_sequence
    )
    allocated_peak_kv_bytes = peak_hbm_kv_bytes + peak_remote_kv_bytes
    fragmentation_pct = (
        (allocated_peak_kv_bytes / exact_peak_kv_bytes - 1) * 100 if exact_peak_kv_bytes else 0.0
    )
    hbm_headroom_bytes = (
        system.hbm_capacity_bytes
        - system.hbm_reserved_bytes
        - workload.weight_bytes
        - peak_hbm_kv_bytes
    )
    remote_headroom_bytes = system.remote_capacity_bytes - peak_remote_kv_bytes

    if workload.weight_bytes > system.usable_hbm_capacity_bytes:
        return _empty_result(
            request,
            "model weights exceed HBM capacity after runtime reserve",
            peak_hbm_kv_bytes,
            peak_remote_kv_bytes,
            fragmentation_pct,
            hbm_headroom_bytes,
            remote_headroom_bytes,
        )
    if hbm_headroom_bytes < 0:
        return _empty_result(
            request,
            "weights, runtime reserve, and resident KV cache exceed HBM capacity",
            peak_hbm_kv_bytes,
            peak_remote_kv_bytes,
            fragmentation_pct,
            hbm_headroom_bytes,
            remote_headroom_bytes,
        )
    if remote_headroom_bytes < 0:
        return _empty_result(
            request,
            "page-allocated cold KV cache exceeds remote-memory capacity",
            peak_hbm_kv_bytes,
            peak_remote_kv_bytes,
            fragmentation_pct,
            hbm_headroom_bytes,
            remote_headroom_bytes,
        )

    steps: list[StepMetrics] = []
    total_energy_j = 0.0
    protocol_multiplier = 1 + system.remote_protocol_overhead_ratio
    new_kv_bytes = workload.concurrent_sequences * workload.kv_bytes_per_token_per_sequence

    for generated_index in range(workload.generated_tokens):
        sequence_length = workload.context_tokens + generated_index
        (
            hot_tokens,
            cold_tokens,
            hot_kv_useful_bytes,
            cold_kv_useful_bytes,
            _,
            cold_kv_physical_bytes,
        ) = _placement(request, sequence_length)
        _, cold_tokens_after_write, _, _, hbm_kv_allocated_bytes, remote_kv_allocated_bytes = (
            _placement(request, sequence_length + 1)
        )

        full_decode_flops = workload.decode_flops(sequence_length)
        useful_cold_attention_flops = 0.0
        physical_cold_attention_flops = 0.0
        if policy.kind == "near_memory" and cold_tokens:
            useful_cold_attention_flops = (
                4
                * workload.layers
                * workload.hidden_size
                * cold_tokens
                * workload.concurrent_sequences
            )
            page_rounded_cold_tokens = _round_up_tokens(cold_tokens, policy.kv_page_tokens)
            physical_cold_attention_flops = (
                4
                * workload.layers
                * workload.hidden_size
                * page_rounded_cold_tokens
                * workload.concurrent_sequences
            )
        accelerator_flops = full_decode_flops - useful_cold_attention_flops
        compute_ms = accelerator_flops / (system.effective_accelerator_tops * 1e12) * 1000

        hbm_read_bytes = workload.weight_bytes + hot_kv_useful_bytes
        hbm_write_bytes = new_kv_bytes
        hbm_ms = (
            (hbm_read_bytes + hbm_write_bytes) / (system.effective_hbm_bandwidth_gbps * 1e9) * 1000
        )

        # A full hot window evicts one token per sequence on average. Initial placement is excluded.
        remote_memory_write_bytes = new_kv_bytes if cold_tokens_after_write else 0.0
        remote_memory_read_bytes = cold_kv_physical_bytes
        interconnect_read_bytes = 0.0
        interconnect_write_bytes = 0.0
        near_memory_compute_ms = 0.0

        if policy.kind == "sliding_window" and cold_tokens:
            interconnect_read_bytes = cold_kv_physical_bytes * protocol_multiplier
        elif policy.kind == "near_memory" and cold_tokens:
            query_bytes = (
                workload.layers
                * workload.concurrent_sequences
                * workload.query_bytes_per_layer_per_sequence
            )
            accumulator_bytes = policy.near_memory_accumulator_bits / 8
            partial_result_bytes = (
                workload.layers
                * workload.concurrent_sequences
                * workload.attention_heads
                * (workload.head_dim + 2)
                * accumulator_bytes
            )
            interconnect_write_bytes += query_bytes * protocol_multiplier
            interconnect_read_bytes += partial_result_bytes * protocol_multiplier
            near_memory_compute_ms = (
                physical_cold_attention_flops / (system.effective_near_memory_tops * 1e12) * 1000
            )

        if remote_memory_write_bytes:
            interconnect_write_bytes += remote_memory_write_bytes * protocol_multiplier

        remote_active = bool(
            remote_memory_read_bytes
            or remote_memory_write_bytes
            or interconnect_read_bytes
            or interconnect_write_bytes
        )
        if remote_active:
            remote_link_ms = (
                (interconnect_read_bytes + interconnect_write_bytes)
                / (system.effective_remote_bandwidth_gbps * 1e9)
                * 1000
            )
            remote_memory_ms = (
                (remote_memory_read_bytes + remote_memory_write_bytes)
                / (system.effective_remote_memory_bandwidth_gbps * 1e9)
                * 1000
            )
            # KV, query, and state operations touch every transformer layer. The batch is
            # coalesced within a layer, but layers remain sequential decode dependencies.
            remote_transactions = workload.layers
            remote_fixed_ms = remote_transactions * system.remote_base_latency_us / 1000
            remote_service_ms = remote_fixed_ms + max(
                remote_link_ms, remote_memory_ms, near_memory_compute_ms
            )
        else:
            remote_link_ms = 0.0
            remote_memory_ms = 0.0
            remote_transactions = 0
            remote_fixed_ms = 0.0
            remote_service_ms = 0.0

        local_floor_ms = max(compute_ms, hbm_ms)
        exposed_remote_ms = remote_service_ms * (1 - policy.transfer_overlap_ratio)
        latency_ms = local_floor_ms + exposed_remote_ms

        local_limiter = "accelerator_compute" if compute_ms >= hbm_ms else "hbm_bandwidth"
        remote_components = {
            "remote_link": remote_link_ms,
            "remote_memory": remote_memory_ms,
            "near_memory_compute": near_memory_compute_ms,
            "remote_fixed_latency": remote_fixed_ms,
        }
        remote_limiter = max(remote_components, key=remote_components.__getitem__)
        bottleneck = remote_limiter if exposed_remote_ms > local_floor_ms else local_limiter

        remote_media_bytes = remote_memory_read_bytes + remote_memory_write_bytes
        link_bytes = interconnect_read_bytes + interconnect_write_bytes
        total_energy_j += (
            accelerator_flops * system.accelerator_compute_energy_pj_per_flop
            + physical_cold_attention_flops * system.near_memory_compute_energy_pj_per_flop
            + (hbm_read_bytes + hbm_write_bytes) * system.hbm_energy_pj_per_byte
            + remote_media_bytes * system.remote_memory_energy_pj_per_byte
            + link_bytes * system.remote_link_energy_pj_per_byte
        ) * 1e-12

        steps.append(
            StepMetrics(
                token_index=generated_index,
                sequence_length=sequence_length,
                compute_ms=compute_ms,
                hbm_ms=hbm_ms,
                remote_link_ms=remote_link_ms,
                remote_memory_ms=remote_memory_ms,
                remote_fixed_ms=remote_fixed_ms,
                near_memory_compute_ms=near_memory_compute_ms,
                remote_service_ms=remote_service_ms,
                exposed_remote_ms=exposed_remote_ms,
                latency_ms=latency_ms,
                hbm_kv_allocated_gib=hbm_kv_allocated_bytes / GIB,
                remote_kv_allocated_gib=remote_kv_allocated_bytes / GIB,
                cold_kv_useful_gib=cold_kv_useful_bytes / GIB,
                hbm_read_gib=hbm_read_bytes / GIB,
                hbm_write_gib=hbm_write_bytes / GIB,
                remote_memory_read_gib=remote_memory_read_bytes / GIB,
                remote_memory_write_gib=remote_memory_write_bytes / GIB,
                interconnect_read_gib=interconnect_read_bytes / GIB,
                interconnect_write_gib=interconnect_write_bytes / GIB,
                remote_transactions=remote_transactions,
                bottleneck=bottleneck,
            )
        )

    mean_latency = fmean(step.latency_ms for step in steps)
    sorted_latency = sorted(step.latency_ms for step in steps)
    p95_index = max(0, math.ceil(len(sorted_latency) * 0.95) - 1)
    p95_latency = sorted_latency[p95_index]
    bottleneck = Counter(step.bottleneck for step in steps).most_common(1)[0][0]

    total_cold_kv_useful_gib = sum(step.cold_kv_useful_gib for step in steps)
    total_remote_memory_read_gib = sum(step.remote_memory_read_gib for step in steps)
    total_interconnect_read_gib = sum(step.interconnect_read_gib for step in steps)
    media_read_amplification = (
        total_remote_memory_read_gib / total_cold_kv_useful_gib if total_cold_kv_useful_gib else 0.0
    )
    interconnect_ratio = (
        total_interconnect_read_gib / total_remote_memory_read_gib
        if total_remote_memory_read_gib
        else 0.0
    )

    return _finite_result(
        SimulationResult(
            workload_name=workload.name,
            system_name=system.name,
            policy_name=policy.name,
            input_hardware_profile=request.provenance.hardware_profile,
            feasible=True,
            rejection_reason=None,
            mean_decode_latency_ms=mean_latency,
            p95_step_latency_ms=p95_latency,
            throughput_tokens_s=workload.concurrent_sequences * 1000 / mean_latency,
            total_hbm_read_gib=sum(step.hbm_read_gib for step in steps),
            total_hbm_write_gib=sum(step.hbm_write_gib for step in steps),
            total_cold_kv_useful_gib=total_cold_kv_useful_gib,
            total_remote_memory_read_gib=total_remote_memory_read_gib,
            total_remote_memory_write_gib=sum(step.remote_memory_write_gib for step in steps),
            total_interconnect_read_gib=total_interconnect_read_gib,
            total_interconnect_write_gib=sum(step.interconnect_write_gib for step in steps),
            remote_memory_read_amplification=media_read_amplification,
            interconnect_read_per_remote_byte=interconnect_ratio,
            mean_remote_service_ms=fmean(step.remote_service_ms for step in steps),
            mean_exposed_remote_ms=fmean(step.exposed_remote_ms for step in steps),
            estimated_energy_j=total_energy_j,
            peak_hbm_kv_allocated_gib=peak_hbm_kv_bytes / GIB,
            peak_remote_kv_allocated_gib=peak_remote_kv_bytes / GIB,
            peak_kv_fragmentation_pct=fragmentation_pct,
            hbm_capacity_headroom_gib=hbm_headroom_bytes / GIB,
            remote_capacity_headroom_gib=remote_headroom_bytes / GIB,
            bottleneck=bottleneck,
            assumptions=ASSUMPTIONS,
            steps=tuple(steps),
        )
    )
