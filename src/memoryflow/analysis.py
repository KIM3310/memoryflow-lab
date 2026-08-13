from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from memoryflow.domain import MIN_EFFECTIVE_RATE, PlacementPolicy, PolicyKind, SimulationRequest
from memoryflow.simulator import simulate

ANALYSIS_SCHEMA_VERSION = "1.0"
Winner = Literal["sliding_window", "near_memory", "tie"]
BreakEvenStatus = Literal["within_bounds", "at_or_below_lower_bound", "not_reached", "invalid_axis"]
CounterexampleStatus = Literal[
    "constructed", "baseline_already_loses", "not_reachable_within_bounds"
]
SensitivityParameter = Literal[
    "remote_bandwidth_gbps",
    "remote_memory_bandwidth_gbps",
    "near_memory_tops",
    "hbm_bandwidth_efficiency",
    "remote_bandwidth_efficiency",
    "remote_memory_bandwidth_efficiency",
    "accelerator_compute_efficiency",
    "near_memory_compute_efficiency",
    "remote_base_latency_us",
    "remote_protocol_overhead_ratio",
    "hbm_reserved_gib",
    "transfer_overlap_ratio",
]

SENSITIVITY_PARAMETERS: tuple[SensitivityParameter, ...] = (
    "remote_bandwidth_gbps",
    "remote_memory_bandwidth_gbps",
    "near_memory_tops",
    "hbm_bandwidth_efficiency",
    "remote_bandwidth_efficiency",
    "remote_memory_bandwidth_efficiency",
    "accelerator_compute_efficiency",
    "near_memory_compute_efficiency",
    "remote_base_latency_us",
    "remote_protocol_overhead_ratio",
    "hbm_reserved_gib",
    "transfer_overlap_ratio",
)


@dataclass(frozen=True)
class DesignPoint:
    remote_link_bandwidth_gbps: float
    near_memory_tops: float
    feasible: bool
    rejection_reason: str | None
    sliding_window_latency_ms: float | None
    near_memory_latency_ms: float | None
    near_memory_speedup: float | None
    winner: Winner | None
    sliding_window_bottleneck: str | None
    near_memory_bottleneck: str | None


@dataclass(frozen=True)
class BreakEvenPoint:
    remote_link_bandwidth_gbps: float
    near_memory_break_even_tops: float | None
    search_lower_tops: float
    status: BreakEvenStatus
    rejection_reason: str | None


@dataclass(frozen=True)
class SensitivityPoint:
    parameter: SensitivityParameter
    multiplier: float
    input_value: float | None
    feasible: bool
    rejection_reason: str | None
    sliding_window_latency_ms: float | None
    near_memory_latency_ms: float | None
    near_memory_speedup: float | None
    winner: Winner | None
    hbm_capacity_headroom_gib: float | None
    remote_capacity_headroom_gib: float | None


@dataclass(frozen=True)
class SensitivityEnvelope:
    minimum_near_memory_speedup: float | None
    maximum_near_memory_speedup: float | None
    observed_winners: tuple[Winner, ...]
    feasible_points: int
    infeasible_points: int


@dataclass(frozen=True)
class Counterexample:
    status: CounterexampleStatus
    varied_parameter: Literal["near_memory_tops"]
    baseline_near_memory_tops: float
    baseline_near_memory_speedup: float
    baseline_winner: Winner
    counterexample_near_memory_tops: float | None
    counterexample_near_memory_speedup: float | None
    counterexample_winner: Winner | None
    conclusion: str


@dataclass(frozen=True)
class DesignSpaceReport:
    methodology: str
    uncertainty_boundary: str
    points: tuple[DesignPoint, ...]
    break_even: tuple[BreakEvenPoint, ...]
    sensitivity_multipliers: tuple[float, ...]
    sensitivity: tuple[SensitivityPoint, ...]
    one_at_a_time_envelope: SensitivityEnvelope
    counterexample: Counterexample

    def to_dict(self) -> dict[str, Any]:
        body = asdict(self)
        _assert_finite_analysis_payload(body)
        return {"schema_version": ANALYSIS_SCHEMA_VERSION, **body}


def _policy(request: SimulationRequest, kind: PolicyKind, name: str) -> PlacementPolicy:
    return PlacementPolicy(
        name=name,
        kind=kind,
        hbm_window_tokens=request.policy.hbm_window_tokens,
        kv_page_tokens=request.policy.kv_page_tokens,
        transfer_overlap_ratio=request.policy.transfer_overlap_ratio,
        near_memory_accumulator_bits=request.policy.near_memory_accumulator_bits,
    )


def _request_for(
    request: SimulationRequest,
    kind: PolicyKind,
    remote_link_bandwidth_gbps: float,
    near_memory_tops: float,
) -> SimulationRequest:
    system = replace(
        request.system,
        remote_bandwidth_gbps=remote_link_bandwidth_gbps,
        near_memory_tops=near_memory_tops,
    )
    return replace(request, system=system, policy=_policy(request, kind, kind))


def _is_positive_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _assert_finite_analysis_payload(value: object) -> None:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("analysis output contains a non-finite number")
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)


def _validate_axis(values: tuple[float, ...], label: str) -> None:
    if not values:
        raise ValueError(f"{label} must contain at least one value")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} values must be unique")
    if any(not _is_positive_finite(value) for value in values):
        raise ValueError(f"{label} values must be positive and finite")


def _winner(sliding_latency_ms: float, near_latency_ms: float) -> Winner:
    if math.isclose(sliding_latency_ms, near_latency_ms, rel_tol=1e-9, abs_tol=1e-9):
        return "tie"
    return "near_memory" if near_latency_ms < sliding_latency_ms else "sliding_window"


def _paired_result(request: SimulationRequest) -> tuple[float, float, Winner, str, str]:
    sliding = simulate(
        replace(request, policy=_policy(request, "sliding_window", "sliding_window"))
    )
    near = simulate(replace(request, policy=_policy(request, "near_memory", "near_memory")))
    if not sliding.feasible or not near.feasible:
        raise ValueError("analysis requires both remote placement policies to be feasible")
    return (
        sliding.mean_decode_latency_ms,
        near.mean_decode_latency_ms,
        _winner(sliding.mean_decode_latency_ms, near.mean_decode_latency_ms),
        sliding.bottleneck,
        near.bottleneck,
    )


def find_near_memory_break_even_tops(
    request: SimulationRequest,
    remote_link_bandwidth_gbps: float,
    *,
    lower_tops: float = 0.001,
    upper_tops: float = 10_000.0,
    iterations: int = 64,
) -> float | None:
    request.validate()
    _validate_axis((remote_link_bandwidth_gbps,), "remote link bandwidth")
    if (
        not _is_positive_finite(lower_tops)
        or not _is_positive_finite(upper_tops)
        or upper_tops <= lower_tops
    ):
        raise ValueError("break-even TOPS bounds must be positive, finite, and increasing")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or not 1 <= iterations <= 256
    ):
        raise ValueError("iterations must be an integer between 1 and 256")

    # Search only raw peak values whose efficiency-adjusted rate is in the valid domain.
    minimum_valid_tops = (
        MIN_EFFECTIVE_RATE * (1 + 1e-12) / request.system.near_memory_compute_efficiency
    )
    effective_lower_tops = max(lower_tops, minimum_valid_tops)
    if upper_tops <= effective_lower_tops:
        raise ValueError("break-even bounds contain no valid effective near-memory rate")

    tiered = simulate(
        _request_for(
            request,
            "sliding_window",
            remote_link_bandwidth_gbps,
            request.system.near_memory_tops,
        )
    )
    if not tiered.feasible:
        return None

    def wins(tops: float) -> bool:
        candidate = simulate(_request_for(request, "near_memory", remote_link_bandwidth_gbps, tops))
        return candidate.feasible and candidate.mean_decode_latency_ms <= (
            tiered.mean_decode_latency_ms * (1 + 1e-12)
        )

    if wins(effective_lower_tops):
        return effective_lower_tops
    if not wins(upper_tops):
        return None

    low = effective_lower_tops
    high = upper_tops
    for _ in range(iterations):
        midpoint = math.sqrt(low * high)
        if wins(midpoint):
            high = midpoint
        else:
            low = midpoint
    return high


def _scaled_request(
    request: SimulationRequest, parameter: SensitivityParameter, multiplier: float
) -> tuple[SimulationRequest, float]:
    if parameter == "transfer_overlap_ratio":
        value = min(1.0, request.policy.transfer_overlap_ratio * multiplier)
        return replace(request, policy=replace(request.policy, transfer_overlap_ratio=value)), value

    current = float(getattr(request.system, parameter))
    value = current * multiplier
    if parameter.endswith("_efficiency") or parameter == "remote_protocol_overhead_ratio":
        value = min(1.0, value)
    if not math.isfinite(value):
        raise ValueError(f"scaled {parameter} is outside the finite input domain")
    updates: Any = {parameter: value}
    return replace(request, system=replace(request.system, **updates)), value


def analyze_sensitivity(
    request: SimulationRequest,
    multipliers: tuple[float, ...] = (0.5, 0.8, 1.0, 1.2, 1.5),
) -> tuple[tuple[SensitivityPoint, ...], SensitivityEnvelope]:
    request.validate()
    _validate_axis(multipliers, "sensitivity multipliers")
    points: list[SensitivityPoint] = []
    for parameter in SENSITIVITY_PARAMETERS:
        for multiplier in multipliers:
            input_value: float | None = None
            try:
                varied, input_value = _scaled_request(request, parameter, multiplier)
                sliding = simulate(
                    replace(
                        varied,
                        policy=_policy(varied, "sliding_window", "sliding_window"),
                    )
                )
                near = simulate(
                    replace(varied, policy=_policy(varied, "near_memory", "near_memory"))
                )
            except ValueError as exc:
                points.append(
                    SensitivityPoint(
                        parameter=parameter,
                        multiplier=multiplier,
                        input_value=input_value,
                        feasible=False,
                        rejection_reason=f"invalid perturbed input: {exc}",
                        sliding_window_latency_ms=None,
                        near_memory_latency_ms=None,
                        near_memory_speedup=None,
                        winner=None,
                        hbm_capacity_headroom_gib=None,
                        remote_capacity_headroom_gib=None,
                    )
                )
                continue

            hbm_headroom = min(
                sliding.hbm_capacity_headroom_gib,
                near.hbm_capacity_headroom_gib,
            )
            remote_headroom = min(
                sliding.remote_capacity_headroom_gib,
                near.remote_capacity_headroom_gib,
            )
            if not sliding.feasible or not near.feasible:
                reasons = tuple(
                    dict.fromkeys(
                        reason
                        for reason in (sliding.rejection_reason, near.rejection_reason)
                        if reason is not None
                    )
                )
                points.append(
                    SensitivityPoint(
                        parameter=parameter,
                        multiplier=multiplier,
                        input_value=input_value,
                        feasible=False,
                        rejection_reason="; ".join(reasons) or "perturbed point is infeasible",
                        sliding_window_latency_ms=None,
                        near_memory_latency_ms=None,
                        near_memory_speedup=None,
                        winner=None,
                        hbm_capacity_headroom_gib=hbm_headroom,
                        remote_capacity_headroom_gib=remote_headroom,
                    )
                )
                continue

            winner = _winner(sliding.mean_decode_latency_ms, near.mean_decode_latency_ms)
            points.append(
                SensitivityPoint(
                    parameter=parameter,
                    multiplier=multiplier,
                    input_value=input_value,
                    feasible=True,
                    rejection_reason=None,
                    sliding_window_latency_ms=sliding.mean_decode_latency_ms,
                    near_memory_latency_ms=near.mean_decode_latency_ms,
                    near_memory_speedup=(
                        sliding.mean_decode_latency_ms / near.mean_decode_latency_ms
                    ),
                    winner=winner,
                    hbm_capacity_headroom_gib=hbm_headroom,
                    remote_capacity_headroom_gib=remote_headroom,
                )
            )
    speedups = [
        point.near_memory_speedup for point in points if point.near_memory_speedup is not None
    ]
    winner_order: tuple[Winner, ...] = ("sliding_window", "near_memory", "tie")
    observed = tuple(winner for winner in winner_order if any(p.winner == winner for p in points))
    feasible_points = sum(point.feasible for point in points)
    return (
        tuple(points),
        SensitivityEnvelope(
            min(speedups) if speedups else None,
            max(speedups) if speedups else None,
            observed,
            feasible_points,
            len(points) - feasible_points,
        ),
    )


def find_counterexample(request: SimulationRequest) -> Counterexample:
    request.validate()
    sliding_ms, baseline_near_ms, baseline_winner, _, _ = _paired_result(request)
    baseline_speedup = sliding_ms / baseline_near_ms

    if baseline_winner == "sliding_window":
        return Counterexample(
            status="baseline_already_loses",
            varied_parameter="near_memory_tops",
            baseline_near_memory_tops=request.system.near_memory_tops,
            baseline_near_memory_speedup=baseline_speedup,
            baseline_winner=baseline_winner,
            counterexample_near_memory_tops=request.system.near_memory_tops,
            counterexample_near_memory_speedup=baseline_speedup,
            counterexample_winner=baseline_winner,
            conclusion=(
                "The valid baseline already demonstrates the conditional result: ordinary "
                "remote tiering is faster at the declared near-memory compute rate."
            ),
        )

    threshold = find_near_memory_break_even_tops(request, request.system.remote_bandwidth_gbps)
    minimum_tops = MIN_EFFECTIVE_RATE * (1 + 1e-12) / request.system.near_memory_compute_efficiency
    candidate_tops = threshold / 2 if threshold is not None else minimum_tops
    candidate_tops = max(candidate_tops, minimum_tops)
    counter_sliding_ms = sliding_ms
    counter_near_ms = baseline_near_ms
    counter_winner: Winner = baseline_winner

    for _ in range(256):
        candidate = replace(
            request, system=replace(request.system, near_memory_tops=candidate_tops)
        )
        counter_sliding_ms, counter_near_ms, counter_winner, _, _ = _paired_result(candidate)
        if counter_winner == "sliding_window":
            return Counterexample(
                status="constructed",
                varied_parameter="near_memory_tops",
                baseline_near_memory_tops=request.system.near_memory_tops,
                baseline_near_memory_speedup=baseline_speedup,
                baseline_winner=baseline_winner,
                counterexample_near_memory_tops=candidate_tops,
                counterexample_near_memory_speedup=counter_sliding_ms / counter_near_ms,
                counterexample_winner=counter_winner,
                conclusion=(
                    "Near-memory placement is conditional: insufficient effective near-memory "
                    "compute makes ordinary remote tiering faster even though partial-state link "
                    "traffic is lower."
                ),
            )
        next_tops = max(candidate_tops / 2, minimum_tops)
        if next_tops >= candidate_tops:
            break
        candidate_tops = next_tops

    return Counterexample(
        status="not_reachable_within_bounds",
        varied_parameter="near_memory_tops",
        baseline_near_memory_tops=request.system.near_memory_tops,
        baseline_near_memory_speedup=baseline_speedup,
        baseline_winner=baseline_winner,
        counterexample_near_memory_tops=None,
        counterexample_near_memory_speedup=None,
        counterexample_winner=None,
        conclusion=(
            "No ordinary-tiering win was reachable by lowering near-memory compute without "
            "crossing the model's minimum valid effective-rate bound. This input therefore does "
            "not supply the advertised slow-compute counterexample."
        ),
    )


def analyze_design_space(
    request: SimulationRequest,
    remote_link_bandwidths_gbps: tuple[float, ...] = (32.0, 64.0, 128.0, 256.0),
    near_memory_tops_values: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 3.0, 12.0),
    sensitivity_multipliers: tuple[float, ...] = (0.5, 0.8, 1.0, 1.2, 1.5),
) -> DesignSpaceReport:
    request.validate()
    _validate_axis(remote_link_bandwidths_gbps, "remote link bandwidth")
    _validate_axis(near_memory_tops_values, "near-memory TOPS")
    _validate_axis(sensitivity_multipliers, "sensitivity multipliers")

    points: list[DesignPoint] = []
    break_even: list[BreakEvenPoint] = []
    search_lower_tops = max(
        0.001,
        MIN_EFFECTIVE_RATE * (1 + 1e-12) / request.system.near_memory_compute_efficiency,
    )
    for bandwidth in remote_link_bandwidths_gbps:
        tiered = None
        tiered_error: str | None = None
        try:
            tiered = simulate(
                _request_for(
                    request,
                    "sliding_window",
                    bandwidth,
                    request.system.near_memory_tops,
                )
            )
        except ValueError as exc:
            tiered_error = f"invalid remote-link grid input: {exc}"
        if tiered is not None and not tiered.feasible:
            tiered_error = tiered.rejection_reason or "design-space baseline is infeasible"

        for tops in near_memory_tops_values:
            if tiered is None or tiered_error is not None:
                points.append(
                    DesignPoint(
                        remote_link_bandwidth_gbps=bandwidth,
                        near_memory_tops=tops,
                        feasible=False,
                        rejection_reason=tiered_error,
                        sliding_window_latency_ms=None,
                        near_memory_latency_ms=None,
                        near_memory_speedup=None,
                        winner=None,
                        sliding_window_bottleneck=None,
                        near_memory_bottleneck=None,
                    )
                )
                continue
            try:
                near = simulate(_request_for(request, "near_memory", bandwidth, tops))
            except ValueError as exc:
                points.append(
                    DesignPoint(
                        remote_link_bandwidth_gbps=bandwidth,
                        near_memory_tops=tops,
                        feasible=False,
                        rejection_reason=f"invalid near-memory grid input: {exc}",
                        sliding_window_latency_ms=tiered.mean_decode_latency_ms,
                        near_memory_latency_ms=None,
                        near_memory_speedup=None,
                        winner=None,
                        sliding_window_bottleneck=tiered.bottleneck,
                        near_memory_bottleneck=None,
                    )
                )
                continue
            if not near.feasible:
                points.append(
                    DesignPoint(
                        remote_link_bandwidth_gbps=bandwidth,
                        near_memory_tops=tops,
                        feasible=False,
                        rejection_reason=near.rejection_reason,
                        sliding_window_latency_ms=tiered.mean_decode_latency_ms,
                        near_memory_latency_ms=None,
                        near_memory_speedup=None,
                        winner=None,
                        sliding_window_bottleneck=tiered.bottleneck,
                        near_memory_bottleneck=near.bottleneck,
                    )
                )
                continue
            points.append(
                DesignPoint(
                    remote_link_bandwidth_gbps=bandwidth,
                    near_memory_tops=tops,
                    feasible=True,
                    rejection_reason=None,
                    sliding_window_latency_ms=tiered.mean_decode_latency_ms,
                    near_memory_latency_ms=near.mean_decode_latency_ms,
                    near_memory_speedup=(
                        tiered.mean_decode_latency_ms / near.mean_decode_latency_ms
                    ),
                    winner=_winner(tiered.mean_decode_latency_ms, near.mean_decode_latency_ms),
                    sliding_window_bottleneck=tiered.bottleneck,
                    near_memory_bottleneck=near.bottleneck,
                )
            )

        if tiered is None or tiered_error is not None:
            break_even.append(
                BreakEvenPoint(
                    remote_link_bandwidth_gbps=bandwidth,
                    near_memory_break_even_tops=None,
                    search_lower_tops=search_lower_tops,
                    status="invalid_axis",
                    rejection_reason=tiered_error,
                )
            )
            continue
        try:
            threshold = find_near_memory_break_even_tops(request, bandwidth)
        except ValueError as exc:
            break_even.append(
                BreakEvenPoint(
                    remote_link_bandwidth_gbps=bandwidth,
                    near_memory_break_even_tops=None,
                    search_lower_tops=search_lower_tops,
                    status="invalid_axis",
                    rejection_reason=str(exc),
                )
            )
            continue
        status: BreakEvenStatus
        if threshold is None:
            status = "not_reached"
        elif math.isclose(threshold, search_lower_tops, rel_tol=0.0, abs_tol=1e-15):
            status = "at_or_below_lower_bound"
        else:
            status = "within_bounds"
        break_even.append(
            BreakEvenPoint(
                remote_link_bandwidth_gbps=bandwidth,
                near_memory_break_even_tops=threshold,
                search_lower_tops=search_lower_tops,
                status=status,
                rejection_reason=None,
            )
        )

    sensitivity, envelope = analyze_sensitivity(request, sensitivity_multipliers)
    return DesignSpaceReport(
        methodology=(
            "Deterministic grid, geometric-bisection break-even, and one-at-a-time input "
            "multipliers. Invalid or capacity-infeasible candidates remain explicit records; "
            "there is no sampling or fitted probability distribution."
        ),
        uncertainty_boundary=(
            "The sensitivity envelope spans only feasible listed input perturbations and is not "
            "a statistical confidence interval or hardware validation."
        ),
        points=tuple(points),
        break_even=tuple(break_even),
        sensitivity_multipliers=sensitivity_multipliers,
        sensitivity=sensitivity,
        one_at_a_time_envelope=envelope,
        counterexample=find_counterexample(request),
    )
