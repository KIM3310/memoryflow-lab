from __future__ import annotations

from dataclasses import replace

from memoryflow.domain import PlacementPolicy, SimulationRequest, SimulationResult
from memoryflow.simulator import simulate


def pareto_front(results: list[SimulationResult]) -> list[SimulationResult]:
    feasible = [result for result in results if result.feasible]
    frontier: list[SimulationResult] = []
    for candidate in feasible:
        dominated = any(
            other.mean_decode_latency_ms <= candidate.mean_decode_latency_ms
            and other.estimated_energy_j <= candidate.estimated_energy_j
            and (
                other.mean_decode_latency_ms < candidate.mean_decode_latency_ms
                or other.estimated_energy_j < candidate.estimated_energy_j
            )
            for other in feasible
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda item: (
            item.mean_decode_latency_ms,
            item.estimated_energy_j,
            item.policy_name,
        ),
    )


def _validate_windows(windows: tuple[int, ...], page_tokens: int) -> None:
    if not windows:
        raise ValueError("windows must contain at least one token count")
    if len(set(windows)) != len(windows):
        raise ValueError("window token counts must be unique")
    if any(
        isinstance(window, bool)
        or not isinstance(window, int)
        or window <= 0
        or window % page_tokens != 0
        for window in windows
    ):
        raise ValueError("windows must be positive integers divisible by kv_page_tokens")


def sweep_hbm_windows(
    request: SimulationRequest,
    windows: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096),
) -> list[SimulationResult]:
    request.validate()
    _validate_windows(windows, request.policy.kv_page_tokens)
    results: list[SimulationResult] = []
    for kind in ("sliding_window", "near_memory"):
        for window in windows:
            policy = PlacementPolicy(
                name=f"{kind}-{window}",
                kind=kind,
                hbm_window_tokens=window,
                kv_page_tokens=request.policy.kv_page_tokens,
                transfer_overlap_ratio=request.policy.transfer_overlap_ratio,
                near_memory_accumulator_bits=request.policy.near_memory_accumulator_bits,
            )
            results.append(simulate(replace(request, policy=policy)))
    return results
