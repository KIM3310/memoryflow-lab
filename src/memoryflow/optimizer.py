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
    return sorted(frontier, key=lambda item: (item.mean_decode_latency_ms, item.estimated_energy_j))


def sweep_hbm_windows(
    request: SimulationRequest,
    windows: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096),
) -> list[SimulationResult]:
    results: list[SimulationResult] = []
    for kind in ("sliding_window", "near_memory"):
        for window in windows:
            policy = PlacementPolicy(
                name=f"{kind}-{window}",
                kind=kind,
                hbm_window_tokens=window,
                transfer_overlap_ratio=request.policy.transfer_overlap_ratio,
                near_memory_reduction_ratio=request.policy.near_memory_reduction_ratio,
            )
            results.append(simulate(replace(request, policy=policy)))
    return results
