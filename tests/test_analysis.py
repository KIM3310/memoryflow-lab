from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from memoryflow.analysis import (
    SENSITIVITY_PARAMETERS,
    analyze_design_space,
    analyze_sensitivity,
    find_counterexample,
    find_near_memory_break_even_tops,
)
from memoryflow.domain import MIN_EFFECTIVE_RATE
from memoryflow.io import load_request

ROOT = Path(__file__).resolve().parents[1]
REQUEST = load_request(ROOT / "scenarios" / "7b-long-context-tiered.json")


def test_design_space_grid_break_even_and_serialization_are_deterministic() -> None:
    kwargs = {
        "remote_link_bandwidths_gbps": (32.0, 64.0),
        "near_memory_tops_values": (0.1, 0.5, 12.0),
        "sensitivity_multipliers": (0.5, 1.0, 1.5),
    }
    report = analyze_design_space(REQUEST, **kwargs)
    assert len(report.points) == 6
    assert len(report.break_even) == 2
    assert len(report.sensitivity) == len(SENSITIVITY_PARAMETERS) * 3
    assert all(point.status == "within_bounds" for point in report.break_even)
    assert report.counterexample.baseline_winner == "near_memory"
    assert report.counterexample.counterexample_winner == "sliding_window"
    assert report.to_dict()["schema_version"] == "1.0"
    assert report.to_dict() == analyze_design_space(REQUEST, **kwargs).to_dict()


def test_break_even_ties_policies_and_is_monotone_with_link_rate() -> None:
    slow_link = find_near_memory_break_even_tops(REQUEST, 32)
    base_link = find_near_memory_break_even_tops(REQUEST, 64)
    assert slow_link is not None and base_link is not None
    assert slow_link < base_link
    assert base_link == pytest.approx(0.2844049821, rel=1e-8)


def test_break_even_bounds_and_iterations_are_validated() -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        find_near_memory_break_even_tops(REQUEST, 0)
    for bounds in ((0, 1), (1, 1), (2, 1), (1, float("inf"))):
        with pytest.raises(ValueError, match="bounds"):
            find_near_memory_break_even_tops(
                REQUEST, 64, lower_tops=bounds[0], upper_tops=bounds[1]
            )
    for iterations in (0, 257, True):
        with pytest.raises(ValueError, match="iterations"):
            find_near_memory_break_even_tops(REQUEST, 64, iterations=iterations)  # type: ignore[arg-type]


def test_break_even_can_be_below_or_above_search_bounds() -> None:
    assert find_near_memory_break_even_tops(REQUEST, 64, lower_tops=1, upper_tops=2) == 1
    assert find_near_memory_break_even_tops(REQUEST, 64, lower_tops=0.001, upper_tops=0.1) is None


def test_sensitivity_covers_effective_and_peak_inputs_with_finite_envelope() -> None:
    points, envelope = analyze_sensitivity(REQUEST, (0.5, 1.0, 1.5))
    assert {point.parameter for point in points} == set(SENSITIVITY_PARAMETERS)
    assert envelope.minimum_near_memory_speedup is not None
    assert envelope.maximum_near_memory_speedup is not None
    assert envelope.minimum_near_memory_speedup > 0
    assert envelope.maximum_near_memory_speedup >= envelope.minimum_near_memory_speedup
    assert envelope.feasible_points + envelope.infeasible_points == len(points)
    # Efficiency scaling clamps at one rather than constructing an invalid system.
    hbm_high = next(
        point
        for point in points
        if point.parameter == "hbm_bandwidth_efficiency" and point.multiplier == 1.5
    )
    assert hbm_high.input_value == 1


def test_sensitivity_records_overflow_without_non_finite_output() -> None:
    points, envelope = analyze_sensitivity(REQUEST, (1e308,))
    assert any(point.input_value is None for point in points)
    assert all(point.input_value is None or math.isfinite(point.input_value) for point in points)
    report = analyze_design_space(
        REQUEST,
        remote_link_bandwidths_gbps=(64.0,),
        near_memory_tops_values=(12.0,),
        sensitivity_multipliers=(1e308,),
    )
    json.dumps(report.to_dict(), allow_nan=False)
    with pytest.raises(ValueError, match="non-finite"):
        replace(report, sensitivity_multipliers=(float("inf"),)).to_dict()
    assert envelope.infeasible_points > 0


def test_counterexample_also_reports_a_baseline_that_already_loses() -> None:
    slow = replace(REQUEST, system=replace(REQUEST.system, near_memory_tops=0.1))
    result = find_counterexample(slow)
    assert result.status == "baseline_already_loses"
    assert result.baseline_winner == "sliding_window"
    assert result.counterexample_winner == "sliding_window"
    assert result.counterexample_near_memory_tops == 0.1


def test_counterexample_steps_below_an_exact_break_even_tie() -> None:
    threshold = find_near_memory_break_even_tops(REQUEST, REQUEST.system.remote_bandwidth_gbps)
    assert threshold is not None
    tied = replace(REQUEST, system=replace(REQUEST.system, near_memory_tops=threshold))
    result = find_counterexample(tied)
    assert result.baseline_winner == "tie"
    assert result.status == "constructed"
    assert result.counterexample_winner == "sliding_window"
    assert result.counterexample_near_memory_tops is not None
    assert result.counterexample_near_memory_tops < threshold


def test_counterexample_reports_not_reachable_at_safe_rate_floor() -> None:
    minimum_link_peak = (
        MIN_EFFECTIVE_RATE * (1 + 1e-12) / REQUEST.system.remote_bandwidth_efficiency
    )
    link_floor = replace(
        REQUEST,
        system=replace(REQUEST.system, remote_bandwidth_gbps=minimum_link_peak),
    )
    result = find_counterexample(link_floor)
    assert result.status == "not_reachable_within_bounds"
    assert result.counterexample_near_memory_tops is None
    assert result.counterexample_near_memory_speedup is None
    assert result.counterexample_winner is None


def test_counterexample_reports_not_reachable_when_no_cold_kv_exists() -> None:
    no_cold = replace(
        REQUEST,
        workload=replace(REQUEST.workload, context_tokens=1024, generated_tokens=1),
        policy=replace(REQUEST.policy, hbm_window_tokens=2048),
    )
    result = find_counterexample(no_cold)
    assert result.baseline_winner == "tie"
    assert result.status == "not_reachable_within_bounds"
    assert result.counterexample_winner is None


def test_design_grid_records_effective_rate_axis_violations() -> None:
    tiny_efficiency = replace(
        REQUEST,
        system=replace(
            REQUEST.system,
            near_memory_compute_efficiency=1e-12,
            near_memory_tops=10_000.0,
        ),
    )
    threshold = find_near_memory_break_even_tops(tiny_efficiency, 64.0)
    minimum_peak = MIN_EFFECTIVE_RATE * (1 + 1e-12) / 1e-12
    assert threshold is None or threshold >= minimum_peak
    report = analyze_design_space(
        tiny_efficiency,
        remote_link_bandwidths_gbps=(64.0,),
        near_memory_tops_values=(0.1, 10_000.0),
        sensitivity_multipliers=(1.0,),
    )
    assert report.points[0].feasible is False
    assert "effective rates" in str(report.points[0].rejection_reason)
    assert report.points[1].feasible is True


def test_design_grid_records_invalid_remote_link_axis_without_aborting() -> None:
    tiny_link_efficiency = replace(
        REQUEST,
        system=replace(
            REQUEST.system,
            remote_bandwidth_efficiency=1e-12,
            remote_bandwidth_gbps=10_000.0,
        ),
    )
    report = analyze_design_space(
        tiny_link_efficiency,
        remote_link_bandwidths_gbps=(64.0,),
        near_memory_tops_values=(12.0,),
        sensitivity_multipliers=(1.0,),
    )
    assert report.points[0].feasible is False
    assert report.break_even[0].status == "invalid_axis"
    assert "effective rates" in str(report.break_even[0].rejection_reason)


def test_sensitivity_records_capacity_infeasibility_instead_of_aborting() -> None:
    close_to_capacity = replace(REQUEST, system=replace(REQUEST.system, hbm_reserved_gib=6.0))
    points, envelope = analyze_sensitivity(close_to_capacity, (1.0, 1.5))
    reserved_high = next(
        point
        for point in points
        if point.parameter == "hbm_reserved_gib" and point.multiplier == 1.5
    )
    assert reserved_high.feasible is False
    assert reserved_high.rejection_reason is not None
    assert "HBM capacity" in reserved_high.rejection_reason
    assert reserved_high.hbm_capacity_headroom_gib is not None
    assert reserved_high.hbm_capacity_headroom_gib < 0
    assert reserved_high.near_memory_speedup is None
    assert envelope.infeasible_points >= 1


@pytest.mark.parametrize(
    ("axis", "message"),
    [
        ({"remote_link_bandwidths_gbps": ()}, "at least one"),
        ({"remote_link_bandwidths_gbps": (64, 64)}, "unique"),
        ({"near_memory_tops_values": (float("nan"),)}, "positive"),
        ({"sensitivity_multipliers": (0,)}, "positive"),
    ],
)
def test_design_axes_are_validated(axis: dict[str, tuple[float, ...]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        analyze_design_space(REQUEST, **axis)  # type: ignore[arg-type]


def test_analysis_rejects_capacity_infeasible_remote_baseline() -> None:
    infeasible = replace(REQUEST, system=replace(REQUEST.system, remote_capacity_gib=1))
    with pytest.raises(ValueError, match="infeasible|feasible"):
        analyze_design_space(
            infeasible,
            remote_link_bandwidths_gbps=(64,),
            near_memory_tops_values=(1,),
            sensitivity_multipliers=(1,),
        )
