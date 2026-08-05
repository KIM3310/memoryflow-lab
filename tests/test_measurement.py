from __future__ import annotations

import pytest

from memoryflow.measurement import (
    AttentionRooflineModel,
    AttentionSample,
    AttentionShape,
    TransferModel,
    TransferSample,
    attention_samples_from_payload,
    bandwidth_only_model,
    compare_attention_affine,
    compare_attention_roofline,
    compare_transfer_model,
    fit_affine_transfer_model,
    fit_attention_affine_model,
    samples_from_payload,
)


def sample(size_bytes: int, median_ms: float, p95_ms: float | None = None) -> TransferSample:
    return TransferSample(size_bytes, median_ms, p95_ms or median_ms)


def test_transfer_prediction_uses_decimal_bandwidth_units() -> None:
    model = TransferModel(bandwidth_gbps=10, base_latency_us=100)
    assert model.predict_ms(10_000_000) == pytest.approx(1.1)


def test_bandwidth_only_model_uses_largest_calibration_point() -> None:
    model = bandwidth_only_model((sample(1_000_000, 1), sample(10_000_000, 2)))
    assert model.bandwidth_gbps == pytest.approx(5)
    assert model.base_latency_us == 0


def test_affine_fit_recovers_latency_and_bandwidth() -> None:
    expected = TransferModel(bandwidth_gbps=20, base_latency_us=50)
    calibration = tuple(
        sample(size, expected.predict_ms(size)) for size in (1_000_000, 4_000_000, 16_000_000)
    )
    fitted = fit_affine_transfer_model(calibration)
    assert fitted.bandwidth_gbps == pytest.approx(20)
    assert fitted.base_latency_us == pytest.approx(50)


def test_comparison_reports_holdout_error() -> None:
    model = TransferModel(bandwidth_gbps=10, base_latency_us=0)
    summary = compare_transfer_model(model, (sample(10_000_000, 2),))
    assert summary.mean_absolute_percentage_error_pct == pytest.approx(50)
    assert summary.max_absolute_percentage_error_pct == pytest.approx(50)
    assert summary.points[0].predicted_ms == pytest.approx(1)


def test_samples_are_selected_by_split() -> None:
    payload = {
        "samples": [
            {"split": "calibration", "size_bytes": 1, "median_ms": 1, "p95_ms": 2},
            {"split": "validation", "size_bytes": 2, "median_ms": 2, "p95_ms": 3},
        ]
    }
    assert samples_from_payload(payload, "validation") == (sample(2, 2, 3),)


@pytest.mark.parametrize(
    "invalid",
    [
        TransferSample(0, 1, 1),
        TransferSample(1, 0, 1),
        TransferSample(1, 2, 1),
    ],
)
def test_invalid_samples_are_rejected(invalid: TransferSample) -> None:
    with pytest.raises(ValueError):
        invalid.validate()


def test_affine_fit_rejects_duplicate_sizes() -> None:
    with pytest.raises(ValueError, match="distinct"):
        fit_affine_transfer_model((sample(1, 1), sample(1, 2)))


def attention_sample(
    context_tokens: int, median_ms: float, p95_ms: float | None = None
) -> AttentionSample:
    return AttentionSample(context_tokens, median_ms, p95_ms or median_ms)


def test_attention_shape_counts_decode_flops_and_kv_bytes() -> None:
    shape = AttentionShape(batch_size=1, heads=8, head_dim=64, dtype_bytes=2)
    assert shape.flops(1024) == 4 * 8 * 64 * 1024
    assert shape.modeled_bytes(1024) == (2 * 8 * 1024 * 64 + 2 * 8 * 64) * 2


def test_attention_roofline_uses_slower_floor_plus_base_latency() -> None:
    shape = AttentionShape(1, 1, 1, 1)
    model = AttentionRooflineModel(
        compute_tops=1e-6,
        memory_bandwidth_gbps=1e-3,
        base_latency_us=100,
    )
    compute_ms, memory_ms = model.components_ms(shape, 1000)
    assert compute_ms == pytest.approx(4)
    assert memory_ms == pytest.approx(2.002)
    assert model.predict_ms(shape, 1000) == pytest.approx(4.1)


def test_attention_affine_fit_uses_modeled_bytes() -> None:
    shape = AttentionShape(1, 1, 1, 1)
    expected = TransferModel(bandwidth_gbps=0.001, base_latency_us=50)
    calibration = tuple(
        attention_sample(context, expected.predict_ms(shape.modeled_bytes(context)))
        for context in (100, 400, 1600)
    )
    fitted = fit_attention_affine_model(shape, calibration)
    assert fitted.bandwidth_gbps == pytest.approx(expected.bandwidth_gbps)
    assert fitted.base_latency_us == pytest.approx(expected.base_latency_us)


def test_attention_comparisons_report_holdout_error() -> None:
    shape = AttentionShape(1, 1, 1, 1)
    validation = (attention_sample(1000, 8.2),)
    roofline = AttentionRooflineModel(1e-6, 1e-3, 100)
    roofline_summary = compare_attention_roofline(roofline, shape, validation)
    affine_summary = compare_attention_affine(TransferModel(1e-3, 100), shape, validation)
    assert roofline_summary.mean_absolute_percentage_error_pct == pytest.approx(50)
    assert roofline_summary.points[0].compute_floor_ms == pytest.approx(4)
    assert affine_summary.points[0].predicted_ms == pytest.approx(2.102)


def test_attention_samples_are_selected_by_split() -> None:
    payload = {
        "samples": [
            {
                "split": "calibration",
                "context_tokens": 128,
                "median_ms": 1,
                "p95_ms": 2,
            },
            {
                "split": "validation",
                "context_tokens": 256,
                "median_ms": 2,
                "p95_ms": 3,
            },
        ]
    }
    assert attention_samples_from_payload(payload, "validation") == (attention_sample(256, 2, 3),)


def test_invalid_attention_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        AttentionShape(1, 0, 64, 2).validate()
