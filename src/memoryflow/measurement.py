from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any


@dataclass(frozen=True)
class TransferSample:
    size_bytes: int
    median_ms: float
    p95_ms: float

    def validate(self) -> None:
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if self.median_ms <= 0 or self.p95_ms <= 0:
            raise ValueError("latencies must be positive")
        if self.p95_ms < self.median_ms:
            raise ValueError("p95_ms cannot be lower than median_ms")

    @property
    def observed_bandwidth_gbps(self) -> float:
        return self.size_bytes / (self.median_ms / 1000) / 1_000_000_000


@dataclass(frozen=True)
class TransferModel:
    bandwidth_gbps: float
    base_latency_us: float

    def validate(self) -> None:
        if self.bandwidth_gbps <= 0:
            raise ValueError("bandwidth_gbps must be positive")
        if self.base_latency_us < 0:
            raise ValueError("base_latency_us cannot be negative")

    def predict_ms(self, size_bytes: int) -> float:
        self.validate()
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        transfer_ms = size_bytes / (self.bandwidth_gbps * 1_000_000_000) * 1000
        return self.base_latency_us / 1000 + transfer_ms


@dataclass(frozen=True)
class AttentionShape:
    batch_size: int
    heads: int
    head_dim: int
    dtype_bytes: int

    def validate(self) -> None:
        values = (self.batch_size, self.heads, self.head_dim, self.dtype_bytes)
        if any(value <= 0 for value in values):
            raise ValueError("attention shape values must be positive")

    def flops(self, context_tokens: int) -> int:
        self.validate()
        if context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        return 4 * self.batch_size * self.heads * self.head_dim * context_tokens

    def modeled_bytes(self, context_tokens: int) -> int:
        self.validate()
        if context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        kv_elements = 2 * self.batch_size * self.heads * context_tokens * self.head_dim
        query_and_output_elements = 2 * self.batch_size * self.heads * self.head_dim
        return (kv_elements + query_and_output_elements) * self.dtype_bytes


@dataclass(frozen=True)
class AttentionSample:
    context_tokens: int
    median_ms: float
    p95_ms: float

    def validate(self) -> None:
        if self.context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        if self.median_ms <= 0 or self.p95_ms <= 0:
            raise ValueError("latencies must be positive")
        if self.p95_ms < self.median_ms:
            raise ValueError("p95_ms cannot be lower than median_ms")


@dataclass(frozen=True)
class AttentionRooflineModel:
    compute_tops: float
    memory_bandwidth_gbps: float
    base_latency_us: float

    def validate(self) -> None:
        if self.compute_tops <= 0 or self.memory_bandwidth_gbps <= 0:
            raise ValueError("compute and bandwidth calibration must be positive")
        if self.base_latency_us < 0:
            raise ValueError("base_latency_us cannot be negative")

    def components_ms(self, shape: AttentionShape, context_tokens: int) -> tuple[float, float]:
        self.validate()
        compute_ms = shape.flops(context_tokens) / (self.compute_tops * 1_000_000_000_000) * 1000
        memory_ms = (
            shape.modeled_bytes(context_tokens)
            / (self.memory_bandwidth_gbps * 1_000_000_000)
            * 1000
        )
        return compute_ms, memory_ms

    def predict_ms(self, shape: AttentionShape, context_tokens: int) -> float:
        compute_ms, memory_ms = self.components_ms(shape, context_tokens)
        return self.base_latency_us / 1000 + max(compute_ms, memory_ms)


@dataclass(frozen=True)
class ComparisonPoint:
    size_bytes: int
    measured_ms: float
    predicted_ms: float
    absolute_percentage_error_pct: float


@dataclass(frozen=True)
class ComparisonSummary:
    mean_absolute_percentage_error_pct: float
    max_absolute_percentage_error_pct: float
    points: tuple[ComparisonPoint, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttentionComparisonPoint:
    context_tokens: int
    measured_ms: float
    predicted_ms: float
    absolute_percentage_error_pct: float
    compute_floor_ms: float | None = None
    memory_floor_ms: float | None = None


@dataclass(frozen=True)
class AttentionComparisonSummary:
    mean_absolute_percentage_error_pct: float
    max_absolute_percentage_error_pct: float
    points: tuple[AttentionComparisonPoint, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bandwidth_only_model(calibration_samples: tuple[TransferSample, ...]) -> TransferModel:
    if not calibration_samples:
        raise ValueError("at least one calibration sample is required")
    for sample in calibration_samples:
        sample.validate()
    largest = max(calibration_samples, key=lambda sample: sample.size_bytes)
    return TransferModel(largest.observed_bandwidth_gbps, 0.0)


def fit_affine_transfer_model(
    calibration_samples: tuple[TransferSample, ...],
) -> TransferModel:
    if len(calibration_samples) < 2:
        raise ValueError("at least two calibration samples are required")
    for sample in calibration_samples:
        sample.validate()

    xs = [float(sample.size_bytes) for sample in calibration_samples]
    ys = [sample.median_ms for sample in calibration_samples]
    mean_x = fmean(xs)
    mean_y = fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("calibration samples must use distinct sizes")
    slope_ms_per_byte = (
        sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator
    )
    if slope_ms_per_byte <= 0:
        raise ValueError("measured latency must increase with transfer size")

    intercept_ms = mean_y - slope_ms_per_byte * mean_x
    if intercept_ms < 0:
        intercept_ms = 0.0
        slope_ms_per_byte = sum(x * y for x, y in zip(xs, ys, strict=True)) / sum(x * x for x in xs)
    bandwidth_gbps = 1 / (slope_ms_per_byte * 1_000_000)
    return TransferModel(bandwidth_gbps, intercept_ms * 1000)


def compare_transfer_model(
    model: TransferModel,
    validation_samples: tuple[TransferSample, ...],
) -> ComparisonSummary:
    model.validate()
    if not validation_samples:
        raise ValueError("at least one validation sample is required")

    points: list[ComparisonPoint] = []
    for sample in validation_samples:
        sample.validate()
        predicted_ms = model.predict_ms(sample.size_bytes)
        error_pct = abs(predicted_ms - sample.median_ms) / sample.median_ms * 100
        points.append(
            ComparisonPoint(
                size_bytes=sample.size_bytes,
                measured_ms=sample.median_ms,
                predicted_ms=predicted_ms,
                absolute_percentage_error_pct=error_pct,
            )
        )

    errors = [point.absolute_percentage_error_pct for point in points]
    return ComparisonSummary(
        mean_absolute_percentage_error_pct=fmean(errors),
        max_absolute_percentage_error_pct=max(errors),
        points=tuple(points),
    )


def fit_attention_affine_model(
    shape: AttentionShape,
    calibration_samples: tuple[AttentionSample, ...],
) -> TransferModel:
    shape.validate()
    converted: list[TransferSample] = []
    for sample in calibration_samples:
        sample.validate()
        converted.append(
            TransferSample(
                size_bytes=shape.modeled_bytes(sample.context_tokens),
                median_ms=sample.median_ms,
                p95_ms=sample.p95_ms,
            )
        )
    return fit_affine_transfer_model(tuple(converted))


def compare_attention_roofline(
    model: AttentionRooflineModel,
    shape: AttentionShape,
    validation_samples: tuple[AttentionSample, ...],
) -> AttentionComparisonSummary:
    model.validate()
    shape.validate()
    if not validation_samples:
        raise ValueError("at least one validation sample is required")
    points: list[AttentionComparisonPoint] = []
    for sample in validation_samples:
        sample.validate()
        compute_ms, memory_ms = model.components_ms(shape, sample.context_tokens)
        predicted_ms = model.predict_ms(shape, sample.context_tokens)
        error_pct = abs(predicted_ms - sample.median_ms) / sample.median_ms * 100
        points.append(
            AttentionComparisonPoint(
                context_tokens=sample.context_tokens,
                measured_ms=sample.median_ms,
                predicted_ms=predicted_ms,
                absolute_percentage_error_pct=error_pct,
                compute_floor_ms=compute_ms,
                memory_floor_ms=memory_ms,
            )
        )
    errors = [point.absolute_percentage_error_pct for point in points]
    return AttentionComparisonSummary(fmean(errors), max(errors), tuple(points))


def compare_attention_affine(
    model: TransferModel,
    shape: AttentionShape,
    validation_samples: tuple[AttentionSample, ...],
) -> AttentionComparisonSummary:
    model.validate()
    shape.validate()
    if not validation_samples:
        raise ValueError("at least one validation sample is required")
    points: list[AttentionComparisonPoint] = []
    for sample in validation_samples:
        sample.validate()
        predicted_ms = model.predict_ms(shape.modeled_bytes(sample.context_tokens))
        error_pct = abs(predicted_ms - sample.median_ms) / sample.median_ms * 100
        points.append(
            AttentionComparisonPoint(
                context_tokens=sample.context_tokens,
                measured_ms=sample.median_ms,
                predicted_ms=predicted_ms,
                absolute_percentage_error_pct=error_pct,
            )
        )
    errors = [point.absolute_percentage_error_pct for point in points]
    return AttentionComparisonSummary(fmean(errors), max(errors), tuple(points))


def samples_from_payload(payload: dict[str, Any], split: str) -> tuple[TransferSample, ...]:
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("measurement payload samples must be a list")
    samples: list[TransferSample] = []
    for raw in raw_samples:
        if not isinstance(raw, dict) or raw.get("split") != split:
            continue
        samples.append(
            TransferSample(
                size_bytes=int(raw["size_bytes"]),
                median_ms=float(raw["median_ms"]),
                p95_ms=float(raw["p95_ms"]),
            )
        )
    if not samples:
        raise ValueError(f"measurement payload has no {split} samples")
    return tuple(samples)


def attention_samples_from_payload(
    payload: dict[str, Any], split: str
) -> tuple[AttentionSample, ...]:
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("attention payload samples must be a list")
    samples: list[AttentionSample] = []
    for raw in raw_samples:
        if not isinstance(raw, dict) or raw.get("split") != split:
            continue
        samples.append(
            AttentionSample(
                context_tokens=int(raw["context_tokens"]),
                median_ms=float(raw["median_ms"]),
                p95_ms=float(raw["p95_ms"]),
            )
        )
    if not samples:
        raise ValueError(f"attention payload has no {split} samples")
    return tuple(samples)
