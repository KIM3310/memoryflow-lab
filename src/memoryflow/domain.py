from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

GIB = 1024**3
RESULT_SCHEMA_VERSION = "2.0"
MAX_NAME_LENGTH = 160
MAX_SOURCE_LENGTH = 500
MAX_CONTEXT_TOKENS = 4_194_304
MAX_GENERATED_TOKENS = 4_096
MAX_CONCURRENT_SEQUENCES = 4_096
MAX_PARAMETER_COUNT_B = 100_000.0
MAX_LAYERS = 4_096
MAX_HIDDEN_SIZE = 1_048_576
MAX_ATTENTION_HEADS = 16_384
MAX_HEAD_DIM = 65_536
MAX_NUMERIC_INPUT = 1e15
# 1e-9 GB/s is one byte/s and 1e-9 TOPS is 1,000 FLOP/s. Inputs below
# this broad analytical floor are not meaningful for this model and risk overflow.
MIN_EFFECTIVE_RATE = 1e-9
MAX_HBM_WINDOW_TOKENS = MAX_CONTEXT_TOKENS + MAX_GENERATED_TOKENS
MAX_KV_PAGE_TOKENS = 65_536


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _is_positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_text(value: object, label: str, maximum: int = MAX_NAME_LENGTH) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must be a non-empty string of at most {maximum} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must contain valid Unicode scalar values") from exc


def _stable_numbers(value: Any) -> Any:
    if isinstance(value, float):
        # Significant digits preserve very small non-zero metrics while keeping evidence compact.
        return float(format(value, ".12g"))
    if isinstance(value, dict):
        return {key: _stable_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stable_numbers(item) for item in value]
    return value


@dataclass(frozen=True)
class Workload:
    name: str
    parameter_count_b: float
    layers: int
    hidden_size: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    weight_bits: int
    kv_bits: int
    activation_bits: int
    context_tokens: int
    generated_tokens: int
    concurrent_sequences: int

    def validate(self) -> None:
        _validate_text(self.name, "workload name")
        if not _is_finite_number(self.parameter_count_b) or self.parameter_count_b <= 0:
            raise ValueError("parameter_count_b must be a positive finite number")
        if self.parameter_count_b > MAX_PARAMETER_COUNT_B:
            raise ValueError(f"parameter_count_b cannot exceed {MAX_PARAMETER_COUNT_B:,.0f}")

        positive_integers = {
            "layers": self.layers,
            "hidden_size": self.hidden_size,
            "attention_heads": self.attention_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "weight_bits": self.weight_bits,
            "kv_bits": self.kv_bits,
            "activation_bits": self.activation_bits,
            "context_tokens": self.context_tokens,
            "generated_tokens": self.generated_tokens,
            "concurrent_sequences": self.concurrent_sequences,
        }
        invalid = [
            name for name, value in positive_integers.items() if not _is_positive_integer(value)
        ]
        if invalid:
            raise ValueError(f"workload values must be positive integers: {', '.join(invalid)}")
        if self.context_tokens > MAX_CONTEXT_TOKENS:
            raise ValueError(f"context_tokens cannot exceed {MAX_CONTEXT_TOKENS:,}")
        if self.generated_tokens > MAX_GENERATED_TOKENS:
            raise ValueError(f"generated_tokens cannot exceed {MAX_GENERATED_TOKENS:,}")
        if self.concurrent_sequences > MAX_CONCURRENT_SEQUENCES:
            raise ValueError(f"concurrent_sequences cannot exceed {MAX_CONCURRENT_SEQUENCES:,}")
        dimension_limits = {
            "layers": (self.layers, MAX_LAYERS),
            "hidden_size": (self.hidden_size, MAX_HIDDEN_SIZE),
            "attention_heads": (self.attention_heads, MAX_ATTENTION_HEADS),
            "kv_heads": (self.kv_heads, MAX_ATTENTION_HEADS),
            "head_dim": (self.head_dim, MAX_HEAD_DIM),
        }
        oversized = [name for name, (value, maximum) in dimension_limits.items() if value > maximum]
        if oversized:
            raise ValueError(f"workload dimensions exceed supported bounds: {', '.join(oversized)}")
        if self.kv_heads > self.attention_heads:
            raise ValueError("kv_heads cannot exceed attention_heads")
        if self.attention_heads % self.kv_heads != 0:
            raise ValueError("attention_heads must be divisible by kv_heads")
        if self.attention_heads * self.head_dim != self.hidden_size:
            raise ValueError("attention_heads * head_dim must equal hidden_size")
        precisions = (self.weight_bits, self.kv_bits, self.activation_bits)
        if any(bits not in {4, 8, 16, 32} for bits in precisions):
            raise ValueError(
                "weight_bits, kv_bits, and activation_bits must be one of 4, 8, 16, 32"
            )

    @property
    def weight_bytes(self) -> float:
        return self.parameter_count_b * 1_000_000_000 * self.weight_bits / 8

    @property
    def kv_bytes_per_layer_per_token_per_sequence(self) -> float:
        # One key and one value vector. GQA/MQA use kv_heads, not attention_heads.
        return 2 * self.kv_heads * self.head_dim * self.kv_bits / 8

    @property
    def kv_bytes_per_token_per_sequence(self) -> float:
        return self.layers * self.kv_bytes_per_layer_per_token_per_sequence

    @property
    def query_bytes_per_layer_per_sequence(self) -> float:
        return self.attention_heads * self.head_dim * self.activation_bits / 8

    def decode_flops(self, sequence_length: int) -> float:
        if not _is_positive_integer(sequence_length):
            raise ValueError("sequence_length must be a positive integer")
        # First-order parameter pass plus QK and AV attention work.
        parameter_flops = 2 * self.parameter_count_b * 1_000_000_000
        attention_flops = 4 * self.layers * self.hidden_size * sequence_length
        return (parameter_flops + attention_flops) * self.concurrent_sequences


@dataclass(frozen=True)
class MemorySystem:
    name: str
    hbm_capacity_gib: float
    hbm_bandwidth_gbps: float
    remote_capacity_gib: float
    remote_bandwidth_gbps: float
    remote_memory_bandwidth_gbps: float
    remote_base_latency_us: float
    accelerator_tops: float
    near_memory_tops: float = 12.0
    hbm_reserved_gib: float = 2.0
    hbm_bandwidth_efficiency: float = 0.75
    remote_bandwidth_efficiency: float = 0.70
    remote_memory_bandwidth_efficiency: float = 0.70
    accelerator_compute_efficiency: float = 0.55
    near_memory_compute_efficiency: float = 0.60
    remote_protocol_overhead_ratio: float = 0.05
    hbm_energy_pj_per_byte: float = 3.0
    remote_memory_energy_pj_per_byte: float = 12.0
    remote_link_energy_pj_per_byte: float = 8.0
    accelerator_compute_energy_pj_per_flop: float = 0.35
    near_memory_compute_energy_pj_per_flop: float = 0.20

    def validate(self) -> None:
        _validate_text(self.name, "system name")
        positive_values = {
            "hbm_capacity_gib": self.hbm_capacity_gib,
            "hbm_bandwidth_gbps": self.hbm_bandwidth_gbps,
            "remote_capacity_gib": self.remote_capacity_gib,
            "remote_bandwidth_gbps": self.remote_bandwidth_gbps,
            "remote_memory_bandwidth_gbps": self.remote_memory_bandwidth_gbps,
            "accelerator_tops": self.accelerator_tops,
            "near_memory_tops": self.near_memory_tops,
        }
        invalid = [
            name
            for name, value in positive_values.items()
            if not _is_finite_number(value) or value <= 0
        ]
        if invalid:
            raise ValueError(f"system values must be positive finite numbers: {', '.join(invalid)}")

        non_negative_values = {
            "hbm_reserved_gib": self.hbm_reserved_gib,
            "remote_base_latency_us": self.remote_base_latency_us,
            "hbm_energy_pj_per_byte": self.hbm_energy_pj_per_byte,
            "remote_memory_energy_pj_per_byte": self.remote_memory_energy_pj_per_byte,
            "remote_link_energy_pj_per_byte": self.remote_link_energy_pj_per_byte,
            "accelerator_compute_energy_pj_per_flop": (self.accelerator_compute_energy_pj_per_flop),
            "near_memory_compute_energy_pj_per_flop": (self.near_memory_compute_energy_pj_per_flop),
        }
        invalid_non_negative = [
            name
            for name, value in non_negative_values.items()
            if not _is_finite_number(value) or value < 0
        ]
        if invalid_non_negative:
            raise ValueError(
                "system values must be non-negative finite numbers: "
                + ", ".join(invalid_non_negative)
            )
        oversized = [
            name
            for name, value in {**positive_values, **non_negative_values}.items()
            if value > MAX_NUMERIC_INPUT
        ]
        if oversized:
            raise ValueError(f"system values exceed supported bounds: {', '.join(oversized)}")
        if self.hbm_reserved_gib >= self.hbm_capacity_gib:
            raise ValueError("hbm_reserved_gib must be lower than hbm_capacity_gib")

        efficiencies = {
            "hbm_bandwidth_efficiency": self.hbm_bandwidth_efficiency,
            "remote_bandwidth_efficiency": self.remote_bandwidth_efficiency,
            "remote_memory_bandwidth_efficiency": self.remote_memory_bandwidth_efficiency,
            "accelerator_compute_efficiency": self.accelerator_compute_efficiency,
            "near_memory_compute_efficiency": self.near_memory_compute_efficiency,
        }
        invalid_efficiencies = [
            name
            for name, value in efficiencies.items()
            if not _is_finite_number(value) or not 0 < value <= 1
        ]
        if invalid_efficiencies:
            raise ValueError(
                "efficiency values must be finite numbers in (0, 1]: "
                + ", ".join(invalid_efficiencies)
            )
        if (
            not _is_finite_number(self.remote_protocol_overhead_ratio)
            or not 0 <= self.remote_protocol_overhead_ratio <= 1
        ):
            raise ValueError("remote_protocol_overhead_ratio must be a finite number in [0, 1]")

        effective_rates = {
            "effective_hbm_bandwidth_gbps": self.effective_hbm_bandwidth_gbps,
            "effective_remote_bandwidth_gbps": self.effective_remote_bandwidth_gbps,
            "effective_remote_memory_bandwidth_gbps": (self.effective_remote_memory_bandwidth_gbps),
            "effective_accelerator_tops": self.effective_accelerator_tops,
            "effective_near_memory_tops": self.effective_near_memory_tops,
        }
        unsafe_rates = [
            name
            for name, value in effective_rates.items()
            if not _is_finite_number(value) or value < MIN_EFFECTIVE_RATE
        ]
        if unsafe_rates:
            raise ValueError(
                f"effective rates must be at least {MIN_EFFECTIVE_RATE:g}: "
                + ", ".join(unsafe_rates)
            )

    @property
    def hbm_capacity_bytes(self) -> float:
        return self.hbm_capacity_gib * GIB

    @property
    def hbm_reserved_bytes(self) -> float:
        return self.hbm_reserved_gib * GIB

    @property
    def usable_hbm_capacity_bytes(self) -> float:
        return self.hbm_capacity_bytes - self.hbm_reserved_bytes

    @property
    def remote_capacity_bytes(self) -> float:
        return self.remote_capacity_gib * GIB

    @property
    def effective_hbm_bandwidth_gbps(self) -> float:
        return self.hbm_bandwidth_gbps * self.hbm_bandwidth_efficiency

    @property
    def effective_remote_bandwidth_gbps(self) -> float:
        return self.remote_bandwidth_gbps * self.remote_bandwidth_efficiency

    @property
    def effective_remote_memory_bandwidth_gbps(self) -> float:
        return self.remote_memory_bandwidth_gbps * self.remote_memory_bandwidth_efficiency

    @property
    def effective_accelerator_tops(self) -> float:
        return self.accelerator_tops * self.accelerator_compute_efficiency

    @property
    def effective_near_memory_tops(self) -> float:
        return self.near_memory_tops * self.near_memory_compute_efficiency


PolicyKind = Literal["hbm_only", "sliding_window", "near_memory"]
ProfileKind = Literal["synthetic", "user_supplied"]
MeasurementScope = Literal["none", "user_supplied"]


@dataclass(frozen=True)
class PlacementPolicy:
    name: str
    kind: PolicyKind
    hbm_window_tokens: int
    kv_page_tokens: int = 16
    transfer_overlap_ratio: float = 0.35
    near_memory_accumulator_bits: int = 32

    def validate(self) -> None:
        _validate_text(self.name, "policy name")
        if self.kind not in {"hbm_only", "sliding_window", "near_memory"}:
            raise ValueError(f"unsupported policy kind: {self.kind}")
        if not _is_positive_integer(self.hbm_window_tokens):
            raise ValueError("hbm_window_tokens must be a positive integer")
        if not _is_positive_integer(self.kv_page_tokens):
            raise ValueError("kv_page_tokens must be a positive integer")
        if self.hbm_window_tokens > MAX_HBM_WINDOW_TOKENS:
            raise ValueError(f"hbm_window_tokens cannot exceed {MAX_HBM_WINDOW_TOKENS:,}")
        if self.kv_page_tokens > MAX_KV_PAGE_TOKENS:
            raise ValueError(f"kv_page_tokens cannot exceed {MAX_KV_PAGE_TOKENS:,}")
        if self.hbm_window_tokens % self.kv_page_tokens != 0:
            raise ValueError("hbm_window_tokens must be divisible by kv_page_tokens")
        if (
            not _is_finite_number(self.transfer_overlap_ratio)
            or not 0 <= self.transfer_overlap_ratio <= 1
        ):
            raise ValueError("transfer_overlap_ratio must be a finite number between 0 and 1")
        if (
            not isinstance(self.near_memory_accumulator_bits, int)
            or isinstance(self.near_memory_accumulator_bits, bool)
            or self.near_memory_accumulator_bits not in {16, 32}
        ):
            raise ValueError("near_memory_accumulator_bits must be 16 or 32")


@dataclass(frozen=True)
class ScenarioProvenance:
    hardware_profile: ProfileKind = "synthetic"
    source: str = "programmatic synthetic input"
    measurement_scope: MeasurementScope = "none"

    def validate(self) -> None:
        if self.hardware_profile not in {"synthetic", "user_supplied"}:
            raise ValueError(f"unsupported hardware_profile: {self.hardware_profile}")
        _validate_text(self.source, "provenance source", MAX_SOURCE_LENGTH)
        if self.measurement_scope not in {"none", "user_supplied"}:
            raise ValueError(f"unsupported measurement_scope: {self.measurement_scope}")
        if self.hardware_profile == "synthetic" and self.measurement_scope != "none":
            raise ValueError("synthetic hardware profiles must use measurement_scope 'none'")


@dataclass(frozen=True)
class SimulationRequest:
    workload: Workload
    system: MemorySystem
    policy: PlacementPolicy
    provenance: ScenarioProvenance = ScenarioProvenance()

    def validate(self) -> None:
        self.workload.validate()
        self.system.validate()
        self.policy.validate()
        self.provenance.validate()


@dataclass(frozen=True)
class StepMetrics:
    token_index: int
    sequence_length: int
    compute_ms: float
    hbm_ms: float
    remote_link_ms: float
    remote_memory_ms: float
    remote_fixed_ms: float
    near_memory_compute_ms: float
    remote_service_ms: float
    exposed_remote_ms: float
    latency_ms: float
    hbm_kv_allocated_gib: float
    remote_kv_allocated_gib: float
    cold_kv_useful_gib: float
    hbm_read_gib: float
    hbm_write_gib: float
    remote_memory_read_gib: float
    remote_memory_write_gib: float
    interconnect_read_gib: float
    interconnect_write_gib: float
    remote_transactions: int
    bottleneck: str


@dataclass(frozen=True)
class SimulationResult:
    workload_name: str
    system_name: str
    policy_name: str
    input_hardware_profile: ProfileKind
    feasible: bool
    rejection_reason: str | None
    mean_decode_latency_ms: float
    p95_step_latency_ms: float
    throughput_tokens_s: float
    total_hbm_read_gib: float
    total_hbm_write_gib: float
    total_cold_kv_useful_gib: float
    total_remote_memory_read_gib: float
    total_remote_memory_write_gib: float
    total_interconnect_read_gib: float
    total_interconnect_write_gib: float
    remote_memory_read_amplification: float
    interconnect_read_per_remote_byte: float
    mean_remote_service_ms: float
    mean_exposed_remote_ms: float
    estimated_energy_j: float
    peak_hbm_kv_allocated_gib: float
    peak_remote_kv_allocated_gib: float
    peak_kv_fragmentation_pct: float
    hbm_capacity_headroom_gib: float
    remote_capacity_headroom_gib: float
    bottleneck: str
    assumptions: tuple[str, ...]
    steps: tuple[StepMetrics, ...]

    def assert_finite(self) -> None:
        def visit(value: object, path: str) -> None:
            if isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError(f"simulation result contains a non-finite number at {path}")
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    visit(item, f"{path}.{key}")
                return
            if isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    visit(item, f"{path}[{index}]")

        visit(asdict(self), "result")

    def to_dict(self, include_steps: bool = True) -> dict[str, Any]:
        self.assert_finite()
        body = cast(dict[str, Any], _stable_numbers(asdict(self)))
        if not include_steps:
            body.pop("steps")
        return {"schema_version": RESULT_SCHEMA_VERSION, **body}
