from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

GIB = 1024**3


def _stable_numbers(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 10)
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
    context_tokens: int
    generated_tokens: int
    concurrent_sequences: int

    def validate(self) -> None:
        positive = {
            "parameter_count_b": self.parameter_count_b,
            "layers": self.layers,
            "hidden_size": self.hidden_size,
            "attention_heads": self.attention_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "weight_bits": self.weight_bits,
            "kv_bits": self.kv_bits,
            "context_tokens": self.context_tokens,
            "generated_tokens": self.generated_tokens,
            "concurrent_sequences": self.concurrent_sequences,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"workload values must be positive: {', '.join(invalid)}")
        if self.kv_heads > self.attention_heads:
            raise ValueError("kv_heads cannot exceed attention_heads")
        if self.weight_bits not in {4, 8, 16, 32} or self.kv_bits not in {4, 8, 16, 32}:
            raise ValueError("weight_bits and kv_bits must be one of 4, 8, 16, 32")

    @property
    def weight_bytes(self) -> float:
        return self.parameter_count_b * 1_000_000_000 * self.weight_bits / 8

    @property
    def kv_bytes_per_token_per_sequence(self) -> float:
        # One key and one value vector per layer. GQA/MQA use kv_heads, not attention_heads.
        return 2 * self.layers * self.kv_heads * self.head_dim * self.kv_bits / 8

    def decode_flops(self, sequence_length: int) -> float:
        # Transparent first-order model: parameter pass plus QK/AV attention work.
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
    remote_base_latency_us: float
    accelerator_tops: float
    near_memory_tops: float = 12.0
    hbm_energy_pj_per_byte: float = 3.0
    remote_energy_pj_per_byte: float = 15.0
    compute_energy_pj_per_flop: float = 0.35

    def validate(self) -> None:
        values = asdict(self)
        invalid = [name for name, value in values.items() if name != "name" and value <= 0]
        if invalid:
            raise ValueError(f"system values must be positive: {', '.join(invalid)}")

    @property
    def hbm_capacity_bytes(self) -> float:
        return self.hbm_capacity_gib * GIB

    @property
    def remote_capacity_bytes(self) -> float:
        return self.remote_capacity_gib * GIB


PolicyKind = Literal["hbm_only", "sliding_window", "near_memory"]


@dataclass(frozen=True)
class PlacementPolicy:
    name: str
    kind: PolicyKind
    hbm_window_tokens: int
    transfer_overlap_ratio: float = 0.35
    near_memory_reduction_ratio: float = 0.90

    def validate(self) -> None:
        if self.kind not in {"hbm_only", "sliding_window", "near_memory"}:
            raise ValueError(f"unsupported policy kind: {self.kind}")
        if self.hbm_window_tokens <= 0:
            raise ValueError("hbm_window_tokens must be positive")
        if not 0 <= self.transfer_overlap_ratio <= 1:
            raise ValueError("transfer_overlap_ratio must be between 0 and 1")
        if not 0 <= self.near_memory_reduction_ratio < 1:
            raise ValueError("near_memory_reduction_ratio must be in [0, 1)")


@dataclass(frozen=True)
class SimulationRequest:
    workload: Workload
    system: MemorySystem
    policy: PlacementPolicy

    def validate(self) -> None:
        self.workload.validate()
        self.system.validate()
        self.policy.validate()


@dataclass(frozen=True)
class StepMetrics:
    token_index: int
    sequence_length: int
    compute_ms: float
    hbm_ms: float
    remote_transfer_ms: float
    near_memory_compute_ms: float
    latency_ms: float
    hbm_kv_gib: float
    remote_kv_gib: float
    hbm_read_gib: float
    remote_read_gib: float
    bottleneck: str


@dataclass(frozen=True)
class SimulationResult:
    workload_name: str
    system_name: str
    policy_name: str
    feasible: bool
    rejection_reason: str | None
    mean_decode_latency_ms: float
    p95_decode_latency_ms: float
    throughput_tokens_s: float
    total_hbm_read_gib: float
    total_remote_read_gib: float
    estimated_energy_j: float
    peak_hbm_kv_gib: float
    peak_remote_kv_gib: float
    bottleneck: str
    assumptions: tuple[str, ...]
    steps: tuple[StepMetrics, ...]

    def to_dict(self, include_steps: bool = True) -> dict[str, Any]:
        payload = cast(dict[str, Any], _stable_numbers(asdict(self)))
        if not include_steps:
            payload.pop("steps")
        return payload
