"""Step-local attention, normalized GRU recurrence, and step-memory readout.

The generator's token features are frozen. Detector states retain their autograd
graphs during training; use ``torch.no_grad()`` during online inference. A state
snapshot can be reused for a branch or rollback because forward_step never
modifies a prior state or its tensors in place.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int
    state_dim: int = 256
    num_heads: int = 4
    head_dim: int = 64
    dropout: float = 0.1
    backbone: str = "method2"
    memory_enabled: bool = True
    initial_state: str = "zero"
    query_source: str = "gru"
    step_attention_query: str = "step_mean"
    step_gate_context: str = "none"

    def __post_init__(self):
        for name in ("hidden_dim", "state_dim", "num_heads", "head_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_heads * self.head_dim != self.state_dim:
            raise ValueError("num_heads * head_dim must equal state_dim")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.backbone != "method2":
            raise ValueError("This ablation requires backbone='method2'")
        if self.memory_enabled is not True:
            raise ValueError("This ablation requires memory_enabled=True")
        if self.initial_state != "zero":
            raise ValueError("This ablation requires initial_state='zero'")
        if self.query_source != "gru":
            raise ValueError("This ablation requires query_source='gru' for memory retrieval")
        if self.step_attention_query != "step_mean":
            raise ValueError("step_attention_query must be 'step_mean'")
        if self.step_gate_context != "none":
            raise ValueError("step_gate_context must be 'none'")


@dataclass(frozen=True)
class DetectorState:
    """A snapshot after a completed step; treat the contained tensors as read-only.

    c is the recurrent state and p is the prefix prediction readout. Neither
    enters the step encoder. Cached K/V tensors each have shape [heads, head_dim].
    """

    c: Tensor
    p: Tensor
    step_vectors: Tuple[Tensor, ...] = ()
    cached_keys: Tuple[Tensor, ...] = ()
    cached_values: Tuple[Tensor, ...] = ()
    step_ids: Tuple[Any, ...] = ()
    offsets: Tuple[Any, ...] = ()
    step_count: int = 0


@dataclass(frozen=True)
class StepOutput:
    step_logit: Tensor
    prefix_logit: Tensor
    state: DetectorState
    attention: Optional[Tensor]
    memory_gate: Tensor
    z: Tensor
    r: Tensor


class _StepEncoder(nn.Module):
    """Current-step attention with its query derived from the same step's mean.

    This module has no detector-history input. The frozen language-model token
    vectors themselves remain contextualized by the language-model prefix.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.state_dim = config.state_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.key_proj = nn.Linear(config.hidden_dim, config.state_dim)
        self.value_proj = nn.Linear(config.hidden_dim, config.state_dim)
        self.query_proj = nn.Linear(config.state_dim, config.state_dim)
        self.mean_proj = nn.Linear(config.hidden_dim, config.state_dim)
        # Keep Method 2's gate family, but remove its recurrent-state input.
        self.residual_gate = nn.Sequential(
            nn.Linear(config.state_dim * 2, config.state_dim),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(config.state_dim, 1),
        )
        self.output_proj = nn.Identity()
        self.norm = nn.LayerNorm(config.state_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, H: Tensor) -> Tensor:
        length = H.shape[0]
        mean = self.mean_proj(H.mean(0))
        K = self.key_proj(H).view(length, self.num_heads, self.head_dim)
        V = self.value_proj(H).view(length, self.num_heads, self.head_dim)
        Q = self.query_proj(mean).view(self.num_heads, self.head_dim)
        scores = (K * Q.unsqueeze(0)).sum(-1) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=0)
        attended = (weights.unsqueeze(-1) * V).sum(0).reshape(self.state_dim)
        attended = self.output_proj(attended)
        gamma = torch.sigmoid(
            self.residual_gate(torch.cat([mean, attended], -1))
        ).squeeze(-1)
        return self.norm(self.dropout(mean + gamma * attended))


class _StepMemoryAttention(nn.Module):
    """Independent multihead retrieval over cached projected step vectors.

    Slot zero has a learned key and an exactly zero value. The output projection
    has no bias, so selecting only the null slot yields no retrieved signal.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.state_dim = config.state_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.query_proj = nn.Linear(config.state_dim, config.state_dim, bias=False)
        self.key_proj = nn.Linear(config.state_dim, config.state_dim, bias=False)
        self.value_proj = nn.Linear(config.state_dim, config.state_dim, bias=False)
        self.null_key = nn.Parameter(torch.empty(config.num_heads, config.head_dim))
        nn.init.normal_(self.null_key, mean=0.0, std=0.02)
        self.output_proj = nn.Linear(config.state_dim, config.state_dim, bias=False)

    def project_step(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        return (
            self.key_proj(z).view(self.num_heads, self.head_dim),
            self.value_proj(z).view(self.num_heads, self.head_dim),
        )

    def forward(
        self,
        c: Tensor,
        keys: Tuple[Tensor, ...],
        values: Tuple[Tensor, ...],
    ) -> Tuple[Tensor, Tensor]:
        if len(keys) != len(values):
            raise ValueError("Memory keys and values must have equal lengths")
        query = self.query_proj(c).view(self.num_heads, self.head_dim)
        null_scores = (query * self.null_key).sum(-1, keepdim=True)
        if keys:
            key_matrix = torch.stack(keys, dim=1)
            value_matrix = torch.stack(values, dim=1)
            scores = (query.unsqueeze(1) * key_matrix).sum(-1)
            scores = torch.cat([null_scores, scores], dim=1)
        else:
            scores = null_scores
        weights = torch.softmax(scores / math.sqrt(self.head_dim), dim=1)
        if values:
            heads = (weights[:, 1:].unsqueeze(-1) * value_matrix).sum(1)
        else:
            heads = query.new_zeros((self.num_heads, self.head_dim))
        retrieved = self.output_proj(heads.reshape(self.state_dim))
        return retrieved, weights


class StepMemoryDetector(nn.Module):
    """Detector supporting full trajectories and independent online state objects.

    This fixed ablation changes only Method 2's step query and gate inputs.
    Its normalized GRU recurrence, memory retrieval, prefix readout and heads
    retain the Method 2 + memory operations. Memory affects only prefix
    prediction; local step prediction and recurrence do not consume retrieval.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        self.config = config
        self.state_dim = config.state_dim
        self.step_encoder = _StepEncoder(config)
        # Retain the Method 2 module creation order and names where compatible.
        self.step_head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(config.state_dim, 1))
        self.gru = nn.GRUCell(config.state_dim, config.state_dim)
        self.prefix_norm = nn.LayerNorm(config.state_dim)
        self.prefix_head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(config.state_dim, 1))
        self.question_init = None
        self.memory_attention = _StepMemoryAttention(config)
        self.memory_gate = nn.Linear(config.state_dim * 2, 1)
        self.memory_norm = nn.LayerNorm(config.state_dim)

    @property
    def architecture_config(self) -> dict:
        return asdict(self.config)

    def _feature_tensor(self, value: Tensor, *, name: str, ndim: int) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != ndim or value.shape[-1] != self.config.hidden_dim:
            required = f"[tokens, {self.config.hidden_dim}]" if ndim == 2 else f"[{self.config.hidden_dim}]"
            raise ValueError(f"{name} must have shape {required}; got {tuple(value.shape)}")
        if ndim == 2 and value.shape[0] == 0:
            raise ValueError("A completed step must contain at least one token")
        if not value.is_floating_point():
            raise TypeError(f"{name} must contain floating-point hidden features")
        reference = self.step_encoder.mean_proj.weight
        features = value.detach().to(
            device=reference.device, dtype=reference.dtype, non_blocking=True
        )
        # Direct online callers do not pass through the data loader. Validate
        # after conversion so both existing NaN/Inf and dtype overflow fail
        # before they can contaminate the recurrent state or stored memory.
        if not torch.isfinite(features).all().item():
            raise ValueError(f"{name} must remain finite in the model's feature dtype")
        return features

    def initial_state(self, question_mean: Optional[Tensor] = None) -> DetectorState:
        # The interface accepts but deliberately ignores optional question data.
        c = self.step_encoder.mean_proj.weight.new_zeros(self.state_dim)
        return DetectorState(c=c, p=c)

    def _validate_state(self, state: DetectorState):
        if not isinstance(state, DetectorState):
            raise TypeError("state must be a DetectorState returned by this model")
        reference = self.step_encoder.mean_proj.weight
        for name in ("c", "p"):
            value = getattr(state, name)
            if value.shape != (self.state_dim,):
                raise ValueError(f"state.{name} must have shape [{self.state_dim}]")
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"state.{name} device/dtype differs from model; initialize state after moving model")
        lengths = tuple(len(getattr(state, name)) for name in (
            "step_vectors", "cached_keys", "cached_values", "step_ids", "offsets"
        ))
        expected = state.step_count
        if any(length != expected for length in lengths):
            raise ValueError("State memory lengths do not match its completed-step count/configuration")

    def forward_step(
        self,
        hidden_states: Tensor,
        state: DetectorState,
        step_id: Any = None,
        offsets: Any = None,
        return_attention: bool = False,
    ) -> StepOutput:
        """Consume only the newly completed step and return a new state snapshot.

        Step IDs and offsets are opaque metadata; they never enter the model.
        Attention, when requested, has shape [heads, 1 + step_count] with the
        null slot first. Attention weights are retrieval weights, not calibrated
        error-source or dependency probabilities.
        """
        self._validate_state(state)
        H = self._feature_tensor(hidden_states, name="hidden_states", ndim=2)
        z = self.step_encoder(H)
        step_logit = self.step_head(z).squeeze(-1)
        c = self.gru(z.unsqueeze(0), state.c.unsqueeze(0)).squeeze(0)
        c = self.prefix_norm(c)

        key, value = self.memory_attention.project_step(z)
        keys = state.cached_keys + (key,)
        values = state.cached_values + (value,)
        vectors = state.step_vectors + (z,)
        ids = state.step_ids + (state.step_count + 1 if step_id is None else step_id,)
        stored_offsets = state.offsets + (offsets,)
        r, attention = self.memory_attention(c, keys, values)
        gate = torch.sigmoid(self.memory_gate(torch.cat([c, r], -1))).squeeze(-1)
        p = self.memory_norm(c + gate * r)

        prefix_logit = self.prefix_head(p).squeeze(-1)
        new_state = DetectorState(
            c=c, p=p, step_vectors=vectors, cached_keys=keys,
            cached_values=values, step_ids=ids, offsets=stored_offsets,
            step_count=state.step_count + 1,
        )
        return StepOutput(
            step_logit=step_logit, prefix_logit=prefix_logit, state=new_state,
            attention=attention if return_attention else None,
            memory_gate=gate, z=z, r=r,
        )

    def forward_trajectory(self, tr, device=None) -> Tuple[Tensor, Tensor]:
        """Evaluate a sequence of .steps without reading any labels or future data.

        Each step must expose .hidden_states and may expose .step_id/.offsets.
        The optional device argument maintains the existing training-loop interface;
        callers must move the model to that device before creating states.
        """
        if device is not None:
            requested = torch.device(device)
            actual = self.step_encoder.mean_proj.weight.device
            if requested.type != actual.type or (requested.index is not None and requested.index != actual.index):
                raise ValueError(f"Requested device {requested} differs from model device {actual}; move model first")
        state = self.initial_state(getattr(tr, "question_mean", None))
        step_logits, prefix_logits = [], []
        for step in tr.steps:
            output = self.forward_step(
                step.hidden_states, state,
                step_id=getattr(step, "step_id", None),
                offsets=getattr(step, "offsets", None),
            )
            state = output.state
            step_logits.append(output.step_logit)
            prefix_logits.append(output.prefix_logit)
        if not step_logits:
            raise ValueError("A trajectory must contain at least one completed step")
        return torch.stack(step_logits), torch.stack(prefix_logits)

Method4 = StepMemoryDetector


# Only the reported GRU+memory branch is included in this publication.
ARCHITECTURES = ("gru_memory",)
def normalize_config(hidden_dim, config):
    values = dict(config)
    architecture = values.get("architecture", "graph_gru")
    if architecture not in ARCHITECTURES:
        raise ValueError("Unknown architecture: " + str(architecture))
    result = {
        "hidden_dim": hidden_dim,
        "architecture": architecture,
        "state_dim": values.get("state_dim", 128),
        "num_heads": values.get("num_heads", 4),
        "dropout": values.get("dropout", 0.1),
        "graph_layers": values.get("graph_layers", 2),
        "recent_neighbors": values.get("recent_neighbors", 16),
        "semantic_neighbors": values.get("semantic_neighbors", 16),
    }
    for name in ("hidden_dim", "state_dim", "num_heads", "graph_layers"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(name + " must be a positive integer")
    for name in ("recent_neighbors", "semantic_neighbors"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(name + " must be a nonnegative integer")
    if result["state_dim"] % result["num_heads"]:
        raise ValueError("state_dim must be divisible by num_heads")
    dropout = result["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (float, int)) or not 0 <= dropout < 1:
        raise ValueError("dropout must be finite and in [0, 1)")
    return result


class GRUMemoryDetector(StepMemoryDetector):
    def __init__(self, hidden_dim, config):
        chosen = normalize_config(hidden_dim, config)
        super().__init__(ModelConfig(hidden_dim=hidden_dim, state_dim=chosen["state_dim"],
            num_heads=chosen["num_heads"], head_dim=chosen["state_dim"] // chosen["num_heads"],
            dropout=chosen["dropout"]))
        self.graph_config = chosen  # preserves original checkpoint architecture metadata

    @property
    def architecture_config(self):
        return dict(self.graph_config)


def build_model(hidden_dim, config):
    return GRUMemoryDetector(hidden_dim, config)
