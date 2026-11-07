"""Portable research architectures exported from preserved source definitions.

Only import/dependency routing is changed. See MODEL_BINDINGS.json for source hashes.
"""
from __future__ import annotations
import math
from typing import Any
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
EPSILON=1e-12
RELATION_WIDTH=256
HIDDEN_WIDTH=256


class AggregationOnlyStreaming(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        # Keep original Streaming classifier initialization/RNG consumption.
        self.probe = nn.Linear(width, 1)
        # Exact previous content_only scalar-weight network; never value features.
        self.scorer = nn.Sequential(
            nn.Linear(width, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    @staticmethod
    def streaming_l2(values: torch.Tensor) -> torch.Tensor:
        norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
        denominator = torch.where(
            norms < 10 * torch.finfo(values.dtype).eps,
            torch.ones_like(norms),
            norms,
        )
        return values / denominator

    def forward_bags(self, tokens, offsets, indices, device):
        bounds = [(int(offsets[i]), int(offsets[i + 1])) for i in indices]
        assert bounds and all(end > start for start, end in bounds)
        # A single scorer call over only valid tokens of the logical Step batch.
        raw = torch.cat([tokens[start:end] for start, end in bounds]).to(
            device=device, dtype=torch.float32
        )
        logits = self.scorer(raw).flatten()
        pooled, entropies, effective, lengths = [], [], [], []
        cursor = 0
        for start, end in bounds:
            length = end - start
            weights = torch.softmax(logits[cursor:cursor + length], dim=0)
            # The scorer's intermediate values are NEVER pooled.
            pooled.append((raw[cursor:cursor + length] * weights[:, None]).sum(0))
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum()
            entropies.append(entropy)
            effective.append(entropy.exp())
            lengths.append(length)
            cursor += length
        values = self.streaming_l2(torch.stack(pooled))
        return (
            self.probe(values).flatten(),
            torch.stack(entropies),
            torch.stack(effective),
            lengths,
        )

class SharedRepresentationStreaming(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.mapper = nn.Sequential(nn.Linear(width, 256), nn.BatchNorm1d(256), nn.ReLU())
        self.scorer = nn.Linear(256, 1)
        self.probe = nn.Linear(256, 1)

    streaming_l2 = staticmethod(AggregationOnlyStreaming.streaming_l2)

    def forward_bags(self, tokens, offsets, indices, device):
        bounds = [(int(offsets[i]), int(offsets[i + 1])) for i in indices]
        assert bounds and all(hi > lo for lo, hi in bounds)
        flat = torch.cat([tokens[lo:hi] for lo, hi in bounds]).to(device=device, dtype=torch.float32)
        mapped = self.mapper(flat)  # Exactly one shared branch and one BN call.
        scores = self.scorer(mapped).flatten()
        pooled, entropy, effective, lengths = [], [], [], []
        cursor = 0
        for lo, hi in bounds:
            n = hi - lo
            weights = torch.softmax(scores[cursor:cursor + n], dim=0)
            pooled.append((mapped[cursor:cursor + n] * weights[:, None]).sum(0))
            ent = -(weights * weights.clamp_min(1e-12).log()).sum()
            entropy.append(ent)
            effective.append(ent.exp())
            lengths.append(n)
            cursor += n
        values = self.streaming_l2(torch.stack(pooled))
        return self.probe(values).flatten(), torch.stack(entropy), torch.stack(effective), lengths

class LearnedTokenStep(nn.Module):
    """Exact architecture from the completed learned-token Step run."""

    def __init__(self, dimension: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dimension, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
        self.probe = nn.Linear(dimension, 1)

    def pool_indices(
        self,
        cache: dict[str, Any],
        indices: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        offsets = cache["offsets"]
        pieces: list[torch.Tensor] = []
        lengths: list[int] = []
        for index in indices.tolist():
            left, right = int(offsets[index]), int(offsets[index + 1])
            pieces.append(cache["tokens"][left:right])
            lengths.append(right - left)
        flat = torch.cat(pieces).to(device=device, dtype=torch.float32, non_blocking=True)
        importance = self.scorer(flat).flatten()
        pooled: list[torch.Tensor] = []
        cursor = 0
        for length in lengths:
            states = flat[cursor : cursor + length]
            weights = torch.softmax(importance[cursor : cursor + length], dim=0)
            vector = torch.sum(states * weights.unsqueeze(1), dim=0)
            pooled.append(vector / (torch.linalg.vector_norm(vector) + EPSILON))
            cursor += length
        z = torch.stack(pooled)
        return z, self.probe(z).flatten(), lengths

class DynamicPrefixGate(nn.Module):
    """Exact scalar transition and linear Prefix probe from Two-stage."""

    def __init__(self, dimension: int):
        super().__init__()
        self.gate = nn.Linear(2 * dimension, 1)
        self.probe = nn.Linear(dimension, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward_sequence(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefixes = [z[0]]
        gates = [torch.full((), float("nan"), device=z.device)]
        norms = [torch.linalg.vector_norm(z[0])]
        previous = z[0]
        for current in z[1:]:
            gate = torch.sigmoid(self.gate(torch.cat([previous, current])).squeeze())
            mixed = (1.0 - gate) * previous + gate * current
            norm = torch.linalg.vector_norm(mixed)
            previous = mixed / (norm + EPSILON)
            prefixes.append(previous)
            gates.append(gate)
            norms.append(norm)
        states = torch.stack(prefixes)
        return self.probe(states).flatten(), torch.stack(gates), torch.stack(norms)

class StandardSelfAttentionStep(nn.Module):
    """Single-head scaled dot-product attention followed by scalar re-pooling."""

    def __init__(self, dimension: int, attention_dimension: int = 64, epsilon: float = 1e-12):
        super().__init__()
        self.dimension = dimension
        self.attention_dimension = attention_dimension
        self.epsilon = epsilon
        # No learned affine or projection biases: exactly Hbar WQ, Hbar WK, Hbar wV.
        self.layer_norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.q_proj = nn.Linear(dimension, attention_dimension, bias=False)
        self.k_proj = nn.Linear(dimension, attention_dimension, bias=False)
        self.v_proj = nn.Linear(dimension, 1, bias=False)
        self.probe = nn.Linear(dimension, 1)

    def forward_padded(
        self, states: torch.Tensor, valid_mask: torch.Tensor, return_debug: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if states.ndim != 3 or valid_mask.ndim != 2:
            raise ValueError("expected states [B,L,D] and valid_mask [B,L]")
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("every Step must contain at least one token")
        hbar = self.layer_norm(states)
        q = self.q_proj(hbar)
        k = self.k_proj(hbar)
        values = self.v_proj(hbar)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_dimension)
        logits = logits.masked_fill(~valid_mask[:, None, :], -torch.inf)
        attention = torch.softmax(logits, dim=-1)
        scalar_context = torch.matmul(attention, values).squeeze(-1)
        scalar_context = scalar_context.masked_fill(~valid_mask, -torch.inf)
        final_weights = torch.softmax(scalar_context, dim=-1)
        pooled = torch.sum(states * final_weights.unsqueeze(-1), dim=1)
        pooled = pooled / torch.linalg.vector_norm(pooled, dim=1, keepdim=True).clamp_min(self.epsilon)
        detector_logits = self.probe(pooled).flatten()
        debug: dict[str, torch.Tensor] = {}
        if return_debug:
            debug = {
                "attention": attention,
                "final_weights": final_weights,
                "scalar_context": scalar_context,
                "pooled": pooled,
            }
        return detector_logits, debug

class TokenRelationResidualStep(nn.Module):
    """Independent scorer plus zero-initialized Shaw Key-side correction."""

    def __init__(
        self,
        dimension: int,
        attention_dimension: int = 64,
        clipping_distance: int = 4,
        tau: float = 1.0,
        epsilon: float = 1e-12,
        value_side_relation: bool = False,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.attention_dimension = attention_dimension
        self.clipping_distance = clipping_distance
        self.tau = tau
        self.epsilon = epsilon
        self.value_side_relation = value_side_relation

        # This order exactly preserves seed-42 initialization of the archived
        # scorer and probe before any new module consumes RNG state.
        self.scorer = nn.Sequential(
            nn.Linear(dimension, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
        self.probe = nn.Linear(dimension, 1)

        self.token_layer_norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.q_proj = nn.Linear(dimension, attention_dimension, bias=False)
        self.k_proj = nn.Linear(dimension, attention_dimension, bias=False)
        self.v_proj = nn.Linear(dimension, attention_dimension, bias=False)
        self.rel_key = nn.Embedding(2 * clipping_distance + 1, attention_dimension)
        self.context_layer_norm = nn.LayerNorm(attention_dimension)
        self.rel_head = nn.Linear(attention_dimension, 1)
        nn.init.zeros_(self.rel_head.weight)
        nn.init.zeros_(self.rel_head.bias)
        # Instantiate the only new scientific component strictly after every
        # predecessor parameter so seed-42 initialization of the Key-only path
        # remains bit-exact.  E_V uses nn.Embedding's same scheme as E_K.
        self.rel_value = (
            nn.Embedding(2 * clipping_distance + 1, attention_dimension)
            if value_side_relation else None
        )

    def relation_indices(self, length: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(length, device=device)
        relative = positions[None, :] - positions[:, None]  # j - i
        return relative.clamp(-self.clipping_distance, self.clipping_distance) + self.clipping_distance

    def forward_padded(
        self,
        states: torch.Tensor,
        valid_mask: torch.Tensor,
        independent_logits: torch.Tensor,
        return_debug: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if states.ndim != 3 or valid_mask.ndim != 2 or independent_logits.ndim != 2:
            raise ValueError("expected states [B,L,D], mask [B,L], independent logits [B,L]")
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("every Step must contain at least one token")
        hbar = self.token_layer_norm(states)
        q = self.q_proj(hbar)
        k = self.k_proj(hbar)
        v = self.v_proj(hbar)
        length = states.shape[1]
        relation_indices = self.relation_indices(length, states.device)
        relative_keys = self.rel_key(relation_indices)
        content_logits = torch.matmul(q, k.transpose(-1, -2))
        relation_logits = torch.einsum("bid,ijd->bij", q, relative_keys)
        attention_logits = (content_logits + relation_logits) / math.sqrt(self.attention_dimension)
        attention_logits = attention_logits.masked_fill(~valid_mask[:, None, :], -torch.inf)
        attention = torch.softmax(attention_logits, dim=-1)
        content_context = torch.matmul(attention, v)
        if self.rel_value is not None:
            relative_values = self.rel_value(relation_indices)
            relative_value_context = torch.einsum("bij,ijd->bid", attention, relative_values)
            context = content_context + relative_value_context
        else:
            relative_values = torch.zeros((), dtype=states.dtype, device=states.device)
            relative_value_context = torch.zeros_like(content_context)
            context = content_context
        correction = self.rel_head(self.context_layer_norm(context)).squeeze(-1)
        new_logits = independent_logits + correction
        new_logits = new_logits.masked_fill(~valid_mask, -torch.inf)
        final_weights = torch.softmax(new_logits / self.tau, dim=-1)
        pooled = torch.sum(states * final_weights.unsqueeze(-1), dim=1)
        pooled = pooled / torch.linalg.vector_norm(pooled, dim=1, keepdim=True).clamp_min(self.epsilon)
        detector_logits = self.probe(pooled).flatten()
        debug: dict[str, torch.Tensor] = {}
        if return_debug:
            independent_masked = independent_logits.masked_fill(~valid_mask, -torch.inf)
            independent_weights = torch.softmax(independent_masked / self.tau, dim=-1)
            independent_pooled = torch.sum(states * independent_weights.unsqueeze(-1), dim=1)
            independent_pooled = independent_pooled / torch.linalg.vector_norm(
                independent_pooled, dim=1, keepdim=True
            ).clamp_min(self.epsilon)
            debug = {
                "attention": attention,
                "correction": correction,
                "new_logits": new_logits,
                "independent_logits": independent_masked,
                "final_weights": final_weights,
                "independent_weights": independent_weights,
                "pooled": pooled,
                "independent_pooled": independent_pooled,
                "independent_detector_logits": self.probe(independent_pooled).flatten(),
                "relation_indices": relation_indices,
                "relative_values": relative_values,
                "content_context": content_context,
                "relative_value_context": relative_value_context,
                "context": context,
            }
        return detector_logits, debug

    def forward_cache(
        self,
        cache: dict[str, Any],
        indices: torch.Tensor,
        device: torch.device,
        relation_microbatch_size: int,
        return_debug: bool = False,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]], list[int]]:
        offsets = cache["offsets"]
        cpu_pieces: list[torch.Tensor] = []
        lengths: list[int] = []
        for index in indices.tolist():
            start, end = int(offsets[index]), int(offsets[index + 1])
            cpu_pieces.append(cache["tokens"][start:end])
            lengths.append(end - start)
        flat = torch.cat(cpu_pieces, dim=0).to(device=device, dtype=torch.float32, non_blocking=True)
        # BatchNorm sees exactly the same logical batch and token order as the
        # archived independent scorer; relation microbatching is downstream.
        independent_flat = self.scorer(flat).flatten()
        pieces = list(flat.split(lengths))
        independent_pieces = list(independent_flat.split(lengths))
        outputs: list[torch.Tensor] = []
        debug_rows: list[dict[str, torch.Tensor]] = []
        for start in range(0, len(pieces), relation_microbatch_size):
            group_states = pieces[start : start + relation_microbatch_size]
            group_independent = independent_pieces[start : start + relation_microbatch_size]
            group_lengths = lengths[start : start + relation_microbatch_size]
            max_length = max(group_lengths)
            states = torch.zeros(
                (len(group_states), max_length, self.dimension), dtype=torch.float32, device=device
            )
            mask = torch.zeros((len(group_states), max_length), dtype=torch.bool, device=device)
            independent = torch.zeros((len(group_states), max_length), dtype=torch.float32, device=device)
            for row, (piece, scores, piece_length) in enumerate(
                zip(group_states, group_independent, group_lengths, strict=True)
            ):
                states[row, :piece_length] = piece
                independent[row, :piece_length] = scores
                mask[row, :piece_length] = True
            logits, debug = self.forward_padded(states, mask, independent, return_debug=return_debug)
            outputs.append(logits)
            if return_debug:
                debug["valid_mask"] = mask
                debug_rows.append(debug)
        return torch.cat(outputs), debug_rows, lengths

class ContentRelationResidualStep(nn.Module):
    """Independent token scorer plus a gamma-gated content-relation branch."""

    def __init__(
        self,
        dimension: int,
        attention_dimension: int = 64,
        clipping_distance: int = 4,
        position_dimension: int = 16,
        query_chunk_size: int = 64,
        tau: float = 1.0,
        epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.attention_dimension = attention_dimension
        self.clipping_distance = clipping_distance
        self.position_dimension = position_dimension
        self.query_chunk_size = query_chunk_size
        self.tau = tau
        self.epsilon = epsilon

        # Base path is instantiated first and exactly matches the archived
        # Independent seed-42 scorer/probe order.
        self.scorer = nn.Sequential(
            nn.Linear(dimension, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
        self.probe = nn.Linear(dimension, 1)

        # All content-relation parameters are instantiated only after the base.
        self.token_layer_norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.relation_proj = nn.Linear(dimension, attention_dimension, bias=False)
        self.q_proj = nn.Linear(dimension, attention_dimension, bias=False)
        self.k_proj = nn.Linear(dimension, attention_dimension, bias=False)
        self.v_proj = nn.Linear(dimension, attention_dimension, bias=False)
        self.position_embedding = nn.Embedding(
            2 * clipping_distance + 1, position_dimension
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(position_dimension + 2, attention_dimension),
            nn.GELU(),
            nn.Linear(attention_dimension, attention_dimension),
        )
        self.edge_key_proj = nn.Linear(
            attention_dimension, attention_dimension, bias=False
        )
        self.edge_value_proj = nn.Linear(
            attention_dimension, attention_dimension, bias=False
        )
        self.context_layer_norm = nn.LayerNorm(attention_dimension)
        self.rel_head = nn.Linear(attention_dimension, 1)
        self.gamma = nn.Parameter(torch.zeros(()))

    def relation_indices(
        self, query_start: int, query_end: int, length: int, device: torch.device
    ) -> torch.Tensor:
        query = torch.arange(query_start, query_end, device=device)[:, None]
        key = torch.arange(length, device=device)[None, :]
        return (key - query).clamp(
            -self.clipping_distance, self.clipping_distance
        ) + self.clipping_distance

    def _content_context(
        self,
        hbar: torch.Tensor,
        valid_mask: torch.Tensor,
        query_chunk_size: int,
        pair_debug: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.relation_proj(hbar)
        q = self.q_proj(hbar)
        k = self.k_proj(hbar)
        v = self.v_proj(hbar)
        batch, length, _ = hbar.shape
        contexts: list[torch.Tensor] = []
        attentions: list[torch.Tensor] = []
        debug_edges: dict[str, torch.Tensor] = {}

        # Unit vectors are used only inside cosine. Euclidean distance retains
        # the raw projected x, as frozen by the contract.
        x_norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(
            self.epsilon
        )
        x_cosine = x / x_norm

        for query_start in range(0, length, query_chunk_size):
            query_end = min(query_start + query_chunk_size, length)
            relation_indices = self.relation_indices(
                query_start, query_end, length, hbar.device
            )
            position = self.position_embedding(relation_indices)
            cosine = torch.matmul(
                x_cosine[:, query_start:query_end], x_cosine.transpose(-1, -2)
            )
            difference = (
                x[:, query_start:query_end, None, :] - x[:, None, :, :]
            )
            distance = torch.log1p(torch.linalg.vector_norm(difference, dim=-1))
            position_batched = position.unsqueeze(0).expand(batch, -1, -1, -1)
            relation_features = torch.cat(
                (position_batched, cosine.unsqueeze(-1), distance.unsqueeze(-1)),
                dim=-1,
            )
            edge = self.edge_mlp(relation_features)
            edge_key = self.edge_key_proj(edge)
            edge_value = self.edge_value_proj(edge)
            content_logits = torch.einsum(
                "bqd,bkd->bqk", q[:, query_start:query_end], k
            )
            edge_logits = torch.einsum(
                "bqd,bqkd->bqk", q[:, query_start:query_end], edge_key
            )
            logits = (content_logits + edge_logits) / math.sqrt(
                self.attention_dimension
            )
            logits = logits.masked_fill(~valid_mask[:, None, :], -torch.inf)
            attention = torch.softmax(logits, dim=-1)
            content_message = torch.matmul(attention, v)
            edge_message = torch.einsum("bqk,bqkd->bqd", attention, edge_value)
            contexts.append(content_message + edge_message)
            if pair_debug:
                attentions.append(attention)
                if not debug_edges:
                    debug_edges = {
                        "relation_indices": relation_indices,
                        "position": position,
                        "cosine": cosine,
                        "distance": distance,
                        "relation_features": relation_features,
                        "edge": edge,
                        "edge_key": edge_key,
                        "edge_value": edge_value,
                        "content_message": content_message,
                        "edge_message": edge_message,
                    }
        context = torch.cat(contexts, dim=1)
        debug: dict[str, torch.Tensor] = {}
        if pair_debug:
            debug = {**debug_edges, "attention": torch.cat(attentions, dim=1), "x": x}
        return context, debug

    def forward_padded(
        self,
        states: torch.Tensor,
        valid_mask: torch.Tensor,
        independent_logits: torch.Tensor,
        return_debug: bool = False,
        pair_debug: bool = False,
        query_chunk_size_override: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("every Step must contain at least one token")
        chunk_size = query_chunk_size_override or self.query_chunk_size
        hbar = self.token_layer_norm(states)
        context, pair = self._content_context(
            hbar, valid_mask, chunk_size, pair_debug
        )
        delta = self.rel_head(self.context_layer_norm(context)).squeeze(-1)
        delta = delta.masked_fill(~valid_mask, 0.0)
        correction = self.gamma * delta
        new_logits = independent_logits + correction
        new_logits = new_logits.masked_fill(~valid_mask, -torch.inf)
        final_weights = torch.softmax(new_logits / self.tau, dim=-1)
        pooled = torch.sum(states * final_weights.unsqueeze(-1), dim=1)
        pooled = pooled / torch.linalg.vector_norm(
            pooled, dim=1, keepdim=True
        ).clamp_min(self.epsilon)
        detector_logits = self.probe(pooled).flatten()
        debug: dict[str, torch.Tensor] = {}
        if return_debug:
            independent_masked = independent_logits.masked_fill(
                ~valid_mask, -torch.inf
            )
            independent_weights = torch.softmax(independent_masked / self.tau, dim=-1)
            independent_pooled = torch.sum(
                states * independent_weights.unsqueeze(-1), dim=1
            )
            independent_pooled = independent_pooled / torch.linalg.vector_norm(
                independent_pooled, dim=1, keepdim=True
            ).clamp_min(self.epsilon)
            debug = {
                "valid_mask": valid_mask,
                "delta": delta,
                "correction": correction,
                "new_logits": new_logits,
                "independent_logits": independent_masked,
                "final_weights": final_weights,
                "independent_weights": independent_weights,
                "pooled": pooled,
                "independent_pooled": independent_pooled,
                "independent_detector_logits": self.probe(independent_pooled).flatten(),
                "context": context,
                **pair,
            }
        return detector_logits, debug

    def forward_cache(
        self,
        cache: dict[str, Any],
        indices: torch.Tensor,
        device: torch.device,
        relation_microbatch_size: int,
        return_debug: bool = False,
        pair_debug: bool = False,
        query_chunk_size_override: int | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]], list[int]]:
        offsets = cache["offsets"]
        pieces_cpu: list[torch.Tensor] = []
        lengths: list[int] = []
        for index in indices.tolist():
            start, end = int(offsets[index]), int(offsets[index + 1])
            pieces_cpu.append(cache["tokens"][start:end])
            lengths.append(end - start)
        flat = torch.cat(pieces_cpu, dim=0).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        # Preserve the predecessor's exact logical BatchNorm population/order.
        independent_flat = self.scorer(flat).flatten()
        pieces = list(flat.split(lengths))
        independent_pieces = list(independent_flat.split(lengths))
        outputs: list[torch.Tensor] = []
        debug_rows: list[dict[str, torch.Tensor]] = []
        for start in range(0, len(pieces), relation_microbatch_size):
            group_states = pieces[start : start + relation_microbatch_size]
            group_independent = independent_pieces[
                start : start + relation_microbatch_size
            ]
            group_lengths = lengths[start : start + relation_microbatch_size]
            max_length = max(group_lengths)
            states = torch.zeros(
                (len(group_states), max_length, self.dimension),
                dtype=torch.float32,
                device=device,
            )
            mask = torch.zeros(
                (len(group_states), max_length), dtype=torch.bool, device=device
            )
            independent = torch.zeros(
                (len(group_states), max_length), dtype=torch.float32, device=device
            )
            for row, (piece, logits, length) in enumerate(
                zip(group_states, group_independent, group_lengths, strict=True)
            ):
                states[row, :length] = piece
                independent[row, :length] = logits
                mask[row, :length] = True
            logits, debug = self.forward_padded(
                states,
                mask,
                independent,
                return_debug=return_debug,
                pair_debug=pair_debug,
                query_chunk_size_override=query_chunk_size_override,
            )
            outputs.append(logits)
            if return_debug:
                debug_rows.append(debug)
        return torch.cat(outputs), debug_rows, lengths

class FullVectorPrefixGate(nn.Module):
    """Exact full gate W_r:[2d]->d followed by the unchanged linear probe."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.gate = nn.Linear(2 * dimension, dimension)
        self.probe = nn.Linear(dimension, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def states_sequence(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = [z[0]]
        gates = [torch.full((self.dimension,), float("nan"), device=z.device)]
        pre_norms = [torch.linalg.vector_norm(z[0])]
        previous = z[0]
        for current in z[1:]:
            gate = torch.sigmoid(self.gate(torch.cat([previous, current], dim=-1)))
            mixed = (1.0 - gate) * previous + gate * current
            norm = torch.linalg.vector_norm(mixed)
            previous = mixed / (norm + EPSILON)
            states.append(previous)
            gates.append(gate)
            pre_norms.append(norm)
        return torch.stack(states), torch.stack(gates), torch.stack(pre_norms)

    def forward_sequence(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states, vector_gates, norms = self.states_sequence(z)
        return self.probe(states).flatten(), vector_gates.mean(dim=-1), norms

    def forward_padded(
        self, sequences: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorize independent trajectories at each recurrent time index."""
        lengths = torch.tensor([len(value) for value in sequences], device=sequences[0].device)
        padded = pad_sequence(sequences, batch_first=True)
        previous = padded[:, 0]
        logits = [self.probe(previous).flatten()]
        gate_means = [torch.full((len(sequences),), float("nan"), device=previous.device)]
        norms = [torch.linalg.vector_norm(previous, dim=-1)]
        for position in range(1, padded.shape[1]):
            current = padded[:, position]
            vector_gate = torch.sigmoid(self.gate(torch.cat([previous, current], dim=-1)))
            mixed = (1.0 - vector_gate) * previous + vector_gate * current
            row_norm = torch.linalg.vector_norm(mixed, dim=-1)
            candidate = mixed / (row_norm.unsqueeze(-1) + EPSILON)
            active = lengths > position
            previous = torch.where(active.unsqueeze(-1), candidate, previous)
            logits.append(self.probe(previous).flatten())
            gate_means.append(torch.where(active, vector_gate.mean(dim=-1), torch.full_like(row_norm, float("nan"))))
            norms.append(torch.where(active, row_norm, torch.full_like(row_norm, float("nan"))))
        return torch.stack(logits, dim=1), torch.stack(gate_means, dim=1), torch.stack(norms, dim=1), lengths

class RelationConditionedPrefixGate(nn.Module):
    """Scalar predecessor gate plus the frozen shared relation descriptor."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

        # Base path must be first and identical to archived DynamicPrefixGate.
        self.gate = nn.Linear(2 * dimension, 1)
        self.probe = nn.Linear(dimension, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        # Relation parameters are instantiated only after all base parameters.
        self.relation_projection = nn.Linear(
            dimension, RELATION_WIDTH, bias=False
        )
        self.relation_mlp = nn.Sequential(
            nn.Linear(2 * RELATION_WIDTH, RELATION_WIDTH),
            nn.GELU(),
            nn.Linear(RELATION_WIDTH, RELATION_WIDTH),
        )
        self.relation_layer_norm = nn.LayerNorm(RELATION_WIDTH)
        self.relation_readout = nn.Linear(RELATION_WIDTH, 1)

    def relation(
        self, previous: torch.Tensor, current: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # The same parameter object is deliberately called twice so product and
        # absolute difference compare aligned coordinates.
        query = self.relation_projection(previous)
        key = self.relation_projection(current)
        product = query * key
        absolute_difference = torch.abs(query - key)
        match = torch.cat((product, absolute_difference), dim=-1)
        edge = self.relation_mlp(match)
        # No sqrt(width) divisor: this is a normalized learned linear readout.
        relation_delta = self.relation_readout(
            self.relation_layer_norm(edge)
        ).squeeze(-1)
        return relation_delta, {
            "query": query,
            "key": key,
            "product": product,
            "absolute_difference": absolute_difference,
            "match": match,
            "edge": edge,
        }

    def forward_sequence(
        self,
        z: torch.Tensor,
        relation_enabled: bool = True,
        return_debug: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prefixes = [z[0]]
        nan = torch.full((), float("nan"), device=z.device)
        gates = [nan]
        base_gates = [nan]
        base_logits = [nan]
        relation_deltas = [nan]
        pre_norms = [torch.linalg.vector_norm(z[0])]
        product_norms = [nan]
        difference_norms = [nan]
        previous = z[0]
        for current in z[1:]:
            base_logit = self.gate(torch.cat((previous, current))).squeeze()
            relation_delta, relation = self.relation(previous, current)
            effective_delta = (
                relation_delta
                if relation_enabled
                else torch.zeros_like(relation_delta)
            )
            gate = torch.sigmoid(base_logit + effective_delta)
            mixed = (1.0 - gate) * previous + gate * current
            norm = torch.linalg.vector_norm(mixed)
            previous = mixed / (norm + EPSILON)
            prefixes.append(previous)
            gates.append(gate)
            base_gates.append(torch.sigmoid(base_logit))
            base_logits.append(base_logit)
            relation_deltas.append(relation_delta)
            pre_norms.append(norm)
            product_norms.append(torch.linalg.vector_norm(relation["product"]))
            difference_norms.append(
                torch.linalg.vector_norm(relation["absolute_difference"])
            )
        states = torch.stack(prefixes)
        debug = {
            "states": states,
            "gates": torch.stack(gates),
            "base_gates": torch.stack(base_gates),
            "base_logits": torch.stack(base_logits),
            "relation_deltas": torch.stack(relation_deltas),
            "pre_normalization_norms": torch.stack(pre_norms),
            "product_norms": torch.stack(product_norms),
            "absolute_difference_norms": torch.stack(difference_norms),
        }
        logits = self.probe(states).flatten()
        return logits, debug if return_debug else {}

class DirectMLPPrefix(nn.Module):
    def __init__(self, dimension: int, epsilon: float):
        super().__init__()
        self.dimension = dimension
        self.epsilon = epsilon
        self.transition = nn.Sequential(
            nn.Linear(2 * dimension, HIDDEN_WIDTH),
            nn.GELU(),
            nn.Linear(HIDDEN_WIDTH, dimension),
        )
        self.probe = nn.Linear(dimension, 1)

    def states(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if z.ndim != 2 or z.shape[1] != self.dimension or len(z) == 0:
            raise AssertionError(f"invalid sequence shape {tuple(z.shape)}")
        prefixes = [z[0]]
        norms = [torch.linalg.vector_norm(z[0])]
        previous = z[0]
        for current in z[1:]:
            raw = self.transition(torch.cat([previous, current]))
            norm = torch.linalg.vector_norm(raw)
            if not bool(torch.isfinite(norm)) or float(norm.detach()) <= self.epsilon:
                raise FloatingPointError(f"invalid pre-normalization norm {float(norm.detach())}")
            previous = raw / (norm + self.epsilon)
            if not bool(torch.all(torch.isfinite(previous))):
                raise FloatingPointError("non-finite normalized Prefix state")
            prefixes.append(previous)
            norms.append(norm)
        return torch.stack(prefixes), torch.stack(norms)

    def forward_sequence(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        states, norms = self.states(z)
        return self.probe(states).flatten(), norms
