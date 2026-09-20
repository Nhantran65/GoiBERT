"""E17 recipient-excluded key refinement on the 384-wide student.

Only block index 6 is replaced.  The target block uses the original Q/K/V,
output projection, dropout and LayerNorm parameters.  Baseline runs the same
explicit FP32 score/softmax kernel without the preview; full_context and
cavity compute the factorized preview and blend its logits with a
zero-initialized scalar eta.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.models.roberta.modeling_roberta import RobertaAttention, RobertaLayer

from goibert_e2.modeling import build_e2_student


INSERTION_LAYER_INDEX = 6
HIDDEN_SIZE = 384
NUM_HEADS = 6
HEAD_DIM = 64
PAIR_CHUNK_SIZE = 64
LN_EPS = 1.0e-5
GUARD_REL = 1.0e-3
GUARD_ABS = 1.0e-7
ARMS = ("baseline", "full_context", "cavity")


@dataclass(frozen=True)
class E17ModelSpec:
    role: str
    backbone: str
    hidden_size: int
    head_hidden_size: int
    head_dropout: float
    num_labels: int
    num_aspects: int
    aspect_class_counts: tuple[int, ...]
    total_parameters: int
    trainable_parameters: int
    architecture: str
    arm: str
    insertion_layer_index: int
    pair_chunk_size: int
    layer_norm_eps: float
    guard_relative: float
    guard_absolute: float
    eta_init: float | None
    eta_weight_decay: float
    added_parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class E17ExplicitSelfAttention(nn.Module):
    """Original projections plus FP32 preview routing for block 6.

    Q/K/V modules are moved from the original self-attention object and thus
    retain the original parameter names.  The output projection and target
    LayerNorm remain owned by the wrapper's original ``RobertaAttention``.
    """

    def __init__(self, source: nn.Module, *, arm: str, chunk_size: int = PAIR_CHUNK_SIZE) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"Unknown E17 arm {arm!r}")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.query = source.query
        self.key = source.key
        self.value = source.value
        self.dropout = source.dropout
        self.num_attention_heads = int(source.num_attention_heads)
        self.attention_head_size = int(source.attention_head_size)
        self.all_head_size = int(source.all_head_size)
        self.dropout_prob = float(getattr(source, "dropout_prob", source.dropout.p))
        self.require_contiguous_qkv = bool(getattr(source, "require_contiguous_qkv", False))
        self.arm = str(arm)
        self.chunk_size = int(chunk_size)
        if self.arm == "baseline":
            self.eta = None
        else:
            self.eta = nn.Parameter(torch.zeros(self.num_attention_heads, dtype=torch.float32))
        self._query_mask: Tensor | None = None
        self.collect_diagnostics = False
        self.last_diagnostics: dict[str, Any] = {}
        self.last_preview: dict[str, Tensor] = {}

    @property
    def added_parameter_count(self) -> int:
        return 0 if self.eta is None else 1

    def set_query_mask(self, attention_mask: Tensor | None) -> None:
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("E17 query mask must be [batch, sequence]")
        self._query_mask = attention_mask

    def set_diagnostics(self, enabled: bool) -> None:
        self.collect_diagnostics = bool(enabled)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*shape).permute(0, 2, 1, 3)

    def _valid_pair_mask(self, batch: int, seq: int, device: torch.device) -> Tensor:
        if self._query_mask is None:
            return torch.ones(batch, seq, seq, dtype=torch.bool, device=device)
        mask = self._query_mask.to(device=device, dtype=torch.bool)
        return mask[:, :, None] & mask[:, None, :]

    def _qkv_and_attention(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            content_value = self.transpose_for_scores(
                F.linear(hidden_states.float(), self.value.weight.float(), None)
            )
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            scores = torch.matmul(query_layer.float(), key_layer.float().transpose(-1, -2))
            scores = scores / math.sqrt(float(self.attention_head_size))
            if not bool(torch.isfinite(scores).all().item()):
                raise FloatingPointError("E17 unmasked target attention scores are non-finite")
            if attention_mask is not None:
                scores = scores + attention_mask.float()
            elif self._query_mask is not None:
                key_mask = self._query_mask.to(device=scores.device, dtype=torch.bool)
                scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
            probabilities = torch.softmax(scores, dim=-1)
            if not bool(torch.isfinite(probabilities).all().item()):
                raise FloatingPointError("E17 target attention probabilities are non-finite")
        return query_layer, key_layer, value_layer, content_value, probabilities

    @staticmethod
    def _head_weight(weight: Tensor, heads: int, head_dim: int) -> Tensor:
        return weight.reshape(heads, head_dim, weight.shape[1])

    def _bias_relocated_preview_inputs(
        self,
        hidden_states: Tensor,
        content_value: Tensor,
        probabilities: Tensor,
        output_dense: nn.Linear,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # u uses Wv h without value bias; the common value bias is moved into
        # b_eff because every non-empty attention row sums to one.
        # output.dense maps concatenated heads: each head contributes d x p.
        wo = output_dense.weight.float().reshape(self.all_head_size, self.num_attention_heads, self.attention_head_size).permute(1, 0, 2)
        value_bias = self.value.bias.float().reshape(self.num_attention_heads, self.attention_head_size)
        output_bias = output_dense.bias.float() if output_dense.bias is not None else torch.zeros(self.all_head_size, device=hidden_states.device)
        b_eff = output_bias + torch.einsum("hdp,hp->d", wo, value_bias)
        content_value_bnhd = content_value.permute(0, 2, 1, 3)
        u = torch.einsum("bnhp,hdp->bnhd", content_value_bnhd.float(), wo)
        messages = torch.einsum("bhjk,bkhd->bjd", probabilities, u)
        z = hidden_states.float() + b_eff.view(1, 1, -1) + messages
        if not bool(torch.isfinite(z).all().item()):
            raise FloatingPointError("E17 preview z is non-finite")
        return z, b_eff, u

    def _literal_fallback(
        self,
        params: dict[str, Tensor],
        pair_indices: Tensor,
        *,
        output_dense: nn.Linear,
        target_ln: nn.LayerNorm,
    ) -> tuple[Tensor, Tensor]:
        """Literal scores for flagged pairs, processed in chunks <=64 pairs."""
        hidden_states = params["hidden"]
        z = params["z"]
        content_value = params["content_value"]
        probabilities = params["probabilities"]
        q = params["query"]
        batch_size, _, seq_len, _ = q.shape
        a = params["a"]
        u = params["u"]
        scores_out: list[Tensor] = []
        variance_out: list[Tensor] = []
        for start in range(0, pair_indices.shape[0], self.chunk_size):
            indices = pair_indices[start : start + self.chunk_size]
            b_idx, i_idx, j_idx = indices[:, 0], indices[:, 1], indices[:, 2]
            pair_a = a[b_idx, i_idx, j_idx]  # [M,H]
            pair_u = u[b_idx, i_idx]  # [M,H,d]
            removed = torch.einsum("mh,mhd->md", pair_a, pair_u)
            cavity = z[b_idx, j_idx] - removed
            centered = cavity - cavity.mean(dim=-1, keepdim=True)
            variance = centered.square().mean(dim=-1)
            ln_cavity = F.layer_norm(
                cavity,
                (self.all_head_size,),
                weight=target_ln.weight.float(),
                bias=target_ln.bias.float(),
                eps=LN_EPS,
            )
            key_weight = self.key.weight.float().reshape(self.num_attention_heads, self.attention_head_size, self.all_head_size)
            key_bias = self.key.bias.float().reshape(self.num_attention_heads, self.attention_head_size)
            keys = torch.einsum("hpd,md->mhp", key_weight, ln_cavity) + key_bias.unsqueeze(0)
            pair_query = q[b_idx, :, i_idx, :].float()
            scores = (pair_query * keys).sum(dim=-1) / math.sqrt(float(self.attention_head_size))
            scores_out.append(scores)
            variance_out.append(variance)
        if not scores_out:
            return z.new_empty((0, self.num_attention_heads)), z.new_empty((0,))
        return torch.cat(scores_out, dim=0), torch.cat(variance_out, dim=0)

    def _factorized_variant_scores(
        self,
        hidden_states: Tensor,
        query: Tensor,
        content_value: Tensor,
        probabilities: Tensor,
        output_dense: nn.Linear,
        target_ln: nn.LayerNorm,
    ) -> tuple[Tensor, dict[str, Any]]:
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            return self._factorized_variant_scores_fp32(
                hidden_states, query, content_value, probabilities, output_dense, target_ln
            )

    def _factorized_variant_scores_fp32(
        self,
        hidden_states: Tensor,
        query: Tensor,
        content_value: Tensor,
        probabilities: Tensor,
        output_dense: nn.Linear,
        target_ln: nn.LayerNorm,
    ) -> tuple[Tensor, dict[str, Any]]:
        batch_size, _, seq_len, _ = query.shape
        z, b_eff, u = self._bias_relocated_preview_inputs(hidden_states, content_value, probabilities, output_dense)
        a = probabilities.permute(0, 3, 2, 1)
        if self.arm == "cavity":
            diagonal = torch.eye(seq_len, dtype=torch.bool, device=hidden_states.device).view(1, seq_len, seq_len, 1)
            a = a.masked_fill(diagonal, 0.0)
        else:
            a = torch.zeros_like(a)
        w = z - z.mean(dim=-1, keepdim=True)
        e = u - u.mean(dim=-1, keepdim=True)
        ww = w.square().sum(dim=-1)
        we = torch.einsum("bjd,bihd->bijh", w, e)
        ee = torch.einsum("bihd,bigd->bihg", e, e)
        variance_numerator = (
            ww[:, None, :]
            - 2.0 * (a * we).sum(dim=-1)
            + torch.einsum("bijh,bijg,bihg->bij", a, a, ee)
        )
        vnorm = variance_numerator / float(self.all_head_size)
        sigma = torch.sqrt(torch.clamp_min(vnorm, 0.0) + LN_EPS)

        key_weight = self.key.weight.float().reshape(self.num_attention_heads, self.attention_head_size, self.all_head_size)
        key_bias = self.key.bias.float().reshape(self.num_attention_heads, self.attention_head_size)
        gamma = target_ln.weight.float()
        beta = target_ln.bias.float()
        r = torch.einsum("hpd,bhnp,d->bhnd", key_weight, query.float(), gamma)
        rw = torch.einsum("bhnd,bjd->bhnj", r, w)
        re = torch.einsum("bhid,bigd->bhig", r, e)
        # re is [B,target_head,query_i,source_head] after the einsum above.
        nterm = rw - torch.einsum("bijg,bhig->bhij", a, re)
        key_beta = torch.einsum("hpd,d->hp", key_weight, beta) + key_bias
        cterm = torch.einsum("bhnp,hp->bhn", query.float(), key_beta)
        variant_scores = (nterm / sigma[:, None, :, :] + cterm[:, :, :, None]) / math.sqrt(float(self.attention_head_size))
        if not bool(torch.isfinite(variant_scores).all().item()):
            raise FloatingPointError("E17 factorized variant logits are non-finite")

        ww_for_guard = ww[:, None, :]
        gram_abs_term = torch.einsum("bijh,bijg,bihg->bij", a.abs(), a.abs(), ee.abs())
        e_scale = (ww_for_guard + 2.0 * (a * we).abs().sum(dim=-1) + gram_abs_term) / float(self.all_head_size)
        pair_valid = self._valid_pair_mask(batch_size, seq_len, hidden_states.device)
        nonfinite_pair = (~torch.isfinite(vnorm)) | (~torch.isfinite(variant_scores).all(dim=1))
        flagged = pair_valid & ((vnorm <= GUARD_REL * e_scale + GUARD_ABS) | nonfinite_pair)
        if not bool(torch.isfinite(e_scale).all().item() and torch.isfinite(vnorm).all().item()):
            raise FloatingPointError("E17 guard intermediates are non-finite")

        total_pairs = batch_size * seq_len * seq_len
        flat_scores = variant_scores.permute(0, 2, 3, 1).reshape(total_pairs, self.num_attention_heads)
        flat_flagged = flagged.reshape(-1)
        flagged_indices = torch.nonzero(flat_flagged, as_tuple=False).flatten()
        fallback_variance = variant_scores.new_empty((0,))
        if flagged_indices.numel():
            flat = flagged_indices
            per_batch = seq_len * seq_len
            b_idx = torch.div(flat, per_batch, rounding_mode="floor")
            rem = flat - b_idx * per_batch
            i_idx = torch.div(rem, seq_len, rounding_mode="floor")
            j_idx = rem - i_idx * seq_len
            pair_indices = torch.stack((b_idx, i_idx, j_idx), dim=1)
            fallback_scores, fallback_variance = self._literal_fallback(
                {
                    "hidden": hidden_states,
                    "z": z,
                    "content_value": content_value,
                    "probabilities": probabilities,
                    "query": query,
                    "a": a,
                    "u": u,
                },
                pair_indices,
                output_dense=output_dense,
                target_ln=target_ln,
            )
            flat_scores = flat_scores.index_copy(0, flagged_indices, fallback_scores)
        selected_scores = flat_scores.reshape(batch_size, seq_len, seq_len, self.num_attention_heads).permute(0, 3, 1, 2)
        if not bool(torch.isfinite(selected_scores).all().item()):
            raise FloatingPointError("E17 selected preview logits are non-finite")
        stats = {
            "fallback_chunks": int(math.ceil(int(flagged_indices.numel()) / self.chunk_size)) if flagged_indices.numel() else 0,
            "preview_dtype": str(z.dtype),
            "scores_dtype": str(selected_scores.dtype),
        }
        if self.collect_diagnostics:
            stats.update(
                {
                    "flagged_count": int(flagged.sum().item()),
                    "flagged_valid_count": int(flagged.sum().item()),
                    "total_valid_pairs": int(pair_valid.sum().item()),
                    "vnorm_min": float(vnorm.detach().min().item()),
                    "vnorm_max": float(vnorm.detach().max().item()),
                    "e_scale_min": float(e_scale.detach().min().item()),
                    "e_scale_max": float(e_scale.detach().max().item()),
                    "eta": self.eta.detach().float().cpu().tolist() if self.eta is not None else [0.0] * self.num_attention_heads,
                    "a_diagonal_max": float(torch.diagonal(probabilities.mean(dim=1), dim1=1, dim2=2).abs().max().detach().item()),
                    "bias_relocation_norm": float(b_eff.detach().norm().item()),
                }
            )
        return selected_scores, {"stats": stats, "fast_scores": variant_scores, "selected_scores": selected_scores, "flagged": flagged, "vnorm": vnorm, "E": e_scale, "z": z, "u": u, "a": a, "probabilities": probabilities, "query": query, "content_value": content_value, "hidden": hidden_states}

    def route(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        *,
        output_dense: nn.Linear,
        target_ln: nn.LayerNorm,
    ) -> tuple[Tensor, dict[str, Any]]:
        query, key, value, content_value, probabilities = self._qkv_and_attention(hidden_states, attention_mask)
        if self.arm == "baseline":
            context_probabilities = self.dropout(probabilities)
            context = torch.matmul(context_probabilities.to(dtype=value.dtype), value)
            stats = {
                "flagged_count": 0,
                "flagged_valid_count": 0,
                "fallback_chunks": 0,
                "eta": [0.0] * self.num_attention_heads,
                "preview_used": False,
                "probabilities_dtype": str(probabilities.dtype),
                "context_dtype": str(context.dtype),
            }
            if self.collect_diagnostics:
                stats["total_valid_pairs"] = int(self._valid_pair_mask(hidden_states.shape[0], hidden_states.shape[1], hidden_states.device).sum().item())
            self.last_diagnostics = stats
            self.last_preview = {}
            return context.transpose(1, 2).reshape(hidden_states.shape[0], hidden_states.shape[1], self.all_head_size), {"stats": stats}
        variant_logits, preview = self._factorized_variant_scores(
            hidden_states,
            query,
            content_value,
            probabilities,
            output_dense,
            target_ln,
        )
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            original_logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(float(self.attention_head_size))
            if not bool(torch.isfinite(original_logits).all().item()):
                raise FloatingPointError("E17 original unmasked logits are non-finite")
            pair_valid = self._valid_pair_mask(hidden_states.shape[0], hidden_states.shape[1], hidden_states.device)
            blend_mask = pair_valid[:, None, :, :]
            blended = original_logits + self.eta.float().view(1, self.num_attention_heads, 1, 1) * (variant_logits - original_logits)
            raw_logits = torch.where(blend_mask, blended, original_logits)
            if attention_mask is not None:
                logits = raw_logits + attention_mask.float()
            elif self._query_mask is not None:
                logits = raw_logits.masked_fill(
                    ~self._query_mask.to(device=raw_logits.device, dtype=torch.bool)[:, None, None, :],
                    float("-inf"),
                )
            else:
                logits = raw_logits
            final_probabilities = torch.softmax(logits, dim=-1)
            if not bool(torch.isfinite(final_probabilities).all().item()):
                raise FloatingPointError("E17 blended probabilities are non-finite")
            if self.collect_diagnostics:
                metric_mask = pair_valid.clone()
                metric_mask.diagonal(dim1=1, dim2=2).fill_(False)
                metric_mask = metric_mask[:, None, :, :]
                probability_delta_norm = ((final_probabilities - probabilities).float() * metric_mask).norm()
            else:
                probability_delta_norm = None
            final_probabilities = self.dropout(final_probabilities)
        context = torch.matmul(final_probabilities.to(dtype=value.dtype), value)
        stats = dict(preview["stats"])
        stats.update(
            {
                "preview_used": True,
                "probabilities_dtype": str(probabilities.dtype),
                "context_dtype": str(context.dtype),
            }
        )
        if self.collect_diagnostics:
            metric_mask = pair_valid.clone()
            metric_mask.diagonal(dim1=1, dim2=2).fill_(False)
            metric_mask = metric_mask[:, None, :, :]
            stats.update(
                {
                    "orig_logits_norm": float((original_logits.detach() * metric_mask).norm().item()),
                    "variant_delta_norm": float(((variant_logits - original_logits).detach() * metric_mask).norm().item()),
                    "eta_delta_norm": float((self.eta.float().view(1, self.num_attention_heads, 1, 1) * (variant_logits - original_logits).detach() * metric_mask).norm().item()),
                    "attention_probability_delta_norm": float(probability_delta_norm.detach().item()),
                }
            )
        self.last_diagnostics = stats
        self.last_preview = preview if self.collect_diagnostics else {}
        return context.transpose(1, 2).reshape(hidden_states.shape[0], hidden_states.shape[1], self.all_head_size), {"stats": stats, "preview": preview, "orig_logits": original_logits, "variant_logits": variant_logits, "logits": logits}

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[Tensor, ...]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Tensor, ...]:
        if any(value is not None for value in (head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value)) or output_attentions:
            raise ValueError("E17 target attention supports encoder self-attention without output_attentions only")
        raise RuntimeError("E17ExplicitSelfAttention.forward must be invoked by E17RobertaAttention")


class E17RobertaAttention(nn.Module):
    def __init__(self, source: RobertaAttention, *, arm: str, chunk_size: int = PAIR_CHUNK_SIZE) -> None:
        super().__init__()
        self.self = E17ExplicitSelfAttention(source.self, arm=arm, chunk_size=chunk_size)
        self.output = source.output
        self.pruned_heads = source.pruned_heads

    def prune_heads(self, heads: Any) -> None:
        raise NotImplementedError("E17 does not support pruning after construction")

    def set_query_mask(self, attention_mask: Tensor | None) -> None:
        self.self.set_query_mask(attention_mask)

    def set_diagnostics(self, enabled: bool) -> None:
        self.self.set_diagnostics(enabled)

    @property
    def diagnostics(self) -> dict[str, Any]:
        return dict(self.self.last_diagnostics)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[Tensor, ...]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Tensor, ...]:
        context, _ = self.self.route(
            hidden_states,
            attention_mask,
            output_dense=self.output.dense,
            target_ln=self.output.LayerNorm,
        )
        projected = self.output.dense(context)
        projected = self.output.dropout(projected)
        output = self.output.LayerNorm(projected + hidden_states)
        return (output,)


class E17Classifier(nn.Module):
    def __init__(self, base: nn.Module, *, arm: str, chunk_size: int = PAIR_CHUNK_SIZE) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"Unknown E17 arm {arm!r}")
        self.encoder = base.encoder
        self.head = base.head
        self.num_labels = int(base.num_labels)
        self.num_aspects = int(base.num_aspects)
        self.aspect_class_counts = tuple(base.aspect_class_counts)
        self._backbone_name = str(base._backbone_name)
        self._head_dropout = float(base._head_dropout)
        self._head_hidden_size = int(base._head_hidden_size)
        self.arm = str(arm)
        self.insertion_layer_index = INSERTION_LAYER_INDEX
        self.chunk_size = int(chunk_size)
        attention = self.encoder.encoder.layer[INSERTION_LAYER_INDEX].attention
        if not isinstance(attention, E17RobertaAttention):
            raise TypeError("E17 target attention was not installed")
        attention.self.chunk_size = int(chunk_size)

    @property
    def target_attention(self) -> E17RobertaAttention:
        value = self.encoder.encoder.layer[self.insertion_layer_index].attention
        if not isinstance(value, E17RobertaAttention):
            raise TypeError("E17 target attention was replaced")
        return value

    @property
    def target_self(self) -> E17ExplicitSelfAttention:
        return self.target_attention.self

    @property
    def eta(self) -> nn.Parameter | None:
        return self.target_self.eta

    def set_diagnostics(self, enabled: bool) -> None:
        self.target_attention.set_diagnostics(enabled)

    def diagnostics(self) -> dict[str, Any]:
        return self.target_attention.diagnostics

    def forward(self, input_ids: Tensor, attention_mask: Tensor, *, token_type_ids: Tensor | None = None) -> Tensor:
        self.target_attention.set_query_mask(attention_mask)
        kwargs: dict[str, Tensor] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        try:
            outputs = self.encoder(**kwargs, return_dict=True)
        finally:
            self.target_attention.set_query_mask(None)
        logits = self.head(outputs.last_hidden_state)
        if self.num_aspects and len(set(self.aspect_class_counts)) == 1:
            logits = logits.reshape(logits.shape[0], self.num_aspects, self.aspect_class_counts[0])
        return logits

    def spec(self) -> E17ModelSpec:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        layer_norm_eps = float(self.target_attention.output.LayerNorm.eps)
        return E17ModelSpec(
            role="student",
            backbone=self._backbone_name,
            hidden_size=int(self.encoder.config.hidden_size),
            head_hidden_size=self._head_hidden_size,
            head_dropout=self._head_dropout,
            num_labels=self.num_labels,
            num_aspects=self.num_aspects,
            aspect_class_counts=self.aspect_class_counts,
            total_parameters=total,
            trainable_parameters=trainable,
            architecture="original_explicit_target_block6_recipient_excluded_key",
            arm=self.arm,
            insertion_layer_index=INSERTION_LAYER_INDEX,
            pair_chunk_size=self.chunk_size,
            layer_norm_eps=layer_norm_eps,
            guard_relative=GUARD_REL,
            guard_absolute=GUARD_ABS,
            eta_init=0.0 if self.eta is not None else None,
            eta_weight_decay=0.0,
            added_parameter_count=self.target_self.num_attention_heads if self.eta is not None else 0,
        )


def build_e17_student(
    backbone_dir: Path,
    *,
    num_classes: int,
    num_aspects: int = 0,
    aspect_class_counts: Sequence[int] | None = None,
    arm: str,
    attention_implementation: str = "sdpa",
    pair_chunk_size: int = PAIR_CHUNK_SIZE,
) -> tuple[E17Classifier, E17ModelSpec]:
    if attention_implementation != "sdpa":
        raise ValueError("E17 target kernel is locked to explicit FP32 routing with SDPA-compatible model construction")
    if arm not in ARMS:
        raise ValueError(f"Unknown E17 arm {arm!r}")
    base, _ = build_e2_student(
        backbone_dir,
        num_classes=num_classes,
        num_aspects=num_aspects,
        aspect_class_counts=aspect_class_counts if num_aspects else None,
        attention_implementation=attention_implementation,
    )
    layer = base.encoder.encoder.layer[INSERTION_LAYER_INDEX]
    source_attention = layer.attention
    original_layer_norm = source_attention.output.LayerNorm
    with torch.random.fork_rng(devices=[]):
        layer.attention = E17RobertaAttention(source_attention, arm=arm, chunk_size=pair_chunk_size)
    model = E17Classifier(base, arm=arm, chunk_size=pair_chunk_size)
    if model.target_attention.output.LayerNorm is not original_layer_norm:
        raise AssertionError("E17 target LayerNorm was not reused")
    if abs(float(model.target_attention.output.LayerNorm.eps) - LN_EPS) > 1.0e-12:
        raise AssertionError(f"E17 target LayerNorm eps must be {LN_EPS}")
    return model, model.spec()
