"""E10 span and multi-kernel residual adapters.

The module is intentionally isolated from the frozen E1/E2/E7 sources.  It
keeps the original encoder and CLS head objects, replacing only encoder block
2 (index 1) with a wrapper that applies a zero-output residual branch after
the original transformer layer.  Every forward receives its raw-text offset
alignment explicitly; no input-ID lookup or token decoding is used.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from goibert.modeling import HEAD_DROPOUT, HEAD_HIDDEN_SIZE, GoiBERTClassifier
from goibert_e2.modeling import build_e2_student


SPAN_WIDTHS = (1, 2, 3, 4)
SPAN_RANK = 128
CONV_RANK = 140
INSERTION_LAYER_INDEX = 1


@dataclass(frozen=True)
class RawUnit:
    start: int
    end: int
    segment: int


@dataclass(frozen=True)
class TokenAlignment:
    """Alignment for one tokenizer output, derived from exactly one raw text.

    ``token_to_units[j]`` contains ``(unit_index, overlap_fraction)`` pairs.
    Fractions are normalized across units touched by token ``j``.  Empty
    entries represent special tokens, padding, punctuation-only tokens, or
    truncated material and cannot enter a span.
    """

    raw_text: str
    units: tuple[RawUnit, ...]
    token_to_units: tuple[tuple[tuple[int, float], ...], ...]
    _layout_cache: dict[tuple[str, str, str], dict[str, Tensor]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def padded(self, length: int) -> "TokenAlignment":
        if length < len(self.token_to_units):
            raise ValueError("cannot truncate an alignment after tokenizer truncation")
        if length == len(self.token_to_units):
            return self
        padded = TokenAlignment(
            self.raw_text,
            self.units,
            self.token_to_units + ((),) * (length - len(self.token_to_units)),
        )
        # Batch padding changes only the number of trailing empty token rows.
        # Share the immutable raw overlap/layout cache across these views;
        # _layout includes sequence length in its cache key below.
        object.__setattr__(padded, "_layout_cache", self._layout_cache)
        return padded


@dataclass(frozen=True)
class E10ModelSpec:
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
    variant: str
    insertion_layer_index: int
    branch_rank: int
    span_widths: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_separator(character: str) -> bool:
    # Unicode symbols such as emoji remain units.  Whitespace and Unicode
    # punctuation are separators and prevent spans from crossing them.
    return character.isspace() or unicodedata.category(character).startswith("P")


def raw_units(text: str) -> tuple[RawUnit, ...]:
    """Split raw text into non-whitespace, non-punctuation units."""

    units: list[RawUnit] = []
    start: int | None = None
    segment = 0
    for index, character in enumerate(text):
        if character.isspace():
            if start is not None:
                units.append(RawUnit(start, index, segment))
                start = None
            # Whitespace separates units but does not block a local span.
            continue
        if unicodedata.category(character).startswith("P"):
            if start is not None:
                units.append(RawUnit(start, index, segment))
                start = None
            # Punctuation is the structural separator that spans cannot cross.
            segment += 1
            continue
        if start is None:
            start = index
    if start is not None:
        units.append(RawUnit(start, len(text), segment))
    return tuple(units)


def build_token_alignment(
    raw_text: str,
    offsets: Sequence[Sequence[int] | tuple[int, int]],
) -> TokenAlignment:
    """Build normalized tokenizer-token to raw-unit overlap alignment."""

    units = raw_units(raw_text)
    rows: list[tuple[tuple[int, float], ...]] = []
    for offset in offsets:
        if len(offset) < 2:
            raise ValueError(f"offset must have two entries, got {offset!r}")
        token_start, token_end = int(offset[0]), int(offset[1])
        overlaps: list[tuple[int, int]] = []
        if token_end > token_start:
            for unit_index, unit in enumerate(units):
                overlap = min(token_end, unit.end) - max(token_start, unit.start)
                if overlap > 0:
                    overlaps.append((unit_index, overlap))
        total = sum(overlap for _, overlap in overlaps)
        if total <= 0:
            rows.append(())
        else:
            rows.append(tuple((index, overlap / total) for index, overlap in overlaps))
    return TokenAlignment(raw_text, units, tuple(rows))


def _tokenize_with_alignment(tokenizer: Any, texts: Sequence[str], max_length: int) -> tuple[dict[str, Any], list[TokenAlignment]]:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    input_ids = [list(ids) for ids in encoded["input_ids"]]
    alignments = [
        build_token_alignment(text, offsets)
        for text, offsets in zip(texts, encoded["offset_mapping"])
    ]
    padded = tokenizer.pad(
        [{"input_ids": ids} for ids in input_ids],
        padding=True,
        return_tensors="pt",
    )
    target_length = int(padded["input_ids"].shape[1])
    alignments = [alignment.padded(target_length) for alignment in alignments]
    return {key: value for key, value in padded.items()}, alignments


class _RawTextDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        texts: Sequence[str],
        input_ids: Sequence[Sequence[int]],
        alignments: Sequence[TokenAlignment],
        labels: Sequence[Any],
        ids: Sequence[Any],
        gold_labels: Sequence[Any],
    ):
        if not (len(texts) == len(input_ids) == len(alignments) == len(labels) == len(ids) == len(gold_labels)):
            raise ValueError("texts, labels, ids and gold_labels must have equal lengths")
        self.texts = list(texts)
        self.input_ids = [list(values) for values in input_ids]
        self.alignments = list(alignments)
        self.labels = list(labels)
        self.ids = list(ids)
        self.gold_labels = list(gold_labels)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "text": self.texts[index],
            "input_ids": self.input_ids[index],
            "alignment": self.alignments[index],
            "labels": self.labels[index],
            "gold_labels": self.gold_labels[index],
            "id": self.ids[index],
        }


def make_e10_collate_fn(tokenizer: Any, max_length: int):
    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [str(row["text"]) for row in rows]
        padded, alignments = _tokenize_with_alignment(tokenizer, texts, max_length)
        labels = [row["labels"] for row in rows]
        gold_labels = [row["gold_labels"] for row in rows]
        padded["labels"] = torch.tensor(labels, dtype=torch.long)
        padded["gold_labels"] = torch.tensor(gold_labels, dtype=torch.long)
        padded["ids"] = [row["id"] for row in rows]
        padded["raw_texts"] = texts
        padded["alignment"] = alignments
        return padded

    return collate


def make_e10_pretokenized_collate_fn(tokenizer: Any):
    """Pad clean rows while preserving their precomputed raw-text alignment."""

    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        padded = tokenizer.pad(
            [{"input_ids": row["input_ids"]} for row in rows],
            padding=True,
            return_tensors="pt",
        )
        target_length = int(padded["input_ids"].shape[1])
        labels = [row["labels"] for row in rows]
        gold_labels = [row["gold_labels"] for row in rows]
        padded["labels"] = torch.tensor(labels, dtype=torch.long)
        padded["gold_labels"] = torch.tensor(gold_labels, dtype=torch.long)
        padded["ids"] = [row["id"] for row in rows]
        padded["raw_texts"] = [str(row["text"]) for row in rows]
        padded["alignment"] = [row["alignment"].padded(target_length) for row in rows]
        return padded

    return collate


def make_e10_loader(
    tokenizer: Any,
    split: Any,
    *,
    texts: Sequence[str],
    ids: Sequence[Any],
    labels: Sequence[Any],
    gold_labels: Sequence[Any],
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[dict[str, Any]]:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    input_ids = [list(values) for values in encoded["input_ids"]]
    alignments = [
        build_token_alignment(text, offsets)
        for text, offsets in zip(texts, encoded["offset_mapping"])
    ]
    dataset = _RawTextDataset(texts, input_ids, alignments, labels, ids, gold_labels)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=make_e10_pretokenized_collate_fn(tokenizer),
    )


class ResidualCompositionBlock(nn.Module):
    def __init__(self, hidden_size: int, rank: int, variant: str):
        super().__init__()
        if variant not in {"span", "conv"}:
            raise ValueError(f"unknown E10 variant {variant!r}")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.variant = variant
        self.width_projections = nn.ParameterList(
            [nn.Parameter(torch.empty(width, hidden_size, rank)) for width in SPAN_WIDTHS]
        )
        self.width_biases = nn.ParameterList(
            [nn.Parameter(torch.empty(rank)) for _ in SPAN_WIDTHS]
        )
        for weights, bias in zip(self.width_projections, self.width_biases):
            nn.init.normal_(weights, mean=0.0, std=0.02)
            nn.init.normal_(bias, mean=0.0, std=0.02)
        self.output = nn.Linear(rank, hidden_size)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if variant == "span":
            self.query = nn.Linear(hidden_size, rank, bias=False)
            nn.init.normal_(self.query.weight, mean=0.0, std=0.02)
            self.position_bias = nn.Parameter(torch.empty(len(SPAN_WIDTHS), max(SPAN_WIDTHS)))
            nn.init.normal_(self.position_bias, mean=0.0, std=0.02)
            self.null_score = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _starts_by_width(alignment: TokenAlignment) -> tuple[tuple[int, ...], ...]:
        starts: list[tuple[int, ...]] = []
        for width in SPAN_WIDTHS:
            values = []
            for start in range(max(0, len(alignment.units) - width + 1)):
                if len({alignment.units[index].segment for index in range(start, start + width)}) == 1:
                    values.append(start)
            starts.append(tuple(values))
        return tuple(starts)

    def _layout(self, alignment: TokenAlignment, hidden: Tensor) -> dict[str, Tensor]:
        key = (str(hidden.device), str(hidden.dtype), str(len(alignment.token_to_units)))
        cached = alignment._layout_cache.get(key)
        if cached is not None:
            return cached
        unit_count = len(alignment.units)
        token_count = len(alignment.token_to_units)
        overlap = torch.zeros(token_count, unit_count, device=hidden.device, dtype=hidden.dtype)
        token_indices: list[int] = []
        unit_indices: list[int] = []
        weights: list[float] = []
        for token_index, entries in enumerate(alignment.token_to_units):
            for unit_index, weight in entries:
                token_indices.append(token_index)
                unit_indices.append(unit_index)
                weights.append(float(weight))
        if token_indices:
            overlap.index_put_(
                (
                    torch.tensor(token_indices, device=hidden.device, dtype=torch.long),
                    torch.tensor(unit_indices, device=hidden.device, dtype=torch.long),
                ),
                torch.tensor(weights, device=hidden.device, dtype=hidden.dtype),
                accumulate=True,
            )
        starts_by_width = self._starts_by_width(alignment)
        starts: list[int] = []
        width_indices: list[int] = []
        incidence_rows: list[int] = []
        incidence_cols: list[int] = []
        pair_bias_indices: list[int] = []
        for width_index, width_starts in enumerate(starts_by_width):
            width = SPAN_WIDTHS[width_index]
            for start in width_starts:
                span_index = len(starts)
                starts.append(start)
                width_indices.append(width_index)
                for position, unit_index in enumerate(range(start, start + width)):
                    incidence_rows.append(unit_index)
                    incidence_cols.append(span_index)
                    pair_bias_indices.append(width_index * max(SPAN_WIDTHS) + position)
        span_count = len(starts)
        incidence = torch.zeros(unit_count, span_count, device=hidden.device, dtype=hidden.dtype)
        pair_bias = torch.zeros(unit_count, span_count, device=hidden.device, dtype=torch.long)
        if incidence_rows:
            row_tensor = torch.tensor(incidence_rows, device=hidden.device, dtype=torch.long)
            col_tensor = torch.tensor(incidence_cols, device=hidden.device, dtype=torch.long)
            incidence.index_put_((row_tensor, col_tensor), torch.ones_like(row_tensor, dtype=hidden.dtype))
            pair_bias.index_put_(
                (row_tensor, col_tensor),
                torch.tensor(pair_bias_indices, device=hidden.device, dtype=torch.long),
            )
        cached = {
            "overlap": overlap,
            "starts": torch.tensor(starts, device=hidden.device, dtype=torch.long),
            "width_indices": torch.tensor(width_indices, device=hidden.device, dtype=torch.long),
            "incidence": incidence,
            "pair_bias": pair_bias,
        }
        alignment._layout_cache[key] = cached
        return cached

    def _compose_one(self, hidden: Tensor, alignment: TokenAlignment) -> Tensor:
        length = hidden.shape[0]
        layout = self._layout(alignment, hidden)
        overlap = layout["overlap"][:length]
        denominators = overlap.sum(dim=0)
        visible = denominators > 0
        if not bool(visible.any()):
            return torch.zeros(length, self.rank, device=hidden.device, dtype=hidden.dtype)
        units = overlap.transpose(0, 1).matmul(hidden)
        units = units / denominators.clamp_min(torch.finfo(hidden.dtype).tiny).unsqueeze(-1)
        span_count = int(layout["starts"].shape[0])
        if span_count == 0:
            return torch.zeros(length, self.rank, device=hidden.device, dtype=hidden.dtype)
        span_values: list[Tensor] = []
        span_valid: list[Tensor] = []
        for width_index, width in enumerate(SPAN_WIDTHS):
            mask = layout["width_indices"] == width_index
            starts = layout["starts"][mask]
            if starts.numel() == 0:
                continue
            offsets = torch.arange(width, device=hidden.device, dtype=torch.long)
            indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
            values = units.index_select(0, indices.reshape(-1)).reshape(-1, width, self.hidden_size)
            values = torch.einsum("nwd,wdr->nr", values, self.width_projections[width_index])
            values = torch.nn.functional.gelu(values + self.width_biases[width_index])
            span_values.append(values)
            span_valid.append(visible.index_select(0, indices.reshape(-1)).reshape(-1, width).all(dim=1))
        values = torch.cat(span_values, dim=0)
        valid = torch.cat(span_valid, dim=0)
        active_incidence = layout["incidence"] * valid.to(hidden.dtype).unsqueeze(0)
        if self.variant == "conv":
            counts = active_incidence.sum(dim=1)
            messages = active_incidence.matmul(values) / counts.clamp_min(1.0).unsqueeze(-1)
        else:
            query = self.query(units)
            scores = query.float().matmul(values.float().transpose(0, 1)) / math.sqrt(self.rank)
            flat_position_bias = self.position_bias.reshape(-1)
            scores = scores + flat_position_bias[layout["pair_bias"].clamp_min(0)].float()
            scores = scores.masked_fill(active_incidence <= 0, float("-inf"))
            scores_with_null = torch.cat(
                [scores, self.null_score.float().expand(scores.shape[0], 1)], dim=1
            )
            probabilities = torch.softmax(scores_with_null, dim=1)
            messages = probabilities[:, :-1].to(values.dtype).matmul(values)
        distributed = overlap.matmul(messages)
        token_weights = overlap.sum(dim=1)
        return distributed / token_weights.clamp_min(torch.finfo(hidden.dtype).tiny).unsqueeze(-1)

    def forward(self, hidden: Tensor, alignments: Sequence[TokenAlignment]) -> Tensor:
        if len(alignments) != hidden.shape[0]:
            raise ValueError(
                f"E10 alignment batch {len(alignments)} does not match hidden batch {hidden.shape[0]}"
            )
        rows: list[Tensor] = []
        for row, alignment in zip(hidden, alignments):
            delta = self.output(self._compose_one(row, alignment))
            eligible = torch.tensor(
                [bool(entries) for entries in alignment.token_to_units[: row.shape[0]]],
                device=row.device,
                dtype=delta.dtype,
            )
            # A learned output bias must never write into special or padding
            # positions, including examples with no visible raw units.
            rows.append(delta * eligible.unsqueeze(-1))
        deltas = torch.stack(rows)
        return hidden + deltas

    def metadata(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "rank": self.rank,
            "hidden_size": self.hidden_size,
            "span_widths": list(SPAN_WIDTHS),
            "output_projection_init": "zero_weight_and_bias",
            "null_score_init": 0.0 if self.variant == "span" else None,
        }


class E10LayerWrapper(nn.Module):
    """Original transformer layer followed by the E10 residual branch."""

    def __init__(self, original_layer: nn.Module, branch: ResidualCompositionBlock):
        super().__init__()
        self.base_layer = original_layer
        self.e10_block = branch
        self._alignment: Sequence[TokenAlignment] | None = None

    def set_alignment(self, alignment: Sequence[TokenAlignment] | None) -> None:
        self._alignment = alignment

    def forward(self, *args: Any, **kwargs: Any) -> tuple[Tensor, ...]:
        outputs = self.base_layer(*args, **kwargs)
        if self._alignment is None:
            raise RuntimeError("E10 layer called without explicit raw-text alignment")
        hidden = self.e10_block(outputs[0], self._alignment)
        return (hidden, *outputs[1:])


class E10Classifier(nn.Module):
    """E1 classifier modules with the block-2 residual branch inserted."""

    def __init__(self, base: GoiBERTClassifier, layer_wrapper: E10LayerWrapper):
        super().__init__()
        self.encoder = base.encoder
        self.head = base.head
        self.num_labels = base.num_labels
        self.num_aspects = base.num_aspects
        self.aspect_class_counts = base.aspect_class_counts
        self._backbone_name = "goiturn-s-gd"
        self._head_dropout = base._head_dropout
        self._head_hidden_size = base._head_hidden_size
        # The wrapper is already registered below ``encoder.encoder.layer.1``.
        # Keep a non-registering reference so state_dict and E7 grouping expose
        # one canonical parameter path for every branch parameter.
        object.__setattr__(self, "_e10_layer_wrapper", layer_wrapper)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        token_type_ids: Tensor | None = None,
        alignment: Sequence[TokenAlignment] | None = None,
    ) -> Tensor:
        if alignment is None:
            raise ValueError("E10Classifier.forward requires explicit raw-text alignment")
        if len(alignment) != int(input_ids.shape[0]):
            raise ValueError("alignment batch size does not match input_ids")
        if any(len(item.token_to_units) < int(input_ids.shape[1]) for item in alignment):
            raise ValueError("alignment is shorter than input_ids; tokenizer/collator mismatch")
        self._e10_layer_wrapper.set_alignment(alignment)
        encoder_kwargs: dict[str, Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        try:
            outputs = self.encoder(**encoder_kwargs, return_dict=True)
            logits = self.head(outputs.last_hidden_state)
        finally:
            self._e10_layer_wrapper.set_alignment(None)
        if self.num_aspects and len(set(self.aspect_class_counts)) == 1:
            logits = logits.reshape(logits.shape[0], self.num_aspects, self.aspect_class_counts[0])
        return logits

    def spec(self) -> E10ModelSpec:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return E10ModelSpec(
            backbone=self._backbone_name,
            hidden_size=int(self.encoder.config.hidden_size),
            head_hidden_size=int(self._head_hidden_size),
            head_dropout=float(self._head_dropout),
            num_labels=int(self.num_labels),
            num_aspects=int(self.num_aspects),
            aspect_class_counts=tuple(self.aspect_class_counts),
            total_parameters=total,
            trainable_parameters=trainable,
            architecture="e10_residual_after_block2",
            variant=self._e10_layer_wrapper.e10_block.variant,
            insertion_layer_index=INSERTION_LAYER_INDEX,
            branch_rank=self._e10_layer_wrapper.e10_block.rank,
            span_widths=SPAN_WIDTHS,
        )

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "architecture": "e10_residual_after_block2",
            "variant": self._e10_layer_wrapper.e10_block.variant,
            "insertion_layer_index": INSERTION_LAYER_INDEX,
            "branch": self._e10_layer_wrapper.e10_block.metadata(),
            "explicit_alignment": True,
            "alignment_contract": "raw_text_unicode_units_fast_tokenizer_offsets_visible_after_truncation",
        }


def attach_e10_block(base: GoiBERTClassifier, variant: str) -> tuple[E10Classifier, E10ModelSpec]:
    if variant not in {"span", "conv"}:
        raise ValueError(f"unknown E10 variant {variant!r}")
    hidden_size = int(base.encoder.config.hidden_size)
    rank = SPAN_RANK if variant == "span" else CONV_RANK
    # The branch consumes a private RNG fork.  The original encoder and CLS
    # head therefore have exact E1/E8 initialization parity and later RNG
    # consumers see the same stream as the frozen student constructor.
    with torch.random.fork_rng(devices=[]):
        branch = ResidualCompositionBlock(hidden_size, rank, variant)
    original = base.encoder.encoder.layer[INSERTION_LAYER_INDEX]
    wrapper = E10LayerWrapper(original, branch)
    base.encoder.encoder.layer[INSERTION_LAYER_INDEX] = wrapper
    model = E10Classifier(base, wrapper)
    return model, model.spec()


def build_e10_student(
    backbone_dir: Path,
    *,
    variant: str,
    num_classes: int,
    num_aspects: int = 0,
    aspect_class_counts: Sequence[int] | None = None,
    attention_implementation: str = "sdpa",
) -> tuple[E10Classifier, E10ModelSpec]:
    base, _ = build_e2_student(
        backbone_dir,
        num_classes=num_classes,
        num_aspects=num_aspects,
        aspect_class_counts=aspect_class_counts,
        attention_implementation=attention_implementation,
    )
    return attach_e10_block(base, variant)
