"""Small, auditable MLM components for the E6 continuation pilot.

The original GoiBERT student directory contains a pooler-free
``RobertaModel``.  E6 loads that encoder unchanged and attaches the standard
RoBERTa MLM head.  The decoder is tied to the encoder word embedding and the
training loop computes logits only at selected masked positions.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn
from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaLMHead


class E6RobertaMLM(nn.Module):
    """Student encoder plus a standard tied RoBERTa masked-LM head."""

    def __init__(self, encoder: RobertaModel) -> None:
        super().__init__()
        self.roberta = encoder
        self.lm_head = RobertaLMHead(self.roberta.config)
        # Match RobertaPreTrainedModel._init_weights for the newly attached
        # head only.  This happens before tying decoder.weight; touching the
        # tied decoder after this point would overwrite loaded embeddings.
        init_std = float(self.roberta.config.initializer_range)
        nn.init.normal_(self.lm_head.dense.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.lm_head.dense.bias)
        nn.init.ones_(self.lm_head.layer_norm.weight)
        nn.init.zeros_(self.lm_head.layer_norm.bias)
        nn.init.normal_(self.lm_head.decoder.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.lm_head.decoder.bias)
        # Assign the exact loaded embedding Parameter.  This keeps the MLM
        # decoder tied to the student embedding from initialization onward.
        self.lm_head.decoder.weight = self.roberta.embeddings.word_embeddings.weight
        self.lm_head._tie_weights()

    @classmethod
    def from_pretrained_encoder(
        cls,
        model_dir: str,
        *,
        local_files_only: bool = True,
        attention_implementation: str = "sdpa",
    ) -> "E6RobertaMLM":
        kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "add_pooling_layer": False,
        }
        if attention_implementation:
            kwargs["attn_implementation"] = attention_implementation
        encoder = RobertaModel.from_pretrained(model_dir, **kwargs)
        return cls(encoder)

    @property
    def encoder(self) -> RobertaModel:
        return self.roberta

    def forward_masked(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        masked_positions: Tensor,
    ) -> Tensor:
        """Return logits for ``True`` positions only, in row-major order."""

        if masked_positions.dtype is not torch.bool:
            masked_positions = masked_positions.to(dtype=torch.bool)
        if masked_positions.shape != input_ids.shape:
            raise ValueError("masked_positions must have the same shape as input_ids")
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        selected = outputs.last_hidden_state[masked_positions]
        if selected.numel() == 0:
            raise ValueError("A masked-LM batch must contain at least one masked token")
        return self.lm_head(selected)

    def forward_full(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return full sequence logits for the small algebraic contract check."""

        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return self.lm_head(outputs.last_hidden_state)


def _special_mask(
    input_ids: Tensor,
    attention_mask: Tensor,
    special_ids: Sequence[int],
) -> Tensor:
    candidates = attention_mask.to(dtype=torch.bool)
    for token_id in special_ids:
        candidates &= input_ids.ne(int(token_id))
    return candidates


def mask_batch(
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    mask_token_id: int,
    vocab_size: int,
    special_ids: Sequence[int],
    probability: float = 0.15,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply dynamic 80/10/10 MLM masking without touching special tokens.

    The global torch RNG is deliberately used for training batches.  The
    fixed evaluation path below receives pre-generated corruptions from a
    separate CPU generator and therefore does not perturb training RNG state.
    """

    if not 0.0 < probability <= 1.0:
        raise ValueError("probability must be in (0, 1]")
    eligible = _special_mask(input_ids, attention_mask, special_ids)
    selected = (torch.rand(input_ids.shape, device=input_ids.device) < probability) & eligible
    labels = input_ids.clone()
    labels[~selected] = -100
    corrupted = input_ids.clone()
    choices = torch.rand(input_ids.shape, device=input_ids.device)
    replace_mask = selected & choices.lt(0.8)
    replace_random = selected & choices.ge(0.8) & choices.lt(0.9)
    corrupted[replace_mask] = int(mask_token_id)
    if replace_random.any():
        corrupted[replace_random] = torch.randint(
            low=0,
            high=int(vocab_size),
            size=(int(replace_random.sum().item()),),
            device=input_ids.device,
        )
    if not selected.any():
        # Very short batches can occasionally miss the 15% draw.  Select one
        # valid content token deterministically so the optimizer always has a
        # finite objective; this does not mask special/pad positions.
        first = eligible.flatten().nonzero(as_tuple=False)
        if first.numel() == 0:
            raise ValueError("Batch contains no non-special, non-padding token")
        flat_index = int(first[0].item())
        selected.view(-1)[flat_index] = True
        labels.view(-1)[flat_index] = input_ids.view(-1)[flat_index]
        corrupted.view(-1)[flat_index] = int(mask_token_id)
    return corrupted, labels, selected


def fixed_masked_batch(
    input_ids: Tensor,
    attention_mask: Tensor,
    mask_positions: Tensor,
    corrupted_ids: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Prepare an already-corrupted fixed evaluation batch."""

    if input_ids.shape != attention_mask.shape or input_ids.shape != mask_positions.shape:
        raise ValueError("fixed MLM arrays must have matching shapes")
    if corrupted_ids.shape != input_ids.shape:
        raise ValueError("corrupted_ids must match input_ids")
    labels = input_ids.clone()
    labels[~mask_positions.to(dtype=torch.bool)] = -100
    return corrupted_ids, labels, mask_positions.to(dtype=torch.bool)


def masked_loss(logits: Tensor, labels: Tensor) -> tuple[Tensor, int]:
    """Compute CE and return its masked-token count for weighted aggregation."""

    targets = labels[labels.ne(-100)]
    if targets.numel() != logits.shape[0]:
        raise ValueError(
            f"masked target/logit count mismatch: {targets.numel()} != {logits.shape[0]}"
        )
    if targets.numel() == 0:
        raise ValueError("masked_loss requires at least one target token")
    loss = nn.functional.cross_entropy(logits, targets, reduction="sum")
    return loss, int(targets.numel())


def encoder_state_hash(model: E6RobertaMLM) -> str:
    """Hash encoder tensors in key order for the Stage-A immutability check."""

    import hashlib

    digest = hashlib.sha256()
    for name, value in model.roberta.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def assert_tied_decoder(model: E6RobertaMLM) -> None:
    embedding = model.roberta.embeddings.word_embeddings.weight
    decoder = model.lm_head.decoder.weight
    if decoder is not embedding:
        raise AssertionError("MLM decoder is not tied to the encoder word embedding Parameter")
    if decoder.data_ptr() != embedding.data_ptr():
        raise AssertionError("MLM decoder and encoder embedding do not share storage")
