"""Relational hidden-state distillation loss (plan §5) and the MLM masking shared by arms M and G.

For a sequence with n unit spans, h[layer, i] is the mean of the hidden vectors of all subwords overlapping span i
(specials/pad excluded).  Vectors are L2-normalised, C[i, j] = cosine(h_i, h_j); the diagonal and spans that are
not valid in BOTH models are dropped; P[i, :] = softmax(C[i, :] / T) row-wise; the loss is the mean over valid rows
of KL(P_teacher || P_student), averaged over the two layer pairs.  The teacher side is computed under no_grad.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

NEG = -1.0e4   # finite "minus infinity" so that fully-masked rows never produce NaN


def span_pool(hidden: torch.Tensor, span_idx: torch.Tensor, n_spans: int) -> tuple[torch.Tensor, torch.Tensor]:
    """hidden (B, L, H), span_idx (B, L) long with -1 for tokens that belong to no span -> pooled (B, N, H) float32, valid (B, N) bool."""
    B, L, H = hidden.shape
    hidden = hidden.float()
    keep = (span_idx >= 0)
    idx = span_idx.clamp(min=0)
    pooled = hidden.new_zeros(B, n_spans, H)
    pooled.scatter_add_(1, idx.unsqueeze(-1).expand(B, L, H), hidden * keep.unsqueeze(-1).to(hidden.dtype))
    counts = hidden.new_zeros(B, n_spans).scatter_add_(1, idx, keep.to(hidden.dtype))
    valid = counts > 0
    pooled = pooled / counts.clamp(min=1.0).unsqueeze(-1)
    return pooled, valid


def relation_log_probs(pooled: torch.Tensor, valid: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """pooled (B, N, H), valid (B, N) -> row-wise log P over valid off-diagonal columns (B, N, N) and the column mask."""
    B, N, _ = pooled.shape
    z = F.normalize(pooled.float(), dim=-1)
    C = torch.bmm(z, z.transpose(1, 2))
    eye = torch.eye(N, dtype=torch.bool, device=pooled.device).unsqueeze(0)
    col_mask = valid.unsqueeze(1) & ~eye                      # (B, N, N): column j valid and j != i
    logits = (C / temperature).masked_fill(~col_mask, NEG)
    return F.log_softmax(logits, dim=-1), col_mask


def relational_kl(teacher_pooled: torch.Tensor, teacher_valid: torch.Tensor, student_pooled: torch.Tensor, student_valid: torch.Tensor,
                  temperature: float) -> tuple[torch.Tensor, int]:
    """Mean over valid rows of KL(P_T || P_S); a row is valid when its span is valid in both models and has >= 1 other valid column."""
    valid = teacher_valid & student_valid
    logp_t, col_mask = relation_log_probs(teacher_pooled, valid, temperature)
    logp_s, _ = relation_log_probs(student_pooled, valid, temperature)
    row_valid = valid & (valid.sum(-1, keepdim=True) >= 2)
    kl = (logp_t.exp() * (logp_t - logp_s)).masked_fill(~col_mask, 0.0).sum(-1)   # (B, N)
    n_rows = int(row_valid.sum().item())
    if n_rows == 0:
        return student_pooled.sum() * 0.0, 0
    return kl[row_valid].mean(), n_rows


def mask_tokens(input_ids: torch.Tensor, attention_mask: torch.Tensor, special_ids: torch.Tensor, mask_id: int, vocab_size: int, mask_prob: float,
                generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """BERT masking on CPU with an explicit generator so that arms M and G draw identical masks: 15 % of non-special,
    non-pad tokens; of those 80 % -> <mask>, 10 % -> random token, 10 % unchanged.  labels = -100 elsewhere."""
    eligible = attention_mask.bool() & ~torch.isin(input_ids, special_ids)
    prob = torch.rand(input_ids.shape, generator=generator)
    selected = (prob < mask_prob) & eligible
    labels = input_ids.clone(); labels[~selected] = -100
    choice = torch.rand(input_ids.shape, generator=generator)
    out = input_ids.clone()
    out[selected & (choice < 0.8)] = mask_id
    rand_pos = selected & (choice >= 0.8) & (choice < 0.9)
    rand_tok = torch.randint(0, vocab_size, input_ids.shape, generator=generator)
    out[rand_pos] = rand_tok[rand_pos]
    return out, labels


def mlm_loss(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """CE at masked positions only (sparse head: lm_head applied to the selected hidden states)."""
    hidden = model.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    sel = labels.ne(-100)
    n = int(sel.sum().item())
    if n == 0:
        return hidden.sum() * 0.0, 0
    logits = model.lm_head(hidden[sel])
    return F.cross_entropy(logits.float(), labels[sel]), n
