"""Diacritic-factorized embedding block for Pilot B.

E(t) = E_skel[skel_id[t]] + E_pat[pat_id[t]] + U[t] @ W   (U: V x r, W: r x H, r << H)

See docs/PILOT_B_DIACRITIC_FACTORIZED.md.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class DiacriticFactorizedEmbedding(nn.Module):
    def __init__(self, n_groups: int, n_patterns: int, vocab_size: int, hidden: int, rank: int,
                 skel_id: Tensor, pat_id: Tensor):
        super().__init__()
        self.vocab_size, self.hidden, self.rank = vocab_size, hidden, rank
        self.E_skel = nn.Parameter(torch.zeros(n_groups, hidden))
        self.E_pat = nn.Parameter(torch.zeros(n_patterns, hidden))
        self.U = nn.Parameter(torch.zeros(vocab_size, rank))
        self.W = nn.Parameter(torch.zeros(rank, hidden))
        self.register_buffer("skel_id", skel_id.long())
        self.register_buffer("pat_id", pat_id.long())

    def forward(self, ids: Tensor) -> Tensor:
        return self.E_skel[self.skel_id[ids]] + self.E_pat[self.pat_id[ids]] + self.U[ids] @ self.W

    def materialize(self) -> Tensor:
        # E_pat is only added on diacritized rows: gathering row 0 for the ~230k undiacritized tokens is a no-op in the
        # forward but its backward is 230k atomic adds into one row (measured 2-3x slowdown of the whole step).
        if not hasattr(self, "_diac_idx"):
            self._diac_idx = self.pat_id.nonzero(as_tuple=True)[0]
        E = self.E_skel[self.skel_id] + self.U @ self.W
        return E.index_add(0, self._diac_idx, self.E_pat[self.pat_id[self._diac_idx]])

    @classmethod
    def from_pretrained_matrix(cls, E_orig: Tensor, skel_id: Tensor, pat_id: Tensor, n_groups: int,
                                n_patterns: int, rank: int = 64) -> tuple["DiacriticFactorizedEmbedding", dict]:
        """Init from an existing [V,H] embedding matrix: group means + pattern means + rank-r SVD residual.

        Returns (module, stats). stats has n_affected_rows and the relative Frobenius reconstruction error of the
        residual at rank 0 (group+pattern factorization alone) and at the requested rank.
        """
        device = E_orig.device
        V, H = E_orig.shape
        skel_id = skel_id.to(device).long()
        pat_id = pat_id.to(device).long()
        E = E_orig.detach().float()

        # group (skeleton) means
        E_skel = torch.zeros(n_groups, H, device=device, dtype=torch.float32)
        cnt = torch.zeros(n_groups, device=device, dtype=torch.float32)
        E_skel.index_add_(0, skel_id, E)
        cnt.index_add_(0, skel_id, torch.ones(V, device=device))
        E_skel /= cnt.clamp(min=1).unsqueeze(1)

        # pattern means of the group-mean residual (pattern 0 = "no diacritics" is identically zero)
        r1 = E - E_skel[skel_id]
        E_pat = torch.zeros(n_patterns, H, device=device, dtype=torch.float32)
        pcnt = torch.zeros(n_patterns, device=device, dtype=torch.float32)
        E_pat.index_add_(0, pat_id, r1)
        pcnt.index_add_(0, pat_id, torch.ones(V, device=device))
        E_pat /= pcnt.clamp(min=1).unsqueeze(1)
        E_pat[0] = 0.0

        R = r1 - E_pat[pat_id]
        affected = R.norm(dim=1) > 1e-8
        n_affected = int(affected.sum().item())

        U_full = torch.zeros(V, rank, device=device, dtype=torch.float32)
        W = torch.zeros(rank, H, device=device, dtype=torch.float32)
        err_rank0 = err_rankr = 0.0
        if n_affected:
            R_aff = R[affected]
            norm_orig_aff = E[affected].norm().item()
            Uo, S, Vh = torch.linalg.svd(R_aff, full_matrices=False)
            r = min(rank, S.numel())
            U_r, S_r, Vh_r = Uo[:, :r], S[:r], Vh[:r]
            W[:r] = S_r.unsqueeze(1) * Vh_r
            U_full[affected, :r] = U_r
            err_rank0 = (R_aff.norm() / norm_orig_aff).item()
            err_rankr = ((R_aff - U_r @ W[:r]).norm() / norm_orig_aff).item()

        module = cls(n_groups, n_patterns, V, H, rank, skel_id.cpu(), pat_id.cpu())
        module.E_skel.data.copy_(E_skel)
        module.E_pat.data.copy_(E_pat)
        module.U.data.copy_(U_full)
        module.W.data.copy_(W)

        stats = dict(n_affected_rows=n_affected, rel_err_rank0=err_rank0, rel_err_rank=err_rankr, rank=rank)
        print(f"[factorized_embedding] V={V} H={H} groups={n_groups} patterns={n_patterns} rank={rank} "
              f"affected_rows={n_affected} rel_err(rank0)={err_rank0:.4f} rel_err(rank{rank})={err_rankr:.4f}",
              flush=True)
        return module, stats


def apply_diacritic_dropout(input_ids: Tensor, labels: Tensor, ascii_tok: Tensor, p: float,
                             generator: torch.Generator | None = None) -> Tensor:
    """Replace non-masked (labels == -100) diacritized input tokens by their ASCII-skeleton id w.p. p.

    ascii_tok: int64 [V], ascii_tok[t] = ascii-skeleton vocab id of t, or -1. Labels are left unchanged.
    """
    eligible = labels.eq(-100) & ascii_tok[input_ids].ge(0)
    draw = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    drop = eligible & draw.lt(p)
    out = input_ids.clone()
    out[drop] = ascii_tok[input_ids][drop]
    return out


class AdditiveDiacriticEmbedding(nn.Module):
    """Identity-preserving variant (arm b2): E(t) = E_tok[t] + E_skel[skel_id[t]] + E_pat[pat_id[t]].

    E_tok is initialised to the original matrix and the shared channels to zero, so the model starts exactly at the
    backbone; the shared channels only get used through the CPT gradient (ChineseBERT-style extra channels rather than
    the compressive factorisation above, which corrupts common toned words at init — see PILOT_B doc).
    """

    def __init__(self, E_orig: Tensor, n_groups: int, n_patterns: int, skel_id: Tensor, pat_id: Tensor):
        super().__init__()
        V, H = E_orig.shape
        self.vocab_size, self.hidden = V, H
        self.E_tok = nn.Parameter(E_orig.detach().float().clone())
        self.E_skel = nn.Parameter(torch.zeros(n_groups, H))
        self.E_pat = nn.Parameter(torch.zeros(n_patterns, H))
        skel_id, pat_id = skel_id.long().cpu(), pat_id.long().cpu()
        self.register_buffer("skel_id", skel_id)
        self.register_buffer("pat_id", pat_id)
        # shared channels act only on "affected" tokens: diacritized or in a multi-member skeleton group; everything
        # else is exactly E_tok (and stays out of the scatter-add backward, see DiacriticFactorizedEmbedding.materialize)
        gsize = torch.bincount(skel_id, minlength=n_groups)
        self.register_buffer("aff_idx", ((gsize[skel_id] >= 2) | (pat_id != 0)).nonzero(as_tuple=True)[0])
        self.register_buffer("aff_mask", torch.zeros(V).index_fill_(0, self.aff_idx, 1.0))

    def shared(self, ids: Tensor) -> Tensor:
        return self.E_skel[self.skel_id[ids]] + self.E_pat[self.pat_id[ids]]

    def forward(self, ids: Tensor) -> Tensor:
        return self.E_tok[ids] + self.shared(ids) * self.aff_mask[ids].unsqueeze(-1)

    def materialize(self) -> Tensor:
        return self.E_tok.index_add(0, self.aff_idx, self.shared(self.aff_idx))
