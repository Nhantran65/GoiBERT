#!/usr/bin/env python3
"""Literal and factorized checks for the cavity-key algebra specification.

This is a synthetic tensor audit.  It deliberately does not load model data,
checkpoints, or run an optimizer.  All computations are CPU Torch operations.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts" / "research_checks" / "cavity_key_algebra"
SPEC_PATH = ROOT / "docs" / "CAVITY_KEY_ALGEBRA_SPEC.md"
EPS = 1.0e-5
SEED = 20260912
FP64_ABS_TOL = 1.0e-8
FP64_REL_TOL = 1.0e-6


def py_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def tensor_finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def tensor_shape(x: torch.Tensor) -> list[int]:
    return [int(v) for v in x.shape]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Convert tensors/scalars to strict JSON values without NaN/Inf."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return py_float(value)
        return [json_safe(v) for v in value.detach().cpu().tolist()]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return value
    return str(value)


def make_valid_mask(n: int) -> torch.Tensor:
    if n == 1:
        count = 1
    elif n == 2:
        count = 1
    else:
        count = n - 1 if n in (5, 9) else n
    mask = torch.zeros(n, dtype=torch.bool)
    mask[:count] = True
    return mask


def softmax_rows(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    key_mask = valid.view(1, 1, -1)
    return torch.softmax(logits.masked_fill(~key_mask, float("-inf")), dim=-1)


def random_params(
    n: int,
    h: int,
    d: int,
    p: int,
    valid: torch.Tensor,
    dtype: torch.dtype,
    generator: torch.Generator,
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    def normal(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=dtype) * scale

    # gamma/beta remain independently nontrivial at the requested scale; the
    # other leaves receive the explicit stress scale.
    params = {
        "H0": normal(n, d),
        "Q": normal(n, h, p),
        "V": normal(n, h, p),
        "Wo": normal(h, d, p),
        "bo": normal(d),
        "Wk": normal(h, p, d),
        "bk": normal(h, p),
        "gamma": 0.6 + torch.rand(d, generator=generator, dtype=dtype),
        "betaLN": normal(d),
        "logits": torch.randn(h, n, n, generator=generator, dtype=dtype),
    }
    params["valid"] = valid.clone()
    return params


def clone_leaves(params: dict[str, torch.Tensor], requires_grad: bool) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, value in params.items():
        copied = value.detach().clone()
        if name != "valid" and requires_grad:
            copied.requires_grad_(True)
        out[name] = copied
    return out


def compute_u_z(params: dict[str, torch.Tensor], a_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # u[i,h,d] = Wo[h,d,p] V[i,h,p]
    u = torch.einsum("hdp,nhp->nhd", params["Wo"], params["V"])
    messages = torch.einsum("hjk,khd->jd", a_matrix, u)
    z = params["H0"] + params["bo"] + messages
    return u, z


def centered(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=-1, keepdim=True)


def literal_forward(params: dict[str, torch.Tensor], eps: float = EPS) -> dict[str, torch.Tensor]:
    a_matrix = softmax_rows(params["logits"], params["valid"])
    u, z = compute_u_z(params, a_matrix)
    # a[h,i,j] = A[h,j,i], with the cavity diagonal deleted and not renormalized.
    a = a_matrix.permute(0, 2, 1)
    diagonal = torch.eye(z.shape[0], dtype=torch.bool, device=z.device).unsqueeze(0)
    a = a.masked_fill(diagonal, 0.0)
    removed = torch.einsum("hij,ihd->ijd", a, u)
    c = z.unsqueeze(0) - removed
    c_centered = centered(c)
    variance = (c_centered.square()).mean(dim=-1)
    sigma = torch.sqrt(variance + eps)
    ln_c = params["gamma"] * c_centered / sigma.unsqueeze(-1) + params["betaLN"]
    # key[i,t,j,p] is only materialized in the literal reference.
    key = torch.einsum("tpd,ijd->itjp", params["Wk"], ln_c) + params["bk"].view(1, -1, 1, params["Wk"].shape[1])
    scores = (params["Q"].unsqueeze(2) * key).sum(dim=-1) / math.sqrt(params["Q"].shape[-1])
    return {
        "A": a_matrix,
        "u": u,
        "z": z,
        "a": a,
        "removed": removed,
        "C": c,
        "C_centered": c_centered,
        "variance": variance,
        "sigma": sigma,
        "LN_C": ln_c,
        "S": scores,
    }


def factorized_forward(params: dict[str, torch.Tensor], eps: float = EPS) -> dict[str, Any]:
    """Factorized score path with no N,N,d or N,N,H,p tensor."""
    a_matrix = softmax_rows(params["logits"], params["valid"])
    u, z = compute_u_z(params, a_matrix)
    a = a_matrix.permute(0, 2, 1)
    diagonal = torch.eye(z.shape[0], dtype=torch.bool, device=z.device).unsqueeze(0)
    a = a.masked_fill(diagonal, 0.0)

    w = centered(z)  # [j,d]
    e = centered(u)  # [i,h,d]
    ww = w.square().sum(dim=-1)  # [j]
    we = torch.einsum("jd,ihd->ijh", w, e)  # [i,j,h]
    ee = torch.einsum("ihd,igd->ihg", e, e)  # [i,h,g]
    aa = a.permute(1, 2, 0)  # [i,j,h]
    variance_numerator = (
        ww.unsqueeze(0)
        - 2.0 * (aa * we).sum(dim=-1)
        + torch.einsum("ijh,ijg,ihg->ij", aa, aa, ee)
    )
    vnorm = variance_numerator / z.shape[-1]
    # The max is a deliberate FP32 safety operation required by the spec.
    if z.dtype == torch.float32:
        vnorm_for_sqrt = torch.maximum(vnorm, torch.zeros_like(vnorm))
    else:
        vnorm_for_sqrt = vnorm
    sigma = torch.sqrt(vnorm_for_sqrt + eps)

    r = torch.einsum("tpd,itp,d->itd", params["Wk"], params["Q"], params["gamma"])
    rw = torch.einsum("itd,jd->itj", r, w)
    re = torch.einsum("itd,ihd->ith", r, e)
    nterm = rw - torch.einsum("hij,ith->itj", a, re)
    key_beta = torch.einsum("tpd,d->tp", params["Wk"], params["betaLN"]) + params["bk"]
    cterm = torch.einsum("itp,tp->it", params["Q"], key_beta)
    scores = (nterm / sigma.unsqueeze(1) + cterm.unsqueeze(-1)) / math.sqrt(params["Q"].shape[-1])

    intermediate_tensors = {
        "u": u,
        "z": z,
        "a": a,
        "w": w,
        "e": e,
        "ww": ww,
        "we": we,
        "ee": ee,
        "aa": aa,
        "variance_numerator": variance_numerator,
        "vnorm": vnorm,
        "sigma": sigma,
        "r": r,
        "rw": rw,
        "re": re,
        "n": nterm,
        "key_beta": key_beta,
        "c": cterm,
        "S": scores,
    }
    intermediate_shapes = {
        name: {"shape": tensor_shape(tensor), "logical_elements": int(tensor.numel())}
        for name, tensor in intermediate_tensors.items()
    }
    peak_name, peak_info = max(
        intermediate_shapes.items(), key=lambda pair: pair[1]["logical_elements"]
    )
    return {
        "A": a_matrix,
        "u": u,
        "z": z,
        "a": a,
        "w": w,
        "e": e,
        "we": we,
        "ee": ee,
        "vnorm": vnorm,
        "sigma": sigma,
        "r": r,
        "rw": rw,
        "re": re,
        "n": nterm,
        "c": cterm,
        "S": scores,
        "intermediate_shapes": intermediate_shapes,
        "derived_peak_logical_element_count": peak_info["logical_elements"],
        "derived_peak_logical_element_name": peak_name,
    }


def valid_pair_mask(valid: torch.Tensor) -> torch.Tensor:
    return valid[:, None] & valid[None, :]


def compare_scores(
    left: torch.Tensor,
    right: torch.Tensor,
    pair_mask: torch.Tensor,
) -> dict[str, Any]:
    # Scores are [i,t,j], while pair_mask is [i,j].
    x = left.permute(0, 2, 1)[pair_mask]
    y = right.permute(0, 2, 1)[pair_mask]
    diff = (x - y).abs()
    denom = y.abs().clamp_min(torch.finfo(y.dtype).tiny)
    abs_norm = (x - y).norm()
    ref_norm = y.norm()
    return {
        "max_abs": py_float(diff.max()) if diff.numel() else 0.0,
        "max_rel": py_float((diff / denom).max()) if diff.numel() else 0.0,
        "abs_norm": py_float(abs_norm),
        "reference_norm": py_float(ref_norm),
        "factorized_norm": py_float(x.norm()),
        "relative_norm": py_float(abs_norm / ref_norm.clamp_min(torch.finfo(y.dtype).tiny)),
        "finite": bool(tensor_finite(left) and tensor_finite(right)),
        "pairs_compared": int(x.shape[0]),
    }


def compare_tensor(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    diff = (left - right).abs()
    denom = right.abs().clamp_min(torch.finfo(right.dtype).tiny)
    left_norm = left.norm()
    right_norm = right.norm()
    diff_norm = (left - right).norm()
    return {
        "max_abs": py_float(diff.max()) if diff.numel() else 0.0,
        "max_rel": py_float((diff / denom).max()) if diff.numel() else 0.0,
        "abs_norm": py_float(diff_norm),
        "reference_norm": py_float(right_norm),
        "factorized_norm": py_float(left_norm),
        "relative_norm": py_float(diff_norm / right_norm.clamp_min(torch.finfo(right.dtype).tiny)),
        "finite": bool(tensor_finite(left) and tensor_finite(right)),
    }


def within_tolerance(error: dict[str, Any], abs_tol: float, rel_tol: float) -> bool:
    return bool(
        error["finite"]
        and error["max_abs"] <= abs_tol
        and error["max_rel"] <= rel_tol
    )


def forward_case_summary(
    name: str,
    params: dict[str, torch.Tensor],
    eps: float = EPS,
) -> dict[str, Any]:
    ref = literal_forward(params, eps=eps)
    fact = factorized_forward(params, eps=eps)
    pairs = valid_pair_mask(params["valid"])
    score_error = compare_scores(fact["S"], ref["S"], pairs)
    sigma_error = compare_tensor(fact["sigma"], ref["sigma"])
    vnorm_error = compare_tensor(fact["vnorm"], ref["variance"])
    return {
        "name": name,
        "shape": {key: tensor_shape(params[key]) for key in ("H0", "Q", "V", "Wo", "Wk", "logits")},
        "valid_mask": [bool(v) for v in params["valid"].tolist()],
        "score_error": score_error,
        "sigma_error": sigma_error,
        "variance_error": vnorm_error,
        "within_float64_tolerance": within_tolerance(score_error, FP64_ABS_TOL, FP64_REL_TOL),
        "factorized_intermediates": {
            "shapes": fact["intermediate_shapes"],
            "derived_peak_logical_element_count": fact["derived_peak_logical_element_count"],
            "derived_peak_logical_element_name": fact["derived_peak_logical_element_name"],
        },
    }


def backward_case_summary(
    name: str,
    base_params: dict[str, torch.Tensor],
    generator: torch.Generator,
) -> dict[str, Any]:
    ref_params = clone_leaves(base_params, requires_grad=True)
    fact_params = clone_leaves(base_params, requires_grad=True)
    ref = literal_forward(ref_params)
    fact = factorized_forward(fact_params)
    cotangent = torch.randn(ref["S"].shape, generator=generator, dtype=ref["S"].dtype)
    pair_mask = valid_pair_mask(base_params["valid"])
    cotangent = cotangent * pair_mask[:, None, :]
    ref_loss = (ref["S"] * cotangent).sum()
    fact_loss = (fact["S"] * cotangent).sum()
    names = ("H0", "Q", "V", "Wo", "bo", "Wk", "bk", "gamma", "betaLN", "logits")
    ref_grads = torch.autograd.grad(ref_loss, tuple(ref_params[name] for name in names), allow_unused=False)
    fact_grads = torch.autograd.grad(fact_loss, tuple(fact_params[name] for name in names), allow_unused=False)
    gradient_errors: dict[str, Any] = {}
    all_ok = True
    for param_name, ref_grad, fact_grad in zip(names, ref_grads, fact_grads):
        error = compare_tensor(fact_grad, ref_grad)
        error["reference_gradient_norm"] = py_float(ref_grad.norm())
        error["factorized_gradient_norm"] = py_float(fact_grad.norm())
        error["finite"] = bool(error["finite"] and tensor_finite(ref_grad) and tensor_finite(fact_grad))
        error["within_float64_tolerance"] = within_tolerance(error, FP64_ABS_TOL, FP64_REL_TOL)
        gradient_errors[param_name] = error
        all_ok = all_ok and error["within_float64_tolerance"]
    return {
        "name": name,
        "cotangent_valid_pairs": int(pair_mask.sum().item()),
        "loss_reference": py_float(ref_loss),
        "loss_factorized": py_float(fact_loss),
        "gradient_errors": gradient_errors,
        "within_float64_tolerance": all_ok,
    }


def full_context_scores(params: dict[str, torch.Tensor], z: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    ln_z = params["gamma"] * centered(z) / torch.sqrt(centered(z).square().mean(dim=-1, keepdim=True) + eps) + params["betaLN"]
    key = torch.einsum("tpd,jd->jtp", params["Wk"], ln_z) + params["bk"].unsqueeze(0)
    return (params["Q"].unsqueeze(2) * key.permute(1, 0, 2).unsqueeze(0)).sum(dim=-1) / math.sqrt(params["Q"].shape[-1])


def partial_path_check(generator: torch.Generator) -> dict[str, Any]:
    n, h, p, d = 3, 2, 3, 8
    valid = torch.ones(n, dtype=torch.bool)
    params = random_params(n, h, d, p, valid, torch.float64, generator)
    base = literal_forward(params)
    routing_mass = base["A"].sum(dim=0)
    candidate_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    i, j = max(candidate_pairs, key=lambda pair: py_float(routing_mass[pair[1], pair[0]]))
    delta = 2.0 * torch.randn(h, p, generator=generator, dtype=torch.float64)
    changed = {key: value.clone() for key, value in params.items()}
    changed["V"][i] = changed["V"][i] + delta
    perturbed = literal_forward(changed)
    cavity_c_error = (base["C"][i, j] - perturbed["C"][i, j]).abs().max()
    cavity_s_error = (base["S"][i, :, j] - perturbed["S"][i, :, j]).abs().max()
    full_base = full_context_scores(params, base["z"])
    full_changed = full_context_scores(changed, perturbed["z"])
    full_key_delta = centered(base["z"][j]) - centered(perturbed["z"][j])
    full_score_delta = (full_base[i, :, j] - full_changed[i, :, j]).abs().max()
    return {
        "N": n,
        "H": h,
        "d": d,
        "p": p,
        "selected_source_i": i,
        "selected_receiver_j": j,
        "routing_mass_A_source_to_receiver": py_float(routing_mass[j, i]),
        "cavity_C_max_abs_change": py_float(cavity_c_error),
        "cavity_S_max_abs_change": py_float(cavity_s_error),
        "full_context_centered_key_l2_change": py_float(full_key_delta.norm()),
        "full_context_S_max_abs_change": py_float(full_score_delta),
        "cavity_invariant": bool(py_float(cavity_c_error) <= 1.0e-12 and py_float(cavity_s_error) <= 1.0e-12),
        "full_context_changed_clearly": bool(py_float(full_score_delta) > 1.0e-7),
        "scope": "H0,Q,A fixed; V_i perturbed; routing is fixed; valid tokens only",
    }


def diagonal_mask_witness() -> dict[str, Any]:
    # Constructed witness supplied by root.  A[receiver, source] notation below
    # is represented as A[h, receiver, source] in the implementation.
    n, h, p, d = 4, 1, 3, 3
    dtype = torch.float64
    q0 = torch.tensor([1.0, -1.0, 0.0], dtype=dtype)
    b = torch.tensor([1.0, 1.0, -2.0], dtype=dtype)
    H0 = torch.zeros(n, d, dtype=dtype)
    H0[1] = b
    H0[2] = b
    Q = torch.zeros(n, h, p, dtype=dtype)
    Q[0, 0] = q0
    V = torch.zeros(n, h, p, dtype=dtype)
    V[0, 0] = 2.0 * q0
    V[3, 0] = 0.5 * q0
    Wo = torch.eye(3, dtype=dtype).unsqueeze(0)
    Wk = torch.eye(3, dtype=dtype).unsqueeze(0)
    A = torch.zeros(h, n, n, dtype=dtype)
    A[0, 0, 0] = 1.0
    A[0, 1, 0] = 1.0
    A[0, 2, 3] = 1.0
    A[0, 3, 3] = 1.0
    params = {
        "H0": H0,
        "Q": Q,
        "V": V,
        "Wo": Wo,
        "bo": torch.zeros(d, dtype=dtype),
        "Wk": Wk,
        "bk": torch.zeros(h, p, dtype=dtype),
        "gamma": torch.ones(d, dtype=dtype),
        "betaLN": torch.zeros(d, dtype=dtype),
        "logits": torch.zeros(h, n, n, dtype=dtype),
        "valid": torch.ones(n, dtype=torch.bool),
    }
    # The witness uses hard A; explicitly pass it by replacing logits with log A
    # only for this analytic calculation, while A itself is used below.
    u, z = compute_u_z(params, A)
    a = A.permute(0, 2, 1).masked_fill(torch.eye(n, dtype=torch.bool).unsqueeze(0), 0.0)
    C = z.unsqueeze(0) - torch.einsum("hij,ihd->ijd", a, u)
    ln_full = params["gamma"] * centered(z) / torch.sqrt(centered(z).square().mean(dim=-1, keepdim=True) + EPS) + params["betaLN"]
    ln_cavity = params["gamma"] * centered(C) / torch.sqrt(centered(C).square().mean(dim=-1, keepdim=True) + EPS) + params["betaLN"]
    full_score = torch.einsum("itp,tpd,jd->itj", Q, Wk, ln_full) / math.sqrt(p)
    cavity_score = torch.einsum("itp,tpd,ijd->itj", Q, Wk, ln_cavity) / math.sqrt(p)
    f1, f2 = full_score[0, 0, 1], full_score[0, 0, 2]
    c1, c2 = cavity_score[0, 0, 1], cavity_score[0, 0, 2]

    # Additive first-pass control: mask every self edge, while retaining the
    # source paths A[receiver=1,source=0] and A[receiver=2,source=3].
    # Cross edges on rows 0 and 3 keep those rows valid without diagonal mass.
    A_all_diag_masked = torch.zeros_like(A)
    A_all_diag_masked[0, 0, 3] = 1.0
    A_all_diag_masked[0, 1, 0] = 1.0
    A_all_diag_masked[0, 2, 3] = 1.0
    A_all_diag_masked[0, 3, 0] = 1.0
    u_variant, z_variant = compute_u_z(params, A_all_diag_masked)
    a_variant = A_all_diag_masked.permute(0, 2, 1).masked_fill(torch.eye(n, dtype=torch.bool).unsqueeze(0), 0.0)
    C_variant = z_variant.unsqueeze(0) - torch.einsum("hij,ihd->ijd", a_variant, u_variant)
    ln_full_variant = params["gamma"] * centered(z_variant) / torch.sqrt(centered(z_variant).square().mean(dim=-1, keepdim=True) + EPS) + params["betaLN"]
    ln_cavity_variant = params["gamma"] * centered(C_variant) / torch.sqrt(centered(C_variant).square().mean(dim=-1, keepdim=True) + EPS) + params["betaLN"]
    full_score_variant = torch.einsum("itp,tpd,jd->itj", Q, Wk, ln_full_variant) / math.sqrt(p)
    cavity_score_variant = torch.einsum("itp,tpd,ijd->itj", Q, Wk, ln_cavity_variant) / math.sqrt(p)
    vf1, vf2 = full_score_variant[0, 0, 1], full_score_variant[0, 0, 2]
    vc1, vc2 = cavity_score_variant[0, 0, 1], cavity_score_variant[0, 0, 2]
    variant_reversal = bool(vf1 > vf2 and vc1 < vc2)
    return {
        "construction": {
            "N": n,
            "H": h,
            "d": d,
            "p": p,
            "Wo": "I",
            "Wk": "I",
            "bo": 0,
            "gamma": 1,
            "betaLN": 0,
            "q0": q0.tolist(),
            "base_b": b.tolist(),
            "H0_1": b.tolist(),
            "H0_2": b.tolist(),
            "V0": (2.0 * q0).tolist(),
            "V3": (0.5 * q0).tolist(),
            "V1": [0.0, 0.0, 0.0],
            "V2": [0.0, 0.0, 0.0],
            "A_nonzero_receiver_source": [[0, 0], [1, 0], [2, 3], [3, 3]],
        },
        "query_i": 0,
        "compared_receivers_j": [1, 2],
        "full_context_scores": {"j1": py_float(f1), "j2": py_float(f2)},
        "cavity_scores": {"j1": py_float(c1), "j2": py_float(c2)},
        "full_context_ranking": "j1 > j2" if py_float(f1) > py_float(f2) else "j2 >= j1",
        "cavity_ranking": "j1 < j2" if py_float(c1) < py_float(c2) else "j2 <= j1",
        "ranking_reversal": bool(f1 > f2 and c1 < c2),
        "self_edges_in_A": {"receiver0_source0": py_float(A[0, 0, 0]), "receiver3_source3": py_float(A[0, 3, 3])},
        "hard_A_gradient": "not run by design",
        "all_diagonal_masked_variant": {
            "A_nonzero_receiver_source": [[0, 3], [1, 0], [2, 3], [3, 0]],
            "max_abs_diagonal_A": py_float(torch.diagonal(A_all_diag_masked[0], dim1=0, dim2=1).abs().max()),
            "full_context_scores": {"j1": py_float(vf1), "j2": py_float(vf2)},
            "cavity_scores": {"j1": py_float(vc1), "j2": py_float(vc2)},
            "full_context_ranking": "j1 > j2" if py_float(vf1) > py_float(vf2) else "j2 >= j1",
            "cavity_ranking": "j1 < j2" if py_float(vc1) < py_float(vc2) else "j2 <= j1",
            "ranking_reversal": variant_reversal,
            "all_self_edges_masked": bool(torch.diagonal(A_all_diag_masked[0], dim1=0, dim2=1).abs().max() == 0),
        },
    }


def no_ln_identity_check(generator: torch.Generator) -> dict[str, Any]:
    n, h, p, d = 5, 2, 3, 8
    params = random_params(n, h, d, p, torch.ones(n, dtype=torch.bool), torch.float64, generator)
    ref = literal_forward(params)
    q = params["Q"]
    w_k = params["Wk"]
    bk = params["bk"]
    z = ref["z"]
    c = ref["C"]
    full = torch.einsum("itp,tpd,jd->itj", q, w_k, z)
    cavity = torch.einsum("itp,tpd,ijd->itj", q, w_k, c)
    bias = torch.einsum("itp,tp->it", q, bk).unsqueeze(-1)
    full = (full + bias) / math.sqrt(p)
    cavity = (cavity + bias) / math.sqrt(p)
    wu = torch.einsum("tpd,ihd->ithp", w_k, ref["u"])
    direct_term = torch.einsum("itp,ithp->ith", q, wu)
    removed_term = torch.einsum("hij,ith->itj", ref["a"], direct_term) / math.sqrt(p)
    rhs = full - removed_term
    error = compare_scores(cavity, rhs, valid_pair_mask(params["valid"]))
    return {
        "error": error,
        "identity_holds": within_tolerance(error, 1.0e-12, 1.0e-12),
        "definition": "no-LN identity checked with direct linear keys and the same deleted message a[h,i,j]u[i,h]",
    }


def centered_score_rank_measurement(generator: torch.Generator) -> dict[str, Any]:
    """Numerical rank control for the independent-Q/K interface tensors.

    Rows are query tokens i and columns are source/receiver tokens j.  The
    source-axis mean is removed before SVD.  This is a diagnostic measurement,
    not a model claim or a novelty test.
    """
    n, h, p, d = 9, 1, 3, 3
    valid = torch.ones(n, dtype=torch.bool)
    # Use a correlated synthetic interface for this small rank control, as
    # requested by the root analysis: Q and V are projections of H0, and A is
    # produced from Q against a separately projected full-context K.  This is
    # intentionally separate from the independent-input parity suite above.
    H0 = torch.randn(n, d, generator=generator, dtype=torch.float64)
    Wq = torch.randn(p, d, generator=generator, dtype=torch.float64)
    Wv = torch.randn(p, d, generator=generator, dtype=torch.float64)
    Wk_for_logits = torch.randn(p, d, generator=generator, dtype=torch.float64)
    Q_flat = H0 @ Wq.transpose(0, 1)
    V_flat = H0 @ Wv.transpose(0, 1)
    bk = torch.randn(h, p, generator=generator, dtype=torch.float64)
    K_for_logits = H0 @ Wk_for_logits.transpose(0, 1) + bk[0]
    logits = (Q_flat @ K_for_logits.transpose(0, 1) / math.sqrt(p)).unsqueeze(0)
    params = random_params(n, h, d, p, valid, torch.float64, generator)
    params["H0"] = H0
    params["Q"] = Q_flat.unsqueeze(1)
    params["V"] = V_flat.unsqueeze(1)
    params["Wo"] = torch.eye(d, dtype=torch.float64).unsqueeze(0)
    params["Wk"] = Wk_for_logits.unsqueeze(0)
    params["bk"] = bk
    params["logits"] = logits
    params["bo"] = torch.zeros(d, dtype=torch.float64)
    params["gamma"] = torch.ones(d, dtype=torch.float64)
    params["betaLN"] = torch.zeros(d, dtype=torch.float64)
    literal = literal_forward(params)
    full = full_context_scores(params, literal["z"])
    cavity_matrix = literal["S"][:, 0, :]
    full_matrix = full[:, 0, :]
    cavity_centered = cavity_matrix - cavity_matrix.mean(dim=-1, keepdim=True)
    full_centered = full_matrix - full_matrix.mean(dim=-1, keepdim=True)
    cavity_svals = torch.linalg.svdvals(cavity_centered)
    full_svals = torch.linalg.svdvals(full_centered)
    threshold = 1.0e-8
    cavity_rank = int((cavity_svals > threshold).sum().item())
    full_rank = int((full_svals > threshold).sum().item())
    return {
        "N": n,
        "H": h,
        "d": d,
        "p": p,
        "dtype": "torch.float64",
        "row_definition": "score matrix [query i, source/receiver j] after subtracting each row's mean over j",
        "interface_note": "correlated synthetic Q/V/logits control; the core parity suite above still uses independent inputs as permitted by the specification",
        "correlated_construction": "H0 random; Q=H0 Wq^T; V=H0 Wv^T; K_for_logits=H0 Wk^T+bk; logits=Q K_for_logits^T/sqrt(p)",
        "threshold": threshold,
        "cavity_singular_values": [py_float(v) for v in cavity_svals],
        "full_context_singular_values": [py_float(v) for v in full_svals],
        "cavity_numerical_rank": cavity_rank,
        "full_context_numerical_rank": full_rank,
        "full_context_rank_bound_p": bool(full_rank <= p),
        "cavity_rank_exceeds_p": bool(cavity_rank > p),
        "interpretation_scope": "numerical control only; no BERT implementation or novelty claim",
    }


def make_constant_centered_case(generator: torch.Generator) -> dict[str, torch.Tensor]:
    n, h, p, d = 4, 2, 3, 8
    params = random_params(n, h, d, p, torch.ones(n, dtype=torch.bool), torch.float32, generator)
    wo_head = torch.randn(h, 1, p, generator=generator, dtype=torch.float32)
    params["Wo"] = wo_head.expand(h, d, p).clone()
    params["bo"] = torch.randn(1, generator=generator, dtype=torch.float32).expand(d).clone()
    params["H0"] = torch.randn(n, 1, generator=generator, dtype=torch.float32).expand(n, d).clone()
    return params


def make_near_removed_case(generator: torch.Generator) -> dict[str, torch.Tensor]:
    n, h, p, d = 4, 2, 3, 8
    params = random_params(n, h, d, p, torch.ones(n, dtype=torch.bool), torch.float32, generator)
    a_matrix = softmax_rows(params["logits"], params["valid"])
    u, _ = compute_u_z(params, a_matrix)
    source_i, receiver_j = 0, 1
    message_i = torch.einsum("h,hd->d", a_matrix[:, receiver_j, source_i], u[source_i])
    all_message_j = torch.einsum("hk,khd->d", a_matrix[:, receiver_j], u)
    other_message_j = all_message_j - message_i
    near_factor = 1.0e-5
    params["H0"][receiver_j] = near_factor * message_i - params["bo"] - other_message_j
    return params


def make_scaled_near_removed_case(generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Cancellation stress with large source messages and a compensated H0 row."""
    n, h, p, d = 4, 2, 3, 8
    params = random_params(n, h, d, p, torch.ones(n, dtype=torch.bool), torch.float32, generator)
    params["Wo"] = params["Wo"] * 1000.0
    params["V"] = params["V"] * 1000.0
    a_matrix = softmax_rows(params["logits"], params["valid"])
    u, _ = compute_u_z(params, a_matrix)
    source_i, receiver_j = 0, 1
    message_i = torch.einsum("h,hd->d", a_matrix[:, receiver_j, source_i], u[source_i])
    all_message_j = torch.einsum("hk,khd->d", a_matrix[:, receiver_j], u)
    other_message_j = all_message_j - message_i
    near_factor = 1.0e-5
    # Q/Wk/gamma stay at their base scale; only source messages are enlarged
    # before H0 is set to compensate them.
    params["H0"][receiver_j] = near_factor * message_i - params["bo"] - other_message_j
    return params


def fp32_diagnostic(name: str, params: dict[str, torch.Tensor]) -> dict[str, Any]:
    ref = literal_forward(params)
    fact = factorized_forward(params)
    error = compare_scores(fact["S"], ref["S"], valid_pair_mask(params["valid"]))
    z = ref["z"]
    fz = fact["z"]
    negative_vnorm_count = int((fact["vnorm"] < 0).sum().item())
    near_zero_vnorm_count = int((fact["vnorm"].abs() < 1.0e-8).sum().item())
    strict_status = within_tolerance(error, FP64_ABS_TOL, FP64_REL_TOL)
    return {
        "name": name,
        "dtype": str(params["H0"].dtype),
        "shape": {key: tensor_shape(params[key]) for key in ("H0", "Q", "V", "Wo", "Wk", "logits")},
        "score_error": error,
        "strict_float64_tolerance_status": "pass" if strict_status else "fail",
        "arithmetic_classification": "FP32 diagnostic; raw mismatch retained; no tolerance adjustment",
        "negative_vnorm_count": negative_vnorm_count,
        "nearzero_vnorm_count": near_zero_vnorm_count,
        "vnorm_min": py_float(fact["vnorm"].min()),
        "vnorm_max": py_float(fact["vnorm"].max()),
        "z_finite": tensor_finite(z),
        "factorized_z_finite": tensor_finite(fz),
    }


def write_report(results: dict[str, Any]) -> None:
    f64 = results["status"]
    lines = [
        "# Cavity key algebra audit",
        "",
        "This report records a synthetic CPU Torch audit of the literal and factorized formulas in `docs/CAVITY_KEY_ALGEBRA_SPEC.md`. No dataset, checkpoint, optimizer, training, or GPU path was used.",
        "",
        f"Seed: `{results['seed']}`. Torch: `{results['environment']['torch_version']}`. Device: `{results['environment']['device']}`.",
        "",
        "## Float64 checks",
        "",
        f"- Forward parity: **{f64['float64_forward']}** across {len(results['float64_forward_cases'])} random shape cases (`N` in 1/2/5/9 and `H` in 1/2/3). Tolerances were abs `1e-8`, rel `1e-6`.",
        f"- Backward parity: **{f64['float64_backward']}** for gradients of H0, Q, V, Wo, bo, Wk, bk, gamma, betaLN, and A-logits.",
        f"- Token permutation equivariance: **{f64['permutation_equivariance']}**.",
        f"- Partial-path exclusion: **{f64['partial_path_exclusion']}**; the full-context control changed as expected for the selected receiver.",
        f"- No-LN limiting identity: **{f64['no_ln_identity']}**.",
        f"- Centered score rank control: cavity rank `{results['centered_score_rank_measurement']['cavity_numerical_rank']}` and full-context rank `{results['centered_score_rank_measurement']['full_context_numerical_rank']}` at threshold `1e-8`; this is a numerical control only.",
        "",
        "The factorized implementation reports logical intermediate shapes and the derived largest logical element count. This is not a peak VRAM measurement.",
        "",
        "## Constructed diagonal-mask witness",
        "",
        "For the supplied N=4, H=1, d=p=3 construction, the saved values show full-context `j1 > j2` and cavity `j1 < j2` while receiver 0 and receiver 3 retain self edges in A. The hard A witness does not run a gradient check.",
        f"The ranking reversal check (including the all-diagonal-masked additive variant) is **{results['status']['diagonal_mask_witness']}**. Values are in `results.json`.",
        "",
        "## Float32 diagnostics",
        "",
        "Random scale, near-removed-message, and constant-centered-row cases are reported as arithmetic diagnostics. Their raw absolute/relative mismatches, negative/near-zero variance counts, and finite checks are retained. Any failure against the strict Float64 thresholds is recorded as a failure; no tolerance was widened to make a case pass.",
        "",
        "| case | max abs | max rel | strict Float64 threshold | negative vnorm | near-zero vnorm |",
        "|---|---:|---:|:---:|---:|---:|",
    ]
    for case in results["fp32_diagnostics"]:
        err = case["score_error"]
        lines.append(
            f"| {case['name']} | {err['max_abs']:.6g} | {err['max_rel']:.6g} | {case['strict_float64_tolerance_status']} | {case['negative_vnorm_count']} | {case['nearzero_vnorm_count']} |"
        )
    lines.extend(
        [
            "",
            "The mandatory formula/parity conclusions above are separated from Float32 arithmetic behavior. See `results.json` for all case shapes, errors, witnesses, and intermediate metadata.",
            "",
        ]
    )
    (ARTIFACT_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    forward_cases: list[dict[str, Any]] = []
    backward_cases: list[dict[str, Any]] = []
    case_index = 0
    for n in (1, 2, 5, 9):
        for h in (1, 2, 3):
            p = 2 + ((n + h) % 2)
            d = h * p + (1 if (n + h) % 2 == 0 else 0)
            valid = make_valid_mask(n)
            gen = torch.Generator(device="cpu").manual_seed(SEED + case_index)
            params = random_params(n, h, d, p, valid, torch.float64, gen)
            forward_cases.append(forward_case_summary(f"N{n}_H{h}_d{d}_p{p}", params))
            backward_gen = torch.Generator(device="cpu").manual_seed(SEED + 1000 + case_index)
            backward_cases.append(
                backward_case_summary(f"N{n}_H{h}_d{d}_p{p}", params, backward_gen)
            )
            case_index += 1

    perm_gen = torch.Generator(device="cpu").manual_seed(SEED + 2000)
    n, h, p, d = 5, 2, 3, 8
    valid = torch.tensor([True, True, True, True, False])
    perm_params = random_params(n, h, d, p, valid, torch.float64, perm_gen)
    perm = torch.tensor([2, 4, 0, 3, 1], dtype=torch.long)
    base_ref = literal_forward(perm_params)
    base_fact = factorized_forward(perm_params)
    synced = {key: value.clone() for key, value in perm_params.items()}
    for key in ("H0", "Q", "V"):
        synced[key] = perm_params[key].index_select(0, perm)
    synced["valid"] = perm_params["valid"].index_select(0, perm)
    synced["logits"] = perm_params["logits"].index_select(1, perm).index_select(2, perm)
    perm_ref = literal_forward(synced)
    perm_fact = factorized_forward(synced)
    expected_ref = base_ref["S"].index_select(0, perm).index_select(2, perm)
    expected_fact = base_fact["S"].index_select(0, perm).index_select(2, perm)
    perm_pair = valid_pair_mask(synced["valid"])
    perm_ref_error = compare_scores(perm_ref["S"], expected_ref, perm_pair)
    perm_fact_error = compare_scores(perm_fact["S"], expected_fact, perm_pair)
    permutation_result = {
        "permutation_new_to_old": perm.tolist(),
        "reference_error": perm_ref_error,
        "factorized_error": perm_fact_error,
        "equivariant": bool(
            within_tolerance(perm_ref_error, FP64_ABS_TOL, FP64_REL_TOL)
            and within_tolerance(perm_fact_error, FP64_ABS_TOL, FP64_REL_TOL)
        ),
    }

    partial_result = partial_path_check(torch.Generator(device="cpu").manual_seed(SEED + 3000))
    witness_result = diagonal_mask_witness()
    identity_result = no_ln_identity_check(torch.Generator(device="cpu").manual_seed(SEED + 4000))
    rank_result = centered_score_rank_measurement(torch.Generator(device="cpu").manual_seed(SEED + 4500))

    fp32_cases: list[dict[str, Any]] = []
    for scale in (1.0e-3, 1.0, 1.0e3):
        gen = torch.Generator(device="cpu").manual_seed(SEED + 5000 + len(fp32_cases))
        params = random_params(5, 2, 8, 3, torch.ones(5, dtype=torch.bool), torch.float32, gen, scale=scale)
        fp32_cases.append(fp32_diagnostic(f"random_scale_{scale:g}", params))
    near_params = make_near_removed_case(torch.Generator(device="cpu").manual_seed(SEED + 5010))
    fp32_cases.append(fp32_diagnostic("near_removed_message", near_params))
    scaled_near_params = make_scaled_near_removed_case(torch.Generator(device="cpu").manual_seed(SEED + 5012))
    fp32_cases.append(fp32_diagnostic("scaled_near_removed_message", scaled_near_params))
    constant_params = make_constant_centered_case(torch.Generator(device="cpu").manual_seed(SEED + 5011))
    fp32_cases.append(fp32_diagnostic("constant_centered_rows", constant_params))

    f64_forward_ok = all(case["within_float64_tolerance"] for case in forward_cases)
    f64_backward_ok = all(case["within_float64_tolerance"] for case in backward_cases)
    partial_ok = bool(partial_result["cavity_invariant"] and partial_result["full_context_changed_clearly"])
    identity_ok = bool(identity_result["identity_holds"])
    witness_ok = bool(
        witness_result["ranking_reversal"]
        and witness_result["all_diagonal_masked_variant"]["ranking_reversal"]
        and witness_result["all_diagonal_masked_variant"]["all_self_edges_masked"]
    )
    perm_ok = bool(permutation_result["equivariant"])
    results: dict[str, Any] = {
        "schema_version": 1,
        "seed": SEED,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "device": "cpu",
            "torch_num_threads": torch.get_num_threads(),
        },
        "dtypes": {"float64": "torch.float64", "float32": "torch.float32"},
        "tolerances": {
            "float64_forward_abs": FP64_ABS_TOL,
            "float64_forward_rel": FP64_REL_TOL,
            "float64_backward_abs": FP64_ABS_TOL,
            "float64_backward_rel": FP64_REL_TOL,
            "fp32_reporting": "strict Float64 thresholds are shown unchanged; no FP32 tolerance was widened",
        },
        "status": {
            "float64_forward": "pass" if f64_forward_ok else "fail",
            "float64_backward": "pass" if f64_backward_ok else "fail",
            "permutation_equivariance": "pass" if perm_ok else "fail",
            "partial_path_exclusion": "pass" if partial_ok else "fail",
            "no_ln_identity": "pass" if identity_ok else "fail",
            "diagonal_mask_witness": "pass" if witness_ok else "fail",
            "fp32_strict_float64_threshold_cases": sum(
                case["strict_float64_tolerance_status"] == "fail" for case in fp32_cases
            ),
            "mandatory_float64_and_controls": "pass"
            if all((f64_forward_ok, f64_backward_ok, perm_ok, partial_ok, identity_ok, witness_ok))
            else "fail",
            "overall": "pass_with_fp32_arithmetic_diagnostics"
            if all((f64_forward_ok, f64_backward_ok, perm_ok, partial_ok, identity_ok, witness_ok))
            else "fail",
        },
        "float64_forward_cases": forward_cases,
        "float64_backward_cases": backward_cases,
        "permutation_equivariance": permutation_result,
        "partial_path_exclusion": partial_result,
        "diagonal_mask_witness": witness_result,
        "no_ln_identity": identity_result,
        "centered_score_rank_measurement": rank_result,
        "fp32_diagnostics": fp32_cases,
        "sourceSHA": {
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "spec_sha256": sha256_file(SPEC_PATH),
            "git_commit": None,
            "git_note": "repository metadata unavailable in this workspace checkout",
        },
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = ARTIFACT_DIR / "results.json"
    result_path.write_text(json.dumps(json_safe(results), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    write_report(results)
    return results


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["status"], sort_keys=True))
