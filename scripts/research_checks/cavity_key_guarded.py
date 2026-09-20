#!/usr/bin/env python3
"""Guarded numerical evaluation of the same cavity-key operator.

The original audit remains unchanged.  This script imports its reference
implementation and deterministic input constructors, then evaluates the
factorized fast path with a bounded-workspace literal fallback for flagged
pairs.  It does not train, benchmark a model, or access data/checkpoints.
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
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
RESEARCH_CHECK_DIR = ROOT / "scripts" / "research_checks"
if str(RESEARCH_CHECK_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_CHECK_DIR))
import cavity_key_algebra as old  # noqa: E402


ARTIFACT_DIR = ROOT / "artifacts" / "research_checks" / "cavity_key_guarded"
EPS = old.EPS
SEED = old.SEED
GUARD_REL = 1.0e-3
GUARD_ABS = 1.0e-7
CHUNK_PAIRS = 64
FP64_ABS_TOL = 1.0e-8
FP64_REL_NORM_TOL = 1.0e-6
FP32_ELEMENTWISE_ABS = 1.0e-5
FP32_ELEMENTWISE_REL = 1.0e-4


def py_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def tensor_shape(value: torch.Tensor) -> list[int]:
    return [int(v) for v in value.shape]


def tensor_finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return py_float(value)
        return [strict_json(v) for v in value.detach().cpu().tolist()]
    if isinstance(value, dict):
        return {str(k): strict_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


def pair_mask(params: dict[str, torch.Tensor]) -> torch.Tensor:
    valid = params["valid"]
    return valid[:, None] & valid[None, :]


def pair_view(scores: torch.Tensor) -> torch.Tensor:
    """Convert [i,t,j] to [i,j,t] for pair masking and chunking."""
    return scores.permute(0, 2, 1)


def finite_minmax(value: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(value)
    finite_values = value[finite]
    if finite_values.numel() == 0:
        return {"finite": False, "min": None, "max": None, "nonfinite_count": int((~finite).sum().item())}
    return {
        "finite": bool(finite.all().item()),
        "min": py_float(finite_values.min()),
        "max": py_float(finite_values.max()),
        "nonfinite_count": int((~finite).sum().item()),
    }


def literal_scores_for_pairs(
    params: dict[str, torch.Tensor],
    u: torch.Tensor,
    z: torch.Tensor,
    a: torch.Tensor,
    pair_indices: torch.Tensor,
    chunk_size: int = CHUNK_PAIRS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute literal C/LN/key scores for pair chunks only.

    Returns scores [M,H] and literal variance [M].  At no point does this
    construct an N,N,d or N,N,H,p tensor.
    """
    score_chunks: list[torch.Tensor] = []
    variance_chunks: list[torch.Tensor] = []
    d = z.shape[-1]
    for start in range(0, pair_indices.shape[0], chunk_size):
        chunk = pair_indices[start : start + chunk_size]
        ii, jj = chunk[:, 0], chunk[:, 1]
        pair_a = a[:, ii, jj].transpose(0, 1)  # [M,H]
        pair_u = u[ii]  # [M,H,d]
        removed = torch.einsum("mh,mhd->md", pair_a, pair_u)
        cavity = z[jj] - removed  # [M,d]
        centered = cavity - cavity.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1)
        ln_cavity = F.layer_norm(
            cavity,
            (d,),
            weight=params["gamma"],
            bias=params["betaLN"],
            eps=EPS,
        )
        keys = torch.einsum("tpd,md->mtp", params["Wk"], ln_cavity)
        keys = keys + params["bk"].unsqueeze(0)
        scores = (params["Q"][ii] * keys).sum(dim=-1) / math.sqrt(params["Q"].shape[-1])
        score_chunks.append(scores)
        variance_chunks.append(variance)
    if not score_chunks:
        empty_scores = z.new_empty((0, params["Q"].shape[1]))
        return empty_scores, z.new_empty((0,))
    return torch.cat(score_chunks, dim=0), torch.cat(variance_chunks, dim=0)


def guarded_forward(params: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Run fast factorized scores and literal fallback on numerically unsafe pairs."""
    fast = old.factorized_forward(params, eps=EPS)
    n = params["H0"].shape[0]
    h = params["Q"].shape[1]
    fast_pairs = pair_view(fast["S"]).reshape(n * n, h)
    aa = fast["a"].permute(1, 2, 0)  # [i,j,h]
    ww = fast["w"].square().sum(dim=-1)  # [j]
    gram_abs_term = torch.einsum("ijh,ijg,ihg->ij", aa.abs(), aa.abs(), fast["ee"].abs())
    # Absolute terms form the conservative cancellation scale E specified by root;
    # the absolute value is applied per (h,g) term before summation.
    e_scale = (
        ww.unsqueeze(0)
        + 2.0 * (aa * fast["we"]).abs().sum(dim=-1)
        + gram_abs_term
    ) / fast["z"].shape[-1]
    variance = fast["vnorm"]
    nonfinite_pair = (~torch.isfinite(variance)) | (~torch.isfinite(fast_pairs).all(dim=-1).reshape(n, n))
    flagged = (variance <= GUARD_REL * e_scale + GUARD_ABS) | nonfinite_pair
    flagged_flat = flagged.reshape(-1)
    flagged_indices = torch.nonzero(flagged_flat, as_tuple=False).flatten()
    pair_grid = torch.cartesian_prod(torch.arange(n), torch.arange(n))
    if pair_grid.ndim == 1:
        pair_grid = pair_grid.reshape(1, 2)

    literal_flagged_scores, literal_flagged_variance = literal_scores_for_pairs(
        params,
        fast["u"],
        fast["z"],
        fast["a"],
        pair_grid[flagged_flat],
    )

    # Build selected rows explicitly. A flagged row has no selected dependency
    # on its fast value; an unflagged row keeps the factorized graph. Finite
    # gradients were verified on the tested cases; this does not establish
    # immunity to NaN/Inf overflow in untested cases.
    if flagged_indices.numel() == 0:
        selected_pairs = fast_pairs
    else:
        flagged_set = set(int(v) for v in flagged_indices.detach().cpu().tolist())
        literal_cursor = 0
        selected_rows: list[torch.Tensor] = []
        for flat_index in range(n * n):
            if flat_index in flagged_set:
                selected_rows.append(literal_flagged_scores[literal_cursor : literal_cursor + 1])
                literal_cursor += 1
            else:
                selected_rows.append(fast_pairs[flat_index : flat_index + 1])
        selected_pairs = torch.cat(selected_rows, dim=0)
    guarded_scores = selected_pairs.reshape(n, n, h).permute(0, 2, 1)

    guarded_variance = variance.detach().clone()
    if flagged_indices.numel():
        guarded_variance_flat = guarded_variance.reshape(-1)
        guarded_variance_flat[flagged_indices] = literal_flagged_variance.detach()
        guarded_variance = guarded_variance_flat.reshape(n, n)
    return {
        "S": guarded_scores,
        "fast_S": fast["S"],
        "fast": fast,
        "E": e_scale,
        "flagged": flagged,
        "flagged_indices": flagged_indices,
        "flagged_count": int(flagged.sum().item()),
        "literal_flagged_scores": literal_flagged_scores,
        "literal_flagged_variance": literal_flagged_variance,
        "guarded_variance": guarded_variance,
        "chunk_size": CHUNK_PAIRS,
    }


def raw_error(actual: torch.Tensor, reference: torch.Tensor, pairs: torch.Tensor) -> dict[str, Any]:
    error = old.compare_scores(actual, reference, pairs)
    error["relative_l2"] = error["relative_norm"]
    return error


def elementwise_closeness(actual: torch.Tensor, reference: torch.Tensor, pairs: torch.Tensor) -> dict[str, Any]:
    actual_pairs = pair_view(actual)[pairs]
    reference_pairs = pair_view(reference)[pairs]
    bound = FP32_ELEMENTWISE_ABS + FP32_ELEMENTWISE_REL * reference_pairs.abs()
    diff = (actual_pairs - reference_pairs).abs()
    close = diff <= bound
    return {
        "criterion": "abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)",
        "pass": bool(close.all().item()),
        "close_count": int(close.sum().item()),
        "total_count": int(close.numel()),
        "failure_count": int((~close).sum().item()),
        "max_excess": py_float((diff - bound).max().clamp_min(0.0)) if diff.numel() else 0.0,
    }


def diagnostic_case(name: str, params: dict[str, torch.Tensor]) -> dict[str, Any]:
    literal = old.literal_forward(params, eps=EPS)
    guarded = guarded_forward(params)
    pairs = pair_mask(params)
    fast_error = raw_error(guarded["fast_S"], literal["S"], pairs)
    guarded_error = raw_error(guarded["S"], literal["S"], pairs)
    pair_flags = guarded["flagged"] & pairs
    literal_var = literal["variance"]
    fast_var = guarded["fast"]["vnorm"]
    selected_var = guarded["guarded_variance"]
    return {
        "name": name,
        "dtype": str(params["H0"].dtype),
        "shape": {key: tensor_shape(params[key]) for key in ("H0", "Q", "V", "Wo", "Wk", "logits")},
        "valid_pair_count": int(pairs.sum().item()),
        "total_pair_count": int(pairs.numel()),
        "flag_count_total": guarded["flagged_count"],
        "flag_count_valid": int(pair_flags.sum().item()),
        "flag_count_nonfinite": int((guarded["flagged"] & ((~torch.isfinite(fast_var)) | (~torch.isfinite(pair_view(guarded["fast_S"])).all(dim=-1)))).sum().item()),
        "flag_count_threshold": int((guarded["flagged"] & ~((~torch.isfinite(fast_var)) | (~torch.isfinite(pair_view(guarded["fast_S"])).all(dim=-1)))).sum().item()),
        "guard_threshold": {"relative": GUARD_REL, "absolute": GUARD_ABS, "expression": "vnorm <= 1e-3*E + 1e-7 or nonfinite variance/score"},
        "literal_score_error_vs_reference": {"max_abs": 0.0, "relative_l2": 0.0, "finite": tensor_finite(literal["S"])},
        "fast_score_error_vs_literal": fast_error,
        "guarded_score_error_vs_literal": guarded_error,
        "fast_finite": tensor_finite(guarded["fast_S"]),
        "guarded_finite": tensor_finite(guarded["S"]),
        "literal_variance": finite_minmax(literal_var),
        "fast_variance": finite_minmax(fast_var),
        "guarded_variance_used": finite_minmax(selected_var),
        "elementwise_closeness_fast": elementwise_closeness(guarded["fast_S"], literal["S"], pairs),
        "elementwise_closeness_guarded": elementwise_closeness(guarded["S"], literal["S"], pairs),
        "fallback_chunk_size_pairs": guarded["chunk_size"],
        "composition": "explicit selected-row concatenation; flagged rows use literal gradients, unflagged rows use fast gradients",
    }


def make_float64_near_removed(generator: torch.Generator) -> dict[str, torch.Tensor]:
    n, h, p, d = 4, 2, 3, 8
    params = old.random_params(n, h, d, p, torch.ones(n, dtype=torch.bool), torch.float64, generator)
    a_matrix = old.softmax_rows(params["logits"], params["valid"])
    u, _ = old.compute_u_z(params, a_matrix)
    source_i, receiver_j = 0, 1
    message_i = torch.einsum("h,hd->d", a_matrix[:, receiver_j, source_i], u[source_i])
    all_message_j = torch.einsum("hk,khd->d", a_matrix[:, receiver_j], u)
    params["H0"][receiver_j] = 1.0e-12 * message_i - params["bo"] - (all_message_j - message_i)
    return params


def gradient_parity_case(name: str, params: dict[str, torch.Tensor], cotangent_seed: int) -> dict[str, Any]:
    ref_params = old.clone_leaves(params, requires_grad=True)
    guard_params = old.clone_leaves(params, requires_grad=True)
    literal = old.literal_forward(ref_params, eps=EPS)
    guarded = guarded_forward(guard_params)
    pairs = pair_mask(params)
    cotangent = torch.randn(
        literal["S"].shape,
        generator=torch.Generator(device="cpu").manual_seed(cotangent_seed),
        dtype=literal["S"].dtype,
    )
    cotangent = cotangent * pairs[:, None, :]
    ref_loss = (literal["S"] * cotangent).sum()
    guard_loss = (guarded["S"] * cotangent).sum()
    names = ("H0", "Q", "V", "Wo", "bo", "Wk", "bk", "gamma", "betaLN", "logits")
    ref_grads = torch.autograd.grad(ref_loss, tuple(ref_params[k] for k in names), allow_unused=False)
    guard_grads = torch.autograd.grad(guard_loss, tuple(guard_params[k] for k in names), allow_unused=False)
    gradient_errors: dict[str, Any] = {}
    all_ok = True
    zero_grad: dict[str, Any] = {}
    for parameter_name, ref_grad, guard_grad in zip(names, ref_grads, guard_grads):
        error = old.compare_tensor(guard_grad, ref_grad)
        error["relative_l2"] = error["relative_norm"]
        error["within_tolerance"] = bool(
            error["finite"]
            and error["max_abs"] <= FP64_ABS_TOL
            and error["relative_norm"] <= FP64_REL_NORM_TOL
        )
        error["reference_gradient_norm"] = py_float(ref_grad.norm())
        error["guarded_gradient_norm"] = py_float(guard_grad.norm())
        gradient_errors[parameter_name] = error
        all_ok = all_ok and error["within_tolerance"]
        ref_zero = bool(ref_grad.norm() == 0)
        guard_zero = bool(guard_grad.norm() == 0)
        zero_grad[parameter_name] = {
            "reference_zero": ref_zero,
            "guarded_zero": guard_zero,
            "zero_status_match": ref_zero == guard_zero,
        }
    score_error = raw_error(guarded["S"], literal["S"], pairs)
    score_ok = bool(score_error["finite"] and score_error["max_abs"] <= FP64_ABS_TOL and score_error["relative_norm"] <= FP64_REL_NORM_TOL)
    return {
        "name": name,
        "dtype": "torch.float64",
        "flag_count_total": guarded["flagged_count"],
        "flag_count_valid": int((guarded["flagged"] & pairs).sum().item()),
        "score_error_guarded_vs_literal": score_error,
        "score_within_tolerance": score_ok,
        "loss_reference": py_float(ref_loss),
        "loss_guarded": py_float(guard_loss),
        "gradient_errors": gradient_errors,
        "gradient_within_tolerance": all_ok,
        "zero_grad_report": zero_grad,
        "all_gradients_finite": all(error["finite"] for error in gradient_errors.values()),
        "composition": "same ten gradient groups as the original audit; no optimizer step",
    }


def build_input_provenance() -> list[dict[str, Any]]:
    return [
        {"name": "random_scale_0.001", "kind": "old.random_params", "seed": SEED + 5000, "scale": 1.0e-3},
        {"name": "random_scale_1", "kind": "old.random_params", "seed": SEED + 5001, "scale": 1.0},
        {"name": "random_scale_1000", "kind": "old.random_params", "seed": SEED + 5002, "scale": 1.0e3},
        {"name": "near_removed_message", "kind": "old.make_near_removed_case", "seed": SEED + 5010},
        {"name": "scaled_near_removed_message", "kind": "old.make_scaled_near_removed_case", "seed": SEED + 5012},
        {"name": "constant_centered_rows", "kind": "old.make_constant_centered_case", "seed": SEED + 5011},
    ]


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    old_results_path = ROOT / "artifacts" / "research_checks" / "cavity_key_algebra" / "results.json"
    old_results = json.loads(old_results_path.read_text(encoding="utf-8"))
    old_script_path = ROOT / "scripts" / "research_checks" / "cavity_key_algebra.py"
    audit_script_sha = sha256_file(old_script_path)
    audit_results_sha = sha256_file(old_results_path)
    source_audit_sha = old_results["sourceSHA"]["script_sha256"]
    source_audit_unchanged = bool(audit_script_sha == source_audit_sha)

    fp32_params: list[tuple[str, dict[str, torch.Tensor]]] = []
    for scale, seed in ((1.0e-3, SEED + 5000), (1.0, SEED + 5001), (1.0e3, SEED + 5002)):
        fp32_params.append(
            (
                f"random_scale_{scale:g}",
                old.random_params(5, 2, 8, 3, torch.ones(5, dtype=torch.bool), torch.float32, torch.Generator(device="cpu").manual_seed(seed), scale=scale),
            )
        )
    fp32_params.append(("near_removed_message", old.make_near_removed_case(torch.Generator(device="cpu").manual_seed(SEED + 5010))))
    fp32_params.append(("scaled_near_removed_message", old.make_scaled_near_removed_case(torch.Generator(device="cpu").manual_seed(SEED + 5012))))
    fp32_params.append(("constant_centered_rows", old.make_constant_centered_case(torch.Generator(device="cpu").manual_seed(SEED + 5011))))
    fp32_results = [diagnostic_case(name, params) for name, params in fp32_params]

    f64_params = [
        (
            "wellconditioned_N5_H2",
            old.random_params(5, 2, 8, 3, torch.ones(5, dtype=torch.bool), torch.float64, torch.Generator(device="cpu").manual_seed(SEED + 6000)),
            SEED + 7000,
        ),
        (
            "wellconditioned_N2_H1",
            old.random_params(2, 1, 3, 3, torch.ones(2, dtype=torch.bool), torch.float64, torch.Generator(device="cpu").manual_seed(SEED + 6001)),
            SEED + 7001,
        ),
        (
            "near_cancellation_float64",
            make_float64_near_removed(torch.Generator(device="cpu").manual_seed(SEED + 6002)),
            SEED + 7002,
        ),
    ]
    f64_results = [gradient_parity_case(name, params, cotangent_seed) for name, params, cotangent_seed in f64_params]

    fp32_guarded_ok = all(case["elementwise_closeness_guarded"]["pass"] for case in fp32_results)
    fp64_forward_ok = all(case["score_within_tolerance"] for case in f64_results)
    fp64_backward_ok = all(case["gradient_within_tolerance"] for case in f64_results)
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
        "algorithm": {
            "guard_expression": "vnorm <= 1e-3*E + 1e-7 or nonfinite variance/score",
            "E_expression": "(||w_j||^2 + 2 sum_h abs(a_ijh <w_j,e_ih>) + sum_hg abs(a_ijh)*abs(a_ijg)*abs(<e_ih,e_ig>))/d",
            "chunk_max_pairs": CHUNK_PAIRS,
            "gradient_selection": "explicit selected-row concatenation; finite gradients verified only on tested cases",
            "nonfinite_overflow_recovery": "not_verified; NaN/Inf immunity is not claimed",
            "same_operator": True,
            "model_or_training": False,
        },
        "tolerances": {
            "fp64_abs": FP64_ABS_TOL,
            "fp64_relative_l2": FP64_REL_NORM_TOL,
            "fp32_elementwise_abs": FP32_ELEMENTWISE_ABS,
            "fp32_elementwise_relative": FP32_ELEMENTWISE_REL,
            "guard_relative": GUARD_REL,
            "guard_absolute": GUARD_ABS,
        },
        "status": {
            "source_audit_unchanged": source_audit_unchanged,
            "fp32_guarded_elementwise": "pass" if fp32_guarded_ok else "fail",
            "fp64_forward_guarded": "pass" if fp64_forward_ok else "fail",
            "fp64_backward_guarded": "pass" if fp64_backward_ok else "fail",
            "guarded_gradients_finite": all(case["all_gradients_finite"] for case in f64_results),
            "nonfinite_overflow_recovery": "not_verified",
            "overall": "pass" if source_audit_unchanged and fp32_guarded_ok and fp64_forward_ok and fp64_backward_ok else "fail",
        },
        "input_seed_provenance": build_input_provenance(),
        "fp32_diagnostics": fp32_results,
        "fp64_gradient_parity": f64_results,
        "sourceSHA": {
            "guarded_script_sha256": sha256_file(Path(__file__).resolve()),
            "source_audit_script_sha256": audit_script_sha,
            "source_audit_results_sha256": audit_results_sha,
            "source_audit_recorded_script_sha256": source_audit_sha,
            "source_audit_unchanged": source_audit_unchanged,
            "git_commit": None,
            "git_note": "repository metadata unavailable in this workspace checkout",
        },
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "results.json").write_text(
        json.dumps(strict_json(results), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# Guarded cavity key evaluation",
        "",
        "This is a CPU synthetic numerical evaluation of the same cavity-key operator. It imports the prior audit's reference implementation and deterministic input constructors. No model, dataset, checkpoint, optimizer, training, or GPU path is used.",
        "",
        f"Seed: `{SEED}`. Torch: `{torch.__version__}`. Source audit unchanged: **{source_audit_unchanged}**.",
        "",
        "## Guard",
        "",
        "Pairs are flagged when `vnorm <= 1e-3*E + 1e-7` or variance/fast score is nonfinite. Flagged pairs use literal cavity construction and `layer_norm` in chunks of at most 64 pairs; unflagged pairs use the factorized fast path. Selected rows are assembled explicitly. Finite gradients were verified on the tested cases; nonfinite overflow recovery and NaN immunity are not verified or claimed.",
        "",
        "## Results",
        "",
        f"- Six prior FP32 diagnostics: guarded elementwise criterion **{results['status']['fp32_guarded_elementwise']}**; raw fast and guarded errors remain in `results.json`.",
        f"- Float64 forward parity: **{results['status']['fp64_forward_guarded']}** on two well-conditioned and one near-cancellation case.",
        f"- Float64 backward parity: **{results['status']['fp64_backward_guarded']}** for H0, Q, V, Wo, bo, Wk, bk, gamma, betaLN, and A-logits.",
        "",
        "| case | flagged valid/total | fast max abs | fast relative L2 | guarded max abs | guarded relative L2 | guarded elementwise |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for case in fp32_results:
        fast = case["fast_score_error_vs_literal"]
        guarded = case["guarded_score_error_vs_literal"]
        close = case["elementwise_closeness_guarded"]["pass"]
        report_lines.append(
            f"| {case['name']} | {case['flag_count_valid']}/{case['valid_pair_count']} | {fast['max_abs']:.6g} | {fast['relative_l2']:.6g} | {guarded['max_abs']:.6g} | {guarded['relative_l2']:.6g} | {close} |"
        )
    report_lines.extend(
        [
            "",
            "The original strict FP32 failures are preserved in the fast columns. Guarded columns report the same operator with the bounded-workspace fallback; this does not claim universal numerical stability or guaranteed fast runtime.",
            "",
            "Zero-gradient status and per-parameter Float64 errors are recorded in `results.json`.",
            "",
        ]
    )
    (ARTIFACT_DIR / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    return results


if __name__ == "__main__":
    output = run()
    print(json.dumps(output["status"], sort_keys=True))
