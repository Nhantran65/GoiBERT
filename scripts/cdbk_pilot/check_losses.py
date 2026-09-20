"""Small synthetic checks of the pilot loss contract before arm K is launched.

Checks (CPU, fp32, fixed seed): KD(t,t)=0; KD>0 for t!=s; KD equals a hand-written T**2 * KL(teacher||student)
averaged over the 12 slots; direction (KL(t||s) != KL(s||t) on asymmetric inputs); gradient reaches the student
only; slot layout ([batch, 12, 4]: a change in one slot moves KD by 1/12 of that slot's KL); CE includes the absent
class 0 and equals E1 compute_loss; combined total = 0.5 CE + 0.5 KD.  Writes artifacts/cdbk_pilot/checks/loss_checks.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import CHECK_ROOT, KD_ALPHA, KD_TEMPERATURE, e1_train  # noqa: E402

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from goibert_e2.losses import logits_kd_loss, supervised_cross_entropy  # noqa: E402


def manual_kd(student: torch.Tensor, teacher: torch.Tensor, T: float) -> torch.Tensor:
    # mean over slots of mean over batch of KL(softmax(t/T) || softmax(s/T)), times T^2
    p_t = F.log_softmax(teacher / T, dim=-1)
    p_s = F.log_softmax(student / T, dim=-1)
    kl = (p_t.exp() * (p_t - p_s)).sum(-1)  # [B, slots]
    return kl.mean(dim=0).mean() * T * T


def main() -> int:
    torch.manual_seed(0)
    B, S, C = 8, 12, 4
    counts = (C,) * S
    teacher = torch.randn(B, S, C) * 3
    student = torch.randn(B, S, C, requires_grad=True)
    teacher_rg = teacher.clone().requires_grad_(True)
    results: dict[str, object] = {}

    kd_same = logits_kd_loss(teacher, teacher, temperature=KD_TEMPERATURE, slot_class_counts=counts)
    results["kd_teacher_vs_itself"] = float(kd_same)
    results["kd_zero_at_equality"] = abs(float(kd_same)) < 1e-6

    kd = logits_kd_loss(student, teacher_rg, temperature=KD_TEMPERATURE, slot_class_counts=counts)
    results["kd_value"] = float(kd)
    results["kd_positive"] = float(kd) > 0
    ref = manual_kd(student.detach(), teacher, KD_TEMPERATURE)
    results["kd_manual_T2_KL_teacher_student"] = float(ref)
    results["kd_matches_manual"] = abs(float(kd) - float(ref)) < 1e-5
    ref_reverse = manual_kd(teacher, student.detach(), KD_TEMPERATURE)  # KL(student||teacher)
    results["kd_reverse_direction_value"] = float(ref_reverse)
    results["kd_direction_is_teacher_to_student"] = abs(float(kd) - float(ref)) < 1e-5 and abs(float(kd) - float(ref_reverse)) > 1e-3
    # T^2 factor: T=1 vs T=2 relationship checked against manual formula at both temperatures
    kd_t1 = logits_kd_loss(student, teacher, temperature=1.0, slot_class_counts=counts)
    results["kd_T1_matches_manual"] = abs(float(kd_t1) - float(manual_kd(student.detach(), teacher, 1.0))) < 1e-5

    kd.backward()
    results["student_grad_nonzero"] = student.grad is not None and float(student.grad.abs().sum()) > 0
    results["teacher_grad_absent"] = teacher_rg.grad is None or float(teacher_rg.grad.abs().sum()) == 0.0

    # slot layout: perturb slot 3 only
    s2 = teacher.clone(); s2[:, 3, :] += torch.randn(B, C) * 2
    kd_slot = logits_kd_loss(s2, teacher, temperature=KD_TEMPERATURE, slot_class_counts=counts)
    p_t = F.log_softmax(teacher[:, 3] / KD_TEMPERATURE, -1); p_s = F.log_softmax(s2[:, 3] / KD_TEMPERATURE, -1)
    slot3 = (p_t.exp() * (p_t - p_s)).sum(-1).mean() * KD_TEMPERATURE ** 2
    results["kd_single_slot_perturbation"] = float(kd_slot)
    results["kd_single_slot_expected_over_12"] = float(slot3) / S
    results["kd_slot_layout_ok"] = abs(float(kd_slot) - float(slot3) / S) < 1e-5

    # CE includes absent (0)
    labels_absent = torch.zeros(B, S, dtype=torch.long)
    ce_abs = e1_train.compute_loss(student.detach(), labels_absent, num_aspects=S, aspect_class_counts=counts)
    ce_manual = F.cross_entropy(student.detach().reshape(-1, C), labels_absent.reshape(-1))
    results["ce_all_absent_value"] = float(ce_abs)
    results["ce_includes_absent_class0"] = float(ce_abs) > 0 and abs(float(ce_abs) - float(ce_manual)) < 1e-6
    labels = torch.randint(0, C, (B, S))
    ce_e1 = e1_train.compute_loss(student.detach(), labels, num_aspects=S, aspect_class_counts=counts)
    ce_e2 = supervised_cross_entropy(student.detach(), labels, slot_class_counts=counts)
    results["ce_e1_equals_e2_helper"] = abs(float(ce_e1) - float(ce_e2)) < 1e-6

    total = (1 - KD_ALPHA) * ce_e1 + KD_ALPHA * kd.detach()
    results["combined_total"] = float(total)
    results["combined_matches_half_half"] = abs(float(total) - 0.5 * (float(ce_e1) + float(kd))) < 1e-6

    # bf16 autocast path: logits in bf16 -> loss computed in fp32 inside helper, finite
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_b = (student.detach().to(dev) @ torch.eye(C, device=dev)).to(torch.bfloat16).requires_grad_(True)
            kd_b = logits_kd_loss(s_b, teacher.to(dev), temperature=KD_TEMPERATURE, slot_class_counts=counts)
        results["bf16_autocast_kd_dtype"] = str(kd_b.dtype)
        results["bf16_autocast_kd_finite_fp32"] = bool(torch.isfinite(kd_b)) and kd_b.dtype == torch.float32

    flags = {k: v for k, v in results.items() if isinstance(v, bool)}
    results["all_ok"] = all(flags.values())
    results["checked_flags"] = sorted(flags)
    results["temperature"] = KD_TEMPERATURE
    results["alpha"] = KD_ALPHA
    results["created_at"] = common.now()
    common.write_json(CHECK_ROOT / "loss_checks.json", results)
    for k, v in results.items():
        print(f"{k}: {v}")
    return 0 if results["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
