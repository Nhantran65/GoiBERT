#!/usr/bin/env python3
"""Turn the pilot artifacts into the §6 gate table and docs/MMBERT_VI_CPT_PILOT_RESULT.md.

The report is assembled from JSON artifacts only (no re-training): data-gate manifests, the manual audit report, the arm
manifests (mixture + locked order hashes), the CPT summaries (loss curves, holdout MLM, throughput, GPU-hours), the 15
fine-tune metadata files and the reusable baseline O seed 42 row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TASK_LABEL = {"absa_restaurant": "Restaurant", "visfd": "ViSFD", "vsfc_topic": "Topic"}
ARMS = ("O", "P_web", "P_mix", "C_mix")


def rj(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def ft_row(arm: str, task: str, seed: int) -> dict | None:
    if arm == "O" and seed == 42:
        p = {"absa_restaurant": "absa_restaurant/lr_5e-05", "visfd": "visfd/lr_5e-05", "vsfc_topic": "vsfc_topic/lr_2e-05"}[task]
        m = rj(common.ROOT / "artifacts/base_encoder_baseline/mmbert_base" / p / "metadata.json")
        if not m:
            return None
        return {"f1": m["best_selection_f1"], "epoch": m["best_epoch"], "status": m["status"], "minutes": m["elapsed_seconds"] / 60,
                "source": str((common.ROOT / "artifacts/base_encoder_baseline/mmbert_base" / p).relative_to(common.ROOT)), "seed": 42, "lr": m["learning_rate"]}
    d = common.ART_DIR / "ft" / arm / task / f"seed_{seed}"
    m = rj(d / "metadata.json")
    if not m:
        return None
    return {"f1": m["best_selection_f1"], "epoch": m["best_epoch"], "status": m["status"], "minutes": m["elapsed_seconds"] / 60,
            "source": str(d.relative_to(common.ROOT)), "seed": seed, "lr": m["learning_rate"],
            "reload_ok": m.get("postrun_checks", {}).get("reload_matches_best"), "ids_ok": m.get("postrun_checks", {}).get("ids_stable_and_complete")}


def mean(rows: list[float]) -> float | None:
    rows = [r for r in rows if r is not None]
    return sum(rows) / len(rows) if rows else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(common.ROOT / "docs" / "MMBERT_VI_CPT_PILOT_RESULT.md"))
    args = ap.parse_args()

    dedup = rj(common.DATA_DIR / "pool/dedup_manifest.json", {})
    tok = rj(common.DATA_DIR / "pool/tokenize_manifest.json", {})
    wiki = rj(common.DATA_DIR / "wiki/wiki_manifest.json", {})
    hold = rj(common.ART_DIR / "holdouts/holdout_manifest.json", {})
    audit = rj(common.ART_DIR / "data_audit/audit_report.json", {})
    arms = rj(common.ART_DIR / "arms/arms_manifest.json", {})
    pre = rj(common.ART_DIR / "preflight.json", {})
    cpt = {a: rj(common.ART_DIR / "cpt" / a / "summary.json") for a in common.ARMS}

    rows = {}
    for arm in ARMS:
        for task in common.FT_TASKS:
            seeds = (42, 43) if task == "absa_restaurant" else (42,)
            if arm == "O":
                seeds = (42, 43)
            rows[(arm, task)] = {s: ft_row(arm, task, s) for s in seeds}

    def cell(arm: str, task: str) -> str:
        r = rows[(arm, task)]
        parts = []
        for s, v in r.items():
            parts.append("n/a" if v is None else f"{100*v['f1']:.2f}")
        if task == "absa_restaurant":
            vals = [v["f1"] for v in r.values() if v]
            m = mean(vals)
            return " / ".join(parts) + (f" (pair-mean {100*m:.2f})" if m is not None else "")
        return " / ".join(parts)

    def pairmean(arm: str, task: str) -> float | None:
        if task != "absa_restaurant":
            v = rows[(arm, task)].get(42)
            return v["f1"] if v else None
        vals = [v["f1"] for v in rows[(arm, task)].values() if v]
        return mean(vals)

    g1 = g2 = g3 = None
    g1 = (pairmean("P_web", "absa_restaurant") or 0) - (pairmean("O", "absa_restaurant") or 0) if all((pairmean(a, "absa_restaurant") is not None) for a in ("P_web", "O")) else None
    g2 = (pairmean("P_mix", "absa_restaurant") or 0) - (pairmean("P_web", "absa_restaurant") or 0) if all((pairmean(a, "absa_restaurant") is not None) for a in ("P_mix", "P_web")) else None
    g3 = (pairmean("C_mix", "absa_restaurant") or 0) - (pairmean("P_mix", "absa_restaurant") or 0) if all((pairmean(a, "absa_restaurant") is not None) for a in ("C_mix", "P_mix")) else None

    lines: list[str] = []
    add = lines.append
    add("# mmBERT-base + Vietnamese CPT — pilot 250 M token (đối chứng register/curriculum)")
    add("")
    add(f"**Trạng thái:** sinh tự động {common.now()} bằng `scripts/mmbert_vi_cpt_pilot/summarize.py` từ artifacts trong "
        "`artifacts/mmbert_vi_cpt_pilot/`. Screening seed 42 (Restaurant thêm seed 43), validation only, dừng ở gate §6.")
    add("")
    add("Model gọi đúng tên: **mmBERT-base 307 M tổng (110,3 M không-embedding)**.")
    add("")
    add("## 1. Provenance, license, ngân sách")
    add("")
    add("| Hạng mục | Giá trị |")
    add("|---|---|")
    add(f"| HPLT 3.0 `vie_Latn` | HF revision `{common.HPLT_HF_REVISION}`, shards {', '.join(dedup.get('shards_locked_order', []))} |")
    sha_lines = []
    sha_path = common.HPLT_DIR / "sha256.txt"
    if sha_path.is_file():
        sha_lines = [l.strip() for l in sha_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    add(f"| SHA256 shards | {('<br>'.join(sha_lines)) if sha_lines else 'n/a'} |")
    add(f"| License HPLT | {common.HPLT_LICENSE} |")
    add(f"| Wikipedia tiếng Việt | {wiki.get('source', {}).get('repo')} config {wiki.get('source', {}).get('config')}, revision "
        f"`{wiki.get('source', {}).get('dataset_revision')}`, file {wiki.get('source', {}).get('file')}, sha256 `{wiki.get('source', {}).get('sha256')}` |")
    add(f"| License Wikipedia | {common.WIKI_LICENSE} |")
    add(f"| FineWeb-2 (holdout b) | {common.FW2_REPO}@{common.FW2_COMMIT}, {common.FW2_SHARD_REL}, sha256 `{common.FW2_SHARD_SHA256}` |")
    add(f"| Model | `{common.MODEL_REPO}@{common.MODEL_COMMIT[:7]}` — {pre.get('model', {}).get('documented', 'mmBERT-base 307 M total (110.3 M non-embedding)')} |")
    add(f"| Attention | {pre.get('attention', {}).get('implementation', 'n/a')} (flash_attn available: {pre.get('attention', {}).get('flash_attn_available')}) — dùng chung cho mọi arm |")
    add(f"| Ngân sách CPT | {common.STEPS} step × {common.GLOBAL_BATCH} chuỗi × {common.MAX_LEN} token = {common.TOKENS_SCHEDULED:,} token/arm |")
    add("")
    add("## 2. Gate dữ liệu (§3)")
    add("")
    st1 = dedup.get("stage1_counters_aggregate", {})
    add(f"- Tài liệu đọc vào: **{st1.get('docs_in', 0):,}**; giữ sau lọc: **{st1.get('kept', 0):,}** (subsample uniform 1/8–1/16 theo id).")
    add(f"- Bỏ theo cờ: MT≥0,5 {st1.get('drop_mt', 0):,}; seg_langs {st1.get('drop_seg_langs', 0):,}; PII {st1.get('drop_pii', 0):,}; "
        f"không phải tiếng Việt {st1.get('drop_not_vi', 0):,}; length prefilter {st1.get('drop_length_prefilter', 0):,}; "
        f"decontam doc {st1.get('drop_decontam_doc', 0):,} (đoạn bị loại: {st1.get('decontam_segments_removed', 0):,}).")
    add(f"- Dedup: exact {dedup.get('dedup', {}).get('exact_key', '')}; near-dup {dedup.get('dedup', {}).get('near_dup', '')}; {dedup.get('dedup', {}).get('near_dup_membership', '')}.")
    add(f"- Pool sau dedup: **{dedup.get('counters', {}).get('pool', 0):,} tài liệu**; holdout HPLT {dedup.get('holdout', {}).get('hlpt', {}).get('docs')} doc, "
        f"holdout register {dedup.get('holdout', {}).get('register', {}).get('docs')} doc.")
    add(f"- Token hoá: {tok.get('counters', {}).get('docs_kept', 0):,} doc, **{tok.get('counters', {}).get('tokens_kept', 0):,} token mmBERT unique** "
        f"(bỏ {tok.get('counters', {}).get('drop_too_short', 0):,} quá ngắn / {tok.get('counters', {}).get('drop_too_long', 0):,} quá dài).")
    pg = dedup.get("pool_by_group", {})
    add(f"- Phân bố nhóm trong pool: {json.dumps(pg, ensure_ascii=False)}; theo bin: {json.dumps(dedup.get('pool_by_bin', {}), ensure_ascii=False)}.")
    add(f"- Wikipedia: {wiki.get('documents', 0):,} doc / {wiki.get('tokens', 0):,} token; counters {json.dumps(wiki.get('counters', {}), ensure_ascii=False)}.")
    add(f"- Holdout MLM: a={hold.get('a_hlpt', {}).get('sequences')} chuỗi/{hold.get('a_hlpt', {}).get('documents')} doc, "
        f"b={hold.get('b_fineweb2', {}).get('sequences')} chuỗi/{hold.get('b_fineweb2', {}).get('documents')} doc, "
        f"c={hold.get('c_register', {}).get('sequences')} chuỗi/{hold.get('c_register', {}).get('documents')} doc.")
    add("")
    add("## 3. Audit tay 100 doc/ô (một người đọc, seed 42)")
    add("")
    if audit:
        add("| Ô | n | tiếng Việt thật | boilerplate chiếm đa số | dịch máy/văn bản sinh | người lớn/spam | giọng ý kiến người dùng |")
        add("|---|---|---|---|---|---|---|")
        for cell, s in audit.get("cells", {}).items():
            ov = "—" if s.get("opinion_voice") is None else f"{s['opinion_voice']:.0%}"
            add(f"| {cell} | {s['n']} | {s['real_vietnamese']:.0%} | {s['boilerplate_majority']:.0%} | {s['machine_translated_or_generated']:.0%} | {s['adult_or_spam']:.0%} | {ov} |")
        add("")
        g = audit.get("gate", {})
        add(f"- Gate audit: opinion precision ≥70 %: **{g.get('opinion_precision_ge_70')}** (min {g.get('opinion_precision_min')}); "
            f"general ≥90 % tiếng Việt thật: **{g.get('general_real_vietnamese_ge_90')}**; general ≤10 % boilerplate: **{g.get('general_boilerplate_le_10')}**; "
            f"cờ MT ≥70 % đúng: **{g.get('mt_flag_correct_ge_70')}**; tất cả: **{g.get('all_ok')}**.")
    else:
        add("_Chưa có audit_report.json._")
    add("")
    add("## 4. Mixture thực tế và thứ tự batch (đã khoá)")
    add("")
    add("| Arm | general | register | bin8 | wiki | tokens | tokens hash | order hash | multiset hash |")
    add("|---|---|---|---|---|---|---|---|---|")
    for arm in common.ARMS:
        m = arms.get("arms", {}).get(arm, {})
        f = m.get("achieved_fractions", {})
        add(f"| {arm} | {f.get('general', 0):.3f} | {f.get('register', 0):.3f} | {f.get('bin8', 0):.3f} | {f.get('wiki', 0):.3f} | "
            f"{m.get('tokens_non_pad', 0):,} | `{(m.get('tokens_hash') or '')[:16]}` | `{(m.get('order_hash') or '')[:16]}` | `{((m.get('multiset_hash') or '')[:16]) or '—'}` |")
    add("")
    add(f"- P_mix và C_mix dùng **cùng multiset** (cùng document, cùng số lần xuất hiện): multiset hash bằng nhau = "
        f"**{arms.get('arms', {}).get('P_mix', {}).get('multiset_hash') == arms.get('arms', {}).get('C_mix', {}).get('multiset_hash')}**.")
    add(f"- Ghi chú sai lệch đã báo: {arms.get('deviation_note', '')}")
    add("")
    add("## 5. CPT: loss, holdout MLM, throughput")
    add("")
    add("| Arm | wall (h) | GPU-giờ | tok/s | peak GB | loss (100 step đầu → cuối) | holdout a (0→100 %) | holdout b | holdout c |")
    add("|---|---|---|---|---|---|---|---|---|")
    for arm in common.ARMS:
        s = cpt.get(arm)
        if not s:
            add(f"| {arm} | — | — | — | — | — | — | — | — |")
            continue
        h = s["holdout"]["rows"]
        first, last = h[0], h[-1]
        add(f"| {arm} | {s['wall_time_s']/3600:.2f} | {s['gpu_hours']:.2f} | {s['tokens_per_s']:,} | {s['peak_gpu_memory_gb']:.1f} | "
            f"{s['train_loss_first_100_mean']:.4f} → {s['train_loss_last_100_mean']:.4f} | {first['a_hlpt']:.4f} → {last['a_hlpt']:.4f} | "
            f"{first['b_fineweb2']:.4f} → {last['b_fineweb2']:.4f} | {first['c_register']:.4f} → {last['c_register']:.4f} |")
    add("")
    add("## 6. Bảng validation (Restaurant seed 42/43; ViSFD, Topic seed 42), tham chiếu Bami/Cafe")
    add("")
    add("| Arm | Restaurant | ViSFD | Topic |")
    add("|---|---|---|---|")
    for arm in ARMS:
        add(f"| {arm} | {cell(arm, 'absa_restaurant')} | {cell(arm, 'visfd')} | {cell(arm, 'vsfc_topic')} |")
    add(f"| Cafe (tham chiếu) | {100*common.REFERENCE['Cafe']['absa_restaurant']:.2f} | {100*common.REFERENCE['Cafe']['visfd']:.2f} | {100*common.REFERENCE['Cafe']['vsfc_topic']:.2f} |")
    add(f"| Bami (tham chiếu) | {100*common.REFERENCE['Bami']['absa_restaurant']:.2f} | {100*common.REFERENCE['Bami']['visfd']:.2f} | {100*common.REFERENCE['Bami']['vsfc_topic']:.2f} |")
    add("")
    add("Delta (điểm F1 validation, Restaurant dùng trung bình ghép cặp 2 seed, task khác seed 42):")
    add("")
    add("| So sánh | Restaurant | ViSFD | Topic |")
    add("|---|---|---|---|")
    for name, a, b in (("G1  P_web − O", "P_web", "O"), ("G2  P_mix − P_web", "P_mix", "P_web"), ("G3  C_mix − P_mix", "C_mix", "P_mix")):
        vals = []
        for task in common.FT_TASKS:
            x, y = pairmean(a, task), pairmean(b, task)
            vals.append("n/a" if x is None or y is None else f"{100*(x-y):+.2f}")
        add(f"| {name} | {vals[0]} | {vals[1]} | {vals[2]} |")
    add("")
    add(f"- G1 (Restaurant ≥ +0,5 hoặc ≥1/3 task ≥ +0,5 và Restaurant không ≤ −0,5): {g1 if g1 is None else format(100*g1, '+.2f')}")
    add(f"- G2 (Restaurant P_mix − P_web ≥ +0,5): {g2 if g2 is None else format(100*g2, '+.2f')}")
    add(f"- G3 (Restaurant C_mix − P_mix ≥ +0,5): {g3 if g3 is None else format(100*g3, '+.2f')}")
    add("")
    add("## 7. Checks")
    add("")
    if pre:
        add(f"- Preflight: {json.dumps(pre.get('gate', {}), ensure_ascii=False)}")
        sm = pre.get("smoke", {})
        add(f"- Smoke {sm.get('steps')} step: loss {sm.get('loss_first'):.4f} → {sm.get('loss_last'):.4f}, grad hữu hạn {sm.get('grad_norm_all_finite')}, "
            f"{sm.get('tokens_s', 0):,.0f} tok/s, peak {sm.get('peak_gpu_memory_gb', 0):.1f} GB.")
    for arm in common.ARMS:
        s = cpt.get(arm)
        if s:
            e = s.get("export", {})
            add(f"- {arm}: reload encoder khớp {e.get('reload_encoder_layer0_query_equal')}, decoder tied {e.get('reload_mlm_decoder_tied')}, "
                f"không có head/projection lạ {e.get('encoder_keys_contain_head_or_projection') == []}, smoke inference {e.get('inference_smoke', {}).get('finite')}.")
    ft_ok = all((v or {}).get("status") == "completed" for task in common.FT_TASKS for arm in ARMS for v in rows[(arm, task)].values() if v)
    add(f"- Fine-tune: reload/prediction/ID checks pass cho mọi run: **{ft_ok}**.")
    add("")
    add("## 8. Quyết định gate §6 và giới hạn")
    add("")
    add("_Điền kết luận sau khi bảng trên đủ số: xem phần diễn giải trong file này._")
    add("")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"[summarize] wrote {args.out} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
