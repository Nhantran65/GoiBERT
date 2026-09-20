# Execution handoff — CafeBERT-DAPT → BamiBERT

Ngày2026-09-17. Đây là prompt để root giao việc sau khi phạm vi thực thi được xác định.

## Prompt copy cho subagent

Bạn là execution subagent của project `/home/kiettqa/GoiBERT`. Bắt buộc model `gpt-5.6-luna`, reasoning_effort `xhigh`; nếu runtime không đúng thì báo root và không đổi model âm thầm. Đọc `AGENTS.md` trước.

**Nguồn đặc tả chính:** `/home/kiettqa/GoiBERT/docs/CAFEBERT_DAPT_BAMIBERT_KD_PLAN.md`. Đọc toàn bộ, đặc biệt loss/schema/cross-tokenizer cache, gates và trial budget. File này đủ context để thực thi, không cần suy luận lại hướng nghiên cứu.

**Scope được root giao:** `<ROOT ĐIỀN M0–M1 / M2 / M3–M4 / M5–M6 / M7–M8 / end-to-end>`.

**Budget được giao:** `<ROOT ĐIỀN TOKEN/TRIAL/GPU-HOUR CAP; KHÔNG VƯỢT PLAN>`.

Nếu hai trường trên chưa được điền và không có chỉ dẫn execution rõ từ root/user, chỉ đọc và kiểm inventory không training; trả root phần scope còn thiếu. Không coi tài liệu kế hoạch là quyền sử dụng GPU vô hạn.

### Objective

Teacher: CafeBERT gốc → MLM DAPT review/social → fine-tune riêng5tasks. Student: **BamiBERT gốc≈103M**, không phải `goibert-b-dapt` hoặc student26M. Dùng logits KD và thử kết hợp E8 (NC+LLRD). Mục tiêu vượt BamiBERT local được tune công bằng, giữ student size. Không phát minh thêm kiến trúc/loss/data strategy.

### Protocol quan trọng

- 5task IDs: `vsfc_sentiment`, `vsfc_topic`, `visfd`, `absa_restaurant`, `absa_hotel`.
- ABSA0absent/1positive/2negative/3neutral. ViSFD widths `(4,4,4,4,4,2,4,4,4,4,4)`; OTHERS index5 pair-excluded. Softmax theo slot.
- Teacher tokenizer XLM-R; student tokenizer BamiBERT. Join logits cache theo stable raw-text IDs; **không align token IDs**.
- Loss full: `(1-alpha)*CE(clean) + alpha*T²*KL(teacher||student_clean) + lambda*symKL(student_clean,student_noisy)`; T2 choKD, T1 choNC, FP32 loss, teacher detached, cả2studentviews cógrad. E2 helper đã nhânT².
- E8 không chứaKD; B0CE/B1E8/K0KD từCafeBERT/KD từDAPT/KDE8 là các arms riêng.
- LLRD gamma.8, student12layers (default E7=10 phải override). Embedding `.8^12`, topblock/head1.
- FT30epochs, effbatch32, maxlen256, BF16SDPA, LRgrid{1e-5,2e-5,5e-5}, seed42; confirmation43/44 không re-grid. Chi tiết và caps nằm ởplan.
- DAPT pilot W300/D300 cùngsource/config; expansion1B cógate và bắt đầu lại từsource với schedule1B. Không tự train nhiều hơn.
- Cache KD chỉtrain text, khônggold, rawFP32logits; teacher eval. Teacherseed42 cố định cho studentconfirmation.
- Không sửa gold/split/history. Test inference chỉM8 saulockedmanifest. Text-only decontamination test tạiM3 cóaccesslog riêng.

### Engineering

- Tạo `src/goibert_cdbk`, `scripts/cdbk`, `configs/cdbk`, `data/cdbk`, `artifacts/cdbk`, `models/cdbk` mới. Không overwrite frozen source/artifacts.
- Tái dùng semantics của E2losses/noise/cache, E7optimizer và frozenloader/evaluator; audit hardcodedteacher/tokenizer/provenance trước khi reuse runner.
- CafeBERT probe cũ không save teacherweights. Phải save/reload best checkpoint thật.
- M0 pin HF revisions/hashes, inspect GPU/disk/env. Không kill job khác. Không giả định4GPUfree. Workspace hiện không cógit repo; bảo toàn bằngSHAmanifests.
- Durablecontroller, atomiccheckpoints, heartbeat, resumeoptimizer/RNG/sampler, oneowner/trial. Latest khácbest. Retrycap vàcomputeaccounting theoplan.
- Viết RUNBOOK exactcommands/dryrun; không báo command đã chạy nếu mới làtemplate.

### Báo root sau mỗi milestone

1. Scope hoàn tất và checklist pass/fail.
2. File/manifest/checkpoint paths vàhashes.
3. Bảng metrics từngtask/arm/seed, táchvalidation/test.
4. Actualruntime/GPUhours/tokensprocessed vàbudget còn lại.
5. Gate facts, lỗi/blockers; root quyết định diễn giải và nhánh tiếp.

Không nghiên cứu độc lập, không tìm novelty, không mở rộng scope. Nếu gatefail, dùng fallback đãđịnhnghĩa hoặc dừng nhánh; không tự tìm hyperparameter để cứu điểm.

## Cấu hình spawn dành cho root

```json
{
  "task_name": "cdbk_execution",
  "model": "gpt-5.6-luna",
  "reasoning_effort": "xhigh",
  "fork_turns": "none",
  "message": "Đọc AGENTS.md và docs/CAFEBERT_DAPT_BAMIBERT_KD_HANDOFF.md, sau đó toàn bộ docs/CAFEBERT_DAPT_BAMIBERT_KD_PLAN.md. Scope và budget thực thi: ROOT PHẢI ĐIỀN THEO ỦY QUYỀN TRƯỚC KHI SPAWN. Chỉ làm phần được giao; báo root artifacts và gate facts."
}
```

JSON trên làtemplate bàn giao, không phải một toolcall đã thực hiện.
