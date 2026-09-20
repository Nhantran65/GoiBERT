# CafeBERT → BamiBERT representation distillation — kết quả pilot (protocol v2)

Plan `docs/CAFEBERT_TO_BAMIBERT_REP_DISTILL_PILOT_PLAN.md` + Phụ lục A (gate §5 sửa, khóa trước launch); protocol `CafeBERT-BamiBERT-repdistill-pilot-v2`; seed 42; validation-only; ngày 2026-09-18. Source `scripts/cafe_bami_repdistill_pilot/`; dữ liệu `data/cafe_bami_repdistill_pilot/`; artifacts `artifacts/cafe_bami_repdistill_pilot/` (tổng hợp máy đọc: `summary.json`).

**Quyết định gate §6: KHÔNG ĐẠT.** G (MLM + quan hệ CafeBERT) không hơn `max(O, M)` ở task nào; thấp hơn O quá 0,5 điểm ở Restaurant và ViSFD. Cấu hình pilot này (λ=1, T=0,2, cặp layer T12→S6/T24→S12, 30M token FineWeb2) **không có tín hiệu để đầu tư**. Dừng tại gate: không seed 43, không Sentiment/Hotel, không test, không λ/whitening, không thêm teacher.

## 1. Kết quả chính (validation seed 42, F1×100, best epoch)

| Task | O: Bami E14 | M: +MLM 30M | G: +MLM+rel | CafeBERT | G−M | G−O | G−Cafe | M−O | Metric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Restaurant | 74,74 @29 | 73,31 @28 | 73,65 @25 | 77,72 @21 | **+0,34** | −1,09 | −4,07 | −1,43 | pair micro F1 |
| Topic | 79,53 @13 | 81,13 @8 | 79,72 @7 | 81,74 @23 | **−1,41** | +0,18 | −2,03 | +1,60 | fixed-label macro F1 |
| ViSFD | 84,38 @23 | 83,82 @25 | 83,25 @29 | 85,47 @19 | **−0,57** | −1,14 | −2,22 | −0,57 | pair micro F1 |

Gate §6 (G > max(O,M) ≥ +0,5 ở ≥2/3 task, có Restaurant/Topic; task còn lại không dưới O quá 0,5): G − max(O,M) = −1,09 / −1,41 / −1,14 → **0/3**. Nguồn O: `artifacts/e14/validation_grid.json` (LR đã chọn 5e-5/1e-5/5e-5); Cafe: `artifacts/cdbk_pilot/runs/T_absa_restaurant_seed42`, `artifacts/sealion_benchmark/cafe_runs/{vsfc_topic,visfd}/lr_2e-05/seed_42` (LR 2e-5).

Đường cong (ổn định, không phải một đỉnh đơn lẻ): 
- Restaurant: O top-3 trung bình 74,69 / 5 epoch cuối 74,37; M 73,22 / 72,93; G 73,20 / 72,70. G ≈ M trên toàn đường cong; cả hai thấp hơn O ~1,4 điểm từ epoch 10 trở đi.
- Topic: M đỉnh 81,13 ở epoch 8 rồi dao động 76,6–79,1 (top-3 80,81, 5 epoch cuối 78,50); G đỉnh 79,72 ở epoch 7, 5 epoch cuối 76,99; O 5 epoch cuối 77,49. Gain của M ở Topic (+1,6) là một đỉnh sớm; xét 5 epoch cuối M vẫn hơn O +1,0 và G −0,5.
- ViSFD: đường cong phẳng (O 11 epoch trong 0,5 của đỉnh); thứ tự O > M > G giữ nguyên suốt epoch 10–30 (5 epoch cuối 84,19 / 83,61 / 82,98).

Đọc thẳng: (i) **plain MLM 30M token web (M) làm giảm** Restaurant −1,43 và ViSFD −0,57, tăng Topic +1,60 — cùng hướng với E19 DAPT (Topic tăng, ABSA không); (ii) **thêm quan hệ CafeBERT (G) không sửa được điều đó**: ≈ M ở Restaurant (+0,34, trong nhiễu một seed ~1 điểm), kém M ở Topic −1,41 và ViSFD −0,57.

## 2. Học trung gian (M/G) — `cpt/{M,G}/summary.json`

| | M | G |
|---|---:|---:|
| Token non-pad đã xử lý | 30.008.828 (đúng lịch, 2.536 step × 128 seq, 0,764 pass) | 30.008.828 (giống hệt) |
| Batch/mask hash step 0–3, 500…3000 | — | **trùng M** (`same_batches_and_masks: true`) |
| MLM train 50 step đầu → cuối | 1,395 → 1,210 | 1,398 → 1,209 |
| L_rel 50 step đầu → cuối | — | 0,076 → 0,029 (0,080 @20 → 0,032 @500 → 0,029 @1500 → 0,028 @2536) |
| Holdout MLM (mask cố định seed 1234; O = step 0) | 1,4410 → **1,1211** | 1,4410 → **1,1191** |
| ratio ‖∇L_rel‖/‖∇L_MLM‖ trong train | — | 0,024 @0, 0,056 @100, rồi giảm dần **0,007–0,010 từ step 600** (min 0,006) |
| Wall / peak VRAM (L40S, micro 64×2) | 366 s (0,10 GPU-h) / 8,65 GB | 1.110 s (0,31 GPU-h) / 21,1 GB (teacher cùng GPU) |
| Export | encoder-only 102.360.576 tham số, reload sạch (0 missing/unexpected), không chứa teacher/projection, khác init, smoke một pass hữu hạn | như M |

L_rel giảm nhanh rồi bão hòa (~0,028) sau ~1.000 step, và gradient của nó co về ~1% gradient MLM: objective được "thỏa" sớm bằng cách làm phẳng phân bố quan hệ của student về mức gần-đều của teacher (anisotropy CafeBERT, xem §4 v1 bên dưới), sau đó đóng góp rất ít. Holdout MLM của G thấp hơn M 0,002 — không đáng kể.

## 3. Correctness

- Fine-tune: mỗi task M/G cùng `head_init_sha256` và cùng `train_order_epoch1_sha256` (kiểm true cả 3 task); recipe E14 (CE, 30 epoch, eff. batch 32, maxlen 256, bf16/SDPA, wd 0,01, warmup 10%, clip 1); reload best checkpoint → F1 và predictions trùng; F1 tính lại từ file predictions trùng; IDs ổn định/đủ; strict load 0 missing/unexpected (6/6 run `completed`).
- CPT: cùng checkpoint gốc (hash layer0.query init trùng), cùng corpus/lịch batch (`corpus.npz` sha `manifest.outputs.corpus_sha256`), cùng mask theo (step, micro); G thêm clean forward student + teacher no-grad; teacher frozen `.eval()`.
- GPU: mỗi job một GPU (0–3), không dừng tiến trình người khác. Tổng thực tế **1,38 GPU-h** (CPT 0,41 + 6 fine-tune 58,2 phút; ước lượng ban đầu 3,4 GPU-h vì giả định fine-tune chậm hơn).

## 4. Lịch sử gate (v1 → v2)

- v1 (`gate_decision.json`, `preflight.json`): gate dữ liệu §3 đạt (39,29M token train, 424.960 sequence 8–96 đơn vị, ≤256 token cả hai tokenizer; decontam 407 chunk); gate căn chỉnh §4 đạt (99,89% span map cả hai model, 30 case tay đúng); gate gradient §5 **fail** (ratio 0,015–0,035 < 0,05 tại init). Nguyên nhân: span vector CafeBERT anisotropic (cos 0,70 ± 0,05 @L12, 0,53 ± 0,10 @L24) → P_T gần đều ở T=0,2 (`relation_anisotropy_diagnostic.json`).
- v2 (`gate_decision_v2.json`, plan Phụ lục A): root quyết định bỏ sàn ratio, gate = hữu hạn/khác 0/ổn định trong profile; giữ nguyên λ, T, layer, corpus, ngân sách, gate §6. Đạt → launch. Kết quả downstream ở §1 xác nhận lo ngại của v1: objective có tác động đo được (L_rel giảm 2,7×) nhưng không chuyển thành lợi ích downstream.

## 5. Diễn giải và giới hạn

- Đây là một seed, validation, không significance; khác biệt <1 điểm (G−M Restaurant +0,34) nằm trong nhiễu run-to-run đã quan sát (~1 điểm). Các delta âm 1,1–1,4 so với O trên hai task ABSA nhất quán trên toàn đường cong, không phải một epoch.
- Kết luận đúng phạm vi: *trên corpus web 30M token, chưng cất quan hệ cosine/0,2 từ CafeBERT frozen vào BamiBERT không tạo backbone tốt hơn BamiBERT gốc hay BamiBERT+MLM*. Không kết luận về representation distillation nói chung (objective có centering/whitening, nhiệt độ khác, corpus review, budget lớn hơn đều chưa thử) và không nói gì về test.
- Phát hiện phụ có giá trị: MLM tiếp tục trên web phổ thông làm **giảm** ABSA (Restaurant −1,4, ViSFD −0,6) ở 30M token — trùng E19 (DAPT 1,67 GB: ABSA không đổi/giảm nhẹ, Topic +2,6) và Pilot B (CPT CafeBERT không tăng clean). Với ABSA, thêm web text không phải hướng đi; corpus đúng miền vẫn thiếu (pilot AS-CPT).

## 6. Artifacts

`data/cafe_bami_repdistill_pilot/{manifest.json, corpus.npz, texts.jsonl, documents.jsonl}`; `artifacts/cafe_bami_repdistill_pilot/{summary.json, gate_decision.json, gate_decision_v2.json, preflight.json, alignment_cases.txt, grad_ratio_diagnostic.json, relation_anisotropy_diagnostic.json, cpt/{M,G}/{summary.json, resolved_config.json, log.jsonl, hf_mlm/, hf_encoder/}, cpt/profile_*.json, ft/{M,G}/{absa_restaurant,vsfc_topic,visfd}/{metadata.json, epoch_curves.json, best_checkpoint.pt, best_validation_metrics.json, validation_predictions.json}, run_finetunes.log}`. Không sửa raw gold/split, source hay artifacts cũ.
