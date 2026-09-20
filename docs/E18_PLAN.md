# E18 — Screening 4B: huấn luyện đọc-được-theo-đoạn (fork-consistency) trên student

Người dùng duyệt 2026-09-13: bắt đầu theo plan §6 của [SLOT_READER_ROUTING_PROPOSAL.md](SLOT_READER_ROUTING_PROPOSAL.md); dùng GPU đang rảnh; root (Claude) làm orchestrator/review, một subagent rẻ (Sonnet) implement/chạy. Không sửa E1–E17, không test split, không multi-seed/grid.

## Giả thuyết và câu hỏi

Trần đọc-theo-đoạn (proposal §4.4) cho thấy với student E8 đóng băng, phần lợi thuộc cấu trúc đoạn chỉ ~+2/+1,5 F1 (oracle). 4B hỏi: **nếu huấn luyện encoder để mỗi đoạn tự đọc được (fork theo đoạn ra đúng nhãn của đoạn), pair F1 trên validation có tăng không, và có phải nhờ cấu trúc đoạn không?** Inference không đổi (CLS thường, không tham số mới).

## Model và loss (khóa)

Backbone fresh `models/goiturn-s-gd`, head CLS 384 như E8. Không tham số mới; checkpoint phải load được bằng `build_e2_student` để tái dùng `segment_read_ceiling.py`.

Trong lúc train (chỉ forward "clean" của E2 loop, không áp cho forward noisy và không áp khi eval), với mỗi câu trong micro-batch:

- Liệt kê split ẩn j ∈ J(x): vị trí token ngay sau một token thuộc tập liên từ/dấu câu `{",", ".", ";", "!", "?", " và", " nhưng", " còn", " mà", " tuy", " nên", " thì"}` (id: 15, 17, 30, 4, 34, 258, 643, 617, 572, 2238, 659, 656 của tokenizer goiturn-s-gd), sao cho mỗi đoạn có ≥1 token nội dung; nếu không có thì lấy các vị trí cách đều; tối đa S_max = 6 split/câu.
- Thêm 1 + 2·S_max hàng "fork": hàng 0 = CLS đầy đủ (mask đủ, trùng CLS gốc), hàng (2s+1, 2s+2) = hai đoạn của split s, mỗi hàng chỉ attend vào {chính nó} ∪ tokens của đoạn ∪ {</s>}. Token rows chỉ thấy CLS gốc + tokens (không thấy fork) ⇒ token states và CLS logits **giống hệt** forward thường. Phép tính đúng như `SlotEncoder.forward(..., fork_masks=...)` trong `scripts/research_checks/competitive_slots_probe_v2.py` (K=1) và `collate_latent` trong `..._v3_latent.py`, nhưng chạy BF16 autocast với softmax/logsumexp FP32 để đủ nhanh cho training.
- Với câu có ≥2 gold aspect (trên các slot được chấm; Restaurant/Hotel không có OTHERS): score(s) = Σ_{a∈G} max_i log p(y_a | fork_{s,i}); chọn s* = argmax; **chấp nhận** nếu score(s*) > score(fork 0). Gán mỗi a∈G cho fork có log p(y_a) cao hơn ⇒ G_1, G_2. Target cho fork i: y_a nếu a∈G_i, absent (0) cho mọi aspect còn lại. Không gradient qua bước chọn/gán (argmax).
- Loss micro-batch: L = CE(CLS) + λ · mean_{câu chấp nhận} ½ Σ_i CE(fork_{s*,i}, target_i) + NC như E8 (symmetric KL clean/noisy trên CLS logits). λ = 0,3. Câu không chấp nhận hoặc <2 aspect chỉ đóng góp CE(CLS) (+NC).

Arms (Restaurant + Hotel, seed 42, 30 epochs):

| Arm | Khác biệt | Mục đích |
|---|---|---|
| `b0_baseline` | không fork (λ=0, không tính fork) | E8 replay cùng kernel/micro-batch |
| `b1_fork` | như trên | ứng viên |
| `b2_random_mask` | thay hai đoạn liên tục của mỗi split bằng hai tập token ngẫu nhiên bù nhau cùng kích thước (seed cố định theo epoch/batch) | đối chứng cấu trúc đoạn |
| `b3_fork_detach` | như b1 nhưng fork states `.detach()` trước head ⇒ fork CE chỉ cập nhật head | đối chứng encoder vs head |

Chạy đợt 1: b0, b1, b2 (6 trial). b3 và `b4_composite` (E8 + composite union, data-only) chỉ khi root yêu cầu sau khi đọc đợt 1.

## Recipe (khóa, giống E17)

E8 NC + LLRD gamma 0,8 (`goibert_e7.optim.build_layerwise_optimizer`, validate coverage như run_e8), CE + λ_NC symmetric KL (Restaurant/Hotel λ_NC = 0,1), LR 1e-4, 30 epochs, warmup 10%, AdamW wd 0,01, clip 1, max_len 256, **micro 16 × accum 2 = effective 32**, BF16/SDPA cho forward thường; forward có fork dùng đường attention thủ công (BF16 matmul, FP32 softmax). Data/split/evaluator frozen (E2 loader, `include_test=False`), ABSA 0 absent/1 pos/2 neg/3 neutral.

## Preflight bắt buộc (artifact `artifacts/e18/preflight.json`), root review trước dispatch

1. Parity: b1 với fork bật, CLS logits vs forward SDPA thường trên 64 câu validation Restaurant: FP32 max |Δ| ≤ 1e-4; BF16 ≤ 0,1 (mức battery 0,047 là tham chiếu). Fork hàng 0 (mask đủ) vs CLS: FP32 ≤ 1e-4.
2. b0 với λ=0 cho loss/logits trùng E2 NC replay trên một fixed batch (cùng seed).
3. Gradient: fork CE tạo gradient hữu hạn, khác 0 vào encoder (b1) và chỉ vào head (b3) — kiểm bằng norm gradient của một tham số encoder khi chỉ backprop fork CE.
4. Tỷ lệ chấp nhận split trên 256 câu train ≥2 aspect của mỗi task (kỳ vọng ~0,7–0,85 với checkpoint GD fresh có thể thấp hơn — báo số thực); histogram vị trí split (tại liên từ vs cách đều).
5. Shapes/loader cho cả hai task; strict checkpoint round-trip qua `build_e2_student`; optimizer coverage (mọi tham số đúng một group, LR/wd đúng); không mutate `models/goiturn-s-gd`, E1–E17 source/artifacts.
6. Profile: median step wall-time và peak VRAM b0/b1/b2 trên fixed train batch micro 16 (báo tỉ lệ b1/b0; ngưỡng dừng nếu b1 > 2× b0 hoặc OOM).

## Vận hành

- `src/goibert_e18/` (modeling + patch loss), `configs/e18/protocol.json`, `scripts/run_e18.py` (preflight / profile / worker / run-all), `artifacts/e18/<arm>/<task>/seed_42/`, logs `artifacts/e18/logs/`, `controller_state.json` với heartbeat/PID.
- Python `.venv-e0-cuda/bin/python`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. GPU: chỉ dùng GPU có <5 GiB đã dùng và util ~0 tại lúc dispatch; không bao giờ dừng job khác; 1 worker/GPU.
- Freeze source hash trước run-all; worker lưu status/epoch_curves/best_checkpoint/validation_predictions và telemetry per-epoch: acceptance rate, fork CE trung bình, tỉ lệ split tại liên từ, λ hiệu dụng.
- Subagent báo root sau preflight+profile (không tự launch 6 trial); root duyệt rồi ra lệnh launch. Sau khi 6 trial xong: recompute F1 từ saved predictions, kiểm IDs/gold, chạy `segment_read_ceiling.py` (contiguous + `--random-masks`) trên mỗi best checkpoint, phân rã missed/spurious/wrong theo k và mixed-polarity, composite battery A/B/both cho b0/b1/b2.

## Gate (khóa trước khi chạy)

b1 > b0 và b1 > b2 ≥ 0,5 điểm pair F1 trên cả hai task; spurious của b1 không vượt b0 quá 10%; oracle-gap phần đoạn của b1 (contiguous − random-mask) tăng so b0. Nếu b2 ≈ b1: gain là regularization/đọc fragment, không claim cơ chế. Không task nào tăng: đóng hướng đoạn/reader. Không test, không multi-seed, không grid trong vòng này.
