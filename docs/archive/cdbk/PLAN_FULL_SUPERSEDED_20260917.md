# CafeBERT-DAPT → BamiBERT student: kế hoạch thực thi và bàn giao

Ngày: 2026-09-17. Protocol đề xuất: `cafe_dapt_bami_kd_v1` (viết tắt **CDBK**).

## 0. Trạng thái và phạm vi ủy quyền

- Người dùng yêu cầu **lập kế hoạch chi tiết và tổng hợp context để subagent thực thi**. Tài liệu này chưa phải thông báo đã chạy, chưa phê duyệt chi phí mới và không tự khởi chạy training.
- Các cấu hình/ngân sách bên dưới là đề xuất cụ thể. Khi người dùng yêu cầu thực thi kế hoạch này, agent chạy các milestone trong phạm vi được giao; không xin lại xác nhận cho từng bước đã nằm trong phạm vi đó.
- Root review gate bằng artifacts; đây là kiểm tra nghiên cứu nội bộ, không phải yêu cầu user phê duyệt lại. Chỉ cần quyết định mới nếu thay mục tiêu, tăng ngân sách hoặc đổi nhánh ngoài kế hoạch.
- Không kế thừa quyền training/test từ E9, E14 hay pilot cũ cho một chương trình mới.
- Không chạy thêm thí nghiệm chỉ để cứu điểm; fail gate được ghi lại và áp dụng nhánh fallback đã định nghĩa.

## 1. Quyết định nghiên cứu đã chốt

### Mục tiêu

Tạo các checkpoint task-specific khởi tạo từ **BamiBERT gốc (~103M)**, học từ **CafeBERT (~550–560M)** hoặc **CafeBERT-DAPT**, để vượt BamiBERT local dưới cùng split/evaluator và ngân sách chọn cấu hình công bằng. Mục tiêu mở rộng là ngang/vượt CafeBERT local, nhưng không hứa trước.

Ưu tiên chất lượng và khả năng tái lập; người dùng đã hạ ưu tiên novelty. Không nghiên cứu thêm architecture/tokenizer trong chương trình này. Student không tăng kiến trúc và không dùng teacher khi inference.

### Pipeline

1. CafeBERT gốc → DAPT bằng MLM trên review/social tiếng Việt đã lọc → encoder CafeBERT-DAPT.
2. Fine-tune **riêng từng task** bằng nhãn train → teacher classifier từng task.
3. Teacher eval trên **text train** bằng tokenizer XLM-R → cache raw logits FP32.
4. BamiBERT gốc + tokenizer BamiBERT → train với CE/KD/NC/LLRD theo từng arm.
5. Chọn checkpoint/LR/arm bằng validation; xác nhận seed; khóa manifest trước test.

**E8 = NC + LLRD, không phải KD.** Full arm mới là CE + KD + NC + LLRD. Từng arm cần đối chứng; không giả định E8 giúp model 103M hoặc cộng lợi ích với KD.

Đầu ra là **5 model task-specific**, không phải một model multi-task duy nhất hay một shared student backbone distill tổng quát. Không gọi task KD là general distillation.

## 2. Vai trò và cấu hình subagent bắt buộc

Đọc `AGENTS.md` ở project root trước khi làm việc.

- Root tự thực hiện research, giải thích số liệu, chọn giả thuyết và quyết định gate.
- Subagent chỉ triển khai, kiểm tra correctness, chạy phép đo/training/evaluation theo đặc tả và trả artifacts.
- Mọi execution subagent dùng **model `gpt-5.6-luna`, reasoning_effort `xhigh`**; truyền rõ cả hai trường.
- Dùng `fork_turns: "none"` hoặc số lượt được hỗ trợ; không dùng `all` khi override.
- Nếu cấu hình không khả dụng, báo root; không đổi model âm thầm.
- Ưu tiên một execution agent sở hữu pipeline xuyên suốt. Chạy workers trên GPU không cần tạo nhiều agent. Không giao hai agent sửa cùng source hoặc điều khiển cùng trial.

Workspace `/home/kiettqa/GoiBERT`; Python tham chiếu `.venv-e0-cuda/bin/python`. Lịch sử môi trường: PyTorch 2.6/CUDA 12.4, Transformers 4.49, BF16/SDPA, 4×L40S. **Phải kiểm tra lại thực tế**, không mặc định GPU đang rảnh.

Tại lúc lập kế hoạch, `git status` trả `not a git repository`. Không dùng git rollback làm cơ chế bảo toàn; dùng thư mục riêng và manifests SHA256. Không `git init` hoặc sửa dữ liệu lịch sử để phục vụ kế hoạch này.

## 3. Context và bằng chứng cần hiểu đúng

### Model nguồn

| Model | Nguồn | Vai trò |
|---|---|---|
| BamiBERT gốc | `models/bamibert-base-original` / `Qualcomm-AI-Research/BamiBERT` | Khởi tạo mọi student mới và baseline B0/B1 |
| CafeBERT gốc | `uitnlp/CafeBERT`, resolve/pin HF revision ở M0 | Teacher T0; khởi tạo tất cả CPT arms |
| Goi B DAPT cũ | `models/goibert-b-dapt` | Chỉ tham chiếu, **không khởi tạo student mới** |
| GoiTurn-S GD | `models/goiturn-s-gd` | Student lịch sử 26M, không thuộc training mới |

BamiBERT: 12 layers, hidden768, 12 heads, FFN3072, vocab20481. HF commit lịch sử `57bc1340debbe4e348ec549047a763caebe4a977`; weights SHA256 `c8c8abe5914f2543bd6a65019b685dacf25ad39da5f01b680b082876e1947627`. Kiểm tra lại file trước khi dùng.

CafeBERT: họ XLM-R-large, 24 layers, hidden1024, 16 heads. Repo lịch sử chỉ có sentencepiece; probe dùng `FacebookAI/xlm-roberta-large` tokenizer sau khi kiểm hash sentencepiece. M0 phải pin tokenizer revision, kiểm special IDs/config/vocab/hash và sample token IDs; không coi mọi tokenizer tên XLM-R đều tương đương.

**Không khởi tạo DAPT từ checkpoint CafeBERT đã fine-tune task.** DAPT bắt đầu từ pretrained MLM checkpoint; task head được tạo mới sau DAPT. Audit MLM head/tied weights và mọi missing/unexpected key khi load.

### Kết quả tham chiếu (không thay baseline mới)

- E8 student26M: test ViSFD 81.81±0.12, Restaurant71.88±0.36 (3 seeds), hơn E2 NC lần lượt +1.19/+2.02 điểm. Không phải toàn bộ post-training thất bại.
- E19 backbone Goi DAPT103M + NC/LLRD không đạt gate thắng rộng. Không mặc định chuyển recipe E8 sang BamiBERT sẽ giúp.
- E21/E22 composite: Restaurant validation +1.16±0.18 paired SD; Hotel yếu, ViSFD không giúp. **Composite không nằm trong vòng CDBK này**, để cô lập KD/DAPT.
- CafeBERT probe vs BamiBERT E14, validation seed42: Restaurant77.14 vs74.74; Topic81.75 vs79.53; ViSFD85.14 vs84.38; Sentiment87.24 vs86.33; Hotel73.60 vs73.24. Single-seed/probe protocol, không coi là lợi thế đã xác nhận.
- Pilot B plain web CPT300M chưa tăng sạch rộng; không dùng lại tokenized corpus đó làm data sạch nếu chưa audit lại.
- Teacher Goi DAPT cũ có lịch sử dùng validation text không nhãn. CDBK dùng checkpoint gốc public và corpus mới được audit; không tuyên bố biết hết corpus tiền huấn luyện của các checkpoint public.
- Test hiện tại đã được quan sát nhiều vòng trước. CDBK khóa test trong vòng mới nhưng **không gọi nó virgin holdout**.

## 4. Dữ liệu downstream và metric contract

| Task ID | Train | Validation | Test | Primary metric |
|---|---:|---:|---:|---|
| `vsfc_sentiment` | 11426 | 1583 | 3166 | Fixed-label macro F1, 3 lớp |
| `vsfc_topic` | 11426 | 1583 | 3166 | Fixed-label macro F1, 4 lớp |
| `visfd` | 7786 | 1112 | 2224 | End-to-end aspect–polarity pair micro F1 |
| `absa_restaurant` | 7028 | 771 | 1938 | End-to-end pair micro F1, 12 aspects |
| `absa_hotel` | 7180 | 795 | 2030 | End-to-end pair micro F1, 34 aspects |

- VSFC sentiment/topic dùng cùng văn bản; 5 task nhưng 4 miền văn bản.
- Không đổi raw gold, split, thứ tự aspect hoặc evaluator. Validate ID/order/gold so với frozen loader.
- ABSA polarity: **0 absent, 1 positive, 2 negative, 3 neutral**. Metadata cũ có tên sai; số nguyên benchmark không tự sai vì tên. Đọc `E12_LABEL_MAPPING_AUDIT.md`.
- VSFC sentiment: 0 negative, 1 neutral, 2 positive.
- ViSFD có 11 slots, widths **`(4,4,4,4,4,2,4,4,4,4,4)`**, OTHERS tại index5, raw4→model1; OTHERS tham gia CE/KD/NC/detection, **loại khỏi pair polarity metric**. Tổng42 logits.
- Restaurant48 logits; Hotel136 logits. Softmax riêng từng slot, không softmax trên toàn bộ packed vector.
- Absent0 là target thật. Chỉ bỏ `ignore_index=-100` hoặc valid-slot mask rõ ràng.
- F1 lưu full precision ở [0,1], bảng hiển thị100×F1; delta_pp=100×(candidate−reference). Không average task metrics để che regression.
- Mirror UIT-ABSA chưa xác minh exact lineage tương đương paper. Claim chính so local baselines, không lấy validation so paper test.

## 5. Implementation map: tái dùng gì, không tái dùng mù quáng gì

### Đọc và tái sử dụng semantics

- `src/goibert/data.py`, `metrics.py`, `modeling.py`, `train.py`: loader, schema, metric, head, training utilities.
- `src/goibert_e2/losses.py`, `noise.py`, `cache.py`: đã có combined KD+NC; contract ở `docs/E2_LOSS_CONTRACT.md` và `configs/e2/loss_contract.json`.
- `src/goibert_e7/optim.py`: optimizer LLRD; **truyền `num_layers=12` cho student**, default là10.
- `scripts/run_e14.py`, `src/goibert_e14/modeling.py`: original BamiBERT adapter/provenance.
- `scripts/run_e19.py`: tham khảo adapter NC/LLRD12layers, không sao chép đường dẫn teacher DAPT cũ.
- `scripts/pilot_b/cpt_train.py`: tham khảo sparse MLM head chỉ chấm masked positions, DDP, token accounting.

### Các bẫy đã xác định

1. `scripts/research_checks/probe_mmbert_finetune.py` là probe chưa khóa protocol; **không save best model checkpoint**, chỉ summary/predictions. Không dùng JSON điểm probe như một teacher có weights sẵn.
2. E2 loop từng hardcode teacher/tokenizer/provenance theo BamiBERT family. Không monkey-patch factory rồi để metadata/hash vẫn chỉ model cũ.
3. Cache teacher phải gắn **tokenizer teacher**, không buộc teacher/student có cùng tokenizer/hash. Join cache bằng stable IDs của văn bản gốc, không bằng token index.
4. `goibert_e7.optim` yêu cầu naming `encoder.embeddings`, `encoder.encoder.layer`, `head`. Adapter mới phải tuân theo hoặc triển khai mapping rõ, không bỏ silently các params lạ. Một telemetry key cũ tên `block_09` không đại diện tầng cuối12layers; báo block11 và embeddings đầy đủ trong manifest mới.
5. Pilot B data prep cắt/pack token rồi lấy tail làm MLM val. CDBK phải split **document trước packing**, tránh cùng document/near-duplicate cluster sang train/holdout.
6. Pilot B trainer không cung cấp đầy đủ resume contract optimizer/RNG/data-cursor. CDBK cần resume thật, hoặc ghi rõ restart-from-source; không gọi load weights là exact resume.
7. Không ghi đè `src/goibert*` cũ, `scripts/run_e*`, `configs/e*`, `models/*` hoặc artifacts lịch sử. Import read-only utilities được phép.

### Đường dẫn mới (chưa tồn tại tại lúc viết plan)

```text
src/goibert_cdbk/                # model adapters, losses wrapper, cache, train/CPT utilities
scripts/cdbk/                   # preflight, prepare_data, train_cpt, train_task, cache, evaluate, controller
configs/cdbk/protocol.json       # machine-readable locked contract; root-reviewed
data/cdbk/                      # manifests, documents, packed corpora, contamination reports
artifacts/cdbk/m0 ... m8/        # audit, trial artifacts, reports
models/cdbk/                    # new exports only
docs/CDBK_STATUS.md
docs/CDBK_RESULTS.md
```

M1 phải cung cấp CLI thật, `--help`, dry-run matrix. Lệnh trong handoff chỉ được coi runnable sau khi implementation tồn tại.

## 6. Training contract chung

### Fine-tuning teacher và student

- Fresh task head: CLS → dropout0.1 → dense H→H → tanh → dropout0.1 → task logits. H=1024 teacher,768 student. Không pooler; toàn encoder trainable.
- 30epochs, effective batch32, max_length256, BF16/SDPA; evaluate validation mỗi epoch, chọn best full-precision primary F1; tie chọn epoch sớm nhất.
- AdamW betas(.9,.999), eps1e-8, wd.01; bias/LayerNorm không decay; warmup10% rồi linear decay; clip_grad_norm1.
- Student micro16×accum2; teacher micro8×accum4. OOM fallback micro8/4 và accum tương ứng, giữ eff32; log thay đổi. Dùng cùng micro giữa các arm của một model/task khi so sánh.
- Batch cuối không đủ: chuẩn hóa gradient theo đúng số samples/valid slots thực trong accumulation window; scheduler đếm optimizer steps thực; không giảm sai gradient batch cuối.
- LR grid cố định **{1e-5,2e-5,5e-5}**, seed42 screening. Seed43/44 confirmation không chọn LR lại.
- Tie chọn LR nhỏ hơn; mọi arm có cùng ngân sách grid. Không early stopping trước30epochs, trừ numerical/runtime failure được ghi rõ.
- Train shuffling/head-init RNG ghép cặp giữa student arms cùng seed/task; noise RNG riêng. Teacher cache generation không được thay RNG của student.
- Seed Python/NumPy/Torch/CUDA, `CUBLAS_WORKSPACE_CONFIG=:4096:8`; log versions/device. Không hứa bitwise deterministic giữa mọi kernel/hardware.
- Đo tỷ lệ truncation theo **mỗi tokenizer**; không suy từ thống kê BamiBERT rằng CafeBERT cũng không truncate.

### Loss và optimizer của student

T=2 cho KD; KL **teacher || student**, teacher detached, FP32 probabilities/log-softmax. NC T=1 symmetric KL, cả clean/noisy nhận gradient.

```text
KD = T² * mean_valid_slots KL(softmax(z_teacher/T) || softmax(z_student_clean/T))
NC = 0.5 * mean_valid_slots [KL(p_clean || p_noisy) + KL(p_noisy || p_clean)]

B0: CE(clean)
B1: CE(clean) + lambda_nc * NC                     + LLRD
K0: (1-alpha)*CE(clean) + alpha*KD(T0)
KD: (1-alpha)*CE(clean) + alpha*KD(TD)
KDE8: (1-alpha)*CE(clean) + alpha*KD(TD) + lambda_nc*NC + LLRD
```

`T0`=fine-tuned CafeBERT gốc; `TD`=fine-tuned CafeBERT-DAPT được chọn ở M4. K0 là đối chứng tác dụng DAPT teacher, không phải student khởi tạo CafeBERT.

- Default alpha=.5. Chỉ thêm alpha=.25 theo ngân sách M6, không sweep T.
- E2 `logits_kd_loss` đã nhân T²: **không nhân lần hai**.
- Loss reduction mean trên example-slot hợp lệ theo contract E2; expose component counts/sums để audit accumulation. Không bỏ absent.
- NC weights kế thừa E8: Sentiment.1, Topic.5, ViSFD.5, Restaurant.1, Hotel.1. Đây là candidate cố định, không phải hệ số đã chứng minh tối ưu cho103M.
- LLRD gamma.8; student12layers: embeddings `.8^12`, block i `.8^(11-i)`, head/top block1.0. B0/K0/KD dùng uniform LR.
- Dùng deterministic text noise E2 (50%unchanged,25%strip,20%teencode,5%typo); giữ safeguards phủ định; log **actual changed fraction**. Không noise teacher hoặc lấy cache clean teacher làm target riêng cho noisy student.
- Chỉ logits KD: không align hidden states/attention/token spans giữa hai tokenizer.
- Các arm có loss scale khác do mixture coefficients; mô tả đúng. So sánh family có LR tuning công bằng, không nhận claim tách riêng optimizer/loss tuyệt đối.

## 7. Milestone M0 — Inventory và khóa provenance

**Input:** plan này, AGENTS, models/data/code lịch sử. **Không training.**

Thực hiện:

1. Kiểm environment, GPU processes, free VRAM/disk; ghi khả năng DDP và BF16. Không kill process khác.
2. Resolve BamiBERT/CafeBERT/tokenizers/MLM head; hash nguồn/config/weights; exact param counts total/backbone/head; missing/unexpected keys phải giải thích.
3. Audit downstream counts/IDs/slot schema/metric và mapping; chưa đọc predictions/gold test cho lựa chọn nào.
4. Kiểm artifacts CafeBERT cũ có weights thật không. Chỉ tái sử dụng run nếu đầy đủ provenance và **khớp toàn bộ protocol mới**; mặc định rerun baselines vì probe20epochs khác30epochs.
5. Lập trial ledger và dự toán runtime từ measurements lịch sử; phân biệt wall hours và GPU-hours.

**Deliver:** `m0/inventory.json`, `model_manifest.json`, `environment.json`, `downstream_contract.json`, `reuse_audit.md`, `trial_ledger.csv`.

**Exit:** đúng source models và schemas; nguồn dữ liệu khả dụng; không blocker disk/runtime. Nếu không load được CafeBERT đúng MLM head, resolve trước DAPT; không khởi tạo lại head âm thầm.

## 8. Milestone M1 — Implementation và preflight

**Input:** M0 manifests. **Output:** runner/CPT/cache/evaluator riêng và protocol JSON đã freeze.

Các kiểm tra bắt buộc, nhỏ và có ý nghĩa:

- Output shapes cả5tasks và mapping ViSFD42logits.
- CE/KD/NC số học đối chiếu tensor nhỏ; KL direction/T²; gradient teacher=none; noisy student có gradient; logits giống nhau KD≈0.
- Cache shuffled ID lookup đúng; duplicate/missing IDs/hash mismatch fail rõ ràng; teacher/student tokenizer khác được chấp nhận khi text contract trùng.
- alpha=0/lambda=0/gamma=1 cho loss/update tương đương arm đối chứng ở FP32/dropout-controlled fixture.
- Optimizer coverage mỗi parameter đúng một lần; endpoints12layers đúng; resume giữ optimizer/scheduler/RNG/sampler/step.
- Save/load strict roundtrip cho teacher1024 và student768; reload checkpoint tái tính validation metric khớp.
- Sparse MLM loss/gradient khớp dense MLM trên batch nhỏ; mask không special/pad, labels=-100 nơi không chấm; tied weights không nhân đôi params.
- Actual CUDA forward/backward all schemas; DDP step ngắn; kiểm grad accumulation và masked-token weighting toàn rank. Loss trên masks phải weight theo số masked tokens, không average tùy tiện các microbatches có số mask khác nhau.
- Profile one-device fine-tune teacher/student/full KDE8 và 4GPU CPT. Ghi steps/s, peak memory; xây ETA và dung lượng checkpoints.

**Budget:** tối đa2 aggregate GPU-hours cho smoke/profile. Không tính fixture này là kết quả nghiên cứu.

**Deliver:** `m1/preflight.json`, `runtime_profile.json`, source hashes, exact commands, `configs/cdbk/protocol.json`, `scripts/cdbk/RUNBOOK.md`.

**Exit:** tất cả correctness checks pass và dry-run matrix nằm trong cap. Không launch full training với cache/schemas chưa pass.

## 9. Milestone M2 — Baseline teacher và student mạnh, công bằng

**Arms:** B0, B1, T0 CE. **5tasks ×3LR ×1seed =45 trials tối đa.**

- B0/B1 đều từ BamiBERT gốc. T0 từ CafeBERT gốc.
- Chọn LR/epoch riêng theo validation mỗi arm/task; lưu checkpoint thật, optimizer latest riêng với best.
- T0 chỉ CE, không NC/LLRD/composite. CDBK so với baseline CafeBERT CE được tune trong budget này; không gọi là CafeBERT tối ưu mọi recipe.
- Report T0−B0, T0−B1 theo task, gold errors/disagreement trên validation, thời gian/params.
- T0 không hơn B0 ở task nào vẫn giữ kết quả; KD không bị cấm chỉ vì teacher scalar F1 thấp hơn, nhưng không tuyên bố teacher có headroom ở task đó.

**Deliver:** `m2/validation_grid.{json,csv}`, `selected_checkpoints.json`, saved validation logits/predictions, `baseline_readout.md`.

**Exit:** đủ45trials hoặc reuse hợp lệ được ghi; 15selected checkpoints audit reload. Không test.

## 10. Milestone M3 — Corpus sạch và pilot DAPT/CPT

### M3a: xây dữ liệu

Hai corpus, tokenizer XLM-R giống nhau:

- **D:** review/social tiếng Việt. Ưu tiên nguồn review tự nhiên có provenance, bổ sung social; không gọi toàn bộ social là hotel/restaurant domain. Lập bảng tỷ trọng token theo nguồn và sample audit.
- **W:** FineWeb2 tiếng Việt web tổng quát, đối chứng cùng số tokens/recipe.

Nguồn ứng viên (chưa đồng nghĩa đã được audit/download):

- `https://huggingface.co/datasets/HuggingFaceFW/fineweb-2`
- `https://huggingface.co/datasets/phucdev/ViSoBERT` (bản chia sẻ bên thứ ba: xác minh lineage/điều kiện dùng trước).
- Dataset reviews public có review text thật và điều kiện dùng phù hợp; executor lập manifest các nguồn cụ thể, root chốt mixture trước run. Không coi model card mô tả14GB là một dataset14GB tải được.

Yêu cầu:

1. Pin revision/checksum/license/source URLs, source field mapping và preprocessing.
2. Unicode NFC, bỏ empty/boilerplate/duplicate; giữ dấu/teencode/casing/emoji tự nhiên. Không dùng từ điển normalize teencode trong corpus D mặc định.
3. Exact + near dedup theo document/cluster, ghi thuật toán/ngưỡng. Giữ source metadata để audit.
4. Exclude **tất cả downstream train/val/test text** khỏi CPT mới, để tách DAPT từ TAPT. Dùng decontamination worker text-only; không load nhãn/predictions test. Chỉ trả hashes/counts overlap, không xuất test text vào nghiên cứu lựa chọn. Nếu loader chỉ tải text qua bundle có nhãn, tạo đường text-only riêng.
5. Đây là truy cập **test text chỉ để loại overlap**, phải ghi access log riêng; test inference/gold analysis vẫn cấm đến M8. Không claim chưa từng đọc test data.
6. Check duplicate clusters với benchmark và cross-corpus W/D; nếu cùng doc xuất hiện hai corpora thì gán một phía theo quy tắc cố định và báo số loại.
7. Human/root sample review ≥100docs mỗi corpus, report actual review/social/web fraction; không suy toàn corpus không sai từ100docs.
8. Split doc-cluster-level train/MLM holdout trước tokenization/packing, holdout1% tối đa5M tokens mỗi corpus; freeze sample/masks đánh giá. Packed seq512, có EOS boundary và attention mask đúng padding. Cho phép cross-document attention trong packed sequence ở cả hai arm; mô tả công khai.
9. D phải có ≥150M **training token positions sau lọc**, measured bằng tokenizer thực, cho pilot300M với tối đa2passes. Nếu thiếu: báo thiếu data, không lặp corpus nhỏ hàng chục lần. Đây là floor ngân sách thực dụng, không ngưỡng khoa học.
10. Ghi cả corpus token positions, tokens processed, số passes/repetition và token mixture; không gọi tokens_seen là dữ liệu độc nhất.

Không đưa labeled downstream train vào D; teacher logits trên train ở M5 là giai đoạn khác. Không pseudo-label triệu câu hoặc gọi API LLM trong scope này.

### M3b: CPT pilot

**W300 và D300**, cùng bắt đầu từ CafeBERT gốc, mỗi arm300M processed tokens, seed42, tổng600M.

- Standard MLM15%eligible tokens;80/10/10 mask/random/unchanged; không diacritic factorization/dropout.
- seq512, global batch256; 4GPU nếu trống, micro8/rank×accum8 khi4ranks. Nếu ít GPU, tăng accum phù hợp và giữ global batch; không chiếm GPU người khác để đạt ETA.
- LR5e-5, AdamW betas(.9,.98), eps1e-6, wd.01 (bias/norm no decay), warmup6%, linear decay, clip1, BF16. Đây là config xuất phát theo Pilot B, không sweep CPT LR.
- Full encoder + pretrained MLM head trainable; sparse MLM head được phép sau parity.
- Train tới step đủ budget, sai số≤1global batch; log **non-padding tokens thực** và masks processed.
- Eval cả D-holdout/W-holdout tại t0, mỗi~50M và cuối; cùng fixed masks, không dùng loss để thay downstream criterion.
- Export cuối300M cố định cho M4, không chọn nhiều checkpoint theo downstream. Save recovery checkpoint mỗi~50M + cuối; giữ2latest, best/selected exports riêng; atomic writes.

**Deliver:** corpus manifests/audit/token counts, `m3/{w300,d300}/` logs/curves/resume state/HF exports, initial/final model hashes.

**Exit:** corpus pass và2runs đủ budget, không NaN, load/export parity. DAPT loss giảm không tự pass quality gate.

## 11. Milestone M4 — Teacher downstream gate và tùy chọn1B

### Pilot downstream

Fine-tune W300 và D300 trên **Restaurant, Topic, Hotel** (sentinel positive-headroom + hard task), mỗi3LR, seed42: **18trials**. So với T0 M2 cùng protocol.

Ngưỡng sau là **investment heuristics**, không significance:

- D300 **strong pass**: ≥2/3tasks tăng≥.3pp so T0, không task giảm>.5pp. Sau đó train D300 hai task còn lại ×3LR =6trials. Broad check:≥3/5 tăng≥.3 và không giảm>.5 → giữ TD=D300.
- Nếu D300 chỉ1/3 tăng≥.3, không task giảm>.5: cho phép **một** nhánh D1000 dưới đây. Broad check D300 fail nhưng có≥2/5 tăng≥.3 và không giảm>.5 cũng đủ xét D1000. Không mở nhánh khi có regression lớn chỉ để cứu model.
- Nếu fail các điều kiện trên: **đóng DAPT expansion**, tiếp tục KD từ T0 theo fallback M6; không coi toàn dự án distillation thất bại.
- W300 là control: nếu D không hơn W, không claim domain advantage. Strong pass so T0 vẫn có thể giữ D làm teacher chất lượng, nhưng phải báo W competitive. Không mở W1B trong scope này.

### Nhánh tùy chọn D1000 (tối đa một run)

- Chỉ khi gate cho phép và D có≥500M training tokens sau lọc (tối đa2passes); nếu không có data thì dừng expansion, không âm thầm đổi mixture.
- **Train mới từ CafeBERT gốc tới1B** với schedule1B và cùng mixture/rules đã chốt. Không resume scheduler300M đã decay về0 rồi gọi là continuation tương đương.
- D300 được giữ như pilot; D1000 là run mới, tổng CPT toàn project tối đa**1.6B tokens** (W300+D300+D1000).
- Fine-tune5tasks×3LR=15trials; áp broad gate≥3/5 tăng≥.3 so T0, không task giảm>.5. Nếu fail, TD không được mặc định chọn vì đã tốn compute; fallback T0, hoặc D300 nếu D300 đã pass broad gate trước đó.
- Không chọn mỗi task từ một CPT snapshot khác nhau để gọi một shared CafeBERT-DAPT. Chốt **một encoder DAPT chung**; classifier/LR/epoch được chọn riêng từng task.

**Deliver:** `m4/teacher_comparison.csv`, teacher manifests, `gate_decision.json` root-written có `selected_encoder`, reasons, comparisons và fallback.

**Budget:** tối đa39new FT trials (18+6+15), thường24 hoặc18. Không test.

## 12. Milestone M5 — Cache teacher và audit cross-tokenizer

**Input:** T0 và TD selected task checkpoints, hoặc chỉT0 nếu DAPT fail.

- Teacher `.eval()`, `inference_mode`, BF16 forward được phép; raw logits lưu CPUFP32, không softmax/argmax trước khi cache.
- Chỉ train text, maxlen256, tokenizer đúng teacher. Chạy mỗi teacher/task đúng một lần trừ corruption audit.
- Cache identity: ordered stable IDs/hash, original text hash, task/slot order/class widths, teacher checkpointSHA, teacher tokenizerSHA/revision, maxlen/truncation/precision, source hashes.
- Student tokenizer manifest tách riêng. Cùng text/ID không yêu cầu cùng token IDs hoặc số token.
- Cache không chứa gold; labels train do loader student đọc riêng. Không cache val/test teacher targets cho training.
- Kiểm random32IDs/cache so live teacher inference (tolerance ghi rõ), duplicate IDs fail; shuffle minibatch vẫn lookup đúng.
- Không dùng stale cache khi checkpoint teacher thay đổi. Không mặc định index dòng cache giống thứ tự DataLoader shuffled.

**Deliver:** tối đa10caches + `m5/cache_manifest.json`, `cache_audit.json`, inference cost.

**Exit:** cache alignment/hash/shape/numerical checks pass. Không loss KD nào được train trước gate này.

## 13. Milestone M6 — Student distillation screening

### Khi có TD

- B0/B1 reuse đúng M2.
- Train K0, KD, KDE8, alpha.5, mỗi5tasks×3LR×seed42 =**45trials**.
- Từng task: chọn tốt hơn giữa K0/KD theo validation, gọi `K_best_teacher`; full recipe KDE8 giữ riêng. Với mỗi family `K_best_teacher` và KDE8, thử alpha.25 tại LR đã chọn, mỗi task1trial: tối đa**10trials**.
- Tổng≤55new FT trials. Việc alpha.25 có thể thay winner phải lưu; không mở grid LR lại cho alpha.25. B0/B1 không cần alpha trials vì không KD; báo ngân sách tuning khác nhau minh bạch.
- So **KD−K0** tại alpha.5 để đo teacher DAPT dưới cùng ngân sách tuning; **KDE8−B1** để đo thêm KD trong recipe E8; **KD−B0** đo plain KD. Không quy KDE8−KD riêng cho NC vì có cả LLRD.

### Fallback khi DAPT fail

- Train K0 và **K0E8** (T0 logits+NC/LLRD), mỗi5tasks×3LR=30trials, alpha.5.
- Thử alpha.25 ở LR selected cho2families×5tasks=10trials. Tổng≤40. Không tạo hai arms trùng nhau tênKD/K0 để giả như có DAPT comparison.

### Chọn candidate cho confirmation

- Mỗi task chọn student tốt nhất trong KD-containing families bằng validation; ties ưu tiên ít forward passes hơn, rồi alpha nhỏ hơn, LR nhỏ hơn.
- Có thể chọn recipe khác nhau theo task vì output là5task models. Report rõ **per-task selection**, không gọi một recipe chung thắng tất cả. Báo thêm bảng recipe cố định alpha.5 trên5tasks.
- Gate sang M7: selected KD students đạt≥3/5tasks **+.3pp so baseline mạnh hơn giữa B0/B1**, không task giảm>.5pp. Không lấy thắng CE nhưng thua B1 làm thành công chính.
- Nếu chỉ1–2tasks tăng≥.3 không regression>.5: M7 chỉ xác nhận các task đó, scope task-specific; không claim broad win. Nếu0task đạt hoặc regression>.5: dừng chương trình xác nhận rộng, báo full matrix.
- Không tiếp tục pseudo-labeling, ensemble, temperature search, hidden KD hoặc student DAPT để cứu gate trong scope này.

**Deliver:** `m6/student_grid.csv`, selected student manifest (teacher, alpha, NC, gamma,LR,seed,epoch), validation error comparisons, decision root.

## 14. Milestone M7 — Confirmation seeds và khóa kết luận trước test

Với các task vượt gate, train seeds43/44 ở **LR/alpha/recipe đã khóa seed42**, không re-grid:

- B0, B1, T0, TD (nếu có), selected student S*.
- Tối đa5systems×5tasks×2seeds=**50trials**. Nếu chỉ xác nhận subset thì giảm tương ứng; không mở task mới sau khi thấy seed kết quả.
- KD students43/44 dùng **teacher checkpoint seed42 cố định** để đo biến thiên student có điều kiện trên cùng teacher. Teacher43/44 là các run đánh giá độ ổn định chất lượng teacher, không tự thay cache của student cùng seed.
- Do đó báo rõ: student3seed uncertainty **chưa bao gồm biến thiên teacher-training**. Không gọi đây là full-pipeline3seed. Kiểm teacher-seed sensitivity là extension ngoài cap, không tự launch.
- Báo mean±sampleSD, per-seed paired deltas với B0 và B1, positive-count; không đồng nhất SD với confidence interval.
- Gate ứng viên broad:≥3/5tasks mean delta≥.3 so baseline Bami mạnh hơn theo mean validation, ≥2/3paired seeds dương ở mỗi task đó; không task mean regression>.5. Đây vẫn là heuristic.
- Nếu không ổn định, đánh dấu fail và dừng claim broad; không đổi candidate sang seed winner sau khi xem confirmation.
- Test một candidate đã khóa vẫn có thể hoàn tất để báo cáo kết quả âm **nếu toàn bộ execution scope gồm M8**; không dùng test để cứu/select candidate khác.

**Deliver:** `m7/confirmation.csv`, `pretest_readout.md`, `locked_test_manifest.json` chứa all selected hashes/source/config/IDs/seeds, danh sách claims dự định kiểm tra.

## 15. Milestone M8 — Test cuối, package và báo cáo

- Chỉ M8 được chạy model trên test và đọc gold cho metrics. M3 text-only decontamination là ngoại lệ được log riêng, không inference.
- Evaluate các checkpoint selected của B0/B1/T0/TD/S* seeds42–44; tối đa**75passes** nếu5systems×5tasks×3seeds. Fallback4systems≤60; subset nhỏ hơn.
- Không load grid checkpoints để chọn lại bằng test. Checkpoint chọn best epoch validation cho mỗi seed.
- Strict reload, ID/gold/order/counts, recompute F1 từ saved predictions độc lập với training summary. Save test access log.
- Báo từng task và seed; paired bootstrap theo **cùng sample indices cho hai models**, tính lại macro/pair F1 từ predictions, không average per-row F1. Với3seeds, báo sample bootstrap riêng/conditional và seed SD riêng; không giả vờ cả hai là một CI bao quát mọi randomness.
- Nếu dùng nhiều hypothesis tests, report multiplicity procedure hoặc gọi intervals exploratory; không biến heuristic gate thành significance.
- Báo params, checkpoint size, latency/throughput ở batch1 và32, seq lengths thực/256, cùng GPU/precision, warmup20 và≥100measured batches; không tính teacher vào student inference nhưng báo training cost teacher/CPT/cache.
- Kiểm export standalone student chạy với tokenizer BamiBERT; không có dependency teacher/cache trong inference.

**Định nghĩa claim:**

1. `beats_plain_BamiBERT`: hơn B0 local, báo per-task.
2. `beats_tuned_BamiBERT`: hơn baseline mạnh hơn B0/B1 đã chọn bằng validation; đây là mục tiêu chính.
3. `matches_or_beats_CafeBERT`: so T0 trong protocol này; nếu điểm gần nhau nhưng CI rộng thì nói chưa thấy khác biệt rõ, không tuyên bố equivalence.
4. Chỉ nói `wins_all_5` nếu bảng từng task hỗ trợ; không lấy mean5tasks thay thế.
5. Chưa đánh giá independent dataset thì claim giới hạn bộ benchmark hiện có, đã adaptive exposure. Không tự tuyên bố SOTA toàn Vietnamese NLU.

**Deliver:** `docs/CDBK_RESULTS.md`, tablesCSV/JSON táchval/test, paired comparisons, all manifests, `models/cdbk/student/<task>/`, model cards, inference example, source/hash inventory và tổng actual GPU-hours.

## 16. Ngân sách, phụ thuộc, điều kiện dừng

| Milestone | Phụ thuộc | Trần đề xuất |
|---|---|---|
| M0 inventory | execution được giao | Không training |
| M1 implementation/preflight | M0 | ≤2GPU-hours smoke/profile |
| M2 baselines | M1 | ≤45FT trials |
| M3 corpus + pilots | M0/M1; data prep có thể cùng M2 | 2CPT runs ×300M |
| M4 teacher gate | M2/M3 | ≤39FT trials; tùy chọn1CPT run1B |
| M5 cache | M4 | ≤10train-logit caches |
| M6 KD screening | M2/M5 | ≤55FT trials, fallback≤40 |
| M7 confirmation | M6gate | ≤50FT trials |
| M8 test/report | M7locked manifest | ≤75test passes |

**Trần toàn chương trình:**189FT trials + tối đa1.6B CPT processed tokens; không phải189trials bắt buộc. Trần compute đề xuất bổ sung **400aggregate GPU-hours**, gồm CPT/FT/cache/eval/smoke và retry; profile M1 kiểm tính khả thi trước launch. Trần trial/token/compute nào chạm trước thì dừng cấp job mới. Đây chưa phải ngân sách user đã duyệt.

Tính GPU-hours = wall hours×sốGPU dành cho job, kể cảDDP. Không suy4GPU×100hours=100GPU-hours. 400GPU-hours là trần chi tiêu, không ETA cam kết. Nếu profiling cho thấy không đủ, root thu hẹp matrix trước fullrun, không để executor bỏ control tùy ý.

- Disk: M0 đo free; dự toán worst-case checkpoints/cache/corpus. Chừa≥20% dung lượng filesystem hoặc≥50GB (lấy mức lớn hơn) nếu khả thi; nếu không đủ thì lập retention nhỏ hơn trước launch. Không xóa artifact cũ. Chỉ xóa intermediate CDBK theo retention đã ghi; selected checkpoints giữ nguyên.
- One worker/GPU ban đầu; DDP cần reserve toàn bộ GPUs của job, không launch task worker lên GPU DDP. Không dùng utilization threshold vô tình chặn worker mình hoặc đoán ownership theo %util.
- Heartbeat30–60s, PID/start-time/device/trial IDs; atomic state writes; resume/adopt live worker trước khi spawn mới. Filelock chống duplicate trial.
- Numerical fail: dừng arm, lưu batch IDs/config/loss, báo root. Không giảmLR hoặc đổiobjective âm thầm.
- Infra fail: resume đúng state; nếu không thể resume, tối đa1fresh restart cùngconfig/trial, artifact attempt riêng và tínhcompute cả2. Không chọn best giữa các attempts.
- Không tự tăng grid/seed/token budget. Nếu hếtcap, báo incomplete rõ và giữ checkpoints resumable.

## 17. Artifact contract mỗi trial

```text
<milestone>/<arm>/<task_or_corpus>/lr_<lr>/seed_<seed>/attempt_<n>/
  resolved_config.json
  source_manifest.json
  model_tokenizer_manifest.json
  status.json                  # queued/running/completed/failed/interrupted
  heartbeat.json
  epoch_curves.jsonl            # or step_curves for CPT
  latest_checkpoint.pt          # resumable, atomic; retain last2
  best_checkpoint.pt            # FT validation-selected
  best_validation_metrics.json
  validation_predictions.json  # stable IDs, logits/preds, gold; no test
  runtime.json
  stdout.log
```

- `completed` chỉ khi toàn bộ budget chạy xong, checkpoint reload và final metrics audit pass.
- Tách `planned`, `implemented`, `preflight_passed`, `training_completed`, `evaluated`, `root_reviewed`; không dùng một PASS mơ hồ.
- Trial ledger ghi reuse/new/failure/retry, checkpointsource, teacher source, counts và actualGPUhours; restart không nhân bản completed rows.
- Mỗi milestone gửi root: files thay đổi, command/PID nếu live, rows completed/total, artifact links, numbers chính, blockers và gate facts. Executor không tự viết kết luận nhân quả từ một metric.

## 18. Tài liệu đọc tối thiểu cho executor

1. `AGENTS.md` và toàn bộ plan này.
2. `docs/E2_LOSS_CONTRACT.md`, `configs/e2/loss_contract.json`.
3. `docs/E14_BAMIBERT_BASELINE_RESULT.md`, `docs/E14_BAMIBERT_BASELINE_PLAN.md`.
4. `configs/e8/protocol.json`, `configs/e19/protocol.json`, `docs/E19_RESULT.md`.
5. `docs/E12_LABEL_MAPPING_AUDIT.md`, `docs/BENCHMARK_PROTOCOL.md`, `docs/BACKBONE.md`.
6. `docs/CAFEBERT_PROBE.md`, `docs/PILOT_B_RESULT.md` (đọc như lịch sử, không như recipe mới đã chứng minh).
7. Các code ở §5; không cần đọc hết22experiments để bắt đầu.

## 19. Prompt bàn giao

Prompt tự chứa ngắn nằm ở `docs/CAFEBERT_DAPT_BAMIBERT_KD_HANDOFF.md`. Root điền milestone scope và budget đã giao trước khi spawn. Nếu chỉ giao M0/M1, không tự chạy M2 vì đọc thấy trong plan. Nếu giao end-to-end, đi tiếp tự động qua gate đã định nghĩa, root review artifacts, không lặp xin phép user từng job.

## 20. Những gì chưa được làm

Tại ngày viết: chưa tạo module `goibert_cdbk`, chưa tải corpus mới, chưa train CafeBERT-DAPT, chưa tạo cross-tokenizer teacher cache, chưa chạy student CDBK, chưa xem test mới. Các bảng trên là cấu hình/ngân sách, **không phải kết quả**.
