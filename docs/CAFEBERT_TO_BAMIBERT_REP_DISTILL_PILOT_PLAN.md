# CafeBERT → BamiBERT: pilot chưng cất biểu diễn trên văn bản không nhãn

**Trạng thái:** plan bàn giao, chưa chạy. Chỉ một screening seed 42 trên validation; dừng tại gate. Ngày chốt 2026-09-18.

## 1. Câu hỏi và lý do chọn

**Câu hỏi:** cho BamiBERT gốc (~103M) học quan hệ giữa các biểu diễn ngữ cảnh của CafeBERT (~560M) trên văn bản tiếng Việt không nhãn có làm backbone tốt hơn BamiBERT gốc **và** BamiBERT học MLM trên cùng văn bản/ngân sách không? Nếu có, model triển khai vẫn là BamiBERT một encoder pass; CafeBERT chỉ hiện diện khi huấn luyện.

Đây là *general representation distillation* lấy cảm hứng từ [MiniLMv2](https://aclanthology.org/2021.findings-acl.188/) và [Contextual Knowledge Distillation](https://aclanthology.org/2021.emnlp-main.30/), **không** là tái hiện nguyên bản hai paper hay tuyên bố phép tính chưa từng có. Điểm cụ thể của pilot là chuyển quan hệ giữa các span ký tự chung qua **hai tokenizer khác nhau**, giữ MLM của BamiBERT làm neo. [Nghiên cứu cross-tokenizer](https://aclanthology.org/2025.findings-acl.419/) nêu sai lệch token/span và vocab là rủi ro thật; vì vậy preflight căn chỉnh là gate, không chỉ smoke hình thức.

Kết quả local có liên quan:

- CafeBERT tốt hơn BamiBERT trên **4/5 task test seed 42**; khoảng cách Restaurant +3,24, Topic +2,59, ViSFD +0,98, Hotel +1,23 điểm F1. Bami hơn Cafe +0,67 ở Sentiment. Nguồn: `docs/E14_BAMIBERT_BASELINE_RESULT.md`, `docs/SEALION_300M_BENCHMARK_RESULT.md`.
- Task-logit KD từ **CafeBERT đã fine-tune** sang BamiBERT trên Restaurant giảm 0,80 điểm validation; teacher logits trên chính train gần one-hot (`docs/CDBK_PILOT_RESULT.md`). Pilot này dùng **CafeBERT pretrained, không có task head/labels**, và quan hệ hidden-state trên corpus không nhãn. Nó chưa được thử trong project và thất bại task-KD không quyết định kết quả này.
- General distillation lịch sử là **BamiBERT-DAPT → student 26M/10M**, không phải CafeBERT → BamiBERT gốc (`docs/EXPERIMENTS_SUMMARY.md`). Plain CPT CafeBERT trên FineWeb2 không tăng clean scores (`docs/PILOT_B_RESULT.md`); vì vậy cần đối chứng **BamiBERT + MLM cùng corpus** để phân biệt lợi ích teacher với lợi ích đọc thêm web.
- Gate corpus review Restaurant/Hotel vừa fail (`docs/BAMIBERT_ASPECT_SENTIMENT_CPT_PILOT_RESULT.md`). Lượt này kiểm **chuyển tri thức tổng quát**, không gọi FineWeb2 là review đúng miền và không dùng corpus review 346K token để train nhiều vòng.

Mục tiêu là tìm **tín hiệu đủ mạnh để đầu tư**, không hứa vượt CafeBERT với 30M token hoặc kết luận SOTA từ một seed. Chưa thêm teacher thứ hai; chỉ xét multi-teacher sau khi tách được hiệu quả CafeBERT một mình.

## 2. Ba arm và ngân sách tối đa

| Arm | Khởi tạo BamiBERT | Học trung gian trên cùng corpus | Fine-tune downstream |
|---|---|---|---|
| **O** | Gốc | Không | Reuse E14 seed 42, không train lại |
| **M** | Gốc | Random MLM chuẩn | Restaurant, Topic, ViSFD |
| **G** | Gốc | Cùng MLM + quan hệ biểu diễn từ CafeBERT frozen | Cùng ba task |

CafeBERT CE seed 42 là mốc chất lượng, không train lại. Sau gate dữ liệu/căn chỉnh: **2 lượt học trung gian + tối đa 6 fine-tune**, không test, không LR/λ/token sweep. Restaurant và Topic là hai khoảng cách lớn nhất với Cafe; ViSFD kiểm chuyển sang một miền ABSA khác. Sentiment/Hotel, seed tiếp theo, thêm teacher và phép đo latency là vòng sau nếu qua gate.

## 3. Corpus và gate dữ liệu

- Nguồn duy nhất: shard FineWeb2 tiếng Việt **đã có local** `data/raw/fineweb2_vie/data/vie_Latn/train/000_00000.parquet` (~2,319M document; card [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) ghi ODC-By 1.0). Pin SHA256/revision từ `data/bami_as_cpt_pilot/corpus_manifest.json`; đọc raw text và **tokenize lại bằng BamiBERT**, không lấy mảng 320M token XLM-R làm token BamiBERT. Đây là văn bản web phổ thông cho general distillation; không lọc bằng từ khóa review.
- HTML-unescape/NFC/collapse whitespace; loại trống/boilerplate rõ, exact duplicate và near-duplicate ở mức document; chỉ giữ text tiếng Việt. Tách văn bản dài theo câu/đoạn thành sequence có 8–96 đơn vị chữ/âm tiết chung và **≤256 token của cả hai tokenizer**; không để tokenizer âm thầm truncate. Không làm mất dấu hoặc word-segment riêng cho một model.
- Chỉ dùng **text** của train/validation/test cả năm benchmark để loại external text trùng exact hoặc chia sẻ 8-word shingle, như reader D1 `scripts/cdbk_dapt_pilot/prepare_corpus.py`. Không đọc nhãn, không dùng predictions, không đưa bất kỳ benchmark text nào vào corpus học trung gian. Holdout MLM/căn chỉnh theo document ID, tách trước train.
- Sample document bằng hash ID và seed 42 từ **toàn shard**, không chỉ 280K dòng đầu. Khóa cùng ID, text, thứ tự cho M/G. Cần **≥31M vị trí BamiBERT non-pad trong train corpus đã dedup** sau loại trùng/holdout; dùng 30M processed tokens mỗi arm, tối đa một pass kể cả batch cuối. Đây là số lần xuất hiện token trong văn bản đã dedup, không phải số token types trong vocab. Nếu không đủ hoặc nhiều dữ liệu hỏng/ngôn ngữ sai, dừng và báo gate fail. Không tải shard mới hoặc thay nguồn.
- Ghi manifest: file/revision/hash, số document và token trước/sau, decontam theo split, train/holdout ID hashes, chiều dài sequence, ví dụ audit chất lượng, license metadata và giới hạn quyền của trang gốc. Không phát hành raw web text. Điều kiện sản xuất của trang gốc cần đánh giá riêng nếu model sẽ deploy; pilot không tự kết luận quyền đó đã sạch.

## 4. Gate căn chỉnh hai tokenizer — trước lượt train dài

Nguồn checkpoint: student `models/bamibert-base-original` (HF commit `57bc1340…`, weight SHA256 `c8c8abe5…`) với **RobertaForMaskedLM và head MLM pretrained/tied**; teacher `uitnlp/CafeBERT@af76fcf2a04096b2b54b348a3e4eb48253c93c5d`, tokenizer `FacebookAI/xlm-roberta-large@c23d21b0620b635a76227c604d44e43a9f0ee389`. Teacher frozen `.eval()`, không task head. Cả hai giữ FP32 master weights/BF16 autocast. Kiểm load/reload, 12L/768 Bami, 24L/1024 Cafe, tokenizer/special IDs, missing/unexpected keys hợp lệ và tie của student MLM head.

Trên **cùng chuỗi raw NFC**, lấy các span liên tục chữ/số theo vị trí ký tự Unicode (khoảng trắng/dấu câu là ranh giới; tiếng Việt chia theo âm tiết là chấp nhận được trong pilot). Hai fast tokenizer trả offset mapping. Với mỗi span `i`, lấy trung bình hidden vectors của mọi subword có offset giao với span đó, bỏ special/pad. Một span chỉ dùng khi cả hai model map được; giữ các span theo cùng thứ tự. Không so token IDs hoặc logits vocab khác nhau.

Trên ít nhất 500 đoạn lấy trước bằng seed cố định: kiểm offset bằng tay 30 trường hợp có dấu, teencode/dấu câu/chuỗi lẫn số; ≥95% span có mapping hợp lệ ở cả hai model; không có span giao nhầm document/sequence; hidden finite, teacher không có gradient, student có gradient. Nếu gate này fail, **dừng trước lượt học 30M**, viết báo cáo kỹ thuật; không tự đổi sang CLS-only hoặc tokenizer chung vì đó là giả thuyết khác.

## 5. Objective và đối chứng

Với một đoạn có `n` span chung, `h_T[layer,i]` và `h_S[layer,i]` là hidden trung bình trên subwords của span `i`. Dùng hai cặp layer `(teacher 12, student 6)` và `(teacher 24, student 12)` (`hidden_states[0]` là embeddings). Chuẩn hóa L2 từng vector; tạo `C[i,j] = cosine(h[i], h[j])`. Bỏ đường chéo/pad rồi tính `P[i,:] = softmax(C[i,:] / 0.2)` theo hàng. **Loss quan hệ** là trung bình `KL(P_teacher || P_student)` theo hàng và hai layer; teacher stop-gradient. Ma trận có cùng kích thước `n×n` dù teacher/student khác chiều hidden và khác số subword. Dùng tối đa 96 span/đoạn để chặn chi phí `O(n²)`.

- **M:** BamiBERT gốc + 15% random MLM (80/10/10, không special/pad), CE trên masked Bami tokens.
- **G:** BamiBERT gốc, cùng minibatch/order/masks và loss MLM; thêm clean forward của student và clean forward của CafeBERT cho `L_rel`. Tổng `L_G = L_MLM + 1.0 × L_rel`. Không mất thêm inference pass sau training; encoder-only export bỏ MLM head. M chỉ có MLM forward; hai arm khớp **token/optimizer-step budget**, không khớp GPU FLOPs. Nếu G thắng, vòng xác nhận cần control về compute/regularization, chưa được quy toàn bộ gain cho nội dung tri thức Cafe.

Preflight trên vài minibatch cố định: tính tay KL cho ma trận nhỏ, diagonal/pad được bỏ thật, `KL(T,T)≈0`, model/trainer gradients hữu hạn, chỉ student update; `grad_norm(L_rel)/grad_norm(L_MLM)` ở encoder khác 0 và không ngoài `[0.05, 20]`. Nếu loss không tác động hoặc mất ổn định, **dừng**, không dò λ bằng downstream validation. Log MLM/rel loss và gradient ratio trong train để biết objective hoạt động.

Recipe cả hai arm: **30M non-pad Bami input tokens**, cùng số optimizer steps, global batch 128 sequences, maxlen 256, AdamW `lr=2e-5`, betas `(0.9,0.98)`, eps `1e-6`, weight decay `0.01`, warmup 6% + linear decay, BF16/SDPA, clip 1, seed 42. Dynamic padding và đếm non-pad thực; sai số ngân sách ≤1 global batch. Dùng RNG riêng cho mask/data order để clean forward thêm ở G không làm lệch MLM masks so M. Profile ~100 optimizer steps trên bản chạy thử rồi **khởi tạo lại từ checkpoint gốc** trước full run; đo runtime/VRAM, không dừng job người khác. Nếu G cần teacher on separate GPU, ghi mapping và tổng GPU-hours; không tự thay đổi batch/budget giữa M và G. Checkpoint **cuối ngân sách cố định**, không chọn bằng downstream F1. Holdout random-MLM loss cùng mask cố định cho O/M/G chỉ là diagnostic, không là gate chất lượng.

## 6. Fine-tune và quyết định đầu tư

M/G export encoder với tokenizer **BamiBERT gốc**. Gắn head CLS E14 mới; cùng seed 42, head init và data order giữa M/G, cùng E14 loader/split/evaluator, CE plain, effective batch 32, maxlen 256, BF16/SDPA, 30 epochs, validation-best checkpoint. LR đã chọn từ E14: Restaurant/ViSFD `5e-5`, Topic `1e-5`. Không NC/KD task logits/composite/LLRD/aspect-pair. O baseline E14 reuse; CafeBERT baseline reuse; không test.

| Validation seed 42, F1×100 | O BamiBERT | CafeBERT | Metric |
|---|---:|---:|---|
| Restaurant | 74,7418 | 77,7197 | Pair micro F1 |
| Topic | 79,5328 | 81,7446 | Fixed-label macro F1 |
| ViSFD | 84,3834 | 85,4665 | Pair micro F1 |

**Gate screening:** G phải hơn `max(O,M)` ít nhất `+0,5` điểm ở **≥2/3 task**, trong đó ít nhất một task là Restaurant hoặc Topic; task còn lại không giảm quá `0,5` so O. Báo best-epoch và đường cong để phát hiện một đỉnh đơn lẻ. Đây không phải significance hoặc chứng minh thắng CafeBERT. Nếu fail: dừng, kết luận cấu hình pilot không có tín hiệu. Nếu pass: **dừng tại gate**, báo root để quyết định seed 43 ghép cặp, control self-teacher/random relations và mở Sentiment/Hotel trước khi test đã khóa. Không tự chạy các bước đó hoặc thêm foundation teacher khác.

## 7. Correctness, artifacts và báo cáo

- M/G cùng corpus/order/MLM masks/initial checkpoint/head-init theo các giai đoạn; hash các đầu vào và kiểm một batch bitwise. Đảm bảo model G export chỉ chứa encoder Bami, không chứa teacher/projection mới; đếm parameters và chạy một inference smoke one-pass.
- Reload CPT exports; reload best FT checkpoint, recompute F1 từ predictions theo stable IDs; kiểm ABSA absent/ViSFD OTHERS đúng E14. Không thay raw gold/split, không sửa code/artifacts cũ.
- Source mới `scripts/cafe_bami_repdistill_pilot/`; dữ liệu mới `data/cafe_bami_repdistill_pilot/`; runs/checks mới `artifacts/cafe_bami_repdistill_pilot/`; báo cáo `docs/CAFEBERT_TO_BAMIBERT_REP_DISTILL_PILOT_RESULT.md`. Báo cả khi dừng ở data/alignment/gradient gate.
- Báo cáo có provenance/hash, alignment coverage, MLM/rel losses, runtime/GPU-hours, O/M/G/Cafe từng task, `G−M`, `G−O`, `G−Cafe`, best epochs và quyết định. Phân biệt nguồn đã kiểm với suy luận; không gọi FineWeb là review, không gọi validation là SOTA. Subagent thực thi tự làm trong scope, không tạo thêm subagent.

## Phụ lục A — sửa gate §5 trước khi launch (protocol v2, 2026-09-18)

Quyết định của root sau khi đọc `docs/CAFEBERT_TO_BAMIBERT_REP_DISTILL_PILOT_RESULT.md` (v1) và `artifacts/cafe_bami_repdistill_pilot/gate_decision.json`: ngưỡng `grad_norm(L_rel)/grad_norm(L_MLM) ∈ [0,05; 20]` là tiêu chí sàng lọc do ta đặt, không phải thước đo downstream; profile 100 step cho thấy L_rel giảm (0,080 → 0,055) và ratio lên quanh 0,05 khi gradient MLM ổn định. Vì vậy:

- **Gate preflight §5 (v2):** L_MLM, L_rel và gradient của chúng **hữu hạn**, ‖∇_enc L_rel‖ **khác 0** trên các minibatch cố định, và **không mất ổn định** trong profile ~100 step (loss hữu hạn, không phân kỳ, ratio > 0 suốt profile). Bỏ sàn/trần ratio. Ratio vẫn được log trong train để diễn giải, không là điều kiện dừng.
- **Giữ nguyên** λ = 1,0, nhiệt độ 0,2, hai cặp layer (T12→S6, T24→S12), corpus/lịch batch/mask đã khóa, recipe §5 và §6, gate screening §6. Không tăng λ, không whitening/centering (giả thuyết khác), không thêm teacher, không mở test.
- Nếu G không hơn `max(O, M)` theo gate §6: dừng cấu hình này. Nếu có gain: dừng tại gate; vòng sau mới tách tri thức CafeBERT khỏi regularization/compute của clean forward (control self-teacher/random relations).
- Ước lượng chi phí từ profile: ~0,4 GPU-h cho hai lượt 30M + ~3 GPU-h cho sáu fine-tune (ước lượng, không phải runtime thực).

Artifacts v1 (`preflight.json`, `gate_decision.json`, hai profile) được giữ nguyên làm bằng chứng; quyết định v2 ghi ở `artifacts/cafe_bami_repdistill_pilot/gate_decision_v2.json`, được tạo **trước** khi launch M/G.
