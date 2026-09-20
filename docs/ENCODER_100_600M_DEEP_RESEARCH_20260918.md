# Deep research: hướng vượt BamiBERT/CafeBERT trong 100–600M tham số

Ngày kiểm tra: **2026-09-18**. Root tự nghiên cứu; không subagent, không training, không inference model mới. Phạm vi: checkpoint public, paper/model card/code gốc và evidence local; không phải một systematic review toàn bộ literature hoặc chứng nhận SOTA.

## 1. Kết luận và lựa chọn

**Chưa tìm được checkpoint trong giới hạn này có bằng chứng công khai đủ để khẳng định sẽ vượt CafeBERT trên đúng năm task hiện tại.** Có hai ứng viên mới đáng thử với chi phí nhỏ, nhưng không đủ lý do mở tiếp một grid năm task ngay.

1. **Ưu tiên thử backbone: `microsoft/mdeberta-v3-base`, khoảng278M encoder.** Có số đo Vietnamese NLI, objective tiền huấn luyện khác họ MLM đã thử, checkpoint khả dụng và implementation ổn định. Không đồng nhất với ViDeBERTa của Fsoft.
2. **Dự phòng: `LiquidAI/LFM2.5-Encoder-350M`, khoảng354.5M.** Có benchmark fine-tuning đa seed và kiến trúc khác thực chất; nhưng kết quả multilingual công bố không cho thấy hơn XLM-R-large. Không dùng thứ hạng tổng17task để suy thắng tiếng Việt.
3. **Nếu giữ CafeBERT để ưu tiên vượt chính baseline mạnh nhất:** aspect-conditioned sentence-pair classification là một hướng phương pháp chưa thấy được thực thi trong repo. Giữ size~560M, tăng compute; không phải block mới/novelty. Khác E9 vì đưa aspect vào encoder từ input, thay vì query readout sau encoder.

**Pilot ưu tiên duy nhất tôi đề xuất trước mắt:** mDeBERTa trên Restaurant và Topic, haiLR, mộtseed =4FT runs, chưa CPT/KD/test. Phần §8 chốt cách dừng. Không tự chạy cả shortlist.

## 2. Reset bằng chứng local trước khi chọn hướng

### Test mới nhất (không trộn validation)

| Task | BamiBERT E14 | CafeBERT | SEA-LION300M |
|---|---:|---:|---:|
| Sentiment macroF1 | **83.03** | 82.36 | 77.90 |
| Topic macroF1 | 78.18 | **80.77** | 77.29 |
| ViSFD pairmicroF1 | 84.50 | **85.48** | 79.42 |
| Restaurant pairmicroF1 | 74.15 | **77.39** | 65.06 |
| Hotel pairmicroF1 | 73.94 | **75.17** | 71.21 |

Nguồn: [SEA-LION benchmark](SEALION_300M_BENCHMARK_RESULT.md), [E14](E14_BAMIBERT_BASELINE_RESULT.md). Mộtseed; Bami3LR/SEA2LR/Cafe1LR; không kết luận significance. Cafe không thắng Bami ở Sentiment test dù thắng validation. Hotel75.17 là bằng chứng trực tiếp chống cách nói chắc chắn “trần74”.

### Không được đề xuất lại các chương trình đã có

- BamiBERT→DAPT40K→general distillation30K→task stage: đã làm. Lịch sử trong [family report](../tmp/GOI_MODELS_SUMMARY_REPORT_EN.md). DAPT trực tiếp không thắng rộng; VLSP2018 Restaurant không phải UIT-ABSA Restaurant.
- E2 task-logit KD từ teacher DAPT: Restaurant/Hotel giảm qua3seeds; không lấy chất lượng student GD làm bằng chứng task KD thắng.
- CafeBERT→taskKD M1: student73.94 < Bami74.74 validation.
- CafeBERT+300M social DAPT→taskKD D1: teacher77.58≈77.72, student73.34. Social data mismatch, chưa loại đúng-domain DAPT; cũng không tạo lý do chạy lại cùng loại corpus.
- E8 NC/LLRD giúp student26M; E19 không chuyển thành gain rộng ở103M.
- E22 composite giúp Restaurant nhưng không rộng; chưa chứng minh áp sang CafeBERT.
- SEA-LION first-token head thất bại: đóng nhánh theo quyết định user. Native mean pooling chưa đo, nhưng không tự mở lại nhánh đã đóng. Không gọi first-token failure là bằng chứng mọi ModernBERT đều kém.

## 3. Kiểm tổng tham số, không dùng tên model thay phép đếm

Đọc HF metadata và config pinned; với kiến trúc hỗ trợ local, dựng **meta-device model chỉ để đếm shape**, không tải weights hay forward. EuroBERT trừ riêng output MLM head không tied theo code gốc. Đây là accounting audit, không xác minh đã load checkpoint inference thành công.

| Checkpoint | Số tham số/ước lượng đã kiểm | Quyết định |
|---|---:|---|
| BamiBERT local | ~103M với taskhead | Reference |
| CafeBERT local | ~560M với taskhead | Reference mạnh nhất ở4/5testtasks |
| mDeBERTa-v3-base | **278.22M encoder**, chưa head | Shortlist1 |
| LFM2.5-Encoder-350M | **354.48M stored tied-MLM model** | Shortlist2; count downstream lại khi load |
| mGTE multilingual MLM | **306.21M stored MLM** | Dự phòng thấp hơn |
| EuroBERT-210m | **211.77M encoder**; storedMLM310.27M | Đúng giới hạn, evidence classification chưa nổi trội |
| EuroBERT-610m | **607.87M encoder**; storedMLM755.63M | Ngoài trần600M đang chốt; không tự nới |
| ViDeBERTa-base Fsoft | **183.75M encoder**, không phải86M total | Hạ ưu tiên, cần kiểm lineage checkpoint |
| PhoBERT-base-v2 | **134.41M encoder** | Baseline Việt nhỏ đáng lưu ý, không evidence thắngCafe |
| PhoBERT-large | **368.11M encoder** | Evidence Cafe paper đã hơn trên task gần; không ưu tiên mới |
| ViT5-base | **225.95M full encoder-decoder theo config** | Phương án đổi formulation; không plug-in BERT |
| ViT5-large | **791.28M full model theo config** | Ngoài giới hạn |

Audit số: [parameter_accounting.json](../artifacts/research_20260918/parameter_accounting.json), [checkpoint_inventory.json](../artifacts/research_20260918/checkpoint_inventory.json). Số modelcard có thể làm tròn hoặc chỉ non-embedding. Cần đếm lại model thực được deploy, có head và unique tied parameters.

**Lưu ý EuroBERT:** 310M stored không có nghĩa encoder210M quảng cáo sai: untied MLM output table được bỏ lúc classification. Ngược lại, 86M của ViDeBERTa là backbone không bao gồm bảng embedding128K×768. Không lẫn hai kiểu đếm này.

## 4. Shortlist: bằng chứng thuận và nghịch

### A. mDeBERTa-v3-base — ưu tiên thử

Microsoft công bố Vietnamese XNLI zero-shot dev accuracy78.1, so XLM-R-base76.5. Đây là evidence tiếng Việt trực tiếp, nhưng **NLI không phải ABSA và comparator không phải CafeBERT-large**. [Model card](https://huggingface.co/microsoft/mdeberta-v3-base)

DeBERTaV3 dùng replaced-token detection: học phân biệt token gốc/thay thế; generator/discriminator sharing có xử lý gradient riêng. Đây là khác biệt pretraining có sẵn trong checkpoint, không cần chúng ta tự xây lại generator hoặc trả chi phí pretrain lớn. [Paper §3–4](https://arxiv.org/html/2111.09543v4)

Lý do tôi xếp đầu: có Vietnamese evidence, tổngsize thấp hơnCafe khoảng một nửa, inference mộtpass, khác objective của phần lớn checkpoints đã benchmark, ít rủi ro compatibility hơn custom encoder mới. Đây là đánh giá lợi ích/chi phí của phép thử, không phải xác suất thắng đã đo.

Không dùng `microsoft/deberta-v3-large` tiếng Anh thay cho **m**DeBERTa đa ngữ. Cũng không suy ViDeBERTa-base thua trong paperBami thì checkpointMicrosoft này đã bị thử và loại.

### B. LFM2.5-Encoder-350M — ứng viên thực, chưa là lời giải

Model card xác nhận15ngôn ngữ có tiếng Việt, encoder hai chiều MLM, tổng354.5M. Family hybrid conv/attention; chọn **Encoder**, không nhầm Embedding/ColBERT/causal350M. [Model card](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M)

Theo bảng tác giả: mean17tasks81.02 so XLM-R-large81.34; XNLI79.82 vs84.69; Amazon60.57 vs61.58; MASSIVE85.70 vs88.50. Vì vậy claim “model mới350M mạnh hơnCafe” không được hỗ trợ. Không trừ các số này với benchmark local vì khác tasks/metrics. [Bảng và protocol nhà phát hành](https://www.liquid.ai/blog/lfm2-5-encoders)

Đọc code harness: lớp đầu ra dùng masked mean pooling+linear; tokenizer riêng; pinTransformers4.56.2, không ép vào venv4.49 cũ. README nói các multilingual scores average Euro language set; code cóvi trong tập language tải. Chưa có bảngvi riêng để xác nhận. README hiện cũng không kèm toàn bộ run artifacts dù blog nói raw results mở. Giữ mức evidence là **author-reported**, chưa audit độc lập. [Harness](https://github.com/Liquid4All/encoder_eval)

Nếu thử: native pooling, isolatedenv, kiểm bidirectional mask/conv padding. Không sao chép BOS classifier từSEA rồi kết luận về toàn bộ model.

### C. mGTE multilingual MLM — hợp lệ nhưng thấp ưu tiên hơn

`Alibaba-NLP/gte-multilingual-mlm-base` là checkpoint MLM đa ngữ, khác bản embedding retrieval. Card báo GLUE83.47/XTREME-R64.44, so XLM-R-base80.44/62.02. Những số này hỗ trợ encoderNLU, nhưng chưa thắngCafe hoặc đúng5tasks. [Model card](https://huggingface.co/Alibaba-NLP/gte-multilingual-mlm-base)

Paper phân biệt MLM encoder, contrastive embedding và reranker; không mượn MTEB gain của bản đã trainretrieval cho baseMLM. Khả năng8k không phải ưu thế quyết định trên câu ngắn của project. [Paper §2–3](https://arxiv.org/html/2407.19669v2)

### D. EuroBERT / PhoBERT / ViDeBERTa

- **EuroBERT210M:** pretrained cóVietnamese, nhưng paper §3.2 nói classification nhìn chung ngang model cùngcỡ; ưu thế code/retrieval không bảo đảm sentiment. Giữ dự phòng, không coi mới hơn=đáng thử trước. [Paper](https://arxiv.org/html/2503.05500v2)
- **PhoBERT-v2:** có thêm120GB OSCAR ngoài20GB corpus cũ; cần RDRSegmenter/tone normalization đúng preprocessing. Đây là control nhỏ, không evidence hiện tại thắngCafe. [Model card](https://huggingface.co/vinai/phobert-base-v2)
- **ViDeBERTa:** paper gốc đánh giá POS/NER/QA, không phải5task này; word segmentation PyVi trong pretrain. PaperBami dùng RDRSegmenter, có thể là khác preprocessing nhưng chưa chứng minh gây gap. Bami báo Restaurant63.78 choViDeBERTa-base vs74.89Bami; evidence không ủng hộ chọn mù checkpoint này. [ViDeBERTa paper](https://aclanthology.org/2023.findings-eacl.79.pdf), [BamiBERT §3](https://arxiv.org/html/2607.02259v1)
- Link giả định `Fsoft-AIC/videberta-large` trả401; chưa xác minh được checkpointlarge public khả dụng. Không đồng nhất HTTP401 với chứng minh không tồn tại model.

## 5. Hướng phương pháp đáng xem hơn thêm block: input aspect-conditioning

Sun etal.2019 biến ABSA thành sentence-pair classification với auxiliary aspect text. Có kết quả SentiHood/SemEval, nhưng metric category detection/sentiment accuracy không phải local end-to-end pairF1. Không dùng điểm90% trong paper làm target local. [Paper §2 vàTables3–5](https://aclanthology.org/N19-1035.pdf)

### Khác gì E9 đã fail?

- E9: encoder đọc review một lần, sau đó learned queries đọc token states; queries không phải tên aspect bằng ngôn ngữ đi xuyên encoder.
- Phương án này: encoder đọc **review + mô tả aspect** cùng nhau từlayer1; mỗiaspect có contextual states riêng.
- Chưa tìm thấy implementation/run hoàn tất dạng này khi rà docs/code E1–E22. Đây là kết quả tìm kiếm local, không tuyên bố chưa ai thử bên ngoài repo.

Ví dụ: input pair `("Đồ ăn ngon nhưng phục vụ chậm", "chất lượng phục vụ")`; output4classes absent/positive/negative/neutral. Khi inference phải chạy **tất cả12aspects**, không chỉgold-present aspects. Hotel34;ViSFD11 vớiOTHERSbinary xử lý riêng. Tên/mô tả aspect chỉ từ ontology, không lấyvalidation errors để viết prompts.

Giữ CafeBERT pretrained gốc, shared4-way head, không tăng encoderparams. Nhãn phrase mapping cần làm trước training và giữ nguyên. **Chi phí thực tăng khoảng12encoderpasses/review ởRestaurant,34ởHotel**; batch/vectorize chỉ cải thiện throughput, không xóaFLOPs. Cùng560Mweights không có nghĩa cùng inferencecost. Vì user mới giới hạn params, đây là trade-off phải nói rõ.

Pilot nếu chọn hướng này thay backbone: mộttaskRestaurant, cố địnhLR2e-5; matched control đổi aspectphrases thành các mãaspect có identity riêng, cùng sốpasses/steps/head. Cả hai cùng input-pair format để đo lợi ích semantics tênaspect. So thêm Cafe77.72 cũ để đo gainhệ thống, nhưng không gọi là compute-matched vớibaseline mộtpass. Chưalaunch và chưa viết fullplan.

Không mở rộng claim thắng5task từ thí nghiệmABSA. VSFC classification khác, có thể giữ recipe thường; nếu task-specific system thì báo đúng như vậy.

## 6. Nếu tiếp tục pretraining, điều gì thực sự khác lịch sử?

**Không đề xuất plainMLM cùng kiểuweb/social thêm một lần.** Hai hướng khác có tiền lệ:

1. **Chuyển sang checkpoint đã có objectiveRTD:** đây là lý do thửmDeBERTa trước. Không phải claim đã tự pretrain một model mới.
2. **Sentiment-aware intermediate pretraining:** SKEP học từ sentiment masking/wordpolarity/aspect pairs; SCAPT dùng contrastive sentiment trên review cóweaklabels. Khác plainMLM40K cũ vì supervision/objective khác. [SKEP](https://aclanthology.org/2020.acl-main.374.pdf), [SCAPT](https://aclanthology.org/2021.emnlp-main.22.pdf)

SCAPT dùng khoảng1.56M restaurant sentences từYelp và0.51M laptop từAmazon saulọc, cóweaklabels theo stars. **Không thể mượn gain để hứa vài nghìn review Việt sẽ đủ.** NguồnEnglish + adaptationcrosslingual là chương trình khác cần control; ratingtoaspect còn cónoise. Trước khi chọn phải có dữliệu đủ và phương pháp chuyển supervision rõ. Hiện chưa đủ điều kiện ưu tiên hơn4FT probe.

### Kiểm được một nguồn dữ liệu mới, nhưng không phải “mỏ500k reviews”

ViTASA2025 công bố hơn500k target-aspect pairs,3domains. Repo tácgiả có3JSONL; tôi tải để kiểm read-only, tổng~2.7MB. Mỗi file chỉ**2.000rows**, không phải500kdocuments. [Repo gốc](https://github.com/kh4nh12/ViTASA)

| File public | Rows | Unique normalized text | Whitespace units | Exact overlap currenttrain/currentval |
|---|---:|---:|---:|---:|
| restaurant | 2000 | 1998 | 110311 | 0 / 0 |
| hotel | 2000 | 1986 | 59831 | 0 / 0 |
| mobile | 2000 | 1999 | 72355 | **1300 / 192** vớiViSFD |

Normalization:NFC,lowercase,removeBOM,whitespacecollapse; **chưa kiểm near-duplicate hoặctest**, nên0exactkhông chứng minh corpusđộc lập. 500kpairclaim không bị bác bỏ bởi6000rows public; có thể unit/coverage khác, phải hỏi/đọc release đầy đủ. Nhưng không thể dùng claim đó để lên kế hoạch100Mreviewtokens. Mobile overlapvalidation phải loại trước mọitraining nếu dùngnguồn này.

Audit: [text_inventory.json](../artifacts/research_20260918/vitasa/text_inventory.json). Chưa sử dụng corpus này cho model nào. Fullpaper publisher bị403 trong lượt nghiên cứu, nên không suy method/scale ngoàiREADMEvàrelease thựcđọc.

## 7. Những kết luận research này không cho phép

- Không cóprobability “70%thắngCafe”; chưa có thínghiệm đểước lượng.
- Không lấyGLUE/MTEB/XNLI đểtuyên bốSOTA UIT-ABSA.
- Không lấylowerMLMloss làmproxy chắc chắn chohigherF1.
- Không gọi86Mnonembedding là86Mtotal; cũng không tínhuntiedMLMdecoder khôngdeploy vào sizeclassifier.
- Không coi E5meanpoolstudent26M fail là loại meanpoolcho mọicheckpoint; ngược lại, không tự mởSEA lại sauuser đã đóng.
- Không coi modelcardauthorclaims ngang vớiindependentreplication.
- Không tuyên bốbộHotel cótrần74; testCafe75.17đãkhác.
- Không đổi benchmark sangsocial chỉđể tìmwin rồi claimthắngnăm taskcũ.

## 8. Pilot ưu tiên — ít lượt nhưng trả lời được câu hỏi

**Cập nhật sau research:** user đã chọn 5 task × 1 LR và cho phép chạy. Đề xuất 2 task × 2 LR dưới đây được giữ như lịch sử, đã được thay bằng [MDEBERTA_PILOT_PLAN.md](MDEBERTA_PILOT_PLAN.md).

**Chưa launch trong lượt research này.** Nếu user chọn tiếp:

- Model: `microsoft/mdeberta-v3-base`, revision đã lưu trong inventory.
- Restaurant + Topic, seed42, LR `{1e-5, 2e-5}`, 30 epochs, effective batch32, maxlen256: **4 runs**.
- Dùng context pooler/head theo DeBERTa; ABSA softmax 4 trạng thái riêng từng slot. Không NC, KD, LLRD hoặc composite; không áp word segmentation của ViDeBERTa vào mDeBERTa. Giữ FP32 master weights và BF16 autocast; nếu cần thay dependencies thì dùng môi trường riêng.
- Tái dùng baseline validation Bami/Cafe. Ngân sách chọn LR khác nhau phải được ghi rõ, không gọi đây là so sánh hoàn toàn khớp mọi điều kiện.
- Mốc validation Restaurant: Bami74.7418 / Cafe77.7197; Topic: Bami79.5328 / Cafe81.7446.
- **Tín hiệu mạnh:** hơn Cafe ít nhất0.3 điểm ở một task, task kia không giảm quá0.5 → xác nhận seed43 trước khi mở cả năm task.
- **Chỉ có tín hiệu về hiệu quả kích thước:** hơn Bami ít nhất0.3 ở cả hai task nhưng vẫn thua Cafe → báo đúng rằng chưa đạt mục tiêu vượt Cafe; không tự đổi mục tiêu thành SOTA.
- Thua Bami ở cả hai hoặc không có tín hiệu → dừng candidate; không tự mở thêm grid hoặc CPT.
- Các ngưỡng là heuristic đầu tư, không phải significance. Không dùng test để chọn recipe; khi công bố phải nói rõ lịch sử đã quan sát test nhiều vòng.

**Nếu ưu tiên cải thiện CafeBERT trên ABSA hơn việc tìm backbone mới:** chọn phép thử input aspect-conditioning ở §5 thay cho bốn runs trên. Không chạy đồng thời cả hai hướng trong vòng đầu. LFM/mGTE/EuroBERT là danh sách dự phòng, không phải queue đã được ủy quyền.

## 9. Hồ sơ bằng chứng và hạn chế tìm kiếm

- `artifacts/research_20260918/checkpoint_inventory.json`: HFAPIstatuses, revisions/configs/files, safetensors totals.
- `parameter_accounting.json`: shape-onlycounts và phươngpháp đếm.
- `lfm_card.md`, `liquid_tasks.py`, `liquid_loaders.py`: sourcesnapshot đểđọcprotocol/pooling/languageaggregation, khôngexecute mãtải về.
- `videberta_card.md`, `phobert_card.md`, `eurobert_model.py`: đốichiếucardvàimplementation.
- `vitasa/` và `vitasa_tree.json`: publicrelease/read-onlydataaudit.
- `source_manifest.json`: SHA256cácnguồnđãlưu; cácGitHubmain snapshots cóhash nhưng khôngđồngnghĩa đãpinreleasecủapaper.

Đã đối chiếu nguồn gốc của BamiBERT, CafeBERT/VLUE, ViDeBERTa, PhoBERT-v2, mDeBERTaV3, EuroBERT, mGTE, LFM2.5-Encoder, BERT-pair, SKEP, SCAPT và ViTASA. Không gộp kết quả UIT-ABSA, VLSP, targeted ABSA và sentiment trên aspect đã biết thành một leaderboard. Chưa hoàn thành kiểm toán SOTA mọi công trình2026; kết luận ở đây nhằm chọn phép thử đáng làm, không chứng nhận SOTA.
