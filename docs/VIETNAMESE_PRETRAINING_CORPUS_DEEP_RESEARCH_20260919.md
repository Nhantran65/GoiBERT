# Deep research — corpus tiếng Việt cho continued pretraining mmBERT

**Ngày kiểm tra:** 2026-09-19. Phạm vi: corpus public có thể dùng để tạo ít nhất 1B token cho encoder 100–600M; đối chiếu model card/paper/trang phát hành chính thức và evidence local. Chưa tải corpus mới, chưa training, chưa dùng GPU. “License dataset” bên dưới thường chỉ áp dụng cho **gói dữ liệu/pipeline**, không tự cấp quyền với từng trang web gốc; đây không phải tư vấn pháp lý.

## 1. Kết luận

**Ứng viên chính: HPLT 3.0 `vie_Latn`, lấy các shard quality cao.** Đây là nguồn phù hợp nhất cho bước CPT mmBERT vì:

- Quy mô tiếng Việt: **145,4M document, 3,59B segment, 142,36B token, 234,49GB**; dư sức tạo pool 1–10B token.
- Dữ liệu từ cả Common Crawl và Internet Archive 2012–2024; mmBERT model card không liệt kê HPLT trong nguồn training, dù chắc chắn vẫn có overlap trang web ở tầng crawl.
- Global dedup cho tiếng Việt, OpenLID ở document/segment, PII metadata, adult/credential/robots filtering, `filter`, `doc_scores`, provenance URL/crawl và model register; shard được sắp theo Web Docs Scorer từ bin 10 xuống 5.
- Vietnamese chỉ có **12 shard** (`10_1`, `9_1–2`, `8_1–4`, `7_1–3`, `6_1`, `5_1`), nên có thể tải riêng bin 10/9 rồi đủ bao nhiêu token lấy bấy nhiêu, không cần tải toàn bộ.

Nguồn: [HPLT 3.0 dataset card và statistics](https://huggingface.co/datasets/HPLT/HPLT3.0/blob/main/README.md), [Vietnamese shard map](https://data.hplt-project.org/three/sorted/vie_Latn.map). HPLT ghi CC0 cho packaging nhưng nói rõ họ không sở hữu underlying text và người dùng chịu trách nhiệm tuân thủ pháp luật/quyền nội dung.

**Đối chứng chất lượng tốt nhất: FineWeb2/FineWeb2-HQ.** FineWeb2 có 50,89B Vietnamese words, 61,1M document và 319,8GB raw; thí nghiệm tác giả trên Vietnamese đạt aggregate 21,1 so với HPLT2 19,7, CulturaX 19,2 và raw Common Crawl 17,1. FineWeb2-HQ giữ top 10% theo classifier chất lượng: **4,0M document, 59GB**. Tuy nhiên mmBERT đã được pretrain từ FineWeb2 và FineWeb2-HQ, nên thêm cùng nguồn có độ mới thấp; local CafeBERT plain CPT 300M FineWeb2 cũng không tăng clean F1. Dùng FineWeb2-HQ làm control/anchor hoặc holdout corpus, không làm nguồn duy nhất cho arm mới. Nguồn: [FineWeb2 paper, Vietnamese results](https://arxiv.org/abs/2506.20920), [FineWeb2-HQ card](https://huggingface.co/datasets/epfml/FineWeb2-HQ), [mmBERT card](https://huggingface.co/jhu-clsp/mmBERT-base), local `docs/PILOT_B_RESULT.md`.

**Nguồn phụ thứ hai: MADLAD-400 Vietnamese.** Dataset đã audit thủ công ở mức ngôn ngữ, Vietnamese được giữ với ghi chú “ok”; khoảng 55M document sau lọc và bảng card báo xấp xỉ 48,7B token filtered. Nó hữu ích để tăng diversity ngoài HPLT. Nhưng metadata license trên HF không nhất quán: header ghi ODC-By, phần License trong card ghi CC-BY-4.0. Phải pin revision và giải quyết điều khoản trước khi dùng production. Nguồn: [MADLAD-400 card](https://huggingface.co/datasets/allenai/MADLAD-400), [paper NeurIPS Datasets & Benchmarks](https://papers.neurips.cc/paper_files/paper/2023/hash/d49042a5d49818711c401d34172f9900-Abstract-Datasets_and_Benchmarks.html).

## 2. Bảng xếp hạng

| Corpus | Quy mô tiếng Việt công bố | Curation | Quyền/phát hành | Mức ưu tiên cho mmBERT |
|---|---:|---|---|---|
| **HPLT 3.0** | 142,36B token; 145,4M docs; 234,5GB | Global dedup, quality bins, LID document/segment, metadata register/PII/provenance | CC0 cho packaging; underlying pages không thuộc HPLT | **A — nguồn chính mới** |
| **FineWeb2-HQ** | 4,00M docs; 59GB, top 10% FineWeb2 | Model-based quality filter; transparent parent pipeline | ODC-By + Common Crawl ToU; underlying rights còn | **A− — control/anchor, không mới với mmBERT** |
| **FineWeb2 full** | 50,89B words; 61,1M docs; 319,8GB | Language-adaptive filtering/dedup; Vietnamese benchmark evidence tốt | ODC-By + Common Crawl ToU | **B+ — pool/control; mmBERT đã thấy** |
| **MADLAD-400** | ~55M docs filtered; ~48,7B token theo card | Manually audited language set; nhiều filter | HF header ODC-By nhưng body CC-BY-4.0 | **B+ — diversity sau audit license** |
| **CulturaX** | 98,45B token; 102,4M docs | mC4 + nhiều OSCAR release; LID, URL/perplexity cleaning, MinHash | Gated; kế thừa điều khoản mC4/OSCAR | **B — fallback, overlap lớn/đời cũ hơn** |
| **OSCAR 23.01** | 22,42B words; 33,9M docs; 140,8GB | LID, category/adult signals, TLSH; cần tự filter/dedup | Controlled access/terms | **C+ — CulturaX/HPLT tiện và sạch hơn** |
| **HPLT 2.0 cleaned** | 83,2B words; 100,8M docs; 494,7GB | Cleaned + dedup, older than v3 | CC0 packaging + same underlying caveat | **C — dùng v3 thay thế** |

FineWeb2 Vietnamese evaluation cho thấy không có corpus web nào thắng mọi loại task: FineWeb2 hơn aggregate, nhưng CulturaX/HPLT2 ngang hoặc hơn ở reading comprehension. Vì vậy “quality” phải gắn với mục tiêu downstream; không lấy một quality score làm sự thật tuyệt đối.

## 3. Corpus tiếng Việt chuyên biệt/public khác

### Có thể làm anchor hoặc bổ sung nhỏ

- **`hoanghai2110/vi-pretrain-clean`:** 511K rows, khoảng 230M token/0,69GB; CC0; 74% document là Wikipedia, phần còn lại chủ yếu OpenSubtitles/CCAligned/WikiMatrix. Có 8 rule lọc được mô tả, nhưng quá nhỏ và lệch encyclopedia/parallel/subtitle để làm core 1B. Phù hợp làm clean anchor/holdout. [Card](https://huggingface.co/datasets/hoanghai2110/vi-pretrain-clean)
- **ViLegalText:** 16GB legal Vietnamese, Apache-2.0 theo card; content extraction, cleaning, LID, dedup; có paper và model CPT chứng minh corpus dùng được trong legal domain. Chất lượng/provenance tốt hơn phần lớn community corpus, nhưng domain legal lệch mạnh với review/feedback. Chỉ nên ≤5% diversity hoặc bỏ hẳn. [Card](https://huggingface.co/datasets/ntphuc149/ViLegalText)
- **Vietnamese Wikipedia/Wikisource/public-domain books:** quyền sử dụng rõ hơn và chất lượng câu tốt, nhưng quy mô và register hẹp. Dùng anchor nhỏ; không đủ một mình cho mục tiêu review/social.

### Quy mô lớn nhưng provenance/license chưa đủ tin cậy

- **BlossomsAI `vietnamese-corpus`:** 24,4GB/29M rows, card Apache-2.0; mô tả ~1,3M Wikipedia + >13M news + text. Card không ghi URL nguồn, revision crawl, near-dedup/LID hoặc quyền underlying news; thống kê README còn placeholder/mâu thuẫn với 29M rows. Không dùng trước audit sâu. [Card](https://huggingface.co/datasets/BlossomsAI/vietnamese-corpus)
- **`fsnaix/vietnamese-corpus-large`:** 24,2GB, claim 106M docs (90,8M web), Apache-2.0 và gated. Mô tả gần như không có pipeline/provenance; file config/card không nhất quán về web split. Không dùng làm core chỉ vì số docs lớn. [Card](https://huggingface.co/datasets/fsnaix/vietnamese-corpus-large)
- **`tiennv/vietnamese-corpus`:** 4,23GB/25 parquet shards nhưng không có dataset card/license/provenance đủ để đánh giá. Không shortlist.
- **Binhvq News Corpus:** ~14,9M bài/18,6GB, chính là nguồn news mà CafeBERT dùng sau dedup. Repo ghi MIT nhưng đó không tự cấp quyền cho nội dung của hàng trăm báo; domain news cũng lệch ABSA. Hữu ích cho reproduction CafeBERT/research, không nên là core production. [Repo](https://github.com/binhvq/news-corpus)
- **Vietnamese Book Corpus:** 1,92GB nhưng không thấy license/provenance tác phẩm đủ rõ; loại khỏi production shortlist.

### Không phù hợp làm corpus chính của hướng này

- **ViSoBERT social corpus:** 166M token positions sau local dedup/decontam, research-only/CC-BY-NC; chỉ 2,9% có tín hiệu review và CafeBERT DAPT Restaurant không tăng. `docs/CDBK_DAPT_PILOT_RESULT.md`.
- **Review Restaurant/Hotel public đã audit:** chất lượng v3 tốt nhưng chỉ ~208K token; FineWeb2 filter review chỉ ~40% precision và 138K token. Không đủ pretraining. `docs/BAMIBERT_ASPECT_SENTIMENT_CPT_PILOT_RESULT.md`.
- **Corpora của BamiBERT/ViDeBERTa:** paper báo 129GB/138GB nhưng full corpus/hashes không được phát hành đủ để project tái dùng/audit.

## 4. License và production

Không có web-scale corpus nào trong shortlist cho phép nói đơn giản “commercial-safe” chỉ dựa trên badge CC0/ODC/Apache:

- HPLT nói rõ CC0 áp dụng cho **packaging**, họ không sở hữu underlying text.
- FineWeb2-HQ tuân ODC-By và Common Crawl Terms; card yêu cầu lọc thêm PII/harmful content.
- CulturaX kế thừa license/terms của mC4 và OSCAR.
- Community uploads gắn Apache/CC0 có thể không có quyền tái cấp phép cho báo, sách hoặc website gốc.

Cho pilot nghiên cứu có thể dùng với attribution, provenance và takedown manifest. Trước khi phát hành model production, cần legal review riêng theo nơi triển khai; ưu tiên giữ URL/crawl ID, loại robots opt-out, PII/adult/credentials và hỗ trợ takedown. Không phát hành lại raw text trong artifacts.

## 5. Mixture đề xuất cho pilot mmBERT

### Nguồn dữ liệu

Pool chính nên là **HPLT 3.0 Vietnamese quality bins 10 rồi 9**, không phải tải ngẫu nhiên toàn corpus. Chỉ mở bin 8 nếu hai bin đầu không đủ sau filter/decontam. Giữ document-level provenance và quality metadata.

Mixture 1B token ban đầu:

| Thành phần | Tỷ lệ | Mục đích |
|---|---:|---|
| HPLT v3 WDS 9–10, general high-quality | 65% | Nguồn Việt mới, câu sạch, diversity |
| HPLT v3 register gần user-generated (`OP`, `ID`, narrative/opinion blog/review nếu metadata thực sự có coverage cho vi) | 20% | Review/social style mà không đòi corpus nhà hàng thuần |
| MADLAD-400 Vietnamese audited hoặc HPLT bin 8 nếu license MADLAD chưa giải quyết | 10% | Source/pipeline diversity |
| Clean anchor: Wikipedia/`vi-pretrain-clean` | 5% | Hạn chế drift, giữ formal Vietnamese |

Trước khi dùng register labels, phải kiểm sample và coverage thật của `web-register` trên Vietnamese; schema có trường này nhưng không tự chứng minh classifier chính xác cho mọi document. Nếu coverage/precision không đạt, thay 20% bằng selection theo embedding similarity với **train text only** của bốn miền, giữ exact/shingle decontamination với train/dev/test.

### Đối chứng nghiên cứu

- **O:** mmBERT gốc.
- **P:** plain MLM trên đúng pool 1B token nhưng shuffle uniform.
- **C:** cùng documents/tần suất/token budget/masking/optimizer, chỉ thay **lịch thứ tự/sampling** từ general → user-generated/target-near. Đây mới cô lập được curriculum.

Không dùng FineWeb2 làm toàn bộ P/C vì mmBERT đã học nguồn đó. FineWeb2-HQ có thể làm một fixed external MLM holdout và 5–10% anchor nếu cần, nhưng không nên được trình bày là dữ liệu mới.

## 6. Gate trước khi tải/train lớn

**Data audit, không GPU model:** stream/sample 500 docs cho mỗi source/bin/register; đo Vietnamese precision, boilerplate, machine-translation/generated text, PII/adult, exact/near duplicates và overlap với FineWeb2 shard local + cả năm benchmark. Gate đề xuất:

- ≥95% document tiếng Việt theo audit;
- ≥85% “usable natural text”, không boilerplate/MT/generated/adult;
- exact benchmark overlap = 0 sau decontam; báo shingle drops;
- ≥1,2B unique mmBERT non-pad token trong pool để train 1B ≤1 pass;
- HPLT contribution không trùng exact/near với FineWeb2 quá mức đã khóa trước;
- license/terms/revision/hash/source URL đầy đủ.

Chỉ sau gate dữ liệu mới viết/run CPT. Với 1,5TB disk trống hiện tại, tải subset 1–10B token khả thi; không cần mirror toàn bộ HPLT/FineWeb2 hàng trăm GB nếu stream và materialize đúng mixture.

## 7. Quyết định

1. **Dùng HPLT 3.0 top-quality Vietnamese làm core.**
2. **Dùng FineWeb2-HQ làm quality reference/control, không coi là nguồn mới.**
3. **MADLAD-400 chỉ thêm sau khi khóa được license revision; CulturaX là fallback.**
4. **Không dùng corpus community 24GB/106M docs trước provenance audit.**
5. **Không cần tìm corpus restaurant/hotel thuần đủ 1B nữa:** dùng register/curriculum để tăng tỷ lệ opinion–discussion–review style, rồi kiểm downstream bằng arm plain-MLM cùng corpus.

Kết luận thực tế: project **có đủ dữ liệu tiếng Việt quy mô và chất lượng để thử CPT 1B**, nhưng corpus tốt nhất không phải review dataset; nó là một mixture có provenance, với HPLT 3.0 làm nguồn mới và FineWeb2/MADLAD làm control/diversity. Tính mới của paper nên nằm ở **target-register/domain annealing có đối chứng cùng corpus**, không nằm ở claim “thu thập được web tiếng Việt”.
