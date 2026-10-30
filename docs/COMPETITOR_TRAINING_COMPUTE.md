# Cấu hình và chi phí huấn luyện của các mô hình đối sánh

## Nguồn

| Mô hình       | Nguồn                                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------------- |
| PhoBERT         | Nguyen & Nguyen, Findings of EMNLP 2020, [arXiv:2003.00744](https://arxiv.org/abs/2003.00744)            |
| PhoBERT-base-v2 | [README VinAIResearch/PhoBERT](https://github.com/VinAIResearch/PhoBERT)                                 |
| BamiBERT        | Nguyen et al., 2026, [arXiv:2607.02259](https://arxiv.org/abs/2607.02259)                                |
| ViSoBERT        | Nguyen et al., EMNLP 2023, [arXiv:2310.11166](https://arxiv.org/abs/2310.11166)                          |
| ViDeBERTa       | Tran et al., 2023, [arXiv:2301.10439](https://arxiv.org/abs/2301.10439)                                  |
| XLM-R           | Conneau et al., ACL 2020, [arXiv:1911.02116](https://arxiv.org/abs/1911.02116)                           |
| CafeBERT        | Do et al., Findings of NAACL 2024 (bài báo VLUE), [arXiv:2403.15882](https://arxiv.org/abs/2403.15882) |
| GoiBERT         | Tran et al., 2026, [arXiv:2610.30067](https://arxiv.org/abs/2610.30067)                                  |

## 1. Kiến trúc

| Mô hình                       |                                          Số tham số | Lớp × chiều ẩn × head                    | Bộ từ vựng                       |                Độ dài tối đa | Đầu vào                             |
| ------------------------------- | ----------------------------------------------------: | --------------------------------------------- | ----------------------------------- | --------------------------------: | -------------------------------------- |
| PhoBERT-base / base-v2          |                                                  135M | 12 × 768 × 12                               | 64K (fastBPE)                       |                               256 | Phải tách từ trước (RDRSegmenter) |
| PhoBERT-large                   |                                                  370M | 24 × 1024 × 16                              | 64K                                 |                               256 | Phải tách từ trước                |
| BamiBERT                        |                                                102,4M | 12 × 768 × 12                               | 20.481 (byte-level BPE của PhoGPT) |                             2.048 | Văn bản gốc                         |
| ViSoBERT                        |                                                   97M | 12 × 768 × 12                               | 15.002 (SentencePiece)              | 128 (Bảng 6) hoặc 512 (Bảng 3) | Văn bản gốc                         |
| ViDeBERTa xsmall / base / large |                        22M / 86M / 304M (phần thân) | 6 / 12 / 24 lớp; chiều ẩn 768 / 768 / 1024 | 128K (SentencePiece của DeBERTaV3) |                                — | Phải tách từ trước (PyVi)         |
| XLM-R base                      | 270M (277,5M theo cách đếm của bài báo GoiBERT) | 12 × 768 × 12                               | 250K (SentencePiece)                |                               512 | Văn bản gốc                         |
| XLM-R large                     |                                                  550M | 24 × 1024 × 16                              | 250K                                |                               512 | Văn bản gốc                         |
| CafeBERT                        |                        ~560M (dựa trên XLM-R large) | 24 × 1024 × 16                              | 250K                                |                               512 | Văn bản gốc                         |
| **GoiBERT-base / large**  |                               **26,4M / 49,4M** | 10 × 384 × 6 / 12 × 512 × 8               | 20.481 (dùng lại của BamiBERT)   |                               256 | Văn bản gốc                         |

## 2. Dữ liệu và quy trình huấn luyện

| Mô hình               | Kiểu huấn luyện                       | Dữ liệu                                               |     Số bước |        Batch | Tốc độ học            | Số epoch / token đã xử lý                          |
| ----------------------- | ---------------------------------------- | ------------------------------------------------------- | -------------: | -----------: | ------------------------- | ------------------------------------------------------- |
| PhoBERT-base            | Pretrain từ đầu (RoBERTa MLM)         | 20 GB (Wikipedia + tin tức), ~3 tỷ từ                |           540K |        1.024 | 4e-4, warmup 2 epoch      | 40 epoch; *~142 tỷ token*                            |
| PhoBERT-large           | Pretrain từ đầu                       | 20 GB                                                   |    1,08 triệu |          512 | 2e-4                      | 40 epoch; *~142 tỷ token*                            |
| PhoBERT-base-v2         | Pretrain từ đầu                       | 20 GB + 120 GB OSCAR-2301                               |             — |           — | —                        | —                                                      |
| BamiBERT                | Pretrain từ đầu (RoBERTa MLM)         | 129 GB văn bản tổng quát                            |             — | 1.024 chuỗi | 1,5e-4, warmup 2 epoch    | 20 epoch                                                |
| ViSoBERT                | Pretrain từ đầu (MLM, mask 30%)       | 1 GB bình luận Facebook, TikTok, YouTube              |     1,2 triệu |          128 | 5e-5, warmup 1.000 bước | 12 epoch; *~20 tỷ token nếu chuỗi dài 128*        |
| ViDeBERTa xsmall / base | Pretrain từ đầu (MLM + RTD)           | CC100 tiếng Việt, 138 GB                              |           500K |        8.192 | 6e-4                      | —                                                      |
| ViDeBERTa large         | Pretrain từ đầu (MLM + RTD)           | CC100 tiếng Việt, 138 GB                              |           250K |        8.192 | 3e-4                      | —                                                      |
| XLM-R large             | Pretrain từ đầu, 100 ngôn ngữ       | CC-100 2,5 TB; tiếng Việt 137,3 GiB (~24,8 tỷ token) |     1,5 triệu |        8.192 | —                        | *~6,3 nghìn tỷ token-slot (mọi ngôn ngữ)*        |
| CafeBERT                | Pretrain tiếp từ XLM-R large (MLM 15%) | 18 GB (Wikipedia + tin tức), ~2,8 tỷ từ              |             — |           — | 2e-5                      | —                                                      |
| **GoiBERT**       | **Chưng cất từ BamiBERT**       | **328,6M token (1,7 GB)**                         | **120K** | **32** | **1e-4**            | ~3 lượt qua kho ngữ liệu; **~0,98 tỷ token** |

## 3. Phần cứng và thời gian huấn luyện (theo công bố)

| Mô hình                      | Phần cứng                                                 | Thời gian công bố            |                      GPU-giờ |
| ------------------------------ | ----------------------------------------------------------- | ------------------------------- | ----------------------------: |
| PhoBERT-base                   | 4× V100 16 GB                                              | 3 tuần                         |          *~2.016 giờ V100* |
| PhoBERT-large                  | 4× V100 16 GB (bài báo không ghi riêng cho bản large) | 5 tuần                         |          *~3.360 giờ V100* |
| BamiBERT                       | 8× A100 40 GB                                              | —                              |                            — |
| ViSoBERT                       | 2× RTX 4090 24 GB                                          | khoảng 3 ngày                 |        *~144 giờ RTX 4090* |
| ViDeBERTa                      | 32× A100 80 GB                                             | —                              |                            — |
| XLM-R large                    | 500× V100 32 GB                                            | —                              |                            — |
| CafeBERT                       | 1× A100 40 GB                                              | —                              |                            — |
| **GoiBERT-base / large** | **1× L40S**                                          | **~3,3 giờ / ~6,0 giờ** | **3,3 / 6,0 giờ L40S** |

## 4. Quy đổi về giờ GPU A100 (ước tính)

### 4.1. Phương pháp

**Cách 1: mô hình có công bố thời gian.** Lấy số GPU-giờ nhân với hệ số tốc độ so với A100. Hệ số dưới đây là ước tính cho huấn luyện kiểu BERT với bf16/fp16, sai số khoảng ±30%.

| GPU           | Hệ số so với A100 |
| ------------- | -------------------: |
| V100          |                 ~0,4 |
| RTX 4090      |                ~0,55 |
| L40S          |                 ~0,9 |
| A100 40/80 GB |                  1,0 |

**Cách 2: mô hình không công bố thời gian.** Ước tính khối lượng tính toán theo công thức chuẩn:

$$
\mathrm{FLOPs} \approx 6 \times N \times T
$$

trong đó N là số tham số không tính lớp embedding, T là số token đã xử lý trong quá trình huấn luyện.

Sau đó đổi ra giờ A100 (đỉnh bf16 312 TFLOPS) theo hai mức hiệu suất sử dụng GPU:

- Lạc quan, 30%: tương ứng mã nguồn hiện đại.
- Thận trọng, 10%: gần với các lần chạy thực tế. Mức này được hiệu chỉnh bằng PhoBERT và ViSoBERT, hai mô hình có công bố thời gian; cả hai đều khớp với mức 10%.

### 4.2. Kết quả

| Mô hình               | Cơ sở ước tính                                                                  |                                         Giờ A100 (ước tính) | So với GoiBERT-large |
| ----------------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------: | --------------------: |
| **GoiBERT-base**  | Đo thực tế: 3,3 giờ L40S                                                         |                                                    **~3** |                    — |
| **GoiBERT-large** | Đo thực tế: 6,0 giờ L40S                                                         |                                                  **~5,4** |                   1× |
| ViSoBERT                | Công bố: ~3 ngày × 2 RTX 4090                                                    |                                                             ~80 |                 ~15× |
| PhoBERT-base            | Công bố: 3 tuần × 4 V100                                                         |                                                            ~800 |                ~150× |
| PhoBERT-large           | Công bố: 5 tuần × 4 V100                                                         |                                                          ~1.350 |                ~250× |
| BamiBERT                | Tính toán: 129 GB × 20 epoch ≈ 500 tỷ token                                     |                                                   ~750 – 3.200 |         ~140 – 600× |
| ViDeBERTa-base          | Tính toán: 500K bước × 8.192 × 512                                             |                                                ~4.700 – 14.000 |       ~900 – 2.600× |
| XLM-R base              | Tính toán: 1,5 triệu bước × 8.192 × 512 (số do bài báo ViSoBERT liệt kê) |                                                ~9.500 – 28.500 |     ~1.800 – 5.300× |
| XLM-R large             | Tính toán: 1,5 triệu bước × 8.192 × 512, 100 ngôn ngữ                       |                                              ~34.000 – 115.000 |    ~6.000 – 21.000× |
| CafeBERT                | Tính toán: 18 GB ≈ 3 tỷ token mỗi epoch; số epoch không công bố             | ~16 – 50 mỗi epoch, cộng toàn bộ chi phí của XLM-R large |                    — |
