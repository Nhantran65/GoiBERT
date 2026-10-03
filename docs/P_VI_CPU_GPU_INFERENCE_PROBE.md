# P_vi: phép đo nhanh suy luận CPU và GPU

**Ngày:** 2026-09-21. Đây là phép đo latency ngắn để trả lời khả năng bổ sung CPU fallback; không phải benchmark production hay đánh giá F1.

## Cấu hình

- Checkpoint: P_vi fine-tune VSFC Sentiment, LR `1e-5`, seed 42, best epoch 10; nạp nguyên weights FP32 và head từ `artifacts/bidirlm_vi_mntp/ft/P_vi/vsfc_sentiment/lr_1e-05/seed_42/best_checkpoint.pt`.
- Model: custom BidirLM hai chiều, mean pooling, `attention_mask` từ tokenizer, `eval()` và `torch.inference_mode()`; PyTorch 2.6.0+cu124, Transformers 4.57.6.
- CPU: Intel Xeon Gold 6548Y+ trên máy hiện tại, PyTorch 8 intra-op threads và 1 inter-op thread. GPU: NVIDIA L40S (GPU 2), CUDA BF16 autocast giống cách đánh giá P_vi trước đây.
- Văn bản lấy từ validation Sentiment: một câu gần độ dài trung vị (13 token), hoặc 8 câu gần phân vị 95 (padding tới 36 token). Đo riêng forward của model, sau warmup; không gồm tokenize, nạp model, truyền mạng, API, queue. CPU lấy median 5 lần, GPU median 9 lần và đồng bộ CUDA trước/sau mỗi lần.

| Batch × độ dài đã pad | CPU FP32 | CPU BF16 autocast, weights FP32 | GPU BF16 autocast, weights FP32 |
|---|---:|---:|---:|
| 1 × 13 | **101 ms** | 124 ms | **44 ms** |
| 8 × 36 | **350 ms** | 220 ms | **41 ms** |

Suy ra trong phép đo này CPU FP32 chậm hơn GPU khoảng 2,3 lần ở batch 1 và 8,5 lần ở batch 8. CPU BF16 autocast không luôn nhanh hơn CPU FP32: batch 1 chậm hơn, batch 8 nhanh hơn. Các tỷ lệ này phụ thuộc phần cứng, số thread, độ dài, batch và tải máy; cần đo lại trên máy production. Chưa đo F1 CPU hoặc bản quantized; kết quả latency không chứng minh quality parity.

**Hàm ý:** Có thể dùng cùng checkpoint FP32 để chạy CPU fallback mà không train lại. Nếu cần CPU throughput hoặc giảm RAM, thử INT8 riêng rồi kiểm chứng prediction/F1 trên validation trước khi chọn bản deploy.
