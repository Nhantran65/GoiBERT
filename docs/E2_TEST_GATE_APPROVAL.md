# E2 test gate — user yêu cầu đánh giá trực tiếp

2026-09-09. User yêu cầu chạy test để kiểm tra E2, thay vì tiếp tục suy đoán từ validation. Quyết định này mở test gate cho đúng15studentcheckpoint đã chọn bên dưới; thay thế HOLD test của E2 cho phạm vi này. Root điều phối/review, subagents thực thi và kiểm tra kết quả.

## Selection khóa bằng validation trước test

| Task | Nhánh/hệ số | LR | Seeds |
|---|---|---|---|
| VSFC sentiment | NC lambda0.1 | 1e-5 | 42,43,44 |
| VSFC topic | KD alpha0.75,T2 | 1e-5 | 42,43,44 |
| ViSFD | NC lambda0.5 | 1e-4 | 42,43,44 |
| UIT-ABSA Restaurant | NC lambda0.1 | 1e-4 | 42,43,44 |
| UIT-ABSA Hotel | NC lambda0.1 | 1e-4 | 42,43,44 |

Nhánh chọn theo mean validation của3seeds trong E2_COMPLETED_AUDIT.md và stageB_paired_e1_comparison.json. Hệ số của từng nhánh đã được chọn bằng seed42 ở E2; epoch của từngcheckpoint được chọn bằng validation trong run30epochs. Không đổi lựa chọn sau xem test. Seed42 từstageA,43/44 từstageB. NC aliasseed42 củaSentiment chỉ tính mộtcheckpoint theo pathresolve/hash.

## Thực thi

Agent e2_test_execute khóa manifest có actualUTCtimestamp, identity và hashes checkpoint/metadata/model/tokenizer/evaluator/data/source. Eval-only, model.eval/no_grad,BF16/SDPA,batch32,max_length256 nhưtraining. Không teacher inference/KD/noise khi đánh giástudent; không retrain hoặc thêmseeds45/46. GiữfrozenE1/E2source/config nguyên vẹn, dùngwrapper riêng nếu cần vì metadataE2 khácE1.

Mỗicheckpoint evaluate một lần; kiểm traoutputs tồn tại đểreuse hợp lệ thay vìghiđè. Lưu stableIDs, prediction, rawgold đúngschemaViSFD, per-seedmetric/testaccesslog. Independentagent e2_test_review tính lại từpredictions, không chạy model lầnhai.

## Bảng so sánh

- E1 vàE2 đều báo testmean±sampleSD trên CÙNG seeds42/43/44. E1 dùng lại testoutputs đã có, không inference lại. Không lấy E1mean5seeds làm baselinepaired3seeds.
- Cạnh đó là BamiBERT reportedtest từbenchmarks/paper_targets.csv, gồm2accuracy và8F1rows. Ghi rõ F1nội bộ VSFCmacro,ABSAdetectionmicro/pairmicro; paperaveraging/provenancemirror còn hạn chế, nhưng không dùng sựthiếu này để chặn việc đo và trìnhbày kết quả.
- Bảng giúp trả lời E2test thay đổi bao nhiêu soE1 và vị trí so reportedpaper. Không hứa gainvalidation sẽ chuyển nguyêngiá trị sangtest hoặc khẳng định significance với một điểm paper.

Sau lần này E2selectedmodels đã được đánh giátest. Không dùng testerrors để chọn hyperparameters/branches khác rồi báo cùng test nhưholdout chưa từng xem. Chưa tự mở thử nghiệmkiếntrúc trong turn này.

## Sửa backend của lượt đánh giá

Lượt thực thi đầu đã dùng CPU BF16 vì sandbox không thấy CUDA. Root xác minh trực tiếp trênhost: đủ4L40S, `torch.cuda.is_available=True`. Đây là sai khác backend so với E1CUDA, không phải bằng chứng driver host bị lỗi.

Root cho phép lưu nguyên15CPUoutputs và accesslog vào diagnostic archive cóchecksum, rồi evaluate lại chính15checkpoint đã khóa bằng CUDA/BF16/SDPA trênGPU1/2/3. Không thay weights, branch, seed, LR, max_length hoặc evaluator; không chọn backend theo điểm cao hơn. Bảng canonical dùng CUDA như E1. Manifest lựa chọn ban đầu phải giữ làm bằng chứng selection trước test; lịch sử cảhai lượt backend được ghi rõ. “Một lần” ở trên được hiểu là một lượt CUDA canonical; không được xóa hoặc che lượt CPU đã thực hiện.
