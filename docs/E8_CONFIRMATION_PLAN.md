# E8 — xác nhận seeds43/44 trên ViSFD và Restaurant

Người dùng duyệt thực thi và nhắc rõ luật subagents. Root điều phối/đọc bằng chứng; hai implementator được tạo với model gpt-5.6-luna, reasoning xhigh, fork_turns=none. Không kế thừa phiên chính.

## Phạm vi

Bốn training trials mới: ViSFD43/44 và Restaurant43/44. Giữ nguyên toàn bộ recipe E8: GD student gốc, CLS, CE+NC, gamma0.8 LLRD, lambda0.5 ViSFD/lambda0.1 Restaurant, baseLR1e-4,30epochs,batch32,max256,BF16/SDPA, optimizer/scheduler/noise như E8. Mỗi trial khởi tạo mới từ models/goiturn-s-gd; không warm-start từ seed42 hoặc đổi hệ số.

Kết quả seed42 và các baseline E1/E2 đã có được tái sử dụng. E7 chỉ có seed42, không tuyên bố so sánh E8–E7 đa seed hoặc chạy thêm baseline E7.

## Chọn checkpoint và đánh giá

Chọn checkpoint từng trial mới bằng validation F1 và quy tắc E8, trước khi đọc test. Sau đủ30epochs, khóa manifest4checkpoint với SHA/task/seed/epoch rồi đánh giá test đúng4bản mới. Bước này nằm trong xác nhận mức tăng test mà người dùng đã đồng ý, không cần mở lại test của seed42/baseline.

Inference strictload, eval/inference_mode, CUDA BF16/SDPA,max256,batch32; không textnoise/teacher/optimizer. ViSFD2224testrows,Restaurant1938; ViSFDOTHERSraw4/model1 chỉ được tính detection, không pair. Đối chiếu ID/gold/metric từ predictions một lần.

Aggregate E8 seeds42–44 mean và sampleSD trên validation/test; paired delta theo cùngseed với E1 CE và E2 NC đã có. Không lấy seed tốt nhất. Mean test gain≥0.3điểm soNC và≥2/3seed có delta dương là tín hiệu xác nhận thực dụng; vẫn báo đầy đủ từngseed và mức biến động, không coi3seed là significance hoặc test độc lập mới.

## Phân tích lỗi song song

Chỉ dùng validation predictions E8, E2 NC và teacher đã fine-tune, ưu tiên seed42 trước. Đếm missed/spurious aspects, wrongpolarity trên các slot pair-scored khi gold/pred cùngpresent; peraspect/polarity với train support. Pair wrongpolarity góp cảFP vàFN. ViSFDOTHERS phải loại đúng khỏi conditionalpolarity denominator.

Teacher đúng/E8 sai và chiều ngược lại cần được đo trên cùng ID/gold. Teacher là DAPT derivative seed42 đã chọn theovalidation, không gọi là paper BamiBERT gốc. Khi seeds43/44 sẵn sàng có thể kiểm tra pattern lặp lại bằng các predictions mới, không inference thêm. Không phân tích test errors để chọn phương pháp tiếp theo trong vòng này.

## Thực thi tối thiểu

Source E8 và mọi seed42 artifact giữ nguyên. Adapter xác nhận riêng có hash riêng; override SEED/đường dẫn trong từng process và không ghi đè status của seed42 hoặc worker cùngtask khácseed. Chạy worker detached, distincttask-seedlogs/status, chọn GPU2/1/3 theo tập GPUtrống≥5GiB, không kill workload khác. CUBLAS trướcCUDA, PYTHONPATH đúng.

Không lặp fulltest/preflight cũ; chỉ kiểm tra seed,config,group/source,manifest/checkpoint và dữ liệu. Không chạy thêm task,seed,NC coefficient hoặc một phương pháp mới.

## Bàn giao

artifacts/e8/{visfd,absa_restaurant}/seed_{43,44}; artifacts/e8/confirmation gồm controller/testmanifest/summaryCSVJSON; docs/E8_CONFIRMATION_RESULT.md. Phân tích lỗi ở artifacts/e8/error_analysis và docs/E8_ERROR_READOUT.md. Root tổng hợp mức ổn định và chọn hướng kế tiếp từ bằng chứng; không tự mở thử nghiệm tiếp theo.
