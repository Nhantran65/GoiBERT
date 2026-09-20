# Trạng thái dữ liệu — 2026-09-08

Đã tự tải VSFC và ViSFD từ nguồn phát hành. Không cần chờ bản processed của thí nghiệm cũ. Giữ nguyên bytes, nhãn, thứ tự và split; chưa preprocessing hoặc train.

| Dataset | Đường dẫn | Train / dev / test | Trạng thái |
|---|---|---|---|
| UIT-VSFC | `data/raw/uit-vsfc/{train,dev,test}/` | 11426 / 1583 / 3166 | Đủ `sents.txt`, `sentiments.txt`, `topics.txt` |
| UIT-ViSFD | `data/raw/uit-visfd/` | 7786 / 1112 / 2224 | ZIP gốc và `Train.csv`, `Dev.csv`, `Test.csv` |
| UIT-ABSA Restaurant | `data/raw/uit-absa-public-candidate/dataset/UIT-ABSA-Restaurant/` | 7028 / 771 / 1938 | Đã tải bản sao public bên thứ ba; đang chờ xác minh lineage/label mapping |
| UIT-ABSA Hotel | `data/raw/uit-absa-public-candidate/dataset/UIT-ABSA-Hotel/` | 7180 / 795 / 2030 | Đã tải cùng bản sao; thuộc phạm vi benchmark chính từ 2026-09-08 |
| Corpus distillation cũ | Chưa có | — | Chỉ chặn các bước cần corpus; baseline supervised không cần |

## Nguồn và kiểm tra

- VSFC: [UIT-NLP dataset page](https://nlp.uit.edu.vn/datasets/) và [repository UIT-NLP](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback). Tải chín file Google Drive theo loader của repository; loader chỉ đọc để lấy URL, không thực thi. Revision loader được lưu trong manifest. Nhãn sentiment 0/1/2 = negative/neutral/positive; topic 0/1/2/3 = lecturer/training_program/facility/others theo loader nguồn.
- ViSFD: [repository tác giả LuongPhan](https://github.com/LuongPhan/UIT-ViSFD). ZIP và README lấy từ commit cố định ghi trong manifest. Có 10 aspect kèm polarity và nhãn đặc biệt `{OTHERS}`; cách evaluator xử lý OTHERS còn cần xác minh.
- UIT-ABSA: [trang UIT-NLP](https://nlp.uit.edu.vn/datasets/) ghi liên hệ `thindv@uit.edu.vn` để ký corpus user agreement và nhận dữ liệu. Chưa gửi email hoặc ký thay người dùng. Không thay bằng VLSP2018 Restaurant.

`data/download_manifest.json` lưu URL, source revision, kích thước và SHA256 từng file, gồm file giải nén. `data/validation_summary.json` lưu số dòng, trùng text trong/giữa splits. Đã kiểm tra ZIP CRC, checksum, đủ nhãn VSFC, cấu trúc nhãn ViSFD và không có text trống. Không xóa bản ghi trùng hoặc đổi official splits.

Số lượng khớp là sanity check, chưa chứng minh evaluator tương đương paper BamiBERT hoặc corpus kế thừa không chứa test. VSFC source card ghi license unknown; ViSFD README cho phép nghiên cứu và hướng dẫn liên hệ cho mục đích thương mại. Giữ README nguồn cùng dữ liệu.

Code reference đang nhận mirror CSV có format khác. Cần loader mới cho layout gốc này trước khi train; chưa coi download là hoàn tất E0.

Gói ZIP vẫn là snapshot bàn giao 2026-09-07, chưa chứa dữ liệu mới hoặc phạm vi Hotel được cập nhật. MANIFEST.sha256 ở root đã cập nhật checksum các file handoff hiện tại. Checksum dữ liệu mới nằm trong `data/download_manifest.json`.

## Bổ sung: bản UIT-ABSA public tìm được

Nguồn: [tienquocbao/ABSA-RAG-Vietnamese-Restaurant-Hotel-Bot](https://github.com/tienquocbao/ABSA-RAG-Vietnamese-Restaurant-Hotel-Bot/tree/2ff3fb7ac67ec404f5323cdc30f46e4853969998/dataset), commit `2ff3fb7ac67ec404f5323cdc30f46e4853969998`. Repo [thduy-h](https://github.com/thduy-h/ABSA-RAG-Vietnamese-Restaurant-Hotel-Bot) cũng có các file cùng tên/kích thước; chưa dùng làm bằng chứng độc lập về provenance.

Đã tải sáu CSV và README vào thư mục candidate, giữ nguyên bytes. `data/uit_absa_candidate_manifest.json` lưu URL, commit và SHA256. Restaurant có 12 cột aspect; Hotel có 34. Các CSV có nhãn số 0..3, không có review trống, số dòng thực tế khớp các split mục tiêu.

README của bản sao ghi số lượng dataset không khớp chính các CSV của repo. Do đó chỉ xác nhận đây là bản sao public mang tên UIT-ABSA có cấu trúc/số lượng phù hợp; chưa xác nhận preprocessing, polarity mapping và nội dung từng split trùng bản gốc tác giả/BamiBERT. Dòng MIT trong README không tự chứng minh điều kiện phân phối dataset gốc. Không thay đổi nhãn hoặc chạy training trong lần tải này.
