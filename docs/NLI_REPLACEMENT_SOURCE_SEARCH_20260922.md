# Nguồn NLI thay ViANLI — vòng tìm nguồn có giới hạn

Ngày 2026-09-22. Root tự tìm nguồn và đọc mẫu; không subagent nghiên cứu, không GPU, không đọc/tải test, không gửi email tác giả. Đây là lựa chọn nguồn để audit, không phải gate D đã pass.

## Kết luận

Ứng viên có provenance rõ nhất để tiếp tục là **MultiNLI tiếng Anh gốc**. Cafe kế thừa XLM-R đa ngôn ngữ nên có giả thuyết chuyển giao từ supervision tiếng Anh, với tiền lệ https://aclanthology.org/2020.aacl-main.56/ ; nhưng đây là thay đổi ngôn ngữ intermediate task và cần plan riêng, không âm thầm thay nguồn trong pilot ViANLI đã dừng. Không đảm bảo năm task tăng. Nếu bắt buộc nguồn tiếng Việt nguyên bản, vòng này chưa xác minh được bản phát hành chính thức đủ điều kiện chạy ngay.

## Nguồn và quyết định

| Nguồn | Đã kiểm | Quyết định |
|---|---|---|
| ViNLI, COLING2022 | Paper30.376 cặp,4nhãn E/C/N/OTHER, website nhóm UIT; chưa xác minh được full release trực tiếp từ tác giả | Giữ ứng viên VN có điều kiện provenance, không gọi dataset không tồn tại |
| presencesw/vinli_4_label | Train24.376,6094/class; dev3009/test2991 theo metadata, chỉ tải train; không license/provenance narrative trong card | Bản mirror khớp quy mô chưa chứng minh nguyên bản; không launch |
| tmnam20/ViNLI | Card train2048/dev232/test232, có context/link/annotator_labels; không license rõ | Subset không dùng thay full corpus |
| anti-ai/ViNLI-SimCSE-supervised_v2 | CardMIT,100K–1M và sentence-similarity/retrieval, không mô tả pipeline nguồn | Tên ViNLI không chứng minh cùng corpus30K; không chọn |
| VnNewsNLI | Đọc paper gốc; chưa xác minh direct train download | Không chọn trong vòng ngắn |
| VLSP2021 NLI | Trang chính thức mô tả vi-vi/en-vi và agree/disagree/neutral; không direct train download hiện trên trang đã đọc | Chưa dùng được ngay; không đồng nhất nhãn một cách tự động |
| XNLI train Vietnamese | Official facebook/xnli,392702rows, nguồn dịch máy từ MultiNLI; đã tải chỉtrainVI/EN và đọc30cặp songngữ | Hạ ưu tiên vì lỗi dịch nghĩa rõ; chưa audit300, không gọi fail cùng gate ViANLI |
| MultiNLI English | Website tác giả, HF nyu-mll/multi_nli có train392702 và validation matched/mismatched, quyền dùng theo từng phần được mô tả paper | Ưu tiên cho audit kế tiếp, chưa đạt gate chất lượng |
| SNLI | Trang tác giả Stanford có nguồn public, nhưng phạm vi captions hẹp hơn MultiNLI | Dự phòng; chưa tải/audit |

## Mẫu nội dung đã đọc

XNLI:10/class, random.Random42 liên tục theo label0,1,2; all392702 label VI/EN khớp theo chỉ số (điều này chưa chứng minh alignment toàn corpus hoàn hảo). Files local `data/nli_source_search/screen30.jsonl` chứa cả hai ngôn ngữ, index gốc. Lỗi đáng chú ý:row43176 bashes->đập vỡ;9559 meeting emission constraints->cuộc họp các hạn chế;73276 upsets me->lên đặt tôi;240823 Santa Maria->Ông Già Noel;80748 bỏ sắc thái before làm câu never seen so clearly before đổi nghĩa. Nhiều ví dụ vẫn giữ nhãn, nhưng ngôn ngữ không phù hợp làm chuẩn suy luận VN sạch. Không ước tính tỷ lệ lỗi toàn corpus từ30mẫu; không sửa hoặc lọc để train. Bản Anh cũng có mảnh câu/nhãn đáng tranh luận (vd86533 consistent vs unchanging,250741 fragment HI trust fund balance), không gọi gold hoàn hảo.

ViNLI mirror:8/class, seed42, labels sorted; `vinli_mirror_screen32.jsonl`. Ngôn ngữ tự nhiên hơn bản dịchXNLI trong các mẫu đọc, nhưng có nghi vấn:row24129 thêm 'hoàn toàn' vào việc tái sử dụng2thiết bị;row22169 thêm suy luận cảm nhận nhạc sâu sắc;row980 đổi tất cả832huyện thành hơn1000huyện mà gánneutral. OTHER chứa cặp không liên quan; nếu dùng3nhãn cần theo đúng protocol paper (không tự merge hoặc drop).32mẫu không đủ xác nhận chất lượng và không giải quyết provenance.

## Revisions/artifacts

- facebook/xnli@b8dd5d7af51114dbda02c0e3f6133f332186418e. Chỉ `vi/train-00000-of-00001.parquet` và `en/train-00000-of-00001.parquet`; hashes/URLs/bytes: `data/nli_source_search/manifest.json`. XNLI repo LICENSE CC BY-NC4.0; HF card licensing trống, không suy thành permissive.
- presencesw/vinli_4_label@a885fd1bdeb00099f2215b601ddb0df255503cf1; chỉtrain; manifest riêng `vinli_mirror_manifest.json`.
- tmnam20/ViNLI@88ae73b712cd09c8cac8f8b235b208e749ab1351; anti-ai/ViNLI-SimCSE-supervised_v2@d917790a95a44e44a93f345856a890945e8f4fb6; nyu-mll/multi_nli@da70db2af9d09693783c3320c4249840212ee221: metadata/card only, không downloadtest.
- Raw bên thứ ba giữ local với .gitignore trong thư mục, không phát hành lại. Không sửa pilot ViANLI/gate cũ.

## Bước hợp lý tiếp theo

Nếu chọn English intermediate: dùng MultiNLI chính thức (giữ genre,pairID,promptID), audit300 theo cùng rubric trước training và split premise/group disjoint; không tái dùng mọi chi tiết7epochs của corpus8K cho corpus393K. Pilot phải khóa subset/steps ngân sách mới, kiểm năng lực tiếng Việt sau learning, không chỉ EnglishNLI. Không lấy bản XNLI EN đã tokenize làm corpus canonical khi cần ID/genre. Nếu bắt buộc VNnative: cần nguồn ViNLI được tác giả xác nhận; không tự liên hệ tác giả trong vòng này.

## Nguồn chính

- ViNLI: https://aclanthology.org/2022.coling-1.339/ ; https://nlp.uit.edu.vn/datasets/
- VnNewsNLI: https://pmc.ncbi.nlm.nih.gov/articles/PMC9311348/
- VLSP: https://vlsp.org.vn/vlsp2021/eval/nli
- XNLI: https://github.com/facebookresearch/XNLI ; https://github.com/facebookresearch/XNLI/blob/main/LICENSE ; https://huggingface.co/datasets/facebook/xnli
- XNLI train machine translation xác nhận: https://github.com/google-research/bert/blob/master/multilingual.md
- MultiNLI: https://cims.nyu.edu/~sbowman/multinli/ ; https://aclanthology.org/N18-1101/ ; https://huggingface.co/datasets/nyu-mll/multi_nli
- SNLI: https://nlp.stanford.edu/projects/snli/

MultiNLI paper mô tả không có premise xuất hiện trong nhiều split và giấy phép theo corpus nguồn; train không có đầy đủ4nhãn verifier như dev/test. Không dùng tên nổi tiếng để thay audit local.
