# Kiểm chứng ý tưởng readout bằng dual frame

## Đề xuất

Probe factorial cho thấy polarity của một aspect có thể chi phối aspect khác. Một giả thuyết là attention-pooled vectors bị trộn do attention rows chồng lấn; giả thuyết này chưa được đo trực tiếp trên learned reader của project.

Với H thuộc R^(N×d), A thuộc R^(K×N), pooling là C=AH. Nếu H=AᵀS thì C=(AAᵀ)S. Đề xuất: R=(AAᵀ+lambda I)^(-1)AH. Khi lambda=0 và A full row rank, R=S trong mô hình sinh lý tưởng. Có thể solve hệ tuyến tính thay vì tạo inverse.

## Prior art: trùng phép tính

Intention có dạng ridge-regression KVQ: f(K,V,Q)=Q(KᵀK+lambda I)^(-1)KᵀV. Thay K=Aᵀ, V=H và Q=I cho đúng R. Đây là phép thay biến cụ thể, không chỉ vì cả hai có inverse. Appendix Eq24 và dimensions xác nhận công thức; một dòng PDF extraction ở main text đảo ký hiệu transpose.[^1]

A được sinh bằng attention từ H có thể là lựa chọn embedding cho KVQ; riêng lựa chọn ấy chưa tạo khác biệt thuật toán được chứng minh. Không dùng proposal này làm claim phép tính mới.

## Utility chưa được chứng minh

H=AᵀS là assumption mạnh. BERT token states chứa nội dung riêng, modifiers và thông tin toàn câu; chưa xác nhận mixing weights đúng bằng Aᵀ. Inverse Gram có thể khuếch đại residual/noise; ridge ổn định số học không chứng minh causal binding. Không dùng toy reconstruction được sinh đúng từ AᵀS làm bằng chứng utility trên model thật.

Patching không cứu joint labels trong hai case cũng không chứng minh Gram pooling là nguyên nhân.

## Quyết định

Loại khỏi danh sách ứng viên mới trước implementation. Có thể dùng Intention như comparator nếu một thiết kế khác cần, nhưng không launch thêm trial chỉ để thử. Mục tiêu nghiên cứu vẫn chưa hoàn tất.

## Nguồn

[^1]: Marta Garnelo, Wojciech Marian Czarnecki. *Exploring the Space of Key-Value-Query Models with Intention*. 2023preprint. [Paper](https://arxiv.org/pdf/2305.10203), định nghĩa least-squares KVQ và Appendix Eq24. Đã đối chiếu công thức, không reproduce experiments.
