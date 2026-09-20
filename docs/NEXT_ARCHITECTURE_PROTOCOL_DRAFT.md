# Candidate kế tiếp — context-conditioned identity-gated FFN

Trạng thái: protocol đề xuất đã được root chốt sau [audit E13/E15 hoàn tất](ARCHITECTURE_TRANSFER_FINAL_REVIEW.md). Chưa triển khai hoặc chạy training. Tên file giữ nguyên để các liên kết trước không bị hỏng; không dùng tài liệu này như lệnh tự launch.

## Thiết kế duy nhất và đối chứng

Backbone official BamiBERT E14, không DAPT hay checkpoint downstream. Sửa FFN block8/index7, giữ W1,b1,W2,b2 và toàn bộ encoder/head gốc. z=GELU(W1x+b1), c=output Wo của attention cùngblock trước dropout/residual. G là một ma trận3072×768, no bias, zero init; không factorize trong pilot để tránh zero-factor gradient delay.

Candidate: f_new=W2[z⊙(1+tanh(G c_normalized))]+b2.
Control: thay c_normalized bằng x_normalized, cùng số tham số và init.
Baseline: FFN gốc.

Normalization chỉ dùng hai scalar RMS đóng băng lấy từ official BamiBERT trên2048traintexts cân bằng4domain (512/domain, seed cố định, dedup và ghiactualcount). VSFC không tính hai lần. Cảhai nguồn được đưa về RMS1 bằng cùng quy tắc; epsilon1e-6. Không thêm LayerNorm học được và không lấyvalidation/testcalibration.

G=0 bảo toàn FFN trong số học thực, tanh'(0)=1 tạo gradient trực tiếp thông qua W2/z pretrained. Init parity thực tế cần đo trên runtime. Không claim G nhất định được tối ưu tốt hoặc có semantics đúng từ công thức.

Tính correction bằng W2[z⊙tanh(Gc)] không bias, và cộng vào output FFN gốc; không nhân gate gần1 bằngBF16 trước khi subtract. Gate/correction/fusion FP32 tại targetblock trong cảcandidate/control. Baseline cùng fusion/precision policy cho phần chung, không đổi toànencoderprecision. Giữ originaldropout đúng một lần và biasb2 đúng một lần. Nếu cần thay điểm cast ảnh hưởngbaseline, replaybaseline cùngđiềukiện thay vìso thẳngE14cũ.

## Cách thích nghi đã chọn cho vòng đầu

Không thêm MLM/KD hoặc giai đoạn corpus mới. Fine-tune supervised từ cùng originalcheckpoint theo recipeE14; LR cố định từE14selection,30epochs,seed42. Đây là lựa chọn có chủ đích vì gate đã có identity embedding và gradient trực tiếp. Không dùng feature matching chỉ đểmatch một hàm đãmatch sẵn ởG0.

UniformAdamW cho cảencoder vàgate, cùngLRtask; không nângLRgate riêng. Head768 giữ nguyên. Microbatch16,accumulation2,eff32 cho **mọi** arm/task, không phụ thuộcGPU. Dataloaderorder vàdropoutpolicy được kiểm tra/log. Không dùngchênhlệchmicrobatch làm đối chứng kiến trúc nữa.

## Ngân sách đề xuất

3arms×5tasks=15trials seed42, khônggrid. Baseline replay cần thiết để có cùngmicrobatch/precision, không thay con số E14 lịch sử. Pilottechnical cógradientprobe vàmột smoke ngắn trướcfullrun. Ghi gradient/update ởstep0,1,10,100 vàcuốiepoch; không lưu fullgradienttensor mọi step.

Dùng validation chọn epoch; báo cảvector5tasks, candidate−baseline/control, throughput,VRAM,gate RMS/quantiles,normcorrection vàactivation clipping. Cùng phương thức chọncheckpoint cho cảarms, không chọn thêm architecturetheotest.

Gate đầu tư: ≥4/5tasktăng,≥2gain≥.3pp,nodrop>.3pp vàcó lợi thế rõ hơnreceivercontrol. Nếu hai gate arms cùng tăng gầnnhau thì chưa chứng minhcontextconditioning. Không tự mởmulti-seed/test/MLM saufail.

## Nếu sau này cần adaptation

Phải là protocol riêng architecture×adaptation: baseline vàcandidate nhận cùngtrain-onlycorpus,tokenbudget,masking/objective vàcompute được báo rõ. Không chỉ cho model mới học thêm rồi quy gaincho gate. Tokenbudget chưa được duyệt/chốt; không khởi độngcorpusstage ởpilot này.

## Mức kết luận được phép

Candidate là architecture mới đối với project, có tiền lệ GLU/FiLM/(IA)³. Không claimnovelty toàncầu haySOTA trướcablation/replication. Các đoE13/E15 chỉ giúp quyết định có nên đóngnhánhpair-value và chọn kiểuparameterization khác; chúng không chứng minh gateFFN sẽthắng.
