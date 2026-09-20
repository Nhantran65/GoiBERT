# E11 — Signed versus convex attention trong student encoder

## Giả thuyết và giới hạn

Kiểm tra liệu một đường attention có khả năng trừ thông tin giúp student phân biệt bằng chứng liên quan và distractor tốt hơn. Giữ GD student và recipe E8. Không khẳng định E10 đã chứng minh attention là bottleneck.

Nguồn: [Differential Transformer](https://arxiv.org/abs/2410.05258) và [DEX](https://arxiv.org/html/2505.16333v1). DEX điều chỉnh output head, không đồng nhất với hai softmax attention maps. Chỉ đổi dấu một linear adapter tự do có thể bị hấp thụ vào trọng số, nên không dùng đối chứng cộng/trừ MLP như bằng chứng differential. E11 dưới đây là adaptation lấy cảm hứng từ differential attention, không phải reproduction DEX hay claim novelty.

## Phạm vi thực thi

Đúng một subagent GPT5.6Luna xhigh phụ trách code, diagnostics, training và evaluation; root plan/review. Source/config/artifacts E11 isolated, E1–E10 frozen. Tận dụng GPU0–3 theo free VRAM, giữ tiến trình khác. Mỗi GPU một worker ban đầu, điều phối detached có heartbeat, nhận lại worker đang chạy sau gián đoạn.

## A. Diagnostic ngắn trước training

Theo yêu cầu tiếp theo “Triển khai và bắt đầu training”, diagnostic ngữ nghĩa được phép hoàn thiện song song để không chặn lịch training. Gate bắt buộc trước full run là review kỹ thuật và preflight đạt; không bỏ kiểm tra nhãn của diagnostic hay báo nó đã hoàn tất khi chưa được duyệt.

Tạo một bộ cặp nhỏ được root đọc duyệt trước inference: khoảng 20–30 cặp trên các domain VSFC, điện thoại, nhà hàng, khách sạn. Có hai loại: thay đổi bề mặt/diễn đạt giữ nghĩa và thay đổi phủ định/đối lập làm đổi nghĩa. Với ABSA, thêm mệnh đề có thể thêm aspect nên không giữ nguyên toàn bộ gold một cách máy móc; chỉ đánh giá slot có kỳ vọng rõ. Với Topic, phủ định thường không đổi topic. Ghi nhãn kỳ vọng cụ thể từng cặp/slot và câu mơ hồ bị loại.

Không tự động gán mọi câu distractor là label-preserving. Không đưa câu diagnostic vào train, không dùng test. Dùng E8 seed42 checkpoint inference trên cặp được duyệt; ghi flip rates có điều kiện base đúng, margin thay đổi, và lỗi theo loại. Diagnostic chỉ định hướng, không ước lượng prevalence hoặc thay thế benchmark. Nếu hoàn toàn không thấy tín hiệu, root xem lại trước khi đầu tư10trial; không tự chuyển sang nhánh khác.

## B. Kiến trúc cố định để review

Thay self-attention tại block9 (index8) của encoder10layers. Đây là lựa chọn vùng trên, không phải layer đã chứng minh lỗi. Giữ Q1/K1/V/output projection từ checkpoint GD, các layer khác và headCLS cũ. Thêm Q2/K2 cùng shapes, khởi tạo bằng Q1/K1 cộng nhiễu normal std0.01 trong fork_rng để không đổi head/RNG cũ. V và output projection dùng chung.

Với A1=softmax(Q1K1ᵀ/sqrt64+mask), A2 tương tự Q2K2, O1=A1V, O2=A2V, so sánh:

- **signed:** O = O1 + λh(t) (O1 − O2).
- **convex control:** O = O1 + λh(t) (O2 − O1).

Cả hai cùng tham số, phép tính, mask, init và schedule; control là tổ hợp không âm, signed cho phép trọng số âm. Hệ số tổng trước attention dropout đều bằng1, tránh confound scale đơn giản. Đây không phải hai cách đổi dấu projection tự do vì A2 bị ràng buộc bởi softmax và λ≥0.

λh(t)=min(1,t/Twarmup)*sigmoid(ah), ah khởi tạo log(.1/.9), riêng6heads. T lấy đúng số optimizer warmupsteps E8. t là optimizer step, không microbatch; clean/noisy cùngstep; eval dùng hệ số checkpoint step hiện hành, test dùng checkpoint đã lưu. Tại t0 cảhai đúng baseline; sau đó λ có thể học trong[0,1]. Đây là schedule module mới, không đổi loss/scheduler tham số cũ. Ghi rõ nó là một phần treatment.

Reuse SDPA cho hai O nếu có thể, cùng attention dropout như baseline. So sánh số học mask/dropout và t0 parity trong eval; không claim bitwise training parity vì nhánh mới tiêu thụ RNG/dropout. Không gọi output_attentions=True toànmodel khi train. Các tham số mới nhận LR block9=baseLR*.8, không headLR. Bias/no-decay nhưE8; không áp weightdecay cho logits gate ah.

## C. Gate kỹ thuật và ngân sách

Root review code trước fulltraining, agent tự sửa lỗi trong scope và tiếp tục không hỏi lại user. Preflight nhỏ: all5outputshapes, t0evalparity, padding/mask, finitegradient tớiQ2/K2/gate khiλ>0, optimizerunique/LR, strictcheckpoint roundtrip gồm step/gate, cùng sốparams giữa2variants. Dùng example nhỏ kiểm tra effectiveweights signed có thể âm còn convex khôngâm (dropout off); không bắt mọi sample đãtrain phải âm.

Giữ recipe E8: original models/goiturn-s-gd, seed42, 30epochs,maxlen256,eff/microbatch32,BF16SDPA,NCactualT1,gamma.8,AdamW/warmup10%/clip1. LRVSFC1e-5,ABSA1e-4;lambdaNC Sent.1,Topic.5,Vi.5,Rest.1,Hotel.1. Không E10/E9head/branch,KD,corpusmới.

Sau diagnostic/rootgate: **2variants×5tasks=10traintrials**. Không grid/layer sweep/seed mới. Profile short actual step trước launch và kiểmtraVRAM mỗi dispatch. Không đổi source giữa run; lưu failed attempts riêng nếu sửa lỗi cần rerun. Nếu tốc độ/correctness có vấn đề báo root trước tiếp tục.

## D. Evaluation và quyết định

Mỗi trial chọn bestvalidation, metrics tính lại từsavedpredictions kiểm IDs/gold. E8baseline đọc filefraction0..1,delta_pp=100*(new−old). Báo năm task, signed−E8,signed−control,params,time,VRAM,gate perhead, relative norm correction, diagnostic outcomes. Không dùng gate gần0 làm lý do tự động quétλ khác.

Gate đề xuất: signed tăng≥4/5vsE8,≥2gain≥.3pp,no drop>.3pp; đồng thời thắngcontrol≥3/5 vàkhôngloss>.3pp. Heuristic đầu tư, không significance. Nếu không đạt dừng, khôngmởtest/seeds đểcứu.

Nếu đạt, root reviewkhóa checkpoints rồi cùngagenttest cả2variants trên5tasks trongphạmvi userđãủyquyền, không chọnlại setting bằngtest. So paperBamiBERT cần caveat protocol; chưa tựmởreproductionBamiBERT/multiseed. Root tổnghợp vàđềxuấttiếp.
