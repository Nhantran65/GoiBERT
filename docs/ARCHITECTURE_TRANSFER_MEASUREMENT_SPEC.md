# Đặc tả phép đo E13/E15 trước quyết định kiến trúc tiếp theo

## Ranh giới trách nhiệm và phạm vi

Root chịu trách nhiệm giả thuyết, định nghĩa phép đo, đọc literature và diễn giải. Một subagent Luna xhigh chỉ implement/run đặc tả này và bàn giao số liệu/provenance. Không giao agent lựa chọn hướng kiến trúc. Không train/fine-tune, không optimizer.step, không test data. Không sửa source E1–E15 hoặc checkpoints. Phép backward đo gradient hiện tại không phải lịch sử gradient training.

Dùng cả20bestcheckpoints: E13centered/uncentered ×5 và E15centered/uncentered ×5. Readmetadata vàcode trước. Chạy inference trên validation; gradientprobe chỉ fixedtrainbatch16 đầu tiên, thứ tự không shuffle, dropoutoff để tái lập.

## A. Can thiệp inference

Với mỗicheckpoint, chạy cùngfullvalidation/microbatch16/BF16SDPA ba chế độ:

1. native: hàm gốc.
2. branch_off: zero delta tại fusion, giữ weights còn lại/checkpoint gốc; không nhầm với baseline E8/E14.
3. fp32_fusion: chỉ thực hiện phép cộng Wo.float()+delta.float() rồi shared dropout/residual/LayerNorm, phần còn lại như native. Ghi dtype mọi điểm; không cast delta vềBF16 trướcLN. Đây là biến thể độ chính xác để chẩn đoán, không phải kết quả benchmark được thay thế.

Giữ native results frozen vàghi variant mới riêng. Hooks/wrappers tháo/reset sau mỗi model; querymask vẫn được set trong model.forward. Validate native metric từpassmới với metric savedbest, ghi sai lệch do microbatch/backend nếu có; không gọi deterministicbitwise khi chưa đo. Savepredictions/IDs/gold/metrics tất cảmode, đồng nhất order/split/labelcodes. ViSFD decode packed42 theo slot counts, OTHERS detectiononly.

Report F1 mode−native và prediction disagreements (câu/slot), normalized KL hoặc maxlogitdifference nếu lưu được. Tắt nhánh trêncheckpoint đã co-adapt không phải đối chứng training-without-branch. Không diễn giải F1khôngđổi là branchhoàntoànvôdụng nếu logits/decisions subset đổi.

## B. Fusion telemetry trên 128 validation rows đầu mỗi task

Collect nativeblock tensors ngoài hottraining: Wooutput x, intendeddelta d, hiddenresidual h, y=x+d trướcdropout. Mask querypadding. Dtype x,d,y,h/LNinput phảighi thực tế.

- Frobenius ||d||/||x|| và ||d||/||h||; pertoken quantiles10/50/90/99 vàsamplecounts.
- Tỷ lệ nonzero-delta tọa độ mà native y==x; táchdelta0 để khônginflate swallowing.
- Effective correction `(y.float()-x.float())`, intendeddelta, normratio và relativeerror `||effective-d||/||d||` vớizero-denominatorNA.
- So block output afterLN native/off/fp32 trên cùnginput, không lấyhiddennorm làmWo norm.

Không dùng tỷ lệ tọa độunchanged để suy trực tiếp tỷ lệ thông tin bị mất. Chuẩn hóa theo sameinput/mask vàghi batchshape.

## C. Norm, cập nhật và gradient

- Từ checkpoint: norm U/W/P, norm originalWo/FFNweights, output P đã kháczero chưa.
- Nếu recreate đúng initializer theo frozenrunner/seed42, ghi relative U/W drift và initializerhash; đánh dấu reconstruction assumption nếu không có init snapshot chứng minh. Không gọi drift lịch sử chính xác khi RNGpathkhôngtáitạođược.
- E15originalencoder tensors có thể so officialpretrainedsource trực tiếp; E13 sooriginalGD. Đây là displacement, không đo đường đi của optimizer.
- Fixedtrainbatch16 evalmode: CE backward trênnative đểđo gradientU/W/P vàWo/FFN; label gradient làprobe, không bước training mới. Reportrawgradnorm, grad/weightnorm (P0ởinit thìNA), finite/zero. Probe thêmfreshinitializedmodel cho từngvariant/task nếu rẻ đểxácnhận Pfirst vàUWzero banđầu; không dùng optimizer.step đểmởUW.
- E13loss khi train là NC nhưng probe CE để so tín hiệu task, ghi rõ khác objective training. Không suy gradnorm nhỏhơn do cùngbatch rằng NC sai. Không dùng currentgradient thay lịch sử.
- Ghi LRgroup củabranch, sốsteps,bestepoch,micro/effectivebatch thực, trainloss cuối nếu có. Lập đối chiếu E13vsE8 và E15vsE14 vềrngpath/head/dropout/optimizer, đánh dấu khác biệt cụ thể, không đưa nhận định nghiên cứu.

## D. Thực thi và artifacts

Isolatedscript `scripts/measure_architecture_transfer.py`, output `artifacts/architecture_transfer_audit/`. Saveincrementally mỗi checkpoint/mode để resume không redo; nohelperagents. GPU0–3freeVRAM hostcheck; .venv-e0-cuda/bin/python; CUBLAS_WORKSPACE_CONFIG=:4096:8. Nếu sandboxkhôngcóCUDA thửhostexecescalated. Dùng GPU choinference cầnthiết, không chạyCPUfullval khi hostGPU có.

20models×3validationpasses; giới hạn chính làfullval20checkpoints và fixedprobes trên, khônggridamplitudes hoặc thêmdata. Tối đa3–4workerprocess bên trong cùngagent, đúngmicrobatch16. Baselinefiles rereadonly, originalcheckpointSHA verified before/after ifneeded.

Deliver summaryCSV/JSON, permodelmodepredictions/telemetry, gradient/updateCSV, runtime/source/inputmanifest, executionnotes về failures/limits. Agent chỉ báo các số và việc đã/chưa chạy; root tự viết conclusions. Báo progress nhanh vàfinish mọi phépđo, không dừng ởscript.
