# Bổ sung kiểm chứng đường tính có fallback cho cavity keys

Root đã review phép kiểm đầu: FP64 forward/backward pass, nhưng scaled_near_removed_message FP32 có max abs score error2.79648 và maxrelative.632684. Không đưa fastpath hiện tại vào model. Giữ nguyên script/results/report cũ làm bằng chứng failure.

Đặc tả bổ sung chỉ xử lý numerical evaluation của **cùng operator**, không thay mô hình hoặc nghiên cứu prior art. Tạo script mới `scripts/research_checks/cavity_key_guarded.py`, import reference/inputs từ script cũ. Output mới `artifacts/research_checks/cavity_key_guarded/`.

## Thuật toán root chốt

Từ factorized intermediates w,e,a,we,ee:

E[i,j]=(||w_j||²+2Σh|a_ijh * <w_j,e_ih>|+Σhg|a_ijh*a_ijg*<e_ih,e_ig>|)/d.

Flag pair khi variance_factorized <= 1e-3*E + 1e-7, hoặc variance/score khôngfinite. Ngưỡng này là numerical guard bảo thủ cho prototype, không phải formal roundoff bound. Không điều chỉnh bằng kết quả benchmark.

Fast pairs dùng score factorized. Flagged pairs: tính literal C=z_j−Σh a[h,i,j]u[i,h] theo từng chunk tối đa64pairs, rồi F.layer_norm với gamma/beta/eps1e-5 và Wk/Q trực tiếp cho tất cảheads. Scatter/where vào scores, giữ gradient của nhánh được chọn; guard boolean không cógradient. Không materialize toàn N×N×d hoặc N×N×H×p; chunk C[M,d] được phép. `where` đơn thuần với fast NaN cóthểlàm gradientNaN; lưu limitation nếu implementation không xửlýđược, không giảpass.

Maxscores keymaterialchunk[M,H,p] được phép. Worstcase mọi pairsflagged thì compute cóthể vềliteral complexity; khôngclaim bảođảmfastruntime. Đây là bounded-workspace fallback, không universal stabletheorem.

## Kiểm tra

- Chạy lại đúng inputs 6FP32diagnostics củaaudit cũ; nếu cần dựng từseeds đọc run() cũ, khôngthayrandomstream. Lưu literal/fast/guarded maxabs,relativeL2, flagcounts vàmin/maxvariance.
- Báo elementwise closeness mới theo absdiff<=1e-5+1e-4*abs(reference); đây là criterionFP32 khai báo trước run bổsung, khôngthay hoặc xóa strictFP64-fail flags cũ. Rawerrors phải được giữ.
- Forward/backward FP64 parity trên ít nhất2case wellconditioned và1near-cancellation, dùng cùnggradientgroups, khôngoptimizerstep. Boundabs1e-8/relnorm1e-6; báozero-grad riêng.
- Kiểm sourceSHA cũ khôngđổi; save newscriptSHA vàsourceauditSHA, inputs/seedprovenance.
- Không model, dataset, training/GPU. Không mở thêm phươngpháp. Nếu guardchưa xửlýfailure thì báoroot số liệu và dừng tạiđó.
