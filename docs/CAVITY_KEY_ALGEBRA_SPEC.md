# Spec kiểm chứng đại số: key chuẩn hóa loại đóng góp từ người nhận

Đây là phép kiểm chứng công thức trên tensors tổng hợp, không phải model experiment, không đọc datasets/checkpoints, không training. Root phụ trách nghiên cứu; execution agent chỉ hiện thực hai công thức độc lập và trả số liệu. Không quyết định novelty hoặc hướng nghiên cứu.

## Dữ liệu và quy ước

N tokens, H heads, d hidden, p headwidth; column-vector convention. Inputs H0[N,d], Q[N,H,p], V[N,H,p], Wo[H,d,p], bo[d], Wk[H,p,d], bk[H,p], gamma[d], betaLN[d], routing A[H,N,N]. A[h,j,i] là source i được receiver j đọc, softmax rows với pad keys masked. Q,V,A được phép là các inputs độc lập; A có thể tạo từ logits leaf để kiểm gradient.

u_i^h=Wo^h v_i^h.

z_j=H0_j+bo+sum_(h,k) A[h,j,k] u_k^h.

a[h,i,j]=A[h,j,i] khi i≠j, và0 khi i=j. Không renormalize A sau deletion. Pad queries/keys loại khỏi metric, outputpad có thểzero.

## Reference (được phép materialize cho kích thước nhỏ)

C[i,j,:]=z_j−sum_h a[h,i,j]u_i^h.

LN(C)=gamma*(C−mean(C))/sqrt(mean((C−mean(C))²)+eps)+betaLN.

S[i,t,j]=Q[i,t] dot (Wk[t] LN(C[i,j])+bk[t])/sqrt(p).

## Công thức factorized, không materialize C[N,N,d] hoặc keys[N,N,H,p]

w_j=center(z_j), e_i^h=center(u_i^h).

vnorm[i,j]=(||w_j||²−2 sum_h a[h,i,j] dot(w_j,e_i^h)+sum_(h,g) a[h,i,j]a[g,i,j]dot(e_i^h,e_i^g))/d.

sigma[i,j]=sqrt(max(vnorm,0)+eps) cho FP32 safety; float64 reference không cần max trừ roundoff. Ghi negative variance/count/min và nearzero.

r_i^t=gamma * (Wk[t]^T Q[i,t]).

n[i,t,j]=dot(r_i^t,w_j)−sum_h a[h,i,j]dot(r_i^t,e_i^h).

c[i,t]=Q[i,t] dot (Wk[t] betaLN+bk[t]).

S[i,t,j]=(n[i,t,j]/sigma[i,j]+c[i,t])/sqrt(p).

Dùng precomputed pair dot products và head Gram matrices, không tensors với N,N,d axes trong implementation factorized. Ghi einsum intermediate shapes và derived peak logical element count, không giả đây là đo peakVRAM.

## Phép kiểm bắt buộc

1. Forward parity float64, random cases N=1/2/5/9, H=1/2/3, d>=H*p (có thể d=Hp), LN eps1e-5, nontrivial gamma/beta/bias; softmax random A, causal không cần vì encoder bidirectional; padding cases có ít nhất1validtoken.
2. Backward parity scalar random cotangent trên all valid S, gradients wrt H0,Q,V,Wo,bo,Wk,bk,gamma,betaLN và A-logits. So absolute/relative norms, finite. float64 tolerances1e-8abs/1e-6rel trên wellconditioned cases.
3. Token permutation equivariance khi permutation đồng bộ inputs/A/mask; diagonal exception preserved.
4. Partial-path exclusion: i≠j, giữH0,Q,A cố định, perturb V_i và recompute z. C[i,j] và S[i,:,j] phải invariant (routing fixed). Full context keys LN(z_j) phải có case đổi rõ. Không tuyên bố toànmodel invariant vì Q/K/A được giữfixed và H0 là input độc lập.
5. Không nhầm với diagonal mask: với3nodes,i≠j,cavityscore cócase khác fullcontext, dù selfedges bịmask trong firstpass. Save small numerical witness.
6. NoLN limiting identity: S_cav=S_full−sum_h a[h,i,j]*(Q_it dot Wk_t u_ih)/sqrt(p). Dùng identityLN riêng; đây là control mathematical.
7. FP32 cases random + cancellation/adversarial: z gầnremovedmessage, constantcenteredrows, scale1e-3/1/1e3. Báo sai lệch kể cảfail, không nới tolerance để pass. Chỉ classify arithmetic, không thay thiết kế.
8. Không benchmark model/GPU, không optimizerstep hoặc train. Không cài packages. Dùng môi trường Python/Torch sẵn có nếu cần.

## Deliverables

- `scripts/research_checks/cavity_key_algebra.py` standalone CPU.
- `artifacts/research_checks/cavity_key_algebra/results.json` chứa seed/shapes/dtypes/errors/tolerances/status/invariantwitnesses/intermediates/sourceSHA.
- `artifacts/research_checks/cavity_key_algebra/REPORT.md` factual ngắn, phân biệt parity failures và ổn địnhFP32.
- Không chỉnh E1–E16, không tự mở biến thể hoặc nghiên cứu paper. Nếu công thức root sai, chỉ nêu counterexample để root sửa spec.
