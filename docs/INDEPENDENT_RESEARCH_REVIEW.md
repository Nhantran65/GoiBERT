# Review nghiên cứu độc lập — GoiBERT (snapshot 12/09/2026; cập nhật bước 3 ngày 13/09)

**Trạng thái (13/09):** C1 đã chạy bước 3 và **bị loại**; định vị nhân quả cho thấy điểm nghẽn là mã CLS một-ý-một-vector không chồng chất được (mục 12). Vòng tiếp cho C4 (reader CLS theo phân vùng trên stack chung): qua mechanism check không train, nhưng phép tính đã có (Ruan et al. 2020) — mục 13. Các mục 0–9 giữ nguyên như đề xuất ban đầu để đối chiếu.

Tác giả: cộng sự nghiên cứu độc lập (không phải agent chính, không phải subagent thực thi). Phạm vi: đọc artifacts/docs local, phân tích predictions validation đã lưu (không load checkpoint, không train, không mở test), kiểm tra toán học bằng số, đối chiếu literature công khai tới 12/09/2026. Mọi số F1 trong tài liệu này ở thang điểm [0,100]; delta là điểm F1.

Script và output phân tích mới nằm ở `artifacts/independent_review/` (không sửa bất kỳ artifact E1–E16 nào):

- `multi_aspect_analysis.py/.txt`: lỗi theo số aspect gold mỗi câu, theo câu trộn/đồng polarity, theo aspect hiếm/phổ biến.
- `majority_count_analysis.py/.txt`: lỗi của aspect mang polarity thiểu số theo số aspect đa số cùng câu, và tỷ lệ "đổi sang polarity đa số".
- `minority_polarity_breakdown.py/.txt/.json`: phân rã theo cặp (polarity thiểu số, polarity đa số).
- `toy_multiplicity_check.py/.txt`: kiểm số tính chất bội số của softmax, leverage và kernel-multiplicity.

## 0. Tóm tắt điều hành

1. **Tôi không đồng ý với cách đọc "screen âm = cơ chế vô ích" cho E9–E16.** Với SD 5 seed của E1 là 1,04/0,82/0,50/0,60/0,32 điểm và riêng đổi micro-batch/precision (E16 baseline so E14) đã dịch validation Sentiment −1,13 điểm, gate "≥4/5 task tăng, ≥2 task ≥0,3" ở một seed gần như không có power với hiệu ứng cỡ 0,3–0,5 điểm. Phần lớn screen cũ là *chưa kết luận được*, không phải âm; screen dương E16 đảo chiều trên test đúng như dự đoán từ noise. Hệ quả cho thiết kế: candidate phải có **thước đo cơ chế riêng** (subgroup, paired) có variance thấp hơn F1 tổng, và dự đoán hiệu ứng đủ lớn trên subgroup đó.
2. **Bằng chứng local mới, lặp lại trên 3 seed và có ở cả BamiBERT:** trong câu ABSA nhiều aspect trộn polarity, aspect mang polarity *thiểu số* bị sai polarity 22–30% (so 8–9% chung), **70–78% lỗi đó đổi sang polarity đa số** (kỳ vọng ngẫu nhiên ≈50%), và tỷ lệ tăng theo số aspect đa số (ViSFD, n=393 slot). Ở Hotel, aspect thiểu số bị *bỏ sót* 69% (student) / 54% (BamiBERT) so 30%/21% của aspect đa số (n nhỏ). Đây là chữ ký của **tổng hợp theo số lượng** (count-weighted aggregation): softmax attention là trung bình có trọng số nên hướng cấu trúc tần suất cao lấn át hướng tần suất thấp. Failure mode này đã được ghi nhận cho ABSA tiếng Anh (ARTS, Xing et al. 2020) nhưng biện pháp ở đó là adversarial training, không phải cơ chế.
3. **Candidate ưu tiên (C1, với claim đã hạ bậc ở mục 4): attention chuẩn hóa bội số mềm theo key** — thêm vào logit của key j một bias `−ρ_h·log m̃_j`, với `m̃_j = 1 + Σ_{i≠j} exp(τ(cos(k_i,k_j) − 1))` là "số bản sao mềm" của key j trong chính câu đó. ρ_h=0 cho đúng hàm pretrained; ρ_h=1 cho tổng hợp kiểu *tập hợp* (mỗi hướng key một phiếu) thay vì *đa tập* (mỗi token một phiếu). Không thêm ma trận nào; +1 scalar/head. Phép tính mới nằm ở **bộ ước lượng bội số trong ngữ cảnh và trục điều khiển "đa tập ↔ tập hợp" của hướng tổng hợp**; đã kiểm số rằng biến thể leverage (ridge) *không* làm được việc này khi N < d_k (câu ngắn), còn kernel-multiplicity thì làm được.
4. **Verdict novelty C1 (sau hai vòng tìm kiếm):** phép tính cốt lõi — KDE ước lượng mật độ rồi nhân nghịch đảo vào trọng số tổng hợp — **đã có** trong point-cloud learning (PointConv 2019 "inverse density scale", GACNN 2020 "density attention"), ở không gian tọa độ vật lý để bù lấy mẫu không đều. Cùng *dạng* với proportional attention của ToMe (chiều ngược), cùng *trục* với CPA/NAPC/MAA (họ giữ count, tôi bỏ), anh em full-row là Intention/mesa/TTR. Phần còn lại của C1: chuyển sang *key space đã học* của encoder ngôn ngữ, centering theo câu, số mũ học được với identity chính xác, và gắn với failure mode đo được. **C1 không phải phép tính mới**; nó là một cơ chế chuyển giao có bằng chứng. Đóng góp nếu có nằm ở phát hiện failure mode + chứng minh cơ chế sửa được nó, không ở kiến trúc.
5. **Hai candidate còn lại bị hạ xuống vai trò đối chứng:** C2 readout gán-độc-quyền (softmax theo *nhãn* cho từng token + dustbin) trùng slot attention/OTSeg/SuperGlue; C3 coverage theo chiều sâu cho hàng CLS trùng coverage NMT. Chúng có giá trị chẩn đoán (rò rỉ ở readout hay ở aggregation; thiếu đọc theo độ sâu hay theo bội số) nhưng không phải phát minh.
6. **Kỳ vọng thực dụng, nói thẳng:** trần cải thiện của failure mode này nếu giảm được một nửa là ≈ +0,7…+1,1 điểm pair F1 mỗi ABSA task; không có cơ chế nào tôi biện minh được từ bằng chứng địa phương cho VSFC (lỗi ở đó là ngữ dụng/kiến thức). "Thắng BamiBERT rộng" bằng riêng một cơ chế trên student là không thực tế; trên BamiBERT gốc thì kịch bản khả thi là 3 ABSA tăng, VSFC không đổi trong noise.
7. **Đối chiếu với vòng 3 của agent chính (cavity keys, E17, mục 10):** operator chặt chẽ và prior art thẳng thắn, nhưng không có failure mode đo được đứng sau, hiệu ứng cấu trúc dự đoán nhỏ ở block 6, chi phí cao hơn, và gate screening cùng vấn đề power. Hai candidate không mâu thuẫn; đề nghị chạy chung một mechanism battery không train trước mọi trial 30-epoch.
8. **Kế hoạch rẻ:** bước 0 không train (đo bội số key trên checkpoint đóng băng; probe nhân bản mệnh đề trên checkpoint đóng băng có/không có chuẩn hóa; adjudication 60 lỗi thiểu số) → screen student 13 trial ≈ 3 giờ L40S → xác nhận seed 43/44 ViSFD/Restaurant ≈ 1 giờ → port BamiBERT 15 trial ≈ 4 giờ. Kill criteria khóa trước ở mục 8.

## 1. Nhận định độc lập về diễn giải lịch sử

### Đồng ý

- E13/E15/E16 là kết quả âm hợp lệ *trong protocol đã chạy*; audit 20 checkpoint đã loại "gradient chết" và "BF16 fusion" khỏi vai trò giải thích mặc định. Không cần cứu các nhánh đó.
- Bác bỏ hai ý tưởng A/B của lượt trước là đúng: A trùng Relational/Edge Transformer ở cấp nguyên lý; B là recurrence tuyến tính có điều kiện thuộc họ matrix-space/DeltaNet và, quan trọng hơn, là cơ chế **phụ thuộc thứ tự quét** — một tính chất decoder được nhập vào encoder hai chiều mà không có failure mode nào yêu cầu nó. Đại số trong `IDEA_NOVELTY_AUDIT.md` §4 tôi kiểm lại đúng.
- D1/D2 không chỉ định được block lỗi; E12 không ủng hộ nhánh ký tự/dấu; E9 không bác bỏ routing trong encoder.
- Không được gọi test hiện tại là holdout nguyên vẹn; mọi claim cuối phải khai báo adaptive exposure.

### Phản đối hoặc bổ sung, vì ảnh hưởng quyết định

**(a) Power của protocol screening.** Bảng dưới lấy từ artifact/docs local:

| Nguồn noise | Sentiment | Topic | ViSFD | Restaurant | Hotel |
|---|---:|---:|---:|---:|---:|
| E1 SD test 5 seed (`E1_TEST_BENCHMARK`) | 1,04 | 0,82 | 0,50 | 0,60 | 0,32 |
| E16 baseline − E14 validation, chỉ khác micro/precision (`artifacts/e16/E16_RESULT.md`) | −1,13 | +0,01 | −0,41 | −0,26 | −0,20 |
| E16 FFN-gate: val delta → test delta | +0,25 → −0,77 | +1,04 → +0,46 | +0,07 → −0,38 | −0,39 → −0,03 | +0,60 → −0,40 |

Với noise cỡ này, ngưỡng 0,3 điểm ở một seed không phân biệt được cơ chế với nhiễu trên Sentiment/Topic/Restaurant; ViSFD/Hotel khá hơn. Kết luận nên sửa: các screen âm E9–E16 phần lớn *chưa kết luận*, và một cơ chế chỉ đáng thử nếu (i) dự đoán hiệu ứng ≥ 1 điểm trên ít nhất một task có SD thấp, hoặc (ii) có **thước đo cơ chế** (subgroup, so sánh paired trên cùng rows) mà hiệu ứng dự đoán lớn gấp nhiều lần SE. Tôi thiết kế theo (ii).

**(b) Hai ý tưởng vừa rồi và cả ba research trước không xuất phát từ một failure mode đo được.** Chúng đi từ literature (pair state, operator composition, head mixing) rồi tìm động cơ. Phần 2 dưới đây cung cấp một failure mode có số liệu, lặp lại 3 seed, có mặt ở cả BamiBERT, và có tiền lệ literature (ARTS) — và nó chỉ đúng một phép tính cụ thể cần sửa.

**(c) "Thắng rộng" không thể là tiêu chí screening cho một cơ chế.** VSFC (2/5 task) có lỗi chủ yếu là đánh giá ngầm/ngữ dụng (E12: vsfc:09, vsfc:11) — không có cơ chế encoder nào tôi có thể biện minh từ bằng chứng địa phương. Tiêu chí hợp lý: ABSA tăng đúng subgroup dự đoán, VSFC không giảm quá noise.

**(d) Student hay BamiBERT.** Failure mode ở phần 2 có ở cả hai; student rẻ hơn 2,6× và có baseline 3 seed trên ViSFD/Restaurant, nên dùng student để *kiểm cơ chế*, BamiBERT để *đạt mục tiêu thực dụng*. Một cơ chế không thêm capacity là loại chuyển được sang BamiBERT mà không kéo theo confound "nhiều params".

### Chưa đủ nguồn

- Tỷ lệ lỗi thiểu số-polarity do nhãn mơ hồ (đặc biệt "neutral" trong review khen) chưa được adjudicate; E12 đã thấy nhiều gold khó bảo vệ. Đây là trần cứng của mọi cơ chế; cần bước 0c.
- Evaluator/averaging của paper BamiBERT vẫn chưa xác minh; không ảnh hưởng thiết kế, ảnh hưởng claim SOTA.

## 2. Bản đồ khoảng trống: bằng chứng local mới

### 2.1. Phân tích trên predictions validation đã lưu

Nguồn: E8 seed 42/43/44 (ViSFD, Restaurant), E8 seed 42 (Hotel), E14 BamiBERT LR đã chọn, E16 baseline; gold từ loader đông lạnh; mapping 1 positive/2 negative/3 neutral theo audit E12; ViSFD OTHERS loại khỏi pair. Không có model nào được load.

**(i) Câu trộn polarity có tỷ lệ sai polarity cao gấp 2–4× câu đồng polarity, ở mọi model.** Sai polarity trên gold slot trong câu ≥2 aspect:

| Task | Model | Câu trộn polarity | Câu đồng polarity |
|---|---|---:|---:|
| ViSFD | E8 s42 / s43 / s44 | 12,5 / 13,8 / 13,6 % | 3,0 / 3,0 / 3,1 % |
| ViSFD | BamiBERT E14 | 10,9 % | 3,5 % |
| Restaurant | E8 s42 / s43 / s44 | 14,8 / 15,2 / 12,7 % | 8,8 / 8,5 / 7,1 % |
| Restaurant | BamiBERT E14 | 16,0 % | 7,1 % |
| Hotel | E8 s42 ; BamiBERT | 12,3 % ; 14,9 % | 3,0 % ; 3,9 % |

**(ii) Aspect mang polarity thiểu số bị hấp thụ vào polarity đa số, và mức độ tăng theo số aspect đa số** (ViSFD, chỉ câu có polarity đa số duy nhất; n=176 slot khi có 2 aspect đa số, n=217 khi ≥3):

| Model | 2 aspect đa số: sai polarity (→đa số) | ≥3 aspect đa số: sai polarity (→đa số) | Gộp: sai % ; tỷ lệ →đa số |
|---|---:|---:|---:|
| E8 s42 | 22,7 % (31/40) | 27,6 % (42/60) | 25,4 % ; 73 % |
| E8 s43 | 21,0 % (27/37) | 27,6 % (41/60) | 24,7 % ; 70 % |
| E8 s44 | 25,0 % (33/44) | 30,4 % (47/66) | 28,0 % ; 73 % |
| BamiBERT E14 | 19,9 % (22/35) | 22,1 % (36/48) | 21,1 % ; 70 % |

Kỳ vọng nếu lỗi polarity phân bố ngẫu nhiên giữa hai lớp còn lại là ≈50% đổi sang đa số; quan sát 70–78%. Tỷ lệ sai của slot thiểu số (21–28%) gấp ~3× tỷ lệ sai polarity chung (7,6–8,9%).

**(iii) Phân rã theo cặp polarity** (ViSFD; `minority_polarity_breakdown.txt`): ô lớn nhất là *thiểu số neutral trong câu đa số positive* (n=164): sai 29–33% (student), 24% (BamiBERT), trong đó 76–83% đổi sang positive; *neutral trong đa số negative* (n=34): sai ~50%, 80% đổi sang negative; *negative trong đa số positive* (n=119): sai 17–25%, nhưng chỉ ~45–53% đổi sang positive (gần ngẫu nhiên). Nghĩa là hấp thụ mạnh nhất với aspect **ít dấu hiệu polarity riêng** (neutral); opinion phủ định rõ ràng bị hấp thụ ít hơn.

**(iv) Bỏ sót tăng theo số aspect ở text ngắn** (Restaurant: 12,9→17,8→34,4% cho 1/2/3 aspect gold, student s42; BamiBERT 10,0→13,9→21,1%; Hotel 21,6→28,1→25,2→31,8% vs 18,1→23,1→19,2→26,6%), không tăng ở ViSFD (text dài, câu 1 aspect lại khó nhất: 16%). Ở Hotel, aspect thiểu số-polarity bị bỏ sót 18/26 = 69% (BamiBERT 14/26 = 54%) so 30%/21% của aspect đa số — n nhỏ, chỉ là hướng.

**(v) Aspect hiếm theo tần suất train bị bỏ sót nhiều hơn** (ViSFD 13–14% vs 8%; Restaurant 24% vs 15%; Hotel 37% vs 20%) — đây là imbalance, không đặc thù kiến trúc, ghi để không nhầm với (ii).

### 2.2. Bản đồ

| Vấn đề | Bằng chứng local | Literature | Chưa biết |
|---|---|---|---|
| Hấp thụ polarity đa số / thiểu số bị lấn át trong câu nhiều aspect | (i)–(iv), 3 seed, cả BamiBERT | ARTS (Xing et al. 2020): model ABSA sụt tới 69,7% khi đảo sentiment aspect không phải target; fix bằng adversarial training | Phần nào là nhãn mơ hồ (neutral); rò rỉ ở aggregation, ở readout hay ở binding token-level |
| Bỏ sót aspect (recall) là phía lỗi lớn hơn | `E8_ERROR_READOUT`, (iv) | — | Có phải "thiếu đọc" theo độ sâu không (C3 đo) |
| Binding sai đối tượng (SERVICE→RESTAURANT, phụ kiện→BATTERY) | E12 ví dụ, không tỷ lệ | Binding problem, TP-Transformer | Tần suất; có sửa được ở top layers không |
| Đánh giá ngầm, ngữ dụng (VSFC) | E12 | — | Không có cơ chế encoder rõ; là kiến thức pretraining |
| Có nên giữ hay bỏ thông tin *số lượng* trong attention | — | CPA/NAPC/MAA/ToMe muốn giữ (đếm, evidence accumulation); tôi muốn bỏ *tương đối* cho output kiểu tập hợp | Dấu của hiệu ứng theo task (ABSA vs VSFC) |

## 3. Nguyên lý thiết kế: từ failure mode tới phép tính

Softmax attention của receiver i: `o_i = Σ_j α_ij v_j`, `α_ij = softmax_j(q_i·k_j/√d)`. Nếu m token có key trùng nhau (cùng hướng cấu trúc a) và một token có hướng b, với `q_i·k_a = q_i·k_b`, thì khối a nhận khối lượng `m/(m+1)`, b nhận `1/(m+1)`. Đây là tổng hợp kiểu **đa tập**: mỗi token một phiếu. Với output kiểu tập hợp (aspect nào *có mặt* và polarity *gắn với nó*), điều cần là **mỗi hướng cấu trúc một phiếu** — tổng hợp kiểu *tập hợp*. Phân loại sum/mean/max của GIN (Xu et al. 2019) diễn đạt đúng cặp này: mean nhìn *phân bố*, max nhìn *tập*; softmax là mean có trọng số. Theo Test-time regression (Wang–Shi–Fox 2025, Eq. 50–51), softmax attention là ước lượng Nadaraya–Watson; trong thống kê phi tham số, NW có *design-density bias* — điểm thiết kế tụ lại kéo ước lượng về phía chúng — và cách sửa cổ điển là chia trọng số kernel cho mật độ thiết kế (Priestley–Chao 1972, Gasser–Müller 1979) hoặc dùng hồi quy cục bộ bậc cao hơn. Đó chính xác là hai hướng sửa dưới đây: C1 chia cho mật độ key trong ngữ cảnh; "anh em" full-row là hồi quy ridge (Intention/mesa-layer).

Điểm quan trọng để không overclaim: một mạng nhiều layer softmax **có thể biểu diễn** tổng hợp bất biến bội số (head sắc nét chỉ đọc một token đại diện; hoặc key của token cùng aspect học cách đẩy nhau; hoặc FFN đếm rồi chuẩn hóa). Claim của tôi là **inductive bias/ tham số hóa**, không phải expressivity; bằng chứng rằng model đã train *không* đạt được tính chất này là (ii)–(iii) — có ở cả BamiBERT.

## 4. Candidate cards

### C1 — Attention chuẩn hóa bội số mềm theo key (tên làm việc, không phải tên phương pháp)

**Hạn chế tính toán và bằng chứng.** Softmax là mean có trọng số nên hướng key xuất hiện nhiều lần lấn át hướng xuất hiện ít; ở ABSA điều này hiện ra như hấp thụ polarity đa số (mục 2.1 ii–iii) và thiểu số bị bỏ sót (iv). Bằng chứng lặp 3 seed, có ở BamiBERT.

**Forward.** Tại layer `l ∈ L_mod` (student: index 7–9; BamiBERT: 8–11), head h, keys `K ∈ R^{B×H×N×d_k}` từ đúng `W_K` pretrained, mask hợp lệ `v ∈ {0,1}^{B×N}` (attention_mask, gồm special tokens). Tính trong FP32:

\[
\bar k=\frac{\sum_i v_i k_i}{\sum_i v_i},\qquad
\hat k_j = \frac{k_j-\bar k}{\|k_j-\bar k\|_2+\varepsilon},\qquad
C_{ij} = \hat k_i^\top \hat k_j,\qquad
\tilde m_j = 1 + \sum_{i\neq j} v_i\,\exp\big(\tau\,(C_{ij}-1)\big)\in[1,N_{valid}],
\]
\[
S_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}} - \rho_h \log \tilde m_j + \text{mask}_j,\qquad
A_{ij}=\mathrm{softmax}_j(S_{ij}),\qquad o_i=\sum_j A_{ij} v_j .
\]

Centering theo câu (`k̄`) là bắt buộc vì biểu diễn tầng trên của BERT rất anisotropic (Ethayarajh 2019): nếu không trừ hướng chung, mọi cặp key đều có cosine cao, `m̃` gần đồng đều theo token và bias triệt tiêu trong softmax; softmax bất biến với dịch chung của logit nên hướng chung của cả câu không mang thông tin bội số cần giữ. Bias theo key j, broadcast cho mọi receiver i (kể cả CLS và token). Phần còn lại của block (Wo, dropout, residual, LayerNorm, FFN) không đổi. `τ` cố định (8 trên cosine; chốt trước, độ nhạy {4,8} chỉ báo từ bước 0a trên train). `ρ_h = 1,5·tanh(θ_h)`, `θ_h ∈ R` học được, init 0 (ρ=0 chính xác; đạo hàm tại init bằng 1,5 nên không chết); ρ=1 là điểm "tập hợp", biên ±1,5 chặn over-correction; cho phép âm (âm = hướng "giữ count" như MAA; dấu học được là thông tin).

**Phép tính mới nằm ở đâu.** (1) `m̃_j`: bộ ước lượng *bội số mềm trong ngữ cảnh* của key j từ Gram key–key, không cần merge/prune token; (2) `−ρ log m̃` biến softmax thành họ một tham số nội suy giữa tổng hợp đa tập (ρ=0) và tập hợp (ρ=1): với m bản sao chính xác, khối lượng khối bằng `m^{1−ρ} e^{s}`. Không có ma trận mới, không gate trên value, không nhánh song song.

**Block vs mạng.** Đã nói ở mục 3: đây là inductive bias. Dự đoán kiểm chứng được là model *đã train* với ρ học được sẽ mở ρ>0 ở một số head top layers cho ABSA và giảm đúng subgroup lỗi (ii)–(iii).

**Ví dụ phân biệt và phản ví dụ (kiểm số, `toy_multiplicity_check.txt`).** 5 key gần trùng hướng a (cos 0,95) + 1 key b + 10 token generic, query khớp a và b bằng nhau: softmax cho khối a 0,50, b 0,10; tham chiếu "1 a + 1 b" là 0,17/0,17; C1 (τ=8, ρ=1) cho 0,26/0,15. Biến thể ridge-leverage (đường chéo hat matrix, `ℓ_j = k_jᵀ(KᵀK+λI)⁻¹k_j`) **không** sửa được (0,45–0,49/0,10–0,11 với λ_rel 0,1–0,01) vì với N<d_k mọi key gần trùng vẫn độc lập tuyến tính → leverage ≈ 1; nó chỉ đúng với trùng chính xác hoặc N>d_k. Đây là lý do C1 dùng kernel, không dùng leverage/ridge — khác biệt kỹ thuật có thể kiểm. Kiểm số thứ hai (cùng file): nếu mọi key có chung một thành phần lớn (mô phỏng anisotropy, |3u| so content đơn vị), `m̃` không centering ≈ 8–9,5 cho *mọi* token (bias gần hằng, triệt tiêu), centering theo câu đưa về `m̃_A≈1,3`, `m̃_B≈m̃_generic≈1,0` — đúng hướng nhưng yếu hơn trường hợp đẳng hướng, nên 0a phải đo trên key thật.

Cơ chế **không giúp hoặc phản tác dụng** khi: (a) count là tín hiệu (cường độ; "mostly positive with one complaint" ở VSFC sentiment; các domain của MAA/ToMe) — dự đoán ρ≈0 hoặc Sentiment giảm nếu ρ mở; (b) hai aspect *khác nhau* nhưng key gần trùng (FOOD#QUALITY vs FOOD#STYLE&OPTIONS): C1 gộp chúng thành một phiếu, có thể tăng nhầm lẫn giữa aspect anh em — phải theo dõi confusion giữa aspect anh em; anh em full-row ridge (whitening) tách được nhưng mất tính lồi; (c) head cú pháp/vị trí: key mang thành phần vị trí nên không trùng → không đổi (trung tính); (d) khi khối bị de-dup là khối *đúng* mà query cần, khối lượng dời sang token generic/sink — đúng theo tham chiếu 1-bản-sao nhưng giảm tỷ số tín hiệu tuyệt đối; hại hay không tùy value của token sink; (e) từ chức năng/dấu câu lặp ("và", ",", "rất") có `m̃` lớn nên bị hạ trọng số — có lợi nếu chúng là nhiễu, có hại nếu head dùng chúng làm mốc; (f) nếu key tầng trên gần như đồng hướng ngay cả sau centering, `m̃` không phân biệt được token và C1 thành no-op — đây là kill 0a.

**Checkpoint, init, thích nghi, gradient, precision.** Không đổi weight nào; thêm `θ_h` init 0 → ρ_h=0 → logits đúng baseline (bitwise nếu cùng đường kernel). Không cần stage thích nghi vì không thêm capacity. `∂L/∂θ_h = 1,5·(1−tanh²θ_h)·[−Σ_{b,i,j} (∂L/∂S_{ij})·log m̃_j]` khác 0 tổng quát ngay bước 1 vì `log m̃_j ≤ 0` biến thiên theo j. `C, m̃, log, S, softmax` trong FP32 với autocast tắt tại các layer `L_mod` **cho mọi arm kể cả baseline replay** (bài học E16: không dùng E8/E14 khác đường kernel làm đối chứng duy nhất). LR riêng cho θ: 1e-2, weight decay 0, chốt trước (scalar log-space cần dịch O(1); LR encoder 1e-5 của VSFC sẽ đóng băng nó một cách vô tình); dùng chung cho mọi task/arm.

**Params/FLOPs/bộ nhớ/độ sâu.** +H scalar mỗi layer sửa (student 18, BamiBERT 48). FLOPs thêm ≈ `2·H·N²·d_k` mỗi layer (Gram key–key, bằng đúng chi phí của `QKᵀ`): với H=12, d_k=64, N=64 là ≈6 MFLOP so hai matmul FFN ≈0,6 GFLOP cùng layer (~1%); ở N=256 là ≈0,1 GFLOP so FFN ≈2,4 GFLOP (~4%) vì Gram tăng bậc hai còn FFN tuyến tính theo N. Activation: thêm một tensor `B×H×N×N` FP32 mỗi layer sửa (N=64: 6 MB; N=256: 100 MB tại B=32). Không thêm bước tuần tự; runtime kỳ vọng +2–5% (kernel nhỏ thêm; phải đo).

**Đối chứng cô lập cơ chế.**
- Baseline replay cùng đường FP32 tại `L_mod`, θ cố định 0.
- **C-perm:** cùng công thức, nhưng `m̃` bị hoán vị ngẫu nhiên theo token trong mỗi câu (cùng phân bố bias, cùng số params, cùng độ "làm phẳng" softmax; mất liên kết với bội số thật). Nếu C1 ≈ C-perm thì gain là do đổi nhiệt độ/ nhiễu, không do bội số.
- (Tùy chọn) anh em full-row ridge/Intention có gate: `o_i = (1−g_h)o_i^{soft} + g_h γ_h Vᵀ K (KᵀK+λI)⁻¹ q_i`, g init 0, γ calibration RMS train-only. Tách "whitening giữa hướng gần trùng" khỏi "chuẩn hóa bội số".
Cả ba dùng cùng LR, cùng micro-batch, cùng precision; capacity và compute khớp (C1 và C-perm khớp chính xác).

**Verdict bước 2 (claim chart ở mục 5, cập nhật): phép tính cốt lõi đã có ở point-cloud learning (PointConv/GACNN); C1 khác ở không gian ước lượng, vị trí, cách gắn vào checkpoint và mục đích. Không gọi C1 là phát minh kiến trúc. Giữ để kiểm chứng như *giả thuyết cơ chế có bằng chứng* — giá trị nằm ở việc chứng minh failure mode đo được được sửa bởi đúng cơ chế này, hơn control hoán vị.**

### C2 — Readout gán độc quyền: softmax theo nhãn cho từng token, có dustbin

**Hạn chế và bằng chứng.** Cùng failure mode (ii)–(iii) nhưng giả thuyết rò rỉ nằm ở *readout*: CLS phải mang đồng thời nhiều cặp (aspect, polarity) và head tuyến tính-tanh đọc ra polarity aspect X bằng feature toàn câu.

**Forward.** Hidden cuối `H ∈ R^{N×d}`; query nhãn `U ∈ R^{(A+1)×d}` (A aspect + 1 dustbin, init N(0,0,02)); `S = U Hᵀ/√d`; **softmax theo trục nhãn cho mỗi token**: `P_{aj} = exp(S_{aj})/Σ_{a'} exp(S_{a'j})` trên token hợp lệ; `r_a = Σ_j P_{aj} h_j / (Σ_j P_{aj} + ε)`; logits aspect a từ `r_a` qua dense/tanh/output rows như E9. Dustbin không có logits. VSFC: A = số lớp, hoặc giữ CLS (C2 chỉ có nghĩa cho ABSA).

**Phép tính mới ở đâu.** Trục chuẩn hóa: E9 softmax theo *token* cho mỗi nhãn (một nhãn có thể đọc cùng token mà nhãn khác đọc → rò rỉ); C2 softmax theo *nhãn* cho mỗi token (mỗi token giải thích bởi ~một nhãn hoặc dustbin). Đây đúng là chuẩn hóa của slot attention; không mới.

**Phân biệt/phản ví dụ.** Câu "phòng sạch, nhân viên tệ": E9 để cả query ROOM và SERVICE đọc "sạch"/"tệ"; C2 buộc "tệ" chọn một nhãn. Phản tác dụng khi một opinion thật sự thuộc nhiều aspect (đánh giá chung "khá ổn" phủ nhiều aspect — E12 absa_hotel:01) và khi A lớn (Hotel 34) làm softmax theo nhãn loãng ở init.

**Checkpoint/init/precision.** Encoder nguyên; head mới; scores/softmax FP32 (như E9). Params ≈ (A+1)·d. Chi phí O(N·A·d). Không thêm độ sâu tuần tự.

**Đối chứng.** E9 (softmax theo token) là control trực tiếp — đã có số seed 42 (+1,03 Restaurant, −0,62 Hotel, −0,34 ViSFD); thêm control "C2 không dustbin".

**Verdict bước 2: đã có cơ chế tương đương (slot attention 2020; OTSeg 2024 multi-prompt Sinkhorn attention; SuperGlue dustbin 2020; Sinkformer 2022 cho phiên bản doubly-stochastic).** Giữ như *comparator chẩn đoán*: nếu C2 giảm (ii)–(iii) mà C1 không, rò rỉ ở readout.

### C3 — Coverage theo chiều sâu cho hàng CLS

**Hạn chế và bằng chứng.** Bỏ sót aspect là phía lỗi lớn (E8 readout; (iv)). Giả thuyết: CLS ở các layer trên đọc lặp cùng token và không bao giờ đọc token của aspect yếu.

**Forward.** Với `l ∈ L_mod`, `c_j^{(l)} = Σ_{l'<l, l'∈L_mod} Σ_h A^{(l')}_{h,0→j}` (khối lượng attention tích lũy từ CLS tới token j); tại layer l, chỉ hàng CLS: `S_{0j} += −γ_h c_j^{(l)}`, γ init 0. Hàng token không đổi.

**Phép tính mới ở đâu.** Coverage của Tu et al. 2016 chuyển từ trục thời gian decoder sang trục độ sâu encoder; dấu ngược RealFormer (cộng lại score cũ). Không mới ở cấp phép tính.

**Phản ví dụ.** Khi token cần được đọc lại sau khi đã contextualize (đọc từ cảm xúc sau khi đã gắn đối tượng) coverage phạt đúng token cần thiết.

**Chi phí.** +H scalar/layer; O(N) thêm; cần giữ hàng CLS của A ở các layer trước (đã có khi tính attention thủ công).

**Verdict bước 2: có khác biệt (trục độ sâu) nhưng đóng góp chưa đủ.** Giữ như comparator tách "thiếu đọc theo độ sâu" với "bội số trong một layer" (C1).

## 5. Claim chart và tài liệu gần nhất (C1)

| # | Claim | Paper gần nhất (section/eq.) | Đã biết | Khác biệt thật | Chưa xác minh |
|---|---|---|---|---|---|
| 1 | Softmax attention tổng hợp theo số lượng; bội số tương đối kéo tổ hợp lồi về hướng xuất hiện nhiều | Test-time regression (Wang–Shi–Fox 2025, Eq. 50–51: softmax = Nadaraya–Watson); GIN (Xu et al. 2019, sum/mean/max); CPA (Zhang & Xie, IJCAI 2020); MAA (2026, Eq. `A_p = Σ m_i v_i/‖m‖_p`) | Toàn bộ mệnh đề toán | Cách đọc cho output kiểu tập hợp (presence/binding) | — |
| 2 | Bias logit theo log "khối lượng" của key | ToMe (Bolya et al., ICLR 2023): `softmax(QKᵀ/√d + log s)`, s = số patch đã merge | Dạng công thức | Dấu/mục đích ngược (bỏ count thay vì giữ), ước lượng liên tục trên chuỗi chưa merge | Hiệu quả trên NLU |
| 3 | Bộ ước lượng bội số mềm (KDE) → trọng số nghịch đảo mật độ trong phép tổng hợp | **PointConv (Wu, Qi, Fuxin, CVPR 2019): "inverse density scale"** — KDE trên tọa độ 3D, nhân nghịch đảo mật độ vào trọng số tích chập để bù lấy mẫu không đều (ablation +≈1%); **GACNN (Wen et al. 2020, LiDAR): "density attention"** — KDE Gaussian (Eq. 3–4), nghịch đảo mật độ neighbor chuẩn hóa theo max rồi MLP thành attention weight | **Phép tính cốt lõi đã có** (KDE → 1/density → weight) trong point-cloud learning | Không gian ước lượng là *key space đã học* của encoder ngôn ngữ (không phải tọa độ vật lý); centering theo câu; bias log trong softmax với số mũ ρ_h học được, identity chính xác; mục đích là output kiểu tập hợp/binding, không phải bù lấy mẫu | Hiệu quả trong text; đây là *chuyển domain + mục đích*, không phải phép tính mới |
| 3b | Cosine key–key như thước đo dư thừa; IDF-reweighting attention | ToMe/DART/similarity-aware pruning (merge/prune); KDEformer 2023 (KDE để *xấp xỉ* softmax); Compactor 2025 (leverage để evict); ReAttn 2026 (IDF post-hoc trên attention re-ranking, tần suất rời rạc xuyên tài liệu); Choi 2022 (một số head BERT tương quan IDF) | Dư thừa/tần suất dùng để chọn hoặc reweight | Bội số *mềm, trong câu, trong key space*, không rời rạc, không corpus-level, không bỏ token | — |
| 4 | Anh em full-row: đọc bằng hồi quy ridge trên toàn chuỗi | Intention (Garnelo & Czarnecki 2023, `f_ls = Q[KKᵀ+αI]⁻¹KᵀV`, Thm 2.6, Informer); mesa-layer (von Oswald et al. 2023); TTR Eq. 11–12 (ghi rõ "applies to non-causal"); SGPA (Chen & Li, ICLR 2023: attention = posterior mean của sparse GP, trên text/image/graph) | Toàn bộ phép tính | Chỉ còn chuyển đổi head pretrained có gate identity | Chưa thấy trên BERT pretrained; không phải novelty ở cấp phép tính |
| 5 | Lineage thống kê: NW design-density bias; Priestley–Chao/Gasser–Müller; hồi quy cục bộ bậc 1 = TTR Eq. 52 | Fan & Gijbels 1996; TTR §"Higher-order attention" | Trong thống kê | Chưa thấy nối vào attention encoder | — |
| 6 | Failure mode: sentiment aspect không-target lấn át | ARTS (Xing et al., EMNLP 2020) | Hiện tượng + adversarial training | Số liệu tiếng Việt, 3 seed, cả BamiBERT; biện pháp cấu trúc | Phần do nhãn |
| 7 | Chuyển đổi identity-anchored, +1 scalar/head | Talking-Heads đề xuất trong `ENCODER_REPLACEMENT_GAP_RESEARCH`; ReZero; DEX | Mẫu init | Không có ma trận mới | Học được với LR nào |
| 8 | DPP/volume-based attention phạt code dư thừa | Frankland et al. (eLife 2024) DPP attention trên grid codes | Ý tưởng "phạt dư thừa" khi chọn | Setting neuro/OOD, chọn subset, không phải operator token trong encoder ngôn ngữ | — |

Kết luận ba mức (cập nhật sau vòng tìm kiếm thứ hai): **claim 1, 2, 4, 5 đã biết; claim 3 — phép tính cốt lõi đã có ở point-cloud learning (PointConv 2019, GACNN 2020), phần còn lại của C1 là chuyển sang key space của encoder ngôn ngữ với identity-anchored exponent, cộng claim 6 (failure mode đo được). Vậy C1 thuộc mức "có khác biệt nhưng không phải phép tính mới"; đóng góp bảo vệ được là *phát hiện + cơ chế chuyển giao có bằng chứng*, không phải kiến trúc phát minh.** Bằng chứng encoder/NLU trực tiếp cho họ này: Sinkformer (NLP, doubly-stochastic, trục khác), NAPC (ablation >25.000 Transformer nhỏ, không phải BERT pretrained), Talking-Heads (BERT). Không có bằng chứng nào chuyển trực tiếp sang BERT tiếng Việt.

Claim chart C2/C3 ngắn: C2 ↔ slot attention (Locatello et al. 2020, softmax theo slot), OTSeg (ECCV 2024), SuperGlue dustbin (CVPR 2020), Sinkformer; C3 ↔ Tu et al. 2016, See et al. 2017 (coverage), RealFormer (dấu ngược). Cả hai: cơ chế tương đương đã có.

## 6. Lựa chọn ưu tiên và điều kiện đổi ý

**Chọn C1, nhưng với claim đã hạ bậc.** Lý do: duy nhất trong ba candidate có (a) failure mode đo được ở cả student và BamiBERT với hiệu ứng dự đoán lớn trên subgroup, (b) không thêm capacity nên tách được khỏi "nhiều params", (c) control hoán vị khớp chính xác compute/params, (d) khác biệt thực chất với tiền lệ point-cloud (không gian, mục đích, cách gắn checkpoint) dù không phải phép tính mới. Nếu người dùng chỉ chấp nhận "phép tính chưa ai làm" làm tiêu chí qua bước 2, thì **C1 không qua**, và kết luận trung thực của lượt này là: chưa có candidate đạt tiêu chí đó; tài sản còn lại là phát hiện failure mode ở mục 2 và bộ thước đo/probe để phân xử mọi candidate sau.

Tôi sẽ đổi ý nếu:
1. Bước 0a: median `m̃` (τ=8) trên content token ở `L_mod` < 1,2 trên cả hai checkpoint → key pretrained không có cấu trúc dư thừa để chuẩn hóa → bỏ C1.
2. Bước 0b: probe nhân bản mệnh đề trên checkpoint đóng băng không giảm tỷ lệ flip ≥25% tương đối ở ρ=1 → cơ chế không tác động failure mode trong biểu diễn này → bỏ.
3. Bước 0c: ≥50% lỗi neutral-thiểu-số được adjudicate là "gold cũng bảo vệ được bằng polarity đa số" → trần quá thấp, chuyển sang vấn đề nhãn.
4. Screen: C1 ≈ C-perm trên thước đo cơ chế → gain (nếu có) không do bội số.
5. C2 giảm (ii)–(iii) mà C1 không → rò rỉ ở readout; đổi sang thiết kế readout (novelty thấp hơn nhưng đúng khoa học).

## 7. Đặc tả tối thiểu có thể triển khai (C1)

- **Nơi gắn:** self-attention của các block index 7,8,9 (student `goiturn-s-gd`) hoặc 8–11 (BamiBERT gốc `models/bamibert-base-original`, commit `57bc1340…`, SHA `c8c8abe5…`). Mọi block khác giữ SDPA gốc.
- **Đường tính tại `L_mod` (mọi arm):** attention thủ công FP32: `Q,K,V` từ projection pretrained (BF16) → upcast FP32; `S = QKᵀ/√d_k + bias_key + mask_pad`; softmax FP32; `A` cast về dtype của V rồi `AV`; sau đó Wo/dropout/residual/LN như HF. Baseline replay: `bias_key = 0`. C1: `bias_key_j = −ρ_h log m̃_j`. C-perm: `bias_key_j = −ρ_h log m̃_{π_b(j)}` với π_b hoán vị ngẫu nhiên các vị trí hợp lệ trong câu b, sinh từ generator tách riêng, mới mỗi forward.
- **Shapes:** `k̄`: B×H×1×d_k (mean trên token hợp lệ; giữ gradient qua `k̄`, không detach); `K̂`: B×H×N×d_k; `C`: B×H×N×N (FP32); `m̃`: B×H×N; `θ`: H mỗi layer (Parameter, init 0, FP32; `ρ = 1,5·tanh θ`); τ: buffer scalar.
- **Mask:** `m̃` chỉ tổng trên `v_i=1`, loại self; key pad nhận `−inf` ở S như thường; hàng query pad không cần xử lý riêng (bị loại ở loss/readout).
- **NC (recipe E8):** forward noisy dùng cùng module, `m̃` tính lại trên text noisy.
- **Optimizer:** θ vào group riêng: AdamW lr 1e-2, wd 0, không LLRD, warmup chung (với ~7.000 step, Adam có thể dịch θ nhiều đơn vị; biên tanh giữ ρ trong ±1,5). Các tham số khác giữ E8 (student) hoặc E14/E16 (BamiBERT).
- **Precision/parity:** preflight: (1) logits eval của baseline replay vs E8/E14 checkpoint gốc trên 128 rows, báo max|Δ| (kỳ vọng ≤1e-2 do đổi kernel); (2) C1 tại ρ=0 bitwise bằng baseline replay; (3) `∂L/∂θ` hữu hạn, khác 0 ở batch đầu; (4) strict roundtrip checkpoint với ρ/τ.
- **Chuyển checkpoint:** không có; load pretrained, thêm ρ.
- **Ngân sách thích nghi:** không stage thêm; fine-tune trực tiếp. Nếu sau này muốn MLM continuation, phải là thiết kế architecture × adaptation với baseline nhận cùng data/compute.
- **Baseline equivalence:** ba arm cùng micro-batch (student 32; BamiBERT 16×2 như E16), cùng seed, cùng data order, cùng precision policy.
- **Log:** ρ_h mỗi epoch (và ở step 0/1/10/100); phân bố `m̃` trên một probe train cố định; thời gian/VRAM so baseline replay.
- **Mức trần độ dài:** N ≤ 256; `C` FP32 tối đa 32×12×256²×4 B ≈ 100 MB mỗi layer — chấp nhận được trên L40S theo peak 3–4,6 GB đã đo ở E8/E14.

## 8. Kế hoạch kiểm chứng tiết kiệm

Thời gian ước tính từ elapsed đã đo: student E8 trial 12–16 phút (NC = 2 forward), BamiBERT E14 7–20 phút, E16 13–22 phút, trên một L40S. Không ước tính xác suất thành công.

**Bước 0 — không train (≤1 giờ GPU inference + vài giờ người).**
- 0a. Trên E8 s42 (ViSFD, Hotel) và E14 (cùng task) đóng băng: phân bố cosine key–key trong câu (có và không centering) và `m̃` (τ=4,8) theo layer/head trên 2.000 train rows; kiểm tra nhanh rằng token lặp nguyên văn (ghép câu với chính nó) nhận `m̃≈2`. Chốt centering/τ từ train trước khi nhìn validation. Kill nếu median `m̃` < 1,2 ở mọi head `L_mod` hoặc IQR của `log m̃` trong câu < 0,1 (bias gần hằng → triệt tiêu).
- 0b. Probe nhân bản: từ các câu validation *một aspect* (ViSFD 156, Restaurant 521, Hotel 463), ghép cặp A+B khác aspect, khác polarity; tạo A+B, A+B+B, A+B+B+B (lặp nguyên văn). Đo P(polarity dự đoán của aspect A chuyển sang polarity B) theo số bản sao B, trên checkpoint đóng băng, rồi bật C1 với ρ∈{0,5;1} ở `L_mod` (không train). Dự đoán: baseline flip tăng theo số bản sao; C1 giảm độ dốc ≥25% tương đối. Đây là phép đo cơ chế tách khỏi nhãn và khỏi binding token-level (nhân bản không đổi hai thứ đó). Text ghép là phân bố lệch; không dùng làm benchmark.
- 0c. Adjudication 60 lỗi thiểu số-polarity trên validation (30 neu|pos, 15 neu|neg, 15 neg|pos), người đọc, không đổi gold. Báo tỷ lệ "gold bảo vệ được" vs "model sai rõ".

**Bước 1 — screen student, seed 42 (13 trial ≈ 3 giờ).** Arms: baseline replay ×5 task; C1 ×5; C-perm ×3 ABSA. Chốt trước:
- Thước đo cơ chế chính: trên ViSFD validation, sai polarity của slot thiểu-số trong câu trộn (n=393) và tỷ lệ →đa số; phụ: Hotel bỏ sót thiểu số (n=26, chỉ hướng), Restaurant sai polarity câu trộn (n=250 slot). Dự đoán C1: giảm ≥5 điểm tuyệt đối so baseline replay *và* so C-perm; tỷ lệ →đa số tiến về 50%.
- Ràng buộc: VSFC Sentiment/Topic không giảm quá 1,0/0,8 điểm (≈1 SD E1); ρ_h báo theo head.
- Phụ: vector F1 5 task, win/loss, worst regression — báo, không dùng làm kill vì noise.
- Kill: thước đo cơ chế không hơn cả baseline lẫn C-perm trên ViSFD.
- Tùy chọn nếu còn budget (≈1,5 giờ): C2 ×3 ABSA (đổi một trục softmax trong code E9) và C3 ×3 ABSA, để đọc *nơi* rò rỉ; không dùng chúng để chọn "best mix" theo task.
- Thước đo cơ chế tính trên validation, cũng là tập chọn checkpoint; selection bias nhỏ nhưng có, nên bước 2 (seed mới, cùng thước đo) là bắt buộc trước bất kỳ claim nào.

**Bước 2 — xác nhận (4 trial ≈ 1 giờ).** C1 seeds 43/44 trên ViSFD/Restaurant (E8 đã có 3 seed): hướng thước đo cơ chế 3/3 và F1 mean không dưới E8 mean.

**Bước 3 — port BamiBERT (15 trial ≈ 4 giờ).** Protocol E16 (micro 16×2, FP32 tại `L_mod` cho mọi arm), baseline replay + C1 + C-perm × 5 task, seed 42. Test chỉ sau khi khóa checkpoint theo validation; báo adaptive exposure của test. Nếu đạt, thêm holdout ngoài vòng phát triển (ví dụ probe kiểu ARTS xây từ *test* chỉ sau khi khóa, hoặc một bộ ABSA khác) trước khi nói "encoder tốt hơn".

**Tách capacity/compute/learning benefit:** C1 vs C-perm khớp params/compute chính xác; C1 vs baseline replay chỉ khác `bias_key`; không stage học thêm nên không có learning-benefit riêng.

## 9. Bản tóm tắt để hai cộng sự đối chiếu

- **Top idea:** chuẩn hóa bội số mềm theo key trong softmax attention ở top layers, +1 scalar/head, identity tại init.
- **Claim mới:** (a) phép đo failure mode "hấp thụ polarity đa số" trên 3 ABSA task, 3 seed, cả BamiBERT; (b) họ một tham số "đa tập ↔ tập hợp" cho hướng tổng hợp attention với bộ ước lượng bội số kernel trong ngữ cảnh; (c) chứng minh số rằng leverage/ridge không làm được ở N<d_k.
- **Prior art gần nhất:** **PointConv 2019 / GACNN 2020 (cùng phép tính KDE→nghịch đảo mật độ, ở tọa độ 3D)**, ToMe proportional attention (dạng), MAA/CPA/NAPC (trục, chiều ngược), Intention/mesa/TTR (anh em full-row), ReAttn 2026 (IDF post-hoc), Sinkformer (trục chuẩn hóa khác), ARTS (failure mode), NW design bias (thống kê).
- **Rủi ro:** không phải phép tính mới (point-cloud prior art) → reviewer xem là chuyển giao;  key pretrained có thể không dư thừa theo cách cần; nhãn neutral mơ hồ chiếm phần lỗi; VSFC có thể giảm nếu count là tín hiệu; gộp aspect anh em.
- **Cần đo trước khi train:** 0a (m̃ có cấu trúc?), 0b (flip theo bản sao có giảm khi bật C1 trên frozen?), 0c (nhãn).
- **Điểm bất đồng với agent trước:** (1) screen cũ phần lớn chưa kết luận, không phải âm; (2) cần failure mode đo được trước khi chọn cơ chế, không đi từ literature xuống; (3) Talking-Heads/head mixing chưa có failure mode local nào yêu cầu; (4) "thắng rộng" không nên là gate screening cho cơ chế nhắm ABSA.

## 10. Đối chiếu với vòng 3 của agent chính: recipient-excluded (cavity) keys và E17

Sau snapshot, workspace có thêm `RECIPIENT_EXCLUDED_KEYS_PROPOSAL.md`, `RECIPIENT_EXCLUDED_KEYS_REVIEW.md`, hai spec cavity, `RESEARCH_LOOP_STATUS.md` và `E17_PLAN.md` (ngày 12/09, sau 23:00). Tôi đã đọc chúng; phần này là đối chiếu độc lập, không phải đồng thuận.

**Tóm tắt candidate của agent chính.** Key của nguồn j khi receiver i đọc được tính từ preview `z_j` đã *trừ* message của chính i (`C_{j\i} = z_j − Σ_t A_ji^t u_i^t`), LayerNorm tính lại đúng theo từng cặp bằng thống kê tích vô hướng, blend `L = L_orig + η(L_cav − L_orig)`, η init 0. Đã có kiểm đại số FP64/FP32 (guarded) trên tensor tổng hợp; E17 dự kiến 15 trial student (baseline/full-context/cavity × 5 task, block index 6).

**Điểm tôi đánh giá cao.** Operator định nghĩa bằng một điều kiện chính xác trên đường truyền; tính chất partial-path và rank witness được chứng minh; factorization tránh lưu N²d; prior art (NBA-GNN, SAOBP, CoTNet) được đối chiếu thẳng thắn; control full-context được thiết kế đúng (cùng máy móc, hệ số deletion = 0).

**Điểm tôi không đồng ý hoặc thấy thiếu.**
1. **Không có failure mode đo được đứng sau nó.** Proposal tự thừa nhận "chưa đo được tỷ lệ lỗi do đường quay về này". Với power của protocol (mục 1a), một cơ chế không có dự đoán subgroup sẽ chỉ được đánh giá bằng vector F1 5 task một seed — đúng cách đã cho kết quả không kết luận được ở E9–E16. Đề nghị cụ thể: dùng ngay các thước đo ở mục 2.1 (sai polarity slot thiểu số, tỷ lệ →đa số, bỏ sót thiểu số Hotel) trên `validation_predictions.json` của E17 — không tốn thêm GPU — và probe nhân bản 0b trên checkpoint E17 nếu có. Nếu cavity keys không đổi các thước đo này, ít nhất biết nó *không* tác động failure mode đã đo.
2. **Kỳ vọng hiệu ứng nhỏ về cấu trúc.** Cavity chỉ bỏ contribution *tường minh, cùng layer* của i khỏi key của j; ở block 6/10, phần lớn ảnh hưởng của i lên `h_j` đã nằm trong residual từ các layer trước (proposal cũng ghi). Property đúng nhưng "lượng" bị loại có thể rất nhỏ so với norm của `z_j`; nên đo trước tỷ lệ `‖Σ_t A_ji u_i‖/‖z_j‖` trên checkpoint E8 thật (một lần inference) — nếu median cỡ 1e-2 thì dự đoán η không mở hoặc mở mà không đổi routing.
3. **Chi phí.** O(N²Hd + N²H²) và guard FP32 với fallback theo chunk; gate ≤3× step time là hợp lý nhưng đắt hơn C1 (≈ +1 lần QKᵀ) nhiều lần cho một hiệu ứng chưa có bằng chứng.
4. **Gate ≥4/5, ≥0,3 điểm, một seed** vẫn là gate không có power (mục 1a). Nếu giữ, hãy đọc kết quả E17 như "chưa kết luận" khi rơi trong ±1 SD, không như "âm".
5. **Novelty tier tương đương C1:** cả hai là *chuyên biệt hóa* của một trục đã biết (non-backtracking; đa tập↔tập hợp). Tôi không cho rằng cavity "mới hơn" C1 hay ngược lại; khác biệt quyết định là C1 gắn với failure mode đo được và rẻ hơn.

**Không mâu thuẫn về thực thi.** C1 sửa block 7–9, cavity sửa block 6; cả hai identity tại init; có thể screen tách biệt trên cùng recipe E8. Không gộp hai cơ chế trong một arm cho tới khi từng cơ chế có tín hiệu riêng. Nếu E17 đang chiếm GPU, bước 0 của tôi chỉ cần inference ngắn; bước 1 có thể xếp sau E17.

**Đề nghị chung cho cả hai cộng sự:** trước khi thêm bất kỳ trial 30-epoch nào, chạy chung một "mechanism battery" không train trên các checkpoint đã có (E8 s42, E14): (a) phân bố `m̃` và tỷ lệ message quay về ở các block đích, (b) probe nhân bản A+B(+B…), (c) adjudication 60 lỗi thiểu số. Ba phép đo này phân xử được cả hai giả thuyết với chi phí gần bằng không.

## 12. Bước 3 đã chạy (13/09/2026): C1 rớt; định vị nhân quả của failure mode

Mọi phép đo dưới đây là inference trên checkpoint đóng băng (E8 seed 42 student; E14 LR chọn BamiBERT), validation/train only, không train, không test. Script: `scripts/independent_review/*.py`; output: `artifacts/independent_review/battery/`. Parity: đường attention thủ công FP32 với ρ=0 cho F1 validation đúng bằng artifact (student 3/3 task; BamiBERT lệch ≤0,06 điểm do batching), predictions native khớp artifact 100% ở Restaurant/Hotel.

### 12.1. Kill 0a — không kích hoạt
Key ở các layer đích có cấu trúc bội số: median `m̃`(τ=8) 1,2–3,0, IQR(log m̃) 0,3–0,9; probe lặp nguyên văn cho cosine key giữa hai bản sao 0,4–0,9 và `m̃≈2–3,6`. Cơ chế *đo* được điều nó định đo.

### 12.2. Kill 0b — kích hoạt: C1 không tác động failure mode
Probe nhân bản (A + B×k, 200 cặp/task, A và B khác aspect, ngược polarity):

| Model/task | Baseline: A đúng % (k=0→1→2→3) | A bị *xóa* % | C1 ρ=1 (k=3) A đúng / xóa | Control hoán vị (k=3) |
|---|---|---|---|---|
| Student Restaurant | 66→39→34→29 | 18→35→40→46 | 28,5 / 44,5 | 28,5 / 46 |
| Student Hotel | 60→42→31→23 | 39→49→56→62 | 24 / 62 | 22,5 / 62,5 |
| BamiBERT Restaurant | 73→52→44→36 | 14→25→35→41 | 33 / 42 | 36 / 41 |
| BamiBERT Hotel | 66→51→45→36 | 30→41→52→57 | 27,5 / 62,5 | 36,5 / 56 |

Failure mode được **tái tạo nhân quả** ở cả hai model (thêm mệnh đề đối lập → A đổi polarity và, chủ yếu, bị xóa). Nhưng C1 (ρ ∈ {0,5; 1; 2}, τ ∈ {4; 8}) không khác baseline và không khác control hoán vị; trên validation thật, sai polarity slot thiểu số ViSFD 25,4% → 25,2–25,7% (student), 20,9% → 20,1–22,4% (BamiBERT); F1 đổi −0,1…−1,3. **Một bản B đã gây phần lớn thiệt hại** (66→39%) → vấn đề không phải bội số mà là *cạnh tranh giữa hai aspect khác nhau*. C1 bị loại theo tiêu chí khóa trước.

### 12.3. Định vị nhân quả (activation patching, A+3B)
Patch trạng thái CLS từ lần chạy "A một mình" vào lần chạy ghép, tại đầu ra layer l:

- Student: phục hồi dần L3→L9 (Restaurant 29→33→38→49→54→59→64→66%); patch **token của A** ở mọi layer: **không phục hồi** (26–30%). Token của A vẫn đúng; CLS không đọc được A khi có B.
- BamiBERT: CLS chỉ được xây ở L10–L11 (36→53→72%); patch token A ở L6–L9 phục hồi một phần (36→52%).

### 12.4. Head không giải mã được chồng chất
Đưa `avg(CLS_A, CLS_B)` hoặc `CLS_A+CLS_B` (hoặc LN của tổng) vào head đã train: A đúng 8–34%, B đúng 9–30% (so 60–88% khi đứng riêng), gần như toàn bộ là "absent". `cos(CLS_A, CLS_B)`=0,12–0,31; `CLS(A+B)` có cos 0,78–0,83 với trung bình hai mã và norm không đổi (post-LN). Hình chiếu CLS(A+B) lên hướng "A có mặt" (từ probe tuyến tính) còn 38–68% giá trị khi A đứng riêng. Co/giãn norm λ∈[0,5;2] không đổi quyết định (argmax bất biến tỷ lệ) → không phải ngưỡng biên độ, mà là **mã kiểu con trỏ**: một vector chỉ mã hóa được một "ý".

### 12.5. Probe tuyến tính: mất ở đâu?
Probe presence/polarity huấn luyện trên CLS *train* (encoder đóng băng), đọc trên câu ghép: A_present student Restaurant 84→71,5→66% (k=0/1/3), BamiBERT Hotel 78→68→61,5%; head còn tệ hơn (81,5→65→54,5; 70,5→59→43). ViSFD hầu như không mất (89,5→88→84,5) — ViSFD có 34% câu train ≥4 aspect, model *học được* cách chồng chất; Restaurant/Hotel (chủ yếu 1–2 aspect) thì không. → Thiếu hụt là **học được từ phân bố dữ liệu**, không phải giới hạn biểu diễn của kiến trúc.

### 12.6. Oracle partition — hướng "sửa aggregation trong encoder" bị đóng
Ép hàng CLS chuẩn hóa attention riêng trong phần A và phần B (oracle) rồi lấy trung bình/tổng, ở 3 layer cuối **và** ở mọi layer: A_present (probe) student Restaurant 71,5→73,5%, Hotel 60→52%, BamiBERT không đổi. Không có cách tái phân phối attention nào phục hồi A; chỉ *thay hẳn* CLS bằng trạng thái "A một mình" mới phục hồi (12.3). Kết luận: **mã CLS một-ý-một-vector là điểm nghẽn; sửa cách CLS đọc không đủ; cần readout có vector riêng cho từng aspect hoặc huấn luyện tính chồng chất.** Đây cũng giải thích E9 (readout theo aspect): +1,03 Restaurant (nơi nghẽn nặng), −0,34 ViSFD (đã chồng chất được), −0,62 Hotel (34 query ngẫu nhiên mới + trộn lại với CLS nghẽn).

### 12.7. Adjudication 0c (60 lỗi thiểu số ViSFD, một người đọc)
24/60 hấp thụ rõ vào polarity đa số với gold bảo vệ được; 6/60 làm mềm về neutral; 6/60 aspect *trộn* (khen+chê cùng aspect) bị chọn phía không phải đa số; 13/60 gold mơ hồ; 11/60 model có lý hơn gold. ≈55–60% là lỗi model thật; trần cải thiện ước tính ở mục 0 giảm còn ≈+0,6–0,7 điểm/task. Nhiều "neutral" trong ViSFD thực ra là "trộn" — cùng một cơ chế chồng chất ở cấp trong-aspect.

### 12.8. Kết luận vòng này
1. **C1 bị loại** (kill 0b). Lý do cơ chế: nhắm bội số trong khi lỗi là cạnh tranh giữa aspect khác nhau dưới một mã không chồng chất.
2. **Đóng góp còn lại là chẩn đoán nhân quả** (12.2–12.6): token đúng, CLS trộn, readout không giải mã được trộn, không tái phân phối attention nào sửa được, và thiếu hụt phụ thuộc phân bố train (ViSFD vs Restaurant/Hotel). Theo hiểu biết của tôi, ARTS (2020) chỉ mô tả hành vi; phân tích cơ chế kiểu này chưa thấy cho ABSA. Đây là *finding*, không phải kiến trúc.
3. **Hướng sửa được bằng chứng chỉ ra** đều thuộc họ đã có: (a) readout có vector riêng từng aspect với gán độc quyền token→aspect (C2/slot attention/E9 sửa lỗi thiết kế: không query ngẫu nhiên, không trộn lại CLS); (b) readout kiểu VSA/HRR với khóa cố định (Ganesan et al. 2021); (c) huấn luyện tính chồng chất bằng ghép câu (ARTS/mixup). Không cái nào là phép tính mới.
4. **Chưa tìm được cơ chế đủ mới** theo tiêu chí "phép tính chưa ai làm" trong vòng này. Ba vòng ý tưởng của tôi (bội số, gán độc quyền, coverage) và ba vòng của agent chính (pair state, operator composition, cavity keys) đều rơi vào họ đã có; điểm chung là literature attention 2017–2026 đã phủ gần như mọi phép tổng hợp một-bước. Nếu tiếp tục, vị trí có xác suất cao hơn là **cách encoder học mã chồng chất** (objective/curriculum), không phải operator — trái mong muốn ban đầu, nhưng là điều bằng chứng chỉ ra.

## 13. Vòng lặp tiếp (13/09): C4 — reader CLS theo phân vùng trên token stack chung

### 13.1. Ngã rẽ: thông tin còn hay mất? (`fork_test.py`)
- Probe tuyến tính huấn luyện trên **câu ghép tổng hợp từ train** đọc A trên câu ghép validation tốt hơn probe huấn luyện trên câu đơn (student Restaurant A_present 71,5→82%; Hotel 60→68,5%) → thông tin A *còn* trong CLS(A+B) ở mức đáng kể — nhưng probe đó **giảm** pair F1 trên validation thật (70,3→67,7) vì câu ghép tổng hợp lệch phân bố. Huấn luyện readout bằng ghép câu không phải lối tắt sạch.
- **Phản thực tế mask hàng CLS:** giữ nguyên mọi thứ, chỉ cho hàng CLS (ở *mọi* layer) nhìn token của phần A (+ `<s>`, `</s>`); token của B vẫn contextualize đầy đủ. Head đã train đọc A đúng **65,5% ≈ A một mình 66%** (Restaurant student), Hotel 62 vs 59,5; `cos(CLS_mask, CLS_A)=0,95`. BamiBERT phục hồi một phần (60 vs 72,5; cos 0,76–0,79; B rò 30–38%, khớp với việc token A ở layer giữa đã bị B nhiễm — mục 12.3). → Trên student, "một reader CLS cho mỗi tập token, chung một stack" tái tạo được đúng kết quả như khi tập token đó đứng một mình, **không cần train**.

### 13.2. C4 (định nghĩa)
K reader token (mỗi reader là một hàng CLS) trong cùng một forward: hàng attention của reader k bị giới hạn vào tập token S_k (+ special) ở mọi layer; hàng của content token không giới hạn; mọi weight dùng chung; decode theo slot: reader có xác suất "có mặt" cao nhất. Không tham số mới. S_k: mệnh đề theo dấu câu/liên từ (vòng này) hoặc phân vùng học được (bước sau).

### 13.3. Kiểm chứng không train trên validation thật (`clause_readers.py`)
Reader toàn câu + reader theo mệnh đề (trung bình 3 reader/câu ở Restaurant/Hotel, 5,6 ở ViSFD), hợp nhất bằng max-present:

| Model/task | Baseline F1 (P/R) | C4 không train F1 (P/R) | Bỏ sót ở câu nhiều aspect |
|---|---|---|---|
| Student Restaurant | 71,3 (72,7/70,0) | 67,6–69,7 (63–66/73–74) | 20,7 → 12–13 % |
| Student Hotel | 73,1 (77,1/69,4) | 68,6–70,1 (63–66/75) | 28,1 → 17–18 % |
| Student ViSFD | 82,4 (82,8/82,0) | 78,9–80,8 (74–77/84–85) | 9,5 → 4–5 % |
| BamiBERT Restaurant | 74,7 (74,0/75,5) | 73,5–74,6 (71–73/76–77) | 15,2 → 12,6–13 % |
| BamiBERT Hotel | 73,3 (73,5/73,0) | 72,2–72,5 (69–70/75–76) | 22,8 → 18–20 % |

Cơ chế làm đúng điều dự đoán (recall và bỏ sót), nhưng reader theo mệnh đề chưa được huấn luyện để nói "absent" ở cấp mệnh đề → precision giảm nhiều hơn → F1 tổng giảm. Ngưỡng chấp nhận không hiệu chỉnh được trên train (model fit train 95–100% F1). **Kết luận: C4 cần train reader (loss MIL/max-per-slot), không có lối tắt không train.**

### 13.4. Bước 2 cho C4
| Claim | Prior art gần nhất | Đã biết | Khác biệt |
|---|---|---|---|
| Nhiều token pooled trong một BERT, mỗi token bị mask luồng thông tin theo phân vùng | **Ruan et al. 2020, "Fact-level Extractive Summarization with Hierarchical Graph Mask on BERT"** (arXiv 2011.09739): `[cls_f]`, `[cls_s]` với graph mask giới hạn nhận thông tin theo fact/câu | Phép tính | Bài toán (ABSA readout), decode max-per-slot, tính chất chuyển giao không train từ head đã fine-tune |
| Nhiều class/CLS token toàn chiều sâu | MCTformer (CVPR 2022), Multi-CLS BERT (2023), BERTSUM (một `[CLS]` mỗi câu, attention toàn cục) | Cấu hình token | Reader theo *phân vùng*, không theo lớp; attention bị giới hạn |
| Phân vùng học được (bước sau) | Slot attention; SPARO (2024, slot compositional encodings) | Phép tính | — |
| MIL trên mệnh đề cho ABSA | MILNET (Angelidis & Lapata, TACL 2018) | Ý tưởng mệnh đề = instance | Encoder chung, reader trong stack, không encoder riêng cho segment |

**Verdict: C4 thuộc họ đã có ở cấp phép tính (Ruan et al. 2020).** Phần bảo vệ được: chẩn đoán nhân quả (mục 12) + tính chất "reader theo phân vùng trên stack chung tái tạo kết quả như câu đứng riêng mà không cần train" (13.1) + ứng dụng cho readout tập hợp trong ABSA. Không phải operator mới.

### 13.5. Trạng thái vòng lặp
Sau C1 (rớt kill 0b), C2/C3 (rớt bước 2), C4 (qua mechanism check, rớt tiêu chí "phép tính chưa ai làm"): **chưa có candidate đạt "phép tính chưa ai làm"**. Điều bằng chứng chỉ ra là điểm nghẽn thuộc *cấu trúc readout/mã* — một vùng literature đã bão hòa; vì vậy khả năng tìm operator mới từ failure mode này là thấp. Đề nghị người dùng chọn: (a) chấp nhận đóng góp kiểu "chẩn đoán cơ chế + fix có định hướng" (C4 train, cần execution agent Luna; ~2 giờ GPU; dự đoán khóa trước: Restaurant/Hotel giảm bỏ sót ở câu nhiều aspect ≥5 điểm, precision không giảm quá 1 điểm, ViSFD ≈), hoặc (b) đổi failure mode/nhiệm vụ để tìm operator mới ở nơi khác.

## 11. Phạm vi tìm kiếm và nguồn

Tìm kiếm web ngày 12/09/2026 với các cụm: "test-time regression associative memory", "Intention key-value-query least squares", "mesa layer", "leverage score attention", "Compactor KV leverage", "softmax duplicate repeated keys normalization", "Mass-Aware Attention", "cardinality preservation attention", "Normalized Attention Without Probability Cage", "Sinkformer doubly stochastic", "optimal transport Sinkhorn label attention multi-label", "Token Merging proportional attention", "DPP attention redundancy", "coverage across layers encoder", "Tasty Burgers Soggy Fries ARTS"; vòng hai (13/09): "point cloud density-aware attention", "PointConv inverse density scale", "IDF reweighting inside self-attention". Đã đọc trực tiếp: TTR (PDF, §4 Eq. 11–13, 24–25, 50–52; câu "applies to non-causal"), Intention (PDF, §2 định nghĩa `f_ls`, Thm 2.6, thí nghiệm few-shot/policy distillation), Compactor (PDF, §3.1–3.5: leverage chỉ để evict), MAA (HTML, công thức `A_p`, kết quả `r^{1−1/p}`, domain CTDG/MTPP/TKG/RAG/STPP). Các nguồn khác đọc ở mức abstract/summary; không khẳng định đã rà mọi biến thể.

1. Ke Alexander Wang, Jiaxin Shi, Emily B. Fox. *Test-time regression: a unifying framework for designing sequence models with associative memory*. arXiv:2501.12352, 2025.
2. Marta Garnelo, Wojciech M. Czarnecki. *Exploring the Space of Key-Value-Query Models with Intention*. arXiv:2305.10203, 2023.
3. Johannes von Oswald et al. *Uncovering mesa-optimization algorithms in Transformers*. arXiv:2309.05858, 2023.
4. Daniel Bolya et al. *Token Merging: Your ViT But Faster*. ICLR 2023 (arXiv:2210.09461) — proportional attention `softmax(QKᵀ/√d + log s)`.
5. *What Softmax Throws Away: Mass-Aware Attention for Evidence Accumulation*. arXiv:2607.22781, 07/2026.
6. Shuo Zhang, Lei Xie. *Improving Attention Mechanism in Graph Neural Networks via Cardinality Preservation*. IJCAI 2020.
7. Oliver Richter, Roger Wattenhofer. *Normalized Attention Without Probability Cage*. arXiv:2005.09561, 2020.
8. Keyulu Xu et al. *How Powerful are Graph Neural Networks?* ICLR 2019 (sum/mean/max).
9. Gabriele Corso et al. *Principal Neighbourhood Aggregation for Graph Nets*. NeurIPS 2020 (degree scalers).
10. Michael E. Sander et al. *Sinkformers: Transformers with Doubly Stochastic Attention*. AISTATS 2022.
11. Francesco Locatello et al. *Object-Centric Learning with Slot Attention*. NeurIPS 2020.
12. *OTSeg: Multi-prompt Sinkhorn Attention for Zero-Shot Semantic Segmentation*. ECCV 2024.
13. Paul-Edouard Sarlin et al. *SuperGlue*. CVPR 2020 (dustbin).
14. Xiaoyu Xing et al. *Tasty Burgers, Soggy Fries: Probing Aspect Robustness in Aspect-Based Sentiment Analysis*. EMNLP 2020 (ARTS).
15. *Compactor: Calibrated Query-Agnostic KV Cache Compression with Approximate Leverage Scores*. arXiv:2507.08143, 2025.
16. Amir Zandieh et al. *KDEformer: Accelerating Transformers via Kernel Density Estimation*. ICML 2023.
17. Zhaoyue Tu et al. *Modeling Coverage for Neural Machine Translation*. ACL 2016; Abigail See et al. *Get To The Point*. ACL 2017.
18. Ruining He et al. *RealFormer*. Findings ACL 2021 (đã có trong project).
19. Steven Frankland et al. *Determinantal point process attention over grid cell code supports out of distribution generalization*. eLife 2024 (arXiv:2305.18417).
20. Jianqing Fan, Irène Gijbels. *Local Polynomial Modelling and Its Applications*. 1996 (NW design bias; Priestley–Chao 1972; Gasser–Müller 1979).
21. Nội bộ: `E8_ERROR_READOUT.md`, `E12_SEMANTIC_REVIEW.md`, `E12_LABEL_MAPPING_AUDIT.md`, `E14_BAMIBERT_BASELINE_RESULT.md`, `artifacts/e16/E16_RESULT.md`, `E16_FFN_TEST_RESULT.md`, `IDEA_NOVELTY_AUDIT.md`, `ENCODER_REPLACEMENT_GAP_RESEARCH.md`, `E1_TEST_BENCHMARK.md`, metadata elapsed của E8/E14/E16.
22. Wenlong Chen, Yingzhen Li. *Calibrating Transformers via Sparse Gaussian Processes*. ICLR 2023 (arXiv:2303.02444).
23. Kawin Ethayarajh. *How Contextual are Contextualized Word Representations?* EMNLP 2019 (anisotropy của biểu diễn tầng trên).
24. Wenxuan Wu, Zhongang Qi, Li Fuxin. *PointConv: Deep Convolutional Networks on 3D Point Clouds*. CVPR 2019 — inverse density scale bằng KDE.
25. Congcong Wen et al. *Airborne LiDAR Point Cloud Classification with Graph Attention Convolution Neural Network*. 2020 (arXiv:2004.09057), §3.1.2 density attention, Eq. 3–4 — đã đọc PDF.
26. *ReAttn: Improving Attention-based Re-ranking via Attention Re-weighting*. arXiv:2602.19969, 2026 — IDF reweighting post-hoc; Jaekeol Choi. *Finding Inverse Document Frequency Information in BERT*. arXiv:2202.12191, 2022.
27. Yuxiang Ruan et al. *Fact-level Extractive Summarization with Hierarchical Graph Mask on BERT*. COLING 2020 (arXiv:2011.09739) — nhiều token pooled với graph mask trong một BERT.
28. Stefanos Angelidis, Mirella Lapata. *Multiple Instance Learning Networks for Fine-Grained Sentiment Analysis*. TACL 2018 (MILNET).
29. Lei Xu et al. *Multi-class Token Transformer for Weakly Supervised Semantic Segmentation*. CVPR 2022; Haw-Shiuan Chang et al. *Multi-CLS BERT*. ACL 2023; Yang Liu, Mirella Lapata. *Text Summarization with Pretrained Encoders* (BERTSUM). EMNLP 2019.
