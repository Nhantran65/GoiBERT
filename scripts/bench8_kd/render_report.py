#!/usr/bin/env python3
"""Render the Vietnamese result report from audited artifacts, without rounding."""
import json
from pathlib import Path
import common as C

def fmt_range(xs):return f"{xs[0]}–{xs[1]}"

def main():
    summary=C.read(C.ART/'summary.json');teacher=summary['teacher'];delta=summary['kd_minus_ce'];table=summary['table']
    alignment=C.read(C.ART/'phoner/alignment_check.json');smoke=C.read(C.ART/'phoner/smoke_checks.json');launcher=C.read(C.ART/'launcher_state.json')
    assert launcher['status']=='completed' and not launcher['failures']
    for name in ('student','cafe'):
        for split in ('train','dev'):
            x=alignment['tokenizers'][name][split];assert x['words_without_subword']==x['truncated_words']==x['truncated_sentences']==0
    assert len(alignment['cross_tokenizer_20_sentences'])==20 and all(x['pass'] for x in alignment['cross_tokenizer_20_sentences'])
    assert all(smoke['checks'].values())
    lines=['# BENCH8 E7 + Cafe OOF KD — kết quả validation','','## Phạm vi và recipe','',
       'Chỉ dùng train và validation; không đọc hoặc nạp test. Student dùng E7 LLRD γ = 0,8, KD từ logits Cafe OOF 5 fold với α = 0,5 và T = 2. LR student: ViSpam 5e-5, ViNLI 1e-4, PhoNER 1e-4; seed 42/43/44. Teacher Cafe dùng LR 2e-5, batch 32, 30 epoch, warmup 10%, clip 1, bf16; chọn best epoch theo validation. ViNLI là mirror chưa xác minh; các dataset chỉ dùng cho nghiên cứu. α và T không được tinh chỉnh. `vispam_binary` là chỉ số phụ, không huấn luyện KD.',
       '', 'Điểm plan chưa nói rõ: seed cho từng teacher fold được đặt 42, theo `cafe_oof_kd` hiện có; mỗi fold giữ nguyên thứ tự train từ `bench8/data.py`. AdamW của teacher dùng `goibert.train.make_optimizer`, như teacher Cafe hiện có. Student giữ batch 32, max length 256/512, 30 epoch, warmup 10%, clip 1, bf16 và chọn best validation F1 theo `bench8/train.py`.',
       '', '## PhoNER: căn nhãn và sáu kiểm tra','',
       '21 nhãn BIO lấy đúng thứ tự từ `bench8/data.py` và được assert trong code mới. Cả train (5.027 câu, 167.541 âm tiết) và dev (2.000 câu, 71.325 âm tiết) có **0 âm tiết không sinh subword, 0 âm tiết bị cắt và 0 câu bị cắt** với cả tokenizer student 20.481 token và tokenizer Cafe 250.002 token. Kiểm chéo 20 câu ngẫu nhiên: 20/20 khớp số âm tiết giữa hai tokenizer và dữ liệu gốc. Chi tiết ở `artifacts/bench8_kd/phoner/alignment_check.json`.',
       '', '1. Gom vị trí trên batch hai câu, có padding, subword phụ và âm tiết bị cắt: PASS.',
       '2. Tra nhãn mềm theo (ID câu, chỉ số âm tiết) sau xáo thứ tự batch: PASS.',
       '3. KD loss với teacher one-hot và logits đồng nhất, gradient hữu hạn: PASS.',
       '4. Smoke teacher một epoch với 64 câu train, OOF fold 0 có ID, offset, shape `[n_words,21]`, `has_logit` và giá trị hữu hạn: PASS.',
       '5. Smoke student một epoch với nhãn mềm one-hot gold, loss hữu hạn và postrun checks: PASS.',
       '6. Kiểm chéo số âm tiết của 20 câu ngẫu nhiên: 20/20 PASS.',
       '', '## Chất lượng teacher','',
       '| Task | F1 validation từng fold | Mean | Range | F1 OOF trên train | Bất đồng | Mean max probability | Tỷ lệ max probability ≥0,99 |',
       '|---|---|---:|---:|---:|---:|---:|---:|']
    for task in C.TASKS:
        x=teacher[task];oof=x['oof_entity_f1'] if task=='phoner' else x['oof_macro_f1'];dis=x['word_disagreement_rate'] if task=='phoner' else x['disagreement_rate']
        lines.append(f"| {task} | {', '.join(map(str,x['fold_validation_f1']))} | {x['fold_validation_f1_mean']} | {fmt_range(x['fold_validation_f1_range'])} | {oof} | {dis} | {x['mean_max_prob']} | {x['share_max_prob_ge_0.99']} |")
    lines+=['','Mỗi task có độ phủ OOF 100%, ID train không trùng. Best epoch từng fold:','']
    for task in C.TASKS:lines.append(f"- {task}: {teacher[task]['fold_best_epochs']}")
    lines+=['','F1 OOF ViSpam theo lớp:','']
    for cls,f1 in teacher['vispam']['per_class_f1'].items():lines.append(f'- Lớp {cls}: {f1}')
    n=teacher['phoner'];lines+=['',f"PhoNER: accuracy theo âm tiết {n['word_accuracy']}; bất đồng trên âm tiết gold là thực thể {n['entity_gold_word_disagreement_rate']}; âm tiết có logits {n['valid_words']}; thiếu logits {n['missing_words']}. F1 theo loại thực thể:",'','| Loại | Gold | Dự đoán | TP | F1 |','|---|---:|---:|---:|---:|']
    for typ,x in n['f1_by_entity_type'].items():lines.append(f"| {typ} | {x['gold']} | {x['predicted']} | {x['tp']} | {x['f1']} |")
    for task in ('vispam','vinli'):
        tr=[x['train']['truncation_rate'] for x in teacher[task]['fold_length_stats']]
        val=[x['validation']['truncation_rate'] for x in teacher[task]['fold_length_stats']]
        lines.append(f"\n{task}: tỷ lệ cắt teacher train từng fold {tr}; validation từng fold {val}.")
    lines+=['','## KD − CE, 3 task mới','','Các số dưới đây là phần trăm của chỉ số validation, tính lại từ file dự đoán từng seed.','',
            '| Task | Chỉ số | CE mean (range) | KD mean (range) | KD − CE mean (range) | Seed dương |','|---|---|---:|---:|---:|---:|']
    def f(x):return f"{x['mean']} ({fmt_range(x['range'])})"
    for task,x in delta.items():lines.append(f"| {task} | {x['indicator']} | {f(x['ce'])} | {f(x['kd'])} | {f(x['delta'])} | {x['positive_seeds']}/3 |")
    lines+=['','KD − CE theo seed:','']
    for task,x in delta.items():lines.append(f"- {task}: {x['per_seed_delta']}")
    lines+=['','## 8 dataset / 15 chỉ số','','Trung bình 3 seed trên validation. Tất cả student ở đây dùng E7 + Cafe OOF KD; năm task cũ lấy nhánh KD, gồm cả Sentiment và Topic.','',
            '| Dataset | Chỉ số | Bami mean (range) | Student E7 + KD mean (range) | Student − Bami | Tham số Bami / Student |','|---|---|---:|---:|---:|---:|']
    for x in table:lines.append(f"| {x['dataset']} | {x['indicator']} | {f(x['bami'])} | {f(x['student_e7_kd'])} | {x['student_minus_bami']} | {x['bami_parameters']} / {x['student_parameters']} |")
    lines+=['','## GPU-giờ, kiểm tra và sự cố','',f"Tổng thời gian GPU: {summary['gpu_seconds']} giây = {summary['gpu_hours']} GPU-giờ, gồm 15 teacher fold, 9 student KD và hai smoke PhoNER. Trần: 18 GPU-giờ.",
            '15/15 teacher fold và 9/9 student KD có trạng thái `completed`, postrun checks PASS. Các chỉ số trong bảng được tính lại từ file dự đoán và đối chiếu metric đã lưu. Mỗi GPU 0–3 chỉ có một job của đợt chạy này tại một thời điểm; ngưỡng bộ nhớ trống khi launcher cấp job thấp nhất ghi trong log là 21.422 MiB. Không có check FAIL hoặc sự cố cần chạy lại.',
            '', '## File tạo','',
            'Danh sách đầy đủ ở `artifacts/bench8_kd/files_created.txt`. Các đầu ra chính:',
            '- `scripts/bench8_kd/`: code trainer, launcher, kiểm PhoNER, ghép OOF, tổng hợp, unit test và render báo cáo.',
            '- `artifacts/bench8_kd/teacher/{task}/fold_{0..4}/`: checkpoint, logits OOF, prediction validation, config, metric, curve, metadata và status.',
            '- `artifacts/bench8_kd/soft/{task}.npz` và `artifacts/bench8_kd/{task}/teacher_stats.json`: nhãn mềm và chất lượng teacher.',
            '- `artifacts/bench8_kd/student/{task}/seed_{42,43,44}/`: checkpoint, prediction validation, metric, curve, config, metadata và status.',
            '- `artifacts/bench8_kd/phoner/alignment_check.json` và `smoke_checks.json`.',
            '- `artifacts/bench8_kd/summary.json` và `table_8datasets_kd.md`.',
            '- `docs/BENCH8_KD_RESULT.md`.']
    (C.ROOT/'docs/BENCH8_KD_RESULT.md').write_text('\n'.join(lines)+'\n')
    paths=[str(p.relative_to(C.ROOT)) for p in sorted((C.ROOT/'scripts/bench8_kd').rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    paths+= [str(p.relative_to(C.ROOT)) for p in sorted(C.ART.rglob('*')) if p.is_file() and p.name!='files_created.txt']
    paths+=['docs/BENCH8_KD_RESULT.md']
    (C.ART/'files_created.txt').write_text('\n'.join(paths)+'\n')
    print('report and file manifest written',len(paths),flush=True)

if __name__=='__main__':main()
