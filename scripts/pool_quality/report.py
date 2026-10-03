"""Render the measured Vietnamese report with unrounded values."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/pool_quality'
DOC=ROOT/'docs/POOL_QUALITY_RESULT.md'
S=json.loads((OUT/'summary.json').read_text())
AUDIT=[json.loads(line) for line in (OUT/'audit_180.jsonl').open(encoding='utf-8')]
LABELS=['tin tức/báo chí','bài giới thiệu/kiến thức','diễn đàn/UGC',
        'thương mại/quảng cáo/SEO','truyện/văn học','văn bản pháp luật/hành chính','rác/lỗi/máy dịch']
A={name:{label:sum(r['sample']==name and r['label']==label for r in AUDIT)
         for label in LABELS} for name in ('U','R','C')}
P=S['projection']

def val(x): return str(x) if x is not None else '—'
def row(*xs): return '| '+' | '.join(map(val,xs))+' |'
def pct(x): return x*100

def run():
    S['content_audit']=A
    (OUT/'summary.json').write_text(json.dumps(S,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    d=S['samples']; names=('U','R','C'); lines=[]
    add=lines.append
    add('# Chất lượng pool HPLT so với corpus v2 và v1')
    add('')
    add('Đo bằng CPU trên text trước bước làm sạch của v2. U là 20.000 chunk web được chọn đều từ 284.991 chunk web *đã xuất* của v2, nối về doc raw qua đoạn đầu đã chuẩn hóa; R là 20.000 doc chọn đều từ 609.624 doc manifest HPLT chưa vào lượt chọn seed 42; C là 20.000 khối 30 dòng CC100 liên tiếp sau khi bỏ dòng domain của v1. Seed U/R/C = 1234. Một doc raw có thể tạo nhiều chunk U; vì vậy U có trọng số theo số chunk đầu ra. Các tỷ lệ đoạn có mẫu số là toàn bộ đoạn trong mẫu trừ khi ghi khác.')
    add('')
    add(f"V1 có {S['v1_blocks']['domain_distinct_lines']} dòng domain phân biệt theo heuristic tần suất bội số 20, tương ứng {S['v1_blocks']['domain_physical_lines']} lượt dòng bị bỏ; còn {S['v1_blocks']['cc100_complete_30_line_blocks']} khối CC100 hoàn chỉnh. C được lấy bằng reservoir sampling seed 1234. `split_paragraphs` của v2 còn tách các ký tự xuống dòng Unicode bên trong vài dòng v1, nên số đoạn đo của C có thể lớn hơn 30 lần số khối.")
    add('')
    add('## Truy vết v2')
    add('')
    add(f"`select_general()` với seed 42 chọn {S['selection']['candidate_docs']} doc đầu vào, {S['selection']['candidate_mmbert_tokens']} token mmBERT từ {S['selection']['pool_rows']} doc manifest. Danh sách ID/pool ordinal nằm tại `artifacts/pool_quality/v2_selected_input_ids.jsonl`. Sau lọc, cắt đoạn và lựa chọn ngẫu nhiên lần hai, `doc_sources.jsonl` ghi {S['selection']['v2_web_chunks']} chunk web với tổng {S['selection']['v2_web_bami_tokens']} token Bami, khớp 232889660. Đây là đối chiếu metadata đầu ra, không phải chạy lại toàn bộ bộ lọc v2. Lượt nối raw của mẫu U tìm {S['extraction']['u_found']}/20000 chunk, ứng với {S['selection']['u_unique_raw_ids']} ID raw khác nhau; số khóa đoạn đầu có nhiều doc ứng viên khớp là {S['extraction']['u_collisions']}.")
    add('')
    add('## Chỉ số trước làm sạch')
    add('')
    add(row('Chỉ số','U','R','C')); add('|---|---:|---:|---:|')
    labels=[('docs','Số đơn vị mẫu'),('tokens_bami','Tổng token Bami'),('tokens_per_doc_mean','Token/doc trung bình'),('tokens_per_doc_median','Token/doc trung vị'),('tokens_per_doc_p90','Token/doc p90'),('paragraphs_per_doc_mean','Đoạn/doc trung bình'),('paragraphs_per_doc_median','Đoạn/doc trung vị'),('paragraph_tokens_mean','Token/đoạn trung bình'),('paragraph_tokens_median','Token/đoạn trung vị'),('paragraph_tokens_p90','Token/đoạn p90'),('vietnameseish_token_rate','Token thuộc đoạn vietnameseish (%)'),('exact_duplicate_paragraph_rate','Đoạn trùng exact, lượt lặp (%)'),('boilerplate_paragraph_rate','Đoạn có tần suất ≥5 (%)'),('short_paragraph_rate','Đoạn <5 âm tiết (%)'),('contact_paragraph_rate','Đoạn có URL/email/điện thoại (%)'),('decontam_paragraph_rate','Đoạn trúng decontam (%)'),('decontam_doc_drop_rate','Doc vượt ngưỡng decontam 0,2 (%)')]
    rates={x for x,_ in labels if x.endswith('_rate')}
    for key,label in labels:
        add(row(label,*[(pct(d[n][key]) if key in rates else d[n][key]) for n in names]))
    add('')
    add('Đoạn được tách bằng `split_paragraphs`; độ dài tính bằng Bami. Trùng exact dùng NFC + casefold + gộp khoảng trắng, đếm các lần xuất hiện vượt lần đầu trong mẫu. Boilerplate đếm mọi lần xuất hiện của đoạn có tần suất ≥5 trong mẫu. `vietnameseish` là hàm nguyên bản của v1, tỷ lệ được cân theo token Bami của đoạn. Decontam dùng nguyên `Decontam.hit` và ngưỡng doc >0,2 của `build.py`; mã v2 hiện có 7 khóa task với validation/test, dù mô tả cũ gọi là 8 task. Chỉ text val/test được dùng làm khóa; không đọc nhãn test.')
    add('')
    add('## Metadata HPLT')
    add('')
    add(row('Chỉ số','U','R'));add('|---|---:|---:|')
    for key,label in [('mmbert_tokens','Token mmBERT trong mẫu'),('bami_per_mmbert','Bami/mmBERT'),('host_top20_token_share','Token của top-20 host (%)')]:
        add(row(label,*[(pct(d[n][key]) if key=='host_top20_token_share' else d[n][key]) for n in ('U','R')]))
    for key in ('s1_mean','s2_prob','mt','op','re','pii_n'):
        for stat in (('n','mean','median','p90','positive_n') if key=='pii_n' else ('n','mean','median','p90')):
            add(row(f'{key} {stat}',*[d[n]['metadata'].get(key,{}).get(stat) for n in ('U','R')]))
    add(row('Bin, số doc',*[json.dumps(d[n]['bin_counts'],ensure_ascii=False) for n in ('U','R')]))
    for b in ('7','8','9'):
        add(row(f'Bin {b} (%)',*[pct(d[n]['bin_counts'][b]/d[n]['docs']) for n in ('U','R')]))
    add('')
    add('`s1_mean` và `s2_prob` lấy từ `data/mmbert_vi_cpt_pilot/v2/spun_scores/`; đây là tên trường tương ứng với điểm S1 doc và S2 doc của pipeline. `op`, `re`, `mt`, `pii_n` lấy từ `pool.jsonl`.')
    add('')
    add('## Đọc mù loại nội dung')
    add('')
    add('Seed 7 chọn 60 mục mỗi mẫu, lấy 600 ký tự đầu, trộn 180 trích đoạn và ẩn nguồn khi gán nhãn. Sau khi gán nhãn mới ghép lại U/R/C. Mỗi ô là số mục trên 60 và tỷ lệ phần trăm.')
    add('')
    add(row('Loại','U','R','C'));add('|---|---:|---:|---:|')
    for label in A['U']:
        add(row(label,*[f"{A[n][label]}/60 ({pct(A[n][label]/60)}%)" for n in names]))
    add('')
    add('## Ước lượng dung lượng và thành phần')
    add('')
    add(f"Pool manifest có {P['pool_mmbert_tokens']} token mmBERT. Hệ số R = {P['r_bami_per_mmbert']} Bami/mmBERT cho {P['web_raw_bami_est']} token Bami thô khi ngoại suy toàn pool. Trong mẫu R, decontam (gồm doc và đoạn) loại {P['r_decontam_drop_fraction']*100}% token, tương ứng {P['pool_decontam_drop_bami_est']} token ngoại suy; exact dedup sau decontam loại {P['r_exact_drop_fraction']*100}% token, tương ứng {P['pool_exact_drop_bami_est']} token, trong đó các đoạn boilerplate ≥5 lần đóng góp {P['r_boilerplate_exact_drop_fraction']*100}% token, tương ứng {P['pool_boilerplate_exact_drop_bami_est']} token. Tỷ lệ token giữ = {P['r_retention_after_doc_decontam_paragraph_decontam_exact']}; suy ra {P['web_cleaned_bami_est']} token web. Boilerplate là tập con của exact dedup, không trừ lần thứ hai. Dedup trên toàn pool và giữa các nguồn có thể loại nhiều hơn trong mẫu R; ước lượng web này có xu hướng lạc quan. Cắt đoạn ở 2048 token chưa được ngoại suy.")
    add('')
    add(f"Wiki còn {P['wiki_unused_docs']} doc, {P['wiki_mmbert_unused']} token mmBERT chưa chọn; dùng tỷ lệ Bami/mmBERT của wiki v2 ước toàn wiki {P['wiki_bami_total_est']} token Bami. UGC refined có {P['ugc_mmbert_total']} token mmBERT và v2 đã đưa cả {78886} doc vào lượt lọc, nên không còn doc refined chưa dùng; phần sạch hiện có {P['ugc_bami_cap']} token Bami. Domain base sạch = {P['domain_base_bami']} token, lặp tối đa 20 = {P['domain_max20_bami']} token.")
    add('')
    add(row('Nguồn','v1 token Bami','v1 %','v2 token Bami','v2 %','v2-large token Bami ước','v2-large % ước'));add('|---|---:|---:|---:|---:|---:|---:|')
    for source in ('web','wiki','ugc','domain'):
        v1key='web_cc100' if source=='web' else source
        x=P['v1_tokens_by_source'][v1key]; y=P['v2_tokens_by_source'][source]; z=P['v2_large_tokens_by_source'][source]
        add(row(source,x,pct(x/P['v1_total_bami']),y,pct(y/P['v2_total_bami']),z,pct(P['v2_large_share_by_source'][source])))
    add(row('Tổng',P['v1_total_bami'],100,P['v2_total_bami'],100,P['v2_large_estimated_total'],100))
    add('')
    add(f"Web của v1 là CC100; web của v2 và v2-large là HPLT. Để đủ 1.000.000.000 token với wiki/UGC/domain như trên cần web {P['web_needed_for_1b']} token. Ngoại suy R cho dung lượng web sạch dư {P['web_capacity_headroom']} token so với nhu cầu; nếu tổn thất bổ sung trên toàn pool vượt {P['additional_raw_pool_loss_fraction_to_shortfall']*100}% token Bami thô thì kịch bản 1B bắt đầu thiếu web (giữ nguyên các giả định khác). Tỷ lệ UGC vẫn dưới 10%; domain không vượt 20 lượt. Wiki không thể giữ mức 15% như v2 vì toàn bộ wiki đủ điều kiện ước chỉ có {P['wiki_bami_total_est']} token Bami.")
    add('')
    add('## Kết luận và giới hạn')
    add('')
    add(f"R gần U về tỷ lệ token vietnameseish ({d['R']['vietnameseish_token_rate']} so với {d['U']['vietnameseish_token_rate']}), độ dài đoạn và điểm spun/MT. R có tỷ lệ đoạn trùng exact thấp hơn ({d['R']['exact_duplicate_paragraph_rate']} so với {d['U']['exact_duplicate_paragraph_rate']}), nhưng trong 60 trích đoạn đọc mù có nhiều mục thương mại/SEO hơn (13 so với 6); đây là tín hiệu nhỏ, chưa đủ để kết luận R kém chất lượng rõ rệt. Hai mẫu chịu thiên lệch khác nhau: U là mẫu chunk đã qua lựa chọn sau làm sạch, còn R là mẫu doc đầu vào chưa dùng. v2-large dự kiến tăng web từ {P['v2_tokens_by_source']['web']/P['v2_total_bami']} lên {P['v2_large_share_by_source']['web']}; tỷ trọng tương đối wiki, UGC và domain giảm. Đạt đúng 1B còn phụ thuộc dedup toàn pool.")
    add('')
    add('Giới hạn: không chạy lại toàn bộ v2 để truy vết chính xác mọi chunk về raw ID; đối chiếu 232889660 dựa trên `doc_sources.jsonl`. Khi nhiều doc raw cùng chứa đoạn đầu của một chunk, lượt nối chọn doc xuất hiện trước trong thứ tự quét như `build.py`; một doc trước đó bị loại ở ngưỡng decontam có thể khiến phép nối sai. Trích đoạn đọc mù chỉ 600 ký tự đầu, mỗi nhóm 60 mục. C là khối 30 dòng CC100, không phải doc thật; nhãn rác/lỗi của C chủ yếu phản ánh phép ghép dòng khác chủ đề. Heuristic dòng domain bội số 20 có thể bỏ sót hoặc gộp nhầm. Tỷ lệ lặp/boilerplate trong 20.000 mục thấp hơn khi đo trên toàn pool; các khoản loại trừ có giao nhau. Wiki mở rộng và tương tác dedup liên nguồn được ước bằng hệ số cũ, chưa dựng corpus mới. Các chỉ số metadata chỉ là phân bố mẫu.')
    add('')
    add('## Lệnh tái lập')
    add('')
    add('```bash')
    add("export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false")
    add('.venv-e0-cuda/bin/python scripts/pool_quality/measure.py prepare')
    add('.venv-e0-cuda/bin/python scripts/pool_quality/measure.py extract')
    add('.venv-e0-cuda/bin/python scripts/pool_quality/measure.py v1')
    add('.venv-e0-cuda/bin/python scripts/pool_quality/summarize.py --samples C')
    add('.venv-e0-cuda/bin/python scripts/pool_quality/summarize.py --samples U R')
    add('.venv-e0-cuda/bin/python scripts/pool_quality/merge.py')
    add('.venv-e0-cuda/bin/python scripts/pool_quality/project.py')
    add('.venv-e0-cuda/bin/python scripts/pool_quality/audit.py prepare')
    add('# Đọc mù audit_reading.txt rồi ghi blind_id và mã nhãn 0..6 vào /tmp/goibert_pool_quality_20260928/audit_labels.txt')
    add('.venv-e0-cuda/bin/python scripts/pool_quality/audit.py finish')
    add('.venv-e0-cuda/bin/python scripts/pool_quality/report.py')
    add('```')
    add('')
    DOC.write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':run()
