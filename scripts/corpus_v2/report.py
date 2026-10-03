"""Render the Vietnamese, source-backed audit after the final checker has run."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/corpus_v2'

def main():
    m=json.loads((OUT/'manifest.json').read_text())
    d=json.loads((OUT/'decontam_report.json').read_text())
    c=json.loads((OUT/'final_check.json').read_text())
    samples=json.loads((OUT/'audit_20_docs.json').read_text())
    v1=m['v1']['tokens_bami'];old=m['v1']['domain_tokens_bami']
    rows=['| Nguồn | v1 token Bami | v2 token Bami | v2 doc | Tỷ lệ v2 |','|---|---:|---:|---:|---:|']
    for source in ('web','wiki','ugc','domain'):
        item=m['sources'][source]
        base=old if source=='domain' else v1-old if source=='web' else 0
        rows.append(f"| {source} | {base} | {item['tokens_bami']} | {item['docs']} | {item['tokens_bami']/c['tokens_bami_reread']:.9f} |")
    rows.append(f"| Tổng | {v1} | {c['tokens_bami_reread']} | {c['documents_reread']} | 1 |")
    filters=['| Nguồn | Chỉ số | Giá trị |','|---|---|---:|']
    for source,counts in sorted(m['filter_counts'].items()):
        for k,v in sorted(counts.items()):filters.append(f'| {source} | {k} | {v} |')
    decont=['| Nguồn | Task/split | Đoạn loại | Doc loại |','|---|---|---:|---:|']
    for source in ('web','wiki','ugc','domain'):
        for key in sorted(c['exact_match_table']):
            row=d['by_source_task_split'].get(source,{}).get(key,{})
            decont.append(f"| {source} | {key} | {row.get('paragraphs',0)} | {row.get('docs',0)} |")
    final=['| Kiểm | Trạng thái |','|---|---|']+[f"| {k} | {'PASS' if v else 'FAIL'} |" for k,v in c['checks'].items()]
    exact=['| Task/split | Text unique | Khớp |','|---|---:|---:|']
    for k,v in c['exact_match_table'].items():exact.append(f"| {k} | {v['unique_texts']} | {v['matches']} |")
    samp=['| # | Nguồn | Trích đoạn (≤400 ký tự) |','|---:|---|---|']
    for i,r in enumerate(samples,1):
        text=r['excerpt'].replace('|','\\|').replace('\n',' ')
        samp.append(f"| {i} | {r['source']} | {text} |")
    body=f'''# Kết quả dựng corpus v2 cho general distillation

Plan khóa: `docs/GD_RECIPE_ABLATION_PLAN.md` §2–3. Dựng bằng CPU, seed 42. Tổng token được đếm lại bằng tokenizer Bami trong `models/goiturn-s-gd`.

## Thành phần

{chr(10).join(rows)}

Domain v1 được đo từ file v1 bằng `sort | uniq -c`, chọn dòng vật lý có tần suất ≥20 và chia hết cho 20: {m['v1']['domain_identified_physical_lines']} lượt dòng, {old} token Bami, tỷ lệ {m['v1']['domain_ratio']}. Bộ CSV lịch sử của script v1 không còn tại đường dẫn cũ; phép đo này có thể tính cả dòng CC100 tình cờ lặp bội số 20. Domain v2 lặp đầy đủ {json.loads((OUT/'stats.json').read_text())['domain_repeat']} lần và thêm {json.loads((OUT/'stats.json').read_text())['domain_partial_repeat_docs']} doc ở lượt thứ {json.loads((OUT/'stats.json').read_text())['domain_max_document_repeat']} để đạt tỷ lệ; không doc nào vượt 20 lần.

General web được lấy ngẫu nhiên từ đúng manifest v2 `general_docs.jsonl` gồm {m['parameters']['web_manifest_pool_docs']} doc, đã qua bin 7–9, lọc spun/MT, trừ register, host cap 2% và near-dedup. Nguồn thô HPLT đối chiếu SHA từ `sha256.txt`; Wikipedia lấy text gốc từ parquet. UGC là doc thuộc `ugc_refined` nhưng đoạn được khôi phục bằng bộ lọc v2 trên raw HPLT.

## Các bước lọc

{chr(10).join(filters)}

Đoạn exact dedup dùng NFC + casefold + gộp khoảng trắng trên toàn bộ nguồn; các bản lặp domain được thêm sau dedup. Web boilerplate tần suất ≥5 vì vậy không thể còn trong phần web. Doc được cắt tại ranh giới đoạn, ngưỡng 2.048 token Bami.

## Khử nhiễm

Text validation/test chỉ dùng làm khóa; nhãn test không được đọc. Tám task có bảy nhóm text vì VSFC Sentiment và Topic dùng chung câu, được nạp một lần dưới tên VSFC. Text từ 13 âm tiết trở lên tạo tất cả 13-gram; text ngắn hơn đòi trùng hoàn toàn sau chuẩn hóa. Doc mất >20% đoạn bị loại toàn bộ. Bảng dưới ghi các lần dính theo task/split; một đoạn hoặc doc có thể thuộc nhiều hàng.

{chr(10).join(decont)}

## Kiểm cuối trên file đã ghi

{chr(10).join(final)}

SHA256 `mlm_mix.txt`: `{c['sha256']}`. Trùng đoạn web: {c['web_duplicate_paragraphs']}/{c['web_paragraphs']} = {c['web_duplicate_ratio']}. `READY`: {'đã ghi' if (OUT/'READY').exists() else 'chưa ghi'}.

{chr(10).join(exact)}

100 chuỗi packed đầu dùng đúng `iter_packed_sequences`; ví dụ decode nằm trong `data/corpus_v2/final_check.json`.

## Audit 20 doc ngẫu nhiên

{chr(10).join(samp)}

## Thời gian và giới hạn

Thời gian dựng: {m['elapsed_seconds']} giây (không gồm lượt đo v1, sort và kiểm cuối). Giới hạn: {'; '.join(m['known_limitations'])}. Nguồn và SHA đầy đủ nằm trong `data/corpus_v2/manifest.json`.
'''
    (ROOT/'docs/CORPUS_V2_RESULT.md').write_text(body,encoding='utf-8')

if __name__=='__main__':main()
