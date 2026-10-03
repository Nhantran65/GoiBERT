#!/usr/bin/env python3
"""Render the frozen test and efficiency artifacts without touching test data."""
from __future__ import annotations
import json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];ART=ROOT/'artifacts/final_test';DOC=ROOT/'docs/FINAL_TEST_RESULT.md'
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import TASKS,METRICS,PAPER

def read(p):return json.loads(p.read_text())
def mr(vals):return f"{statistics.fmean(vals)} ({min(vals)}–{max(vals)})"
def main():
 checks=read(ART/'validation_checks.json');manifest=read(ART/'locked_manifest.json');results=read(ART/'test_results.json');eff=read(ART/'efficiency.json');access=read(ART/'test_access_log.json')
 assert checks['completed']==48 and all(x['passed'] for x in checks['checks'])
 assert manifest['checkpoint_count']==48 and results['completed']==48 and len([x for x in access['entries'] if x['status']=='completed'])==48
 rows=results['results'];lookup={(e['task'],e['model'],e['seed']):e for e in rows};lines=['# Kết quả final test: Student E7 + KD và BamiBERT gốc','',f"Khóa manifest: `{manifest['locked_at']}`. Validation: 48/48 checkpoint PASS; sai số chỉ số tối đa `{max(max(c['metric_abs_diffs'].values()) for c in checks['checks'])}`; dự đoán validation khớp hoàn toàn 48/48. Test: 48/48 bộ dự đoán hoàn chỉnh, mỗi checkpoint đúng một lượt inference; một lượt chuẩn bị của seed 43 lỗi trước inference và được ghi log. Chỉ số trong bảng là phần trăm (giá trị evaluator nhân 100).",'','## Test 8 dataset / 15 chỉ số','','| Dataset | Chỉ số | Bami test mean (range) | Student test mean (range) | Student − Bami | Bami validation mean | Student validation mean | Paper BamiBERT test | Tham số Bami / Student |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 deltas=[];del_no_nli=[];wins=0;bmeans=[];smeans=[];b_non=[];s_non=[]
 for task in TASKS:
  for idx,metric in enumerate(METRICS[task]):
   b=[100*lookup[(task,'bami',seed)]['test_metrics'][metric] for seed in (42,43,44)]
   s=[100*lookup[(task,'student',seed)]['test_metrics'][metric] for seed in (42,43,44)]
   vb=[100*lookup[(task,'bami',seed)]['val_metrics'][metric] for seed in (42,43,44)]
   vs=[100*lookup[(task,'student',seed)]['val_metrics'][metric] for seed in (42,43,44)]
   bm,sm=statistics.fmean(b),statistics.fmean(s);delta=sm-bm;wins+=delta>=0;deltas.append(delta);bmeans.append(bm);smeans.append(sm)
   if task!='vinli':del_no_nli.append(delta);b_non.append(bm);s_non.append(sm)
   pb=lookup[(task,'bami',42)]['parameters'];ps=lookup[(task,'student',42)]['parameters']
   lines.append(f'| {task} | {metric} | {mr(b)} | {mr(s)} | {delta} | {statistics.fmean(vb)} | {statistics.fmean(vs)} | {PAPER[task][idx]} | {pb} / {ps} |')
 lines += ['','## Tổng hợp','','| Phạm vi | Student ≥ Bami | Bami mean | Student mean | Student − Bami |','|---|---:|---:|---:|---:|',f'| 15 chỉ số | {wins}/15 | {statistics.fmean(bmeans)} | {statistics.fmean(smeans)} | {statistics.fmean(deltas)} |',f'| Bỏ ViNLI (13 chỉ số) | {sum(x>=0 for x in del_no_nli)}/13 | {statistics.fmean(b_non)} | {statistics.fmean(s_non)} | {statistics.fmean(del_no_nli)} |','','## Tốc độ và bộ nhớ','','Head phân loại câu lấy từ checkpoint `vsfc_sentiment` seed 42. Dung lượng checkpoint dưới đây là `state_dict` encoder + head lưu riêng vào `/tmp`, không bao gồm optimizer; file checkpoint gốc có thể lớn hơn. GPU bf16; CPU fp32 với 8 luồng. Median/p90 tính trên thời gian mỗi lần forward, có đồng bộ CUDA.','','| Model | Tham số | Checkpoint fp32 (byte) | Checkpoint bf16 (byte) | Checkpoint gốc (byte) |','|---|---:|---:|---:|---:|']
 for model in ('bami','student'):
  e=eff['models'][model];lines.append(f"| {model} | {e['parameters']} | {e['checkpoint_bytes']['fp32']} | {e['checkpoint_bytes']['bf16']} | {e['original_checkpoint_bytes']} |")
 lines += ['','| Thiết bị | Batch | Length | Bami median ms | Bami p90 ms | Student median ms | Student p90 ms | Tăng tốc median | Bami mẫu/s | Student mẫu/s | Bami peak VRAM byte | Student peak VRAM byte |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for batch in (1,32):
  for length in (64,128,256):
   key=f'b{batch}_l{length}';b=eff['gpu']['bami'][key];s=eff['gpu']['student'][key]
   lines.append(f"| GPU bf16 | {batch} | {length} | {b['median_ms']} | {b['p90_ms']} | {s['median_ms']} | {s['p90_ms']} | {eff['gpu_student_speedup'][key]}× | {b['throughput_examples_per_second']} | {s['throughput_examples_per_second']} | {b['peak_vram_bytes']} | {s['peak_vram_bytes']} |")
 for length in (64,128):
  key=f'b1_l{length}';b=eff['cpu']['bami'][key];s=eff['cpu']['student'][key]
  lines.append(f"| CPU fp32 | 1 | {length} | {b['median_ms']} | — | {s['median_ms']} | — | {eff['cpu_student_speedup'][key]}× | — | — | — | — |")
 lines += ['',f"GPU đo: `{eff['gpu_name']}`; snapshot VRAM/tải và process khác trước/sau phép đo: `artifacts/final_test/efficiency.json`. Tổng thời gian dùng một GPU ≤ `{eff['gpu_hours_upper_bound']['hours']}` GPU-giờ, tính bảo thủ từ lúc tạo thư mục final_test trước lượt kiểm đầu tiên đến khi ghi xong efficiency, gồm cả thời gian viết code không chạy GPU. Riêng cửa sổ test và efficiency là `{(results['elapsed_seconds']+eff['elapsed_seconds'])/3600}` GPU-giờ.",'','## Kiểm tra và giới hạn','','- Lượt kiểm validation ban đầu dừng ở Bami ViSFD seed 42 do script sử dụng micro batch 32 trong khi checkpoint lưu micro batch 16. Sau khi sửa script đọc batch size từ metadata, toàn bộ validation đã được kiểm lại trước khóa manifest. Không truy cập test trong lượt dừng đó.','- Student ViNLI seed 42 đã lưu đủ 2991 dự đoán trước khi process bị dừng do bước tính metric kéo dài; metric được tính từ file đã lưu, không chạy inference lần hai. Student ViNLI seed 43 lượt đầu lỗi import trước inference; chỉ checkpoint đó chạy lại theo log.','- SHA256 của hai file ViNLI/PhoNER khớp giá trị được chỉ định trước khi đọc test; SHA256 của 5 dataset cũ được đối chiếu sau test với manifest dữ liệu gốc. SHA256 ViSpam được ghi nhận vì manifest trước đó chưa có giá trị tham chiếu.','- Các số Paper BamiBERT chỉ là tham khảo; số local dùng checkpoint, tokenizer, split và evaluator của repo.','- Không train, không sửa checkpoint, không chọn lại cấu hình sau khi xem test.','','## File tạo','','- `scripts/final_test/run.py`, `scripts/final_test/resume.py`, `scripts/final_test/efficiency.py`, `scripts/final_test/report.py`','- `artifacts/final_test/validation_checks.json`, `locked_manifest.json`, `test_access_log.json`, `test_results.json`, `efficiency.json`','- `artifacts/final_test/predictions/`: 48 file; `artifacts/final_test/metrics/`: 48 file',f'- `docs/FINAL_TEST_RESULT.md`','']
 DOC.write_text('\n'.join(lines))
 print('REPORT_COMPLETE',DOC)
if __name__=='__main__':main()
