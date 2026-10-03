#!/usr/bin/env python3
"""Render source-backed Vietnamese family SM report from frozen artifacts."""
import json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];ART=ROOT/'artifacts/family_sm';DOC=ROOT/'docs/FAMILY_SM_RESULT.md'
sys.path.insert(0,str(Path(__file__).resolve().parent))
from summarize import TASKS,FIRST,METRICS,read
def mr(vals):return f'{statistics.mean(vals)} ({min(vals)}–{max(vals)})'
def main():
 pre=read(ART/'preflight.json');d=read(ART/'decisions.json');reg=read(ART/'launcher_jobs.json');v=read(ART/'validation_summary.json')
 test=read(ART/'test/test_results.json');manifest=read(ART/'test/locked_manifest.json');access=read(ART/'test/test_access_log.json');checks=read(ART/'test/validation_checks.json');eff=read(ART/'efficiency.json')
 old=read(ROOT/'artifacts/final_test/test_results.json')
 assert test['completed']==48 and manifest['checkpoint_count']==48 and checks['completed']==48 and all(c['passed'] for c in checks['checks'])
 completed=[a for a in access['entries'] if a['status']=='completed']
 assert len(completed)==48 and len({(a['model'],a['task'],a['seed']) for a in completed})==48
 lookup={(e['task'],e['model'],e['seed']):e for e in test['results']};oldlookup={(e['task'],e['model'],e['seed']):e for e in old['results']}
 recovered=[a for a in access['entries'] if 'technical_interruption' in a]
 lines=['# Kết quả họ S′/M — GD sạch, chọn trên validation, test một lần','',f"Plan khóa: `docs/FAMILY_SM_PLAN.md`. Manifest khóa lúc `{manifest['locked_at']}` trước truy cập test đầu tiên `{min(a['accessed_at'] for a in access['entries'])}`. 48/48 kiểm validation PASS, sai số chỉ số tối đa `{max(max(c['metric_abs_diffs'].values()) for c in checks['checks'])}`; 48/48 bộ dự đoán test hoàn tất. Log có `{len(access['entries'])}` mục cho 48 checkpoint; `{len(recovered)}` mục được tính metric từ predictions đã lưu sau gián đoạn kỹ thuật, không chạy lại inference.",'','## Kiểm tra trước loạt chính','','| Kiểm tra | Giá trị |','|---|---:|']
 lines.append(f"| M-6×768 khớp fp32 sau chép lớp: sai khác lớn nhất | {pre['m6_copy_max_abs_diff']} |")
 lines.append(f"| M-6×768 loss bước 0, chép teacher | {pre['m6_step0_loss']['teacher_layers']} |")
 lines.append(f"| M-6×768 loss bước 0, ngẫu nhiên | {pre['m6_step0_loss']['random']} |")
 for variant in ('s_prime','m6','m12'):
  s=pre['smoke'][variant];lines.append(f"| {variant}: tham số backbone; loss bước 1→200; giây/bước; peak VRAM byte; resume max diff 101–110 | {pre['backbones'][variant]['parameters']}; {s['loss_first']}→{s['loss_last']}; {s['seconds_per_step']}; {s['peak_vram_bytes']}; {s['resume_max_abs_diff_101_110']} |")
 p=pre['parity'];lines.append(f"| Parity Hotel seed 42: ΔF1 epoch 1–5; Δbest | {p['epoch_1_5_abs_f1_point_diffs']}; {p['best_abs_f1_point_diff']} |")
 lines.append(f"| Dự báo tổng sau smoke / trần | {pre['forecast_gpu_hours_after_smoke']} / 45 GPU-giờ |")
 lines+=['','## General distillation','','| Backbone | Bước | Giây/bước | GPU-giờ job | Loss cuối | GPU |','|---|---:|---:|---:|---:|---:|']
 for variant in ('s_prime','m6','m12'):
  m=read(ART/'backbones'/variant/'final/distill_manifest.json');j=reg[f'GD:{variant}'];lines.append(f"| {variant} | {m['completed_steps']} | {m['elapsed_seconds']/m['completed_steps']} | {j['elapsed_seconds']/3600} | {m['loss_curve_tail'][-1]['total']} | {j['gpu']} |")
 lines+=['','## Quyết định validation','','Gate A4:','',f"- P(S′) = {d['s_gate']['P_s_prime']}; P(A4) = {d['s_gate']['P_a4']}; ngưỡng = {d['s_gate']['threshold']}; kích hoạt = {d['s_gate']['triggered']}; S′ chính thức = `{d['s_official']}`.",'',f"- Điểm chọn hình dạng M: M-6×768 = {d['shape_selection']['means']['m6']}; M-12×512 = {d['shape_selection']['means']['m12']}; chênh tuyệt đối = {d['shape_selection']['absolute_delta']}; ngưỡng hòa = {d['shape_selection']['tie_threshold']}; chọn `{d['m_shape']}`.",'','| Task chọn hình dạng | LR student | LR Bami | M-6×768 student | M-6×768 Bami | M-12×512 student | M-12×512 Bami |','|---|---:|---:|---:|---:|---:|---:|']
 from summarize import STUDENT_LR,BAMI_LR
 for task in FIRST:
  scores=d['shape_selection']['scores'];ls=f'{STUDENT_LR[task]:.0e}';lb=f'{BAMI_LR[task]:.0e}'
  lines.append(f"| {task} | {STUDENT_LR[task]} | {BAMI_LR[task]} | {scores['m6'][task][ls]} | {scores['m6'][task][lb]} | {scores['m12'][task][ls]} | {scores['m12'][task][lb]} |")
 lines+=['','| Task chọn LR sau hình dạng | LR student, F1 | LR Bami, F1 | LR chọn |','|---|---:|---:|---:|']
 for task in ('vsfc_topic','visfd','absa_hotel'):
  scores=d['remaining_lr_selection'][task];ls=f'{STUDENT_LR[task]:.0e}';lb=f'{BAMI_LR[task]:.0e}'
  lines.append(f"| {task} | {scores[ls]} | {scores[lb]} | {d['m_lr'][task]} |")
 lines+=['','## Validation: 8 task, 15 chỉ số','','Mỗi ô là mean (min–max) của 3 seed, đơn vị điểm phần trăm; tính lại từ file dự đoán. A4 là kết quả tham chiếu 5 task của vòng trước.','','| Task | Chỉ số | Bami | S gốc | A4 | S′ | M |','|---|---|---:|---:|---:|---:|---:|']
 for task in TASKS:
  for metric in METRICS[task]:
   row=v['metric_table'][f'{task}/{metric}'];cell=lambda name:mr(row[name]['seeds']) if name in row else '—'
   lines.append(f"| {task} | {metric} | {cell('bami')} | {cell('s_original')} | {cell('a4_reference')} | {cell('s_prime')} | {cell('m')} |")
 lines+=['','## Test: 8 task, 15 chỉ số','','Mỗi ô mean (min–max) của 3 seed, điểm phần trăm. Bami và S gốc lấy từ `artifacts/final_test/test_results.json`; S′ và M tính từ dự đoán của lượt test đã khóa.','','| Task | Chỉ số | Bami | S gốc | S′ | M | S′ − Bami | M − Bami |','|---|---|---:|---:|---:|---:|---:|---:|']
 totals={name:[] for name in ('bami','s_original','s_prime','m')};non={name:[] for name in totals};wins={'s_prime':0,'m':0};table=[]
 for task in TASKS:
  for metric in METRICS[task]:
   vals={}
   for label in totals:
    source=oldlookup if label in ('bami','s_original') else lookup;name='student' if label=='s_original' else label
    vals[label]=[100*source[(task,name,seed)]['test_metrics'][metric] for seed in (42,43,44)]
    totals[label].append(statistics.mean(vals[label]))
    if task!='vinli':non[label].append(statistics.mean(vals[label]))
   ds=statistics.mean(vals['s_prime'])-statistics.mean(vals['bami']);dm=statistics.mean(vals['m'])-statistics.mean(vals['bami'])
   wins['s_prime']+=ds>=0;wins['m']+=dm>=0
   lines.append(f"| {task} | {metric} | {mr(vals['bami'])} | {mr(vals['s_original'])} | {mr(vals['s_prime'])} | {mr(vals['m'])} | {ds} | {dm} |")
   table.append({'task':task,'metric':metric,'values':{k:{'mean':statistics.mean(x),'range':[min(x),max(x)]} for k,x in vals.items()},'s_prime_minus_bami':ds,'m_minus_bami':dm})
 lines+=['','| Phạm vi | Bami mean | S gốc mean | S′ mean | M mean | S′ ≥ Bami | M ≥ Bami |','|---|---:|---:|---:|---:|---:|---:|']
 for label,source in [('15 chỉ số',totals),('13 chỉ số bỏ ViNLI',non)]:
  n=len(source['bami']);sw=sum(a>=b for a,b in zip(source['s_prime'],source['bami']));mw=sum(a>=b for a,b in zip(source['m'],source['bami']))
  lines.append(f"| {label} | {statistics.mean(source['bami'])} | {statistics.mean(source['s_original'])} | {statistics.mean(source['s_prime'])} | {statistics.mean(source['m'])} | {sw}/{n} | {mw}/{n} |")
 lines+=['','## Hiệu năng','','Các phép đo dùng checkpoint Sentiment seed 42 trong cùng một phiên. Dung lượng là state_dict encoder và head; GPU bf16, CPU fp32 8 luồng.','','| Model | Tham số | Checkpoint fp32 byte | Checkpoint bf16 byte | Checkpoint gốc byte |','|---|---:|---:|---:|---:|']
 for name in ('bami','s_prime','m'):
  x=eff['models'][name];lines.append(f"| {name} | {x['parameters']} | {x['checkpoint_bytes']['fp32']} | {x['checkpoint_bytes']['bf16']} | {x['original_checkpoint_bytes']} |")
 lines+=['','| Thiết bị | Batch | Length | Bami median ms | S′ median ms | M median ms | Bami throughput/s | S′ throughput/s | M throughput/s | Bami peak VRAM byte | S′ peak VRAM byte | M peak VRAM byte |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for batch in (1,32):
  for length in (64,128,256):
   key=f'b{batch}_l{length}';x={n:eff['gpu'][n][key] for n in ('bami','s_prime','m')}
   lines.append(f"| GPU bf16 | {batch} | {length} | {x['bami']['median_ms']} | {x['s_prime']['median_ms']} | {x['m']['median_ms']} | {x['bami']['throughput_examples_per_second']} | {x['s_prime']['throughput_examples_per_second']} | {x['m']['throughput_examples_per_second']} | {x['bami']['peak_vram_bytes']} | {x['s_prime']['peak_vram_bytes']} | {x['m']['peak_vram_bytes']} |")
 for length in (64,128):
  key=f'b1_l{length}';x={n:eff['cpu'][n][key] for n in ('bami','s_prime','m')}
  lines.append(f"| CPU fp32 | 1 | {length} | {x['bami']['median_ms']} | {x['s_prime']['median_ms']} | {x['m']['median_ms']} | — | — | — | — | — | — |")
 history=read(ART/'retry_history.json') if (ART/'retry_history.json').exists() else []
 jobs=[j for j in reg.values() if j.get('status')=='completed']+history
 gpu_hours=sum(j['elapsed_seconds'] for j in jobs)/3600
 per_gpu={str(g):sum(j['elapsed_seconds'] for j in jobs if j.get('gpu')==g)/3600 for g in (0,1,2,3)}
 smoke_training_hours=sum(x['seconds_per_step']*x['steps'] for x in pre['smoke'].values())/3600
 parity_hours=read(ART/'parity/absa_hotel/seed_42/metadata.json')['elapsed_seconds']/3600
 timer_sum=gpu_hours+test['elapsed_seconds']/3600+eff['elapsed_seconds']/3600+smoke_training_hours+parity_hours
 lines+=['','## GPU-giờ, sự cố và giới hạn','',f"Loạt validation: `{gpu_hours}` GPU-giờ, GPU 0–3: `{per_gpu}`; trong đó 12 run lỗi ban đầu chiếm `{sum(x['elapsed_seconds'] for x in history)/3600}` GPU-giờ. Cửa sổ test trên GPU 2: `{test['elapsed_seconds']/3600}` giờ, gồm khoảng dừng để sửa lỗi metric. Hiệu năng trên GPU 2: `{eff['elapsed_seconds']/3600}` GPU-giờ. Smoke GD trên GPU 0/1/2 theo timer train: `{smoke_training_hours}` GPU-giờ; parity Hotel trên GPU 0: `{parity_hours}` GPU-giờ. Tổng các timer ghi nhận là `{timer_sum}` giờ GPU quy đổi; đây không phải tổng sử dụng GPU chính xác vì cửa sổ test có thời gian dừng, còn ba lượt smoke resume ngắn chưa có timer job riêng. GPU đo hiệu năng: `{eff['gpu_name']}`; snapshot process khác nằm trong `artifacts/family_sm/efficiency.json`.",'',f"Hash dữ liệu test ba task mới được ghi trong `artifacts/family_sm/test/test_access_log.json`: `{access['test_files']}`. ViNLI/PhoNER khớp SHA đã khóa trong protocol `scripts/final_test/run.py`. Manifest có 48 SHA checkpoint và chỉ số validation được chạy lại từ checkpoint.",'','Sự cố: 12 run M seed 42 thất bại trước khi train vì optimizer E7 gốc giả định 10 lớp; đã sửa adapter trong code mới cho 6/12 lớp, chuyển output dở sang `aborted/`, chạy lại thành công và giữ GPU-giờ trong `retry_history.json`. Lệnh test đầu tiên dừng trước manifest do trùng tên module import. Sau khi khóa manifest, checkpoint test thứ 31 đã lưu predictions nhưng bước tính metric lỗi do import `common`; `resume_test.py` chỉ tính metric từ predictions đó rồi chạy 17 checkpoint còn lại. Không checkpoint nào chạy inference test lần hai; tất cả postrun và validation gate cuối đều PASS. Không sửa code hay artifact cũ.','','Giới hạn: mỗi backbone chỉ có một GD seed; S′ giữ LR student; M chọn hình dạng từ 5 task và một seed, LR chỉ trong hai giá trị đã khóa. Các phép đo latency thực hiện trên GPU dùng chung nên chịu tải ngoài; snapshot nằm trong artifact hiệu năng. Test đã dùng hết một lượt này. A4 chỉ là tham chiếu 5 task; teacher DAPT của A4 từng đọc text validation trong vòng trước.','','## Lệnh tái lập','','Chỉ chạy các lệnh test từ một thư mục artifact mới sau khi validation hoàn tất; `test.py` từ chối chạy lại nếu manifest đã tồn tại.','','```bash','PYTHONPATH=src HF_HUB_OFFLINE=1 .venv-e0-cuda/bin/python scripts/family_sm/preflight.py','setsid nohup env PYTHONPATH=src HF_HUB_OFFLINE=1 FSM_GPUS=0,1,2,3 .venv-e0-cuda/bin/python scripts/family_sm/launcher.py > artifacts/family_sm/launcher.log 2>&1 < /dev/null &','PYTHONPATH=src .venv-e0-cuda/bin/python scripts/family_sm/summarize.py','CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src HF_HUB_OFFLINE=1 .venv-e0-cuda/bin/python scripts/family_sm/test.py','CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src HF_HUB_OFFLINE=1 .venv-e0-cuda/bin/python scripts/family_sm/efficiency.py','PYTHONPATH=src .venv-e0-cuda/bin/python scripts/family_sm/report.py','```','','## File tạo','','- `scripts/family_sm/`: GD, aligner, optimizer theo số lớp, fine-tune, launcher, tổng hợp, test/resume, hiệu năng và báo cáo.','- `artifacts/family_sm/`: preflight, smoke log, parity, ba backbone, validation, quyết định, launcher registry, retry history, aborted runs, manifest và log test, 48 predictions, 48 metrics, hiệu năng, báo cáo số liệu.','- `docs/FAMILY_SM_RESULT.md`.','']
 DOC.write_text('\n'.join(lines));(ART/'final_summary.json').write_text(json.dumps({'test_table':table,'test_means_15':{k:statistics.mean(x) for k,x in totals.items()},'test_means_13':{k:statistics.mean(x) for k,x in non.items()},'wins':wins,'gpu_hours_validation':gpu_hours,'gpu_hours_test_window':test['elapsed_seconds']/3600,'gpu_hours_efficiency':eff['elapsed_seconds']/3600,'gpu_hours_smoke_training':smoke_training_hours,'gpu_hours_parity':parity_hours,'gpu_hours_timer_sum':timer_sum,'gpu_hours_by_gpu':per_gpu},indent=2)+'\n')
 print('REPORT_COMPLETE',DOC)
if __name__=='__main__':main()
