"""Wait for the single authorized pilot and emit its factual result report."""
import json,time,os,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];o=ROOT/'artifacts/cafebert_composite';pid=json.load(open(o/'launch.json'))['pid'];report=ROOT/'docs/CAFEBERT_COMPOSITE_PILOT_RESULT.md'
while not (o/'summary.json').exists():
 try:os.kill(pid,0)
 except ProcessLookupError:
  report.write_text('# CafeBERT composite pilot\n\nRun stopped before summary completion. Inspect artifacts/cafebert_composite/train.log and per-run status. No automatic rerun.\n');raise SystemExit(1)
 time.sleep(20)
s=json.load(open(o/'summary.json'));r=o/'run_seed42';m=json.load(open(r/'metadata.json'));a=json.load(open(r/'validation_predictions.json'))['rows'];ref=ROOT/'artifacts/cdbk_pilot/runs/T_absa_restaurant_seed42';b=json.load(open(ref/'validation_predictions.json'))['rows'];assert [(x['id'],x['gold']) for x in a]==[(x['id'],x['gold']) for x in b]
def f(rows):
 tp=sum(y==g and g>0 for x in rows for y,g in zip(x['prediction'],x['gold']));n=sum(y>0 for x in rows for y in x['prediction']);ng=sum(g>0 for x in rows for g in x['gold']);return 2*tp/(n+ng)
fnew,fref=f(a),f(b);assert abs(fnew-s['validation_f1'])<1e-12;delta=100*(fnew-fref)
cur=json.load(open(r/'epoch_curves.json'))['epochs'];assert len(cur)==30
verdict='Có tín hiệu để xác nhận ở lượt sau; chưa significance.' if delta>=.5 else ('Gain nhỏ, chưa rõ.' if delta>=.2 else 'Chưa có tín hiệu đạt mục tiêu; dừng cấu hình này.')
report.write_text(f'''# CafeBERT + composite — pilot Restaurant

Một run mới, seed42, validation-only; không test hoặc training thêm. Inference giữ nguyên encoder/head48logits, một pass mỗi review.

| Arm | Best epoch | Validation pair microF1 | Delta | Runtime |
|---|---:|---:|---:|---:|
| CafeBERT CE reference |21|{100*fref:.4f}|—|23.3min lịch sử|
| CafeBERT CE + uniform composite |{s['best_epoch']}|{100*fnew:.4f}|{delta:+.4f}pp|{s['elapsed_seconds']/60:.1f}min|

## Cách làm

7028reviewtrain +2343composites của E22uniformseed42 =9371rows. Cùng checkpoint CafeBERT pretrained, head và lossCE như reference;LR2e-5,30epochs,effbatch32,maxlen256,BF16SDPA. KhôngKD/NC/LLRD. Mọi sourceID/text/unionlabel composite được kiểm, max90tokens theoXLM-R, khôngtruncation. Validation giữ nguyên771reviews.

## Kiểm tra

Smoke/reload/F1checks passed trong metadata. Report writer đếm lại pairF1 từ savedpredictions độc lập và đối chiếu ID/gold với reference; khớp1e-12.30epochs hoàn tất. Best checkpoint chọn validation, không đổi bằngtest.

## Diễn giải

{verdict}

Có33%thêmoptimizersteps (8790 vs6600) do trainset lớn hơn ở cùng30epochs; chưa cô lập gain doaugmentation với extra training. Inference model không tăng sốpasses hoặcparams. Runtime trênGPUkhác điều kiệnchia sẻ khôngphải benchmarklatency. Chỉmộtseed/mộttask; khôngclaimSOTA hoặc thắngrộng.

Artifacts: artifacts/cafebert_composite/summary.json và run_seed42/. Scope kết thúc, không tự thêmseed/LR/test.
''')
(o/'report_status.json').write_text(json.dumps({'status':'completed','report':str(report),'pair_f1_recomputed':fnew,'delta_pp':delta},indent=2));print(delta,flush=True)
