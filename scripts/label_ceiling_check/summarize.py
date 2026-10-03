#!/usr/bin/env python3
"""Render the validation-only Vietnamese report from analysis artifacts."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts/label_ceiling_check'
TASKS=('vsfc_sentiment','vsfc_topic','visfd','absa_restaurant','absa_hotel')
LABEL={'vsfc_sentiment':'Sentiment','vsfc_topic':'Topic','visfd':'ViSFD','absa_restaurant':'Restaurant','absa_hotel':'Hotel'}
def rd(n):return json.loads((ART/n).read_text())
def pct(x):return '' if x is None else str(100*x)
def num(x):return '' if x is None else str(x)
def table(head,rows):return '| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)

def main():
    a=rd('consensus_analysis.json');p=rd('pool_manifest.json');noise=rd('seed_noise.json');join=rd('blind_reading_join.json');inter=rd('interpretation_inputs.json');manifest=rd('manifest.json')
    gpu_ready=all(noise[t]['range'] is not None for t in TASKS)
    lines=['# Kiểm tra trần nhãn gốc — validation only','',
      '## Trạng thái và protocol','',
      ('**Phần A đã đủ ba seed.** Phần B và C đã hoàn tất; số liệu validation only.' if gpu_ready else '**Phần A đang chờ chạy GPU.** Phiên sandbox chuẩn bị không có GPU và không có mạng. Đã chuẩn bị wrapper, dry-run config, recipe diff và launcher; không train trong phiên chuẩn bị. Phần B và C đã hoàn tất trên CPU. Mọi F1 dưới đây là validation selection F1, không đọc split test. R và mọi điều kiện phụ thuộc R để trống cho đến khi đủ seed 42/43/44.'),'',
      'Mục tiêu là đo nhiễu seed CafeBERT và phân rã lỗi còn lại bằng pool các họ model khác nhau. Đơn vị là câu cho VSFC, slot aspect cho ABSA/ViSFD. CE: gold khác nhãn được ít nhất N−1 phiếu; mức chặt là N phiếu. LE gốc: Cafe sai và ít nhất một model khác đúng; HE: Cafe sai, không model đúng, không thuộc CE.','',
      '**Giao thoa định nghĩa:** CE và LE gốc có thể trùng nhau khi N−1 model cùng sai và model còn lại đúng. Bảng phân rã dùng CE ưu tiên, LE = LE gốc trừ CE, rồi HE để tạo phân hoạch. Cột “giao thoa” và LE gốc được lưu trong `consensus_analysis.json`; H_max sửa toàn bộ LE gốc.','',
      '## A. Recipe và nhiễu seed','',
      'Dry-run seed 43/44 cho cả năm task khớp core recipe seed 42: LR 2e-5, 30 epoch, batch hiệu dụng 32, max length 256, bf16, first-token → dropout 0,1 → dense → tanh → dropout 0,1 → logits, CE thường, best validation. Restaurant T42 và seed 43 cũ khớp hyperparameter, model/tokenizer và evaluator; cách diễn đạt protocol/head/selection, adapter, hash nguồn và runtime metadata khác. Seed 43 cũ được tái dùng qua liên kết tới artifact gốc. Chi tiết tại `recipe_diff.json`.','',
      table(['Task','Cafe 42 (%)','Cafe 43 (%)','Cafe 44 (%)','Mean (%)','SD mẫu (%)','R (%)'],[[LABEL[t],pct(noise[t]['seed_f1'].get('42')),pct(noise[t]['seed_f1'].get('43')),pct(noise[t]['seed_f1'].get('44')),pct(noise[t]['mean']),pct(noise[t]['sample_std']),pct(noise[t]['range'])] for t in TASKS]),'',
      'Restaurant seed 42 → seed 43 đã thấy: 77,71971496437055 → 76,39605818864383, chênh 1,32365677572672 điểm. Đây mới là chênh hai seed, chưa phải R ba seed.','',
      '### Chênh lệch đã thấy với Cafe seed 42','']
    ens=rd('../two_backbone_ensemble/ensemble_results.json') if False else json.loads((ROOT/'artifacts/two_backbone_ensemble/ensemble_results.json').read_text())
    delta=[]
    for t in TASKS:
        cafe=p[t]['CafeBERT']['f1'];r=[LABEL[t],pct(p[t]['P_vi']['f1']-cafe),pct(p[t]['BidirLM-1B']['f1']-cafe) if 'BidirLM-1B' in p[t] else '']
        r.append(str(ens[t]['ensemble']['PAIR']['delta_cafe']) if t in ens else '')
        delta.append(r)
    lines += [table(['Task','P_vi − Cafe (điểm)','BidirLM-1B − Cafe (điểm)','Cafe + P_vi − Cafe, artifact (điểm)'],delta),'',
      ('R ba seed đã có trong bảng trên; đối chiếu trực tiếp với các chênh đã thấy. “PAIR” trong artifact ensemble là Cafe + P_vi.' if gpu_ready else 'So sánh các chênh này với range R sẽ thực hiện sau khi đủ ba seed. “PAIR” trong artifact ensemble là Cafe + P_vi.'),'',
      '## B. Pool và phân rã','',
      'Pool có N=6 cho Restaurant/ViSFD/Topic, N=5 cho Sentiment/Hotel. LR của mmBERT lấy từ `summary.json`; Bami từ locked selection E14; các run khác theo đường dẫn cố định của spec. Đã tính lại F1 bằng `goibert.train.evaluate_predictions` và `selection_score`; ID và gold trùng tuyệt đối với validation bundle và giữa mọi model. XLM-R-large gốc cùng họ Cafe nên chỉ là tham khảo ngoài pool.','',
      table(['Task','Cafe','mmBERT','Bami','mDeBERTa','P_vi','BidirLM-1B'],[[LABEL[t]]+[pct(p[t][m]['f1']) if m in p[t] else '' for m in ('CafeBERT','mmBERT-base','BamiBERT E14','mDeBERTa-v3-base','P_vi','BidirLM-1B')] for t in TASKS]),'']
    xl=json.loads((ROOT/'artifacts/xlmr_large_baseline/summary.json').read_text())
    references=[]
    for task,grid in xl['grid'].items():
        candidates=[(entry['42']['f1'],lr) for lr,entry in grid.items() if entry and entry.get('42') and entry['42'].get('status')=='completed']
        if candidates:
            f1,lr=max(candidates)
            references.append(f'{LABEL[task]} {pct(f1)}% (LR {lr})')
    lines += ['XLM-R-large seed 42 tham khảo: '+', '.join(references)+'; không vào pool.','',
      table(['Task','Seed','Đơn vị','Lỗi Cafe','CE','LE riêng','HE','CE/lỗi','CE∩LE gốc','CE cả N'],[[LABEL[t],s,x['units'],x['errors'],x['CE'],x['LE'],x['HE'],pct(x['CE_rate']),x['CE_LE_overlap'],x['all_N_CE']] for t in TASKS for s,x in a[t]['seeds'].items()]),'',
      'Tỷ lệ LE/HE và LE gốc đầy đủ nằm trong `consensus_analysis.json`. ViSFD OTHERS là slot detection-only: tham gia đếm lỗi/đồng thuận, nhưng bị loại đúng như evaluator khi tính pair micro F1.','',
      table(['Task','Seed','F1 gốc (%)','L_max (điểm)','H_max (điểm)','Oracle pool (%)'],[[LABEL[t],s,pct(x['f1']),pct(x['L_max']),pct(x['H_max']),pct(a[t]['oracle_f1']) if s=='42' else ''] for t in TASKS for s,x in a[t]['seeds'].items()]),'',
      'L_max là thay dự đoán Cafe ở mọi CE mà Cafe sai bằng gold rồi chạy lại evaluator; H_max tương tự với mọi LE gốc. Oracle đúng nếu bất kỳ model nào đúng từng đơn vị: đây là **cận trên lạc quan**, không phải hệ thống khả thi.','',
      '### Bỏ CE khỏi chấm điểm','',
      'Với ABSA/ViSFD, TP, số dự đoán dương và số gold dương được cộng thủ công trên slot còn lại theo pair micro F1 gốc; kiểm bộ đếm không bỏ slot khớp evaluator. Với VSFC, evaluator macro F1 chạy lại trên các câu còn lại.','',
      table(['Task','F1 khi bỏ CE (%) theo thứ tự Cafe / mmBERT / Bami / mDeBERTa / P_vi / B1B','Thứ hạng đổi?'],[[LABEL[t],' / '.join(pct(a[t]['excluded_CE_f1'][m]) if m in a[t]['excluded_CE_f1'] else '' for m in ('CafeBERT','mmBERT-base','BamiBERT E14','mDeBERTa-v3-base','P_vi','BidirLM-1B')),'có' if a[t]['rank_changed'] else 'không'] for t in TASKS]),'',
      'Thứ hạng cụ thể trước/sau nằm trong `consensus_analysis.json`.','',
      '### Đối chiếu đọc mù 14b','',
      'UID được nối qua `*_uid_mapping.json` → ID validation; slot theo `aspect_names` của bundle gốc và vector nhãn. Với Topic nối câu. Reader của đợt cũ gán toàn review; phép đếm dưới đây chỉ giữ các slot/câu CE đã được đọc.','',
      table(['Task','CE đã đọc','Reader = gold','Reader = nhãn CE','Reader khác cả hai'],[[LABEL[t],join[t]['counts'].get('ce_read',0),join[t]['counts'].get('reader_gold',0),join[t]['counts'].get('reader_consensus',0),join[t]['counts'].get('reader_other',0)] for t in ('absa_restaurant','visfd','vsfc_topic')]),'',
      '## C. Gói phân xử mù','',
      'Mỗi task có 24 CE, 8 đối chứng toàn pool đúng, 8 đơn vị pool chia rẽ; tổng 200 mục. `items.jsonl` và `items.csv` chỉ có UID ngẫu nhiên, task, text, aspect, nhãn hợp lệ, ô trống; đã xáo seed 42. `key.jsonl` chứa đáp án và nhóm, không mở trước khi phân xử. `score_adjudication.py` chỉ được chuẩn bị, chưa chạy.','',
      '## Quy tắc diễn giải đã chốt','',
      'Đơn vị R/L/H ở bảng sau là điểm F1; C là tỷ lệ lỗi. Điều kiện trần nhãn của task: C ≥ 0,5 và L ≥ 2R. Kết luận chung cần ≥3/5 task đạt. H < 2R là phép kiểm dư địa học được.','',
      table(['Task','R','C','L','H','C ≥ 0,5','L ≥ 2R','H < 2R','Đạt cả hai'],[[LABEL[t],num(inter[t]['R']),num(inter[t]['C']),num(inter[t]['L']),num(inter[t]['H']),str(inter[t]['C_ge_half']),'' if inter[t]['L_ge_2R'] is None else str(inter[t]['L_ge_2R']),'' if inter[t]['H_lt_2R'] is None else str(inter[t]['H_lt_2R']),'' if inter[t]['label_ceiling_condition'] is None else str(inter[t]['label_ceiling_condition'])] for t in TASKS]),'',
      ('Điều kiện chung ≥3/5 task: '+str(inter['_overall']['at_least_three_of_five']) if gpu_ready else 'Kết luận chung: **chờ R**.'),'',
      '## GPU, kiểm tra và giới hạn','',
      f'- GPU-giờ dùng cho run mới: **{manifest["gpu_hours_used"]}**; GPU đã dùng: **{", ".join(manifest["gpus_used"]) if manifest["gpus_used"] else "không có"}**. Cafe Restaurant seed 43 là artifact có sẵn, không tính vào ngân sách phiên này. Budget launcher: 15 GPU-giờ, ước tính lại sau hai run mới đầu và dừng trước đợt kế nếu vượt trần.','- Check CPU: F1 prediction = F1 best metric; IDs/gold đồng nhất; evaluator thủ công không bỏ CE = F1 gốc; recipe core PASS. Không có check FAIL ở phần đã chạy. Postrun checks của seed 43 cũ PASS theo metadata gốc.','- CE chỉ là ứng viên nhãn sai, chưa phải nhãn sai đã xác nhận. Pool 5–6 model vẫn có thể sai chung vì nguyên nhân khác. Đọc mù cũ chỉ có một người đọc LLM trên mỗi task. Mọi số chỉ thuộc validation.','',
      '## Lệnh tái lập và hoàn tất GPU','',
      'Chạy tại root repo. Launcher tự chọn GPU 1/2 khi VRAM trống ≥20 GiB, tối đa một job của mình trên mỗi GPU, không dùng GPU 0/3. Mỗi lượt kiểm sau ≥5 phút. Lệnh này dùng venv của run Cafe gốc, `PYTHONPATH=src`, `HF_HUB_OFFLINE=1`:','',
      '```bash','PYTHONPATH=src HF_HUB_OFFLINE=1 .venv-e0-cuda/bin/python scripts/label_ceiling_check/recipe_diff.py','nohup env PYTHONPATH=src HF_HUB_OFFLINE=1 .venv-e0-cuda/bin/python scripts/label_ceiling_check/launcher.py > artifacts/label_ceiling_check/logs/launcher.log 2>&1 &','```','',
      'Sau khi launcher hoàn tất hoặc dừng vì budget, chạy lại toàn bộ phân tích CPU và báo cáo bằng một lệnh:','',
      '```bash','PYTHONPATH=src HF_HUB_OFFLINE=1 .venv-e0-cuda/bin/python scripts/label_ceiling_check/analyze.py && PYTHONPATH=src HF_HUB_OFFLINE=1 .venv-e0-cuda/bin/python scripts/label_ceiling_check/summarize.py','```','',
      'Dry-run một cấu hình riêng, không nạp model lên GPU:','',
      '```bash','PYTHONPATH=src HF_HUB_OFFLINE=1 .venv-e0-cuda/bin/python scripts/label_ceiling_check/seed_train.py --task visfd --seed 43 --dry-run-config','```','']
    dest=ROOT/'docs/LABEL_CEILING_CHECK_RESULT.md';dest.write_text('\n'.join(lines)+'\n')
    print(dest)
if __name__=='__main__':main()
