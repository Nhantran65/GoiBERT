#!/usr/bin/env python3
"""Validation-only consensus analysis and blinded adjudication packet."""
import csv
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from goibert import train  # noqa: E402

ART=ROOT/'artifacts/label_ceiling_check'
TASKS=('vsfc_sentiment','vsfc_topic','visfd','absa_restaurant','absa_hotel')
BLIND=ROOT/'artifacts/cafe_bami_paired_error_audit/blind_reading'

def read(path): return json.loads(path.read_text())
def write(name,value):
    p=ART/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');return p

def root_paths(task):
    base=ROOT/'artifacts'
    cafe=base/'cdbk_pilot/runs/T_absa_restaurant_seed42' if task=='absa_restaurant' else base/'sealion_benchmark/cafe_runs'/task/'lr_2e-05/seed_42'
    mm=read(base/'base_encoder_baseline/summary.json')['models']['mmbert_base']['selected'][task]
    bami=next(x for x in read(base/'e14/locked_selection_manifest.json')['selections'] if x['task_id']==task)
    result={'CafeBERT':cafe,'mmBERT-base':ROOT/mm['dir'],'BamiBERT E14':Path(bami['artifact_dir']),
            'mDeBERTa-v3-base':base/'mdeberta_pilot/runs'/task/'lr_2e-05/seed_42',
            'P_vi':base/'bidirlm_vi_mntp/ft/P_vi'/task/'lr_1e-05/seed_42'}
    if task in ('visfd','vsfc_topic','absa_restaurant'):
        result['BidirLM-1B']=base/'bidirlm_1b_probe/phase2/ft/B1B'/task/'lr_5e-06/seed_42'
    return result

def rows(path):
    x=read(path/'validation_predictions.json')['rows']
    ids=[r['id'] for r in x]
    if len(ids)!=len(set(ids)): raise ValueError(f'duplicate ids: {path}')
    return x

def score(task,bundle,gold,pred): return train.selection_score(train.evaluate_predictions(task,bundle,gold,pred,split='validation'))

def units(rows_by_model,task):
    names=list(rows_by_model)
    n=len(rows_by_model[names[0]])
    result=[]
    for i in range(n):
        ref=rows_by_model[names[0]][i]; g=ref['gold']
        for j in range(len(g) if isinstance(g,list) else 1):
            labels={m:(rows_by_model[m][i]['prediction'][j] if isinstance(g,list) else rows_by_model[m][i]['prediction']) for m in names}
            # The evaluator maps ViSFD's binary OTHERS head 1 to raw gold 4.
            # Consensus and exact correctness must use that same label space.
            if task=='visfd' and j==5:
                labels={m:(4 if value==1 else value) for m,value in labels.items()}
            gold=g[j] if isinstance(g,list) else g
            votes=Counter(labels.values()); label,count=votes.most_common(1)[0]
            consensus=(label!=gold and count>=len(names)-1)
            split=(count<len(names)-1)
            result.append({'id':ref['id'],'row':i,'slot':j if isinstance(g,list) else None,'gold':gold,'predictions':labels,'ce':consensus,'all_wrong_same':label!=gold and count==len(names),'consensus_label':label if consensus else None,'all_correct':all(v==gold for v in labels.values()),'split':split})
    return result

def patched(rows_,changes):
    out=[list(r['prediction']) if isinstance(r['prediction'],list) else r['prediction'] for r in rows_]
    for u in changes:
        if u['slot'] is None: out[u['row']]=u['gold']
        else: out[u['row']][u['slot']]=u['gold']
    return out

def excluded_score(task,us,model,bundle,original_score):
    keep=[u for u in us if not u['ce']]
    if task.startswith('vsfc_'):
        result=score(task,bundle,[u['gold'] for u in keep],[u['predictions'][model] for u in keep])
        check=score(task,bundle,[u['gold'] for u in us],[u['predictions'][model] for u in us])
    else:
        aspects=list(train._field(bundle,'aspect_names',[]))
        def pair_f1(qq):
            tp=pg=gg=0
            for u in qq:
                if task=='visfd' and aspects[u['slot']]=='OTHERS': continue
                g,p=u['gold'],u['predictions'][model]
                if task=='visfd' and aspects[u['slot']]=='OTHERS' and p==1:p=4
                tp+=int(g>0 and g==p); pg+=int(p>0);gg+=int(g>0)
            return 2*tp/(pg+gg) if pg+gg else 0.0
        result=pair_f1(keep);check=pair_f1(us)
    if abs(check-original_score)>1e-10: raise ValueError(f'manual counter mismatch {task}/{model}: {check} vs {original_score}')
    return result

def blind_join(task,us):
    if task not in ('absa_restaurant','visfd','vsfc_topic'): return None
    mapping=read(BLIND/f'{task}_uid_mapping.json')
    readers={x['uid']:x for x in map(json.loads,(BLIND/f'{task}_reader1.jsonl').read_text().splitlines())}
    by_id={}
    for u in us: by_id[(u['id'],u['slot'])]=u
    aspects=list(train._field(train.load_task_bundle(task,ROOT,include_test=False),'aspect_names',[]))
    topic={'lecturer':0,'training_program':1,'facility':2,'others':3}
    pol={'positive':1,'negative':2,'neutral':3}
    counts=Counter(); examples=[]
    for m in mapping:
        r=readers.get(m['uid'])
        if r is None: continue
        if task=='vsfc_topic':
            candidates=[by_id.get((m['id'],None))]
        else:
            candidates=[by_id.get((m['id'],j)) for j in range(len(aspects))]
        reader_labels={v['aspect']:pol.get(v.get('polarity')) for v in r.get('labels',[])}
        for u in candidates:
            if u is None or not u['ce']:continue
            if task=='vsfc_topic': val=topic.get(r.get('label'))
            else:
                aspect=aspects[u['slot']]
                val=(4 if r.get('others') else 0) if aspect=='OTHERS' and task=='visfd' else reader_labels.get(aspect,0)
            counts['ce_read']+=1
            if val==u['gold']:counts['reader_gold']+=1
            elif val==u['consensus_label'] or (task=='visfd' and aspect=='OTHERS' and val==4 and u['consensus_label']==1):counts['reader_consensus']+=1
            else:counts['reader_other']+=1
            examples.append({'uid':m['uid'],'id':u['id'],'slot':u['slot'],'reader':val,'gold':u['gold'],'consensus':u['consensus_label']})
    if sum(counts[x] for x in ('reader_gold','reader_consensus','reader_other'))!=counts['ce_read']:raise ValueError('blind join partition')
    return {'counts':dict(counts),'joined_units':examples,'mapping_source':str((BLIND/f'{task}_uid_mapping.json').relative_to(ROOT))}

def packet(all_units,bundles):
    rng=random.Random(42); items=[];keys=[]
    for task,us in all_units.items():
        groups={'CE':[u for u in us if u['ce']], 'all_correct':[u for u in us if u['all_correct']], 'split':[u for u in us if u['split']]}
        sampled=[]
        for group,count in [('CE',24),('all_correct',8),('split',8)]:
            if len(groups[group])<count: raise ValueError(f'insufficient {task}/{group}: {len(groups[group])}')
            sampled.extend((group,u) for u in rng.sample(groups[group],count))
        val=train.split_field(bundles[task],'validation')
        ids=train.split_ids(val);texts=train.split_texts(val);text_by_id=dict(zip(ids,texts))
        aspects=list(train._field(bundles[task],'aspect_names',[]))
        classes=list(train._field(bundles[task],'class_names',train._field(bundles[task],'label_names',[])))
        for group,u in sampled:
            uid=f'LC{rng.getrandbits(64):016x}'
            aspect=aspects[u['slot']] if u['slot'] is not None else ''
            valid=['0 absent','4 present'] if task=='visfd' and aspect=='OTHERS' else ([f'{i} {x}' for i,x in enumerate(classes)] if u['slot'] is None else ['0 absent','1 positive','2 negative','3 neutral'])
            items.append({'uid':uid,'task':task,'text':text_by_id[u['id']],'aspect':aspect,'valid_labels':', '.join(valid),'human_label':''})
            keys.append({'uid':uid,'id':u['id'],'slot':u['slot'],'gold':u['gold'],'consensus_label':(4 if task=='visfd' and aspect=='OTHERS' and u['consensus_label']==1 else u['consensus_label']),'group':group,'task':task})
    order=list(range(len(items)));rng.shuffle(order)
    items=[items[i] for i in order];keys=[keys[i] for i in order]
    p=ART/'adjudication_packet';p.mkdir(parents=True,exist_ok=True)
    with (p/'items.jsonl').open('w') as f:
        for x in items:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    with (p/'key.jsonl').open('w') as f:
        for x in keys:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    with (p/'items.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    (p/'README.md').write_text('''# Gói phân xử mù\n\nChỉ mở `items.jsonl` hoặc `items.csv` trước khi phân xử. **Không mở `key.jsonl` trước khi phân xử xong**: file đó chứa gold, nhãn đồng thuận và nhóm. Điền `human_label` bằng mã số ở `valid_labels`; dùng `score_adjudication.py` sau khi đã điền.\n\n- VSFC Sentiment: 0 negative = ý kiến tiêu cực; 1 neutral = trung tính hoặc không rõ sắc thái; 2 positive = ý kiến tích cực. Gán cảm xúc toàn câu.\n- VSFC Topic: lecturer = giảng viên; training_program = chương trình đào tạo; facility = cơ sở vật chất; others = chủ đề khác.\n- Restaurant/Hotel: xét đúng aspect được nêu, 0 = không nhắc tới; 1 = tích cực; 2 = tiêu cực; 3 = trung tính. FOOD#QUALITY nói về chất lượng/vị; FOOD#STYLE&OPTIONS về loại món/lựa chọn; #PRICES về giá.\n- ViSFD: xét đúng aspect được nêu; cùng mã 0–3. Riêng OTHERS dùng 0 = không có, 4 = có ý kiến về aspect khác. GENERAL = nhận xét chung; PERFORMANCE = hiệu năng; PRICE = giá.\n\nĐọc cả review trước khi gán slot. Nếu mơ hồ, ghi chú ngoài file đáp án. Guideline dựa trên bộ đọc mù 14b và nhãn gốc của dữ liệu validation.\n''')
    return {'items':len(items),'per_task':40,'seed':42}

def main():
    pool_manifest={};analysis={};joins={};all_units={};bundles={};noise={}
    for task in TASKS:
        bundle=train.load_task_bundle(task,ROOT,include_test=False);bundles[task]=bundle
        if task=='visfd' and list(train._field(bundle,'aspect_names',[]))[5]!='OTHERS':
            raise ValueError('ViSFD OTHERS slot index changed')
        paths=root_paths(task);rr={m:rows(p) for m,p in paths.items()}
        ref=rr['CafeBERT'];ids=[x['id'] for x in ref];gold=[x['gold'] for x in ref]
        val=train.split_field(bundle,'validation')
        if ids!=list(train.split_ids(val)) or gold!=json.loads(json.dumps(list(train.split_labels(val)))):raise ValueError(f'Cafe ids/gold differ from validation bundle: {task}')
        for m,x in rr.items():
            if [z['id'] for z in x]!=ids or [z['gold'] for z in x]!=gold:raise ValueError(f'ids/gold mismatch {task}/{m}')
        ps={m:score(task,bundle,gold,[x['prediction'] for x in rows_]) for m,rows_ in rr.items()}
        for m,p in paths.items():
            met=read(p/'best_validation_metrics.json')
            if abs(ps[m]-float(met['selection_f1']))>1e-9:raise ValueError(f'F1 mismatch {task}/{m}')
        pool_manifest[task]={m:{'path':str(p.relative_to(ROOT)),'f1':ps[m]} for m,p in paths.items()}
        us=units(rr,task);all_units[task]=us
        n=len(paths);expected=6 if task in ('visfd','vsfc_topic','absa_restaurant') else 5
        if n!=expected:raise ValueError(f'wrong pool size {task}: {n}')
        ce=[u for u in us if u['ce']]
        f1_no={m:excluded_score(task,us,m,bundle,ps[m]) for m in paths}
        rank_before=sorted(paths,key=lambda m:-ps[m]);rank_after=sorted(paths,key=lambda m:-f1_no[m])
        oracle=patched(ref,[u for u in us if u['gold'] in u['predictions'].values()])
        oracle_f1=score(task,bundle,gold,oracle)
        seeds={};values=[]
        for seed in (42,43,44):
            p=paths['CafeBERT'] if seed==42 else ART/'cafe_seeds'/task/f'seed_{seed}'
            if seed>42 and not (p/'validation_predictions.json').exists():continue
            seed_rows=rows(p)
            if [x['id'] for x in seed_rows]!=ids or [x['gold'] for x in seed_rows]!=gold:raise ValueError(f'seed ids/gold mismatch {task}/{seed}')
            modified=dict(rr);modified['CafeBERT']=seed_rows
            su=units(modified,task)
            errors=[u for u in su if u['predictions']['CafeBERT']!=u['gold']]
            ces=[u for u in errors if u['ce']]
            le_raw=[u for u in errors if any(v==u['gold'] for m,v in u['predictions'].items() if m!='CafeBERT')]
            le=[u for u in le_raw if not u['ce']]
            he=[u for u in errors if not u['ce'] and u not in le_raw]
            if len(errors)!=len(ces)+len(le)+len(he):raise ValueError('CE LE HE do not partition')
            base=score(task,bundle,gold,[x['prediction'] for x in seed_rows]);values.append(base)
            lc=score(task,bundle,gold,patched(seed_rows,ces))-base
            hc=score(task,bundle,gold,patched(seed_rows,le_raw))-base
            seeds[str(seed)]={'f1':base,'units':len(su),'errors':len(errors),'CE':len(ces),'LE':len(le),'LE_raw':len(le_raw),'CE_LE_overlap':len(le_raw)-len(le),'HE':len(he),'CE_rate':len(ces)/len(errors) if errors else 0,'LE_rate':len(le)/len(errors) if errors else 0,'HE_rate':len(he)/len(errors) if errors else 0,'L_max':lc,'H_max':hc,'all_N_CE':sum(u['all_wrong_same'] for u in su)}
        noise[task]={'seed_f1':{s:x['f1'] for s,x in seeds.items()},'mean':statistics.mean(values) if len(values)==3 else None,'sample_std':statistics.stdev(values) if len(values)==3 else None,'range':max(values)-min(values) if len(values)==3 else None}
        analysis[task]={'pool_size':n,'units':len(us),'CE_units':len(ce),'all_N_CE_units':sum(u['all_wrong_same'] for u in us),'seeds':seeds,'oracle_f1':oracle_f1,'excluded_CE_f1':f1_no,'rank_before':rank_before,'rank_after':rank_after,'rank_changed':rank_before!=rank_after}
        joins[task]=blind_join(task,us)
    write('pool_manifest.json',pool_manifest);write('consensus_analysis.json',analysis);write('blind_reading_join.json',joins);write('seed_noise.json',noise)
    interpretation={t:{'R':noise[t]['range'],'C':analysis[t]['seeds']['42']['CE_rate'],'L':analysis[t]['seeds']['42']['L_max'],'H':analysis[t]['seeds']['42']['H_max'],'C_ge_half':analysis[t]['seeds']['42']['CE_rate']>=0.5,'L_ge_2R':(analysis[t]['seeds']['42']['L_max']>=2*noise[t]['range']) if noise[t]['range'] is not None else None,'H_lt_2R':(analysis[t]['seeds']['42']['H_max']<2*noise[t]['range']) if noise[t]['range'] is not None else None} for t in TASKS}
    for t in TASKS:
        interpretation[t]['label_ceiling_condition']=(interpretation[t]['C_ge_half'] and interpretation[t]['L_ge_2R']) if interpretation[t]['L_ge_2R'] is not None else None
    ready=all(interpretation[t]['R'] is not None for t in TASKS)
    interpretation['_overall']={'tasks_meeting_condition':sum(interpretation[t]['label_ceiling_condition'] is True for t in TASKS) if ready else None,
                                'at_least_three_of_five':sum(interpretation[t]['label_ceiling_condition'] is True for t in TASKS)>=3 if ready else None}
    write('interpretation_inputs.json',interpretation)
    info=packet(all_units,bundles)
    new_runs=[]
    for t in TASKS:
        for seed in (43,44):
            run=ART/'cafe_seeds'/t/f'seed_{seed}'
            meta=run/'metadata.json';check=run/'recipe_check.json'
            if meta.exists() and check.exists() and not read(check).get('reuse'):
                new_runs.append({'task':t,'seed':seed,'elapsed_seconds':read(meta)['elapsed_seconds'],
                                 'gpu':read(run/'status.json').get('gpu')})
    manifest={'stage':'CPU complete; GPU complete' if ready else 'CPU complete; GPU pending',
              'include_test':False,'tasks':TASKS,'pool_sizes':{t:analysis[t]['pool_size'] for t in TASKS},
              'packet':info,'new_gpu_runs':new_runs,
              'gpu_hours_used':sum(x['elapsed_seconds'] for x in new_runs)/3600,
              'gpus_used':sorted({str(x['gpu']) for x in new_runs if x['gpu'] is not None})}
    write('manifest.json',manifest)
    print(json.dumps({'analysis':{t:{'seed42':analysis[t]['seeds']['42'],'oracle_f1':analysis[t]['oracle_f1'],'rank_changed':analysis[t]['rank_changed']} for t in TASKS},'blind':{t:v['counts'] if v else None for t,v in joins.items()}},indent=2))
if __name__=='__main__':main()
