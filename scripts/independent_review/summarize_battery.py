import json, glob, os, statistics as st
ROOT='/home/kiettqa/GoiBERT/artifacts/independent_review/battery'
for path in sorted(glob.glob(f'{ROOT}/*.json')):
    r=json.load(open(path)); print('='*110); print(os.path.basename(path), '| parity:', {k:(round(v,4) if isinstance(v,float) else v) for k,v in r['parity'].items()})
    a=r['step0a']
    for tau in ('tau4','tau8'):
        line=f'  0a {tau}:'
        for layer,heads in a['multiplicity'][tau].items():
            med=[h['median_m'] for h in heads.values()]; iqr=[h['iqr_logm'] for h in heads.values()]; fr=[h['frac_m_ge_1.5'] for h in heads.values()]
            line+=f' L{layer}: med_m={min(med):.2f}-{max(med):.2f} iqr_logm={min(iqr):.2f}-{max(iqr):.2f} frac>=1.5={min(fr):.2f}-{max(fr):.2f} |'
        print(line)
    for layer,d in a['duplicate_text_probe'].items():
        print(f'  dup-probe L{layer}: n_tok={d["n_tokens"]} cos(copy1,copy2)/head={[round(x,2) for x in d["mean_cos_copy1_copy2_per_head"]]} median_m(copy1)/head={[round(x,2) for x in d["median_m_copy1_per_head_tau8"]]}')
    print(f'  0b pairs={r["step0b_pairs"]["n_pairs"]} (eligible singles={r["step0b_pairs"]["n_single_aspect_rows_eligible"]})')
    for cond,res in r['step0b'].items():
        for order in ('A_first','B_first'):
            row=[]
            for c in (0,1,2,3):
                d=res[f'{order}|B_copies={c}']; n=d['n']
                row.append(f'c{c}: flip={100*d["A_flip_to_B"]/n:4.1f}% ok={100*d["A_correct"]/n:4.1f}% miss={100*d["A_missed"]/n:4.1f}% oth={100*d["A_other_pol"]/n:4.1f}%')
            print(f'  {cond:16s} {order:8s} ' + ' | '.join(row))
