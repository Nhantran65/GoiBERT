#!/usr/bin/env python3
"""Score completed blinded responses; never run before human answers exist."""
import argparse
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
KEY=ROOT/'artifacts/label_ceiling_check/adjudication_packet/key.jsonl'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('filled',type=Path,help='Completed items.csv or items.jsonl')
    args=ap.parse_args()
    if args.filled.suffix=='.csv':
        with args.filled.open(newline='') as f:answers=list(csv.DictReader(f))
    else:answers=[json.loads(line) for line in args.filled.read_text().splitlines() if line.strip()]
    keys={x['uid']:x for x in map(json.loads,KEY.read_text().splitlines())}
    if len(answers)!=len(keys) or {x['uid'] for x in answers}!=set(keys):raise ValueError('UIDs incomplete, duplicated, or unknown')
    totals=defaultdict(Counter)
    for a in answers:
        k=keys[a['uid']];raw=str(a['human_label']).strip()
        if not raw:raise ValueError(f'missing answer {a["uid"]}')
        try:label=int(raw)
        except ValueError as e:raise ValueError(f'invalid label {a["uid"]}: {raw}') from e
        allowed={int(z.strip().split(' ')[0]) for z in a['valid_labels'].split(',')}
        if label not in allowed:raise ValueError(f'out-of-range label {a["uid"]}: {label}')
        for group in ((k['task'],k['group']),('ALL',k['group'])):
            v=totals[group];v['n']+=1;v['human_eq_gold']+=int(label==k['gold']);v['human_eq_consensus']+=int(k['consensus_label'] is not None and label==k['consensus_label'])
    result={f'{t}/{g}':{'n':c['n'],'human_eq_gold':c['human_eq_gold'],'human_eq_gold_rate':c['human_eq_gold']/c['n'],'human_eq_consensus':c['human_eq_consensus'],'human_eq_consensus_rate':c['human_eq_consensus']/c['n'] if g=='CE' else None} for (t,g),c in sorted(totals.items())}
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
