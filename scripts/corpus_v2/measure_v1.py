"""CPU Bami token census of v1, partitioned at byte boundaries."""
import argparse
import json
import mmap
import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def worker(job):
    start, end = job
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / 'models/goiturn-s-gd'), local_files_only=True)
    tok.model_max_length = 10**9
    n = lines = 0
    with (ROOT / 'data/corpus_v1/mlm_mix.txt').open('rb') as fh:
        fh.seek(start)
        if start:
            fh.readline()
        while fh.tell() < end:
            b = fh.readline()
            if not b:
                break
            line = b.decode('utf-8').strip()
            if line:
                n += len(tok.encode(line, add_special_tokens=False))
                lines += 1
    return n, lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=32)
    a = p.parse_args()
    size = (ROOT / 'data/corpus_v1/mlm_mix.txt').stat().st_size
    jobs = [(size*i//a.workers, size*(i+1)//a.workers) for i in range(a.workers)]
    with mp.Pool(a.workers) as pool:
        result = pool.map(worker, jobs)
    out = {'bami_tokens': sum(x[0] for x in result), 'physical_nonempty_lines': sum(x[1] for x in result), 'workers': a.workers, 'parts': result}
    dest = ROOT / 'data/corpus_v2/v1_measurement.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='parts'}), flush=True)


if __name__ == '__main__':
    main()
