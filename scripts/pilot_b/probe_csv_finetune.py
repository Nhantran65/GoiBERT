"""Fine-tune probe for CSV single-label social-media tasks (ViHSD, ViSpamReviews, UIT-VSMEC), same recipe/head/JSON
format as scripts/research_checks/probe_mmbert_finetune.py; metric = macro-F1 on the dev split; --eval-strip scores the
diacritic-stripped dev text at every epoch. Validation-only, not a locked protocol."""
from __future__ import annotations
import argparse, json, math, sys, time, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd, torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
spec = importlib.util.spec_from_file_location("probe", ROOT / "scripts/research_checks/probe_mmbert_finetune.py"); probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
Head, strip_diacritics, normalize_teencode = probe.Head, probe.strip_diacritics, probe.normalize_teencode
SOC = ROOT / "data/raw/social"
TASKS = {  # name: (train csv, dev csv, text col, label col)
    "vihsd": (SOC / "vihsd/train.csv", SOC / "vihsd/dev.csv", "free_text", "label_id"),
    "vispam": (SOC / "vispamdetection/train.csv", SOC / "vispamdetection/dev.csv", "Comment", "Label"),
    "vispam4": (SOC / "vispamdetection/train.csv", SOC / "vispamdetection/dev.csv", "Comment", "SpamLabel"),
    "vsmec": (SOC / "uit-vsmec/train.csv", SOC / "uit-vsmec/valid.csv", "Sentence", "Emotion"),
}

def load(task):
    tr_p, va_p, tc, lc = TASKS[task]; tr, va = pd.read_csv(tr_p), pd.read_csv(va_p)
    tr, va = tr.dropna(subset=[tc, lc]), va.dropna(subset=[tc, lc])
    labels = sorted(tr[lc].unique().tolist()); lid = {l: i for i, l in enumerate(labels)}
    return tr[tc].astype(str).tolist(), [lid[l] for l in tr[lc]], va[tc].astype(str).tolist(), [lid[l] for l in va[lc]], len(labels)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tokenizer", default=None); ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--micro", type=int, default=8); ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--tag", default=""); ap.add_argument("--out", required=True); ap.add_argument("--eval-strip", action="store_true"); ap.add_argument("--eval-norm", action="store_true"); ap.add_argument("--norm-train", action="store_true", help="teencode-normalize the training text too")
    args = ap.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device); out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    tr_texts, tr_y, va_texts, va_y, C = load(args.task); va_strip = [strip_diacritics(t) for t in va_texts]; va_norm = [normalize_teencode(t) for t in va_texts]
    if args.norm_train: tr_texts = [normalize_teencode(t) for t in tr_texts]
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model); enc = AutoModel.from_pretrained(args.model)
    model = nn.ModuleDict({"enc": enc, "head": Head(enc.config.hidden_size, C)}).to(device)
    Ytr = torch.tensor(tr_y); accum = 32 // args.micro; steps_per_epoch = math.ceil(len(tr_texts) / args.micro / accum)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, round(0.1 * steps_per_epoch * args.epochs), steps_per_epoch * args.epochs)
    best = (-1.0, 0); curves = []; t0 = time.time()
    def evaluate(texts):
        preds = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                e = tok(texts[i:i + 64], padding=True, truncation=True, max_length=args.max_len, return_tensors="pt")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = model["enc"](input_ids=e["input_ids"].to(device), attention_mask=e["attention_mask"].to(device)).last_hidden_state
                preds += model["head"](h.float()).argmax(-1).cpu().tolist()
        return f1_score(va_y, preds, average="macro"), preds
    for ep in range(1, args.epochs + 1):
        model.train(); perm = torch.randperm(len(tr_texts)); opt.zero_grad()
        for bi, i in enumerate(range(0, len(tr_texts), args.micro)):
            idx = perm[i:i + args.micro].tolist()
            e = tok([tr_texts[j] for j in idx], padding=True, truncation=True, max_length=args.max_len, return_tensors="pt")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model["enc"](input_ids=e["input_ids"].to(device), attention_mask=e["attention_mask"].to(device)).last_hidden_state
            loss = nn.functional.cross_entropy(model["head"](h.float()), Ytr[idx].to(device)) / accum; loss.backward()
            if (bi + 1) % accum == 0 or i + args.micro >= len(tr_texts):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad()
        model.eval(); f1, _ = evaluate(va_texts); rec = {"epoch": ep, "f1": round(100 * f1, 3), "elapsed": round(time.time() - t0)}
        if args.eval_strip: rec["f1_strip"] = round(100 * evaluate(va_strip)[0], 3)
        if args.eval_norm: rec["f1_norm"] = round(100 * evaluate(va_norm)[0], 3)
        curves.append(rec)
        if f1 > best[0]: best = (f1, ep)
        print(f"{args.task} {args.model} lr={args.lr} epoch {ep} f1 {100*f1:.2f}" + (f" strip {rec['f1_strip']:.2f}" if args.eval_strip else "") + (f" norm {rec['f1_norm']:.2f}" if args.eval_norm else "") + f" best {100*best[0]:.2f}@{best[1]} ({round(time.time()-t0)}s)", flush=True)
        summary = {"args": vars(args), "best_f1": round(100 * best[0], 3), "best_epoch": best[1], "curves": curves}
        if args.eval_strip: summary["strip_f1_at_best"] = curves[best[1] - 1]["f1_strip"]
        if args.eval_norm: summary["norm_f1_at_best"] = curves[best[1] - 1]["f1_norm"]
        json.dump(summary, open(out_dir / f"{args.task}_{args.model.rstrip('/').split('/')[-1]}_lr{args.lr:g}_s{args.seed}{args.tag}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
