"""Quick probe: fine-tune an off-the-shelf HF encoder (default jhu-clsp/mmBERT-base) on one GoiBERT task with the
plain E14-style recipe (CE, AdamW, warmup 10%, first-token pooling head), validation-only.  Purpose: is a modern
multilingual encoder already competitive with BamiBERT/E19 on Vietnamese ABSA before any Vietnamese continued
pretraining?  Not a locked protocol; no test split.
"""
from __future__ import annotations

import argparse, json, math, sys, time, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from goibert import train as tr  # noqa: E402

OUT = ROOT / "artifacts" / "research_diagnostics" / "mmbert_probe"
_MARKS = set("\u0300\u0301\u0303\u0309\u0323\u0302\u0306\u031b")


_TMAP = None


def normalize_teencode(s):
    """Replace teencode/abbreviation words by their canonical form using data/cpt/teencode_map.json (ViLexNorm-derived):
    'ko đc' -> 'không được'."""
    global _TMAP
    import re
    if _TMAP is None: _TMAP = json.load(open(ROOT / "data" / "cpt" / "teencode_map.json"))
    return re.sub(r"[^\W\d_]+", lambda m: _TMAP.get(m.group(0).lower(), m.group(0)), s)


def strip_diacritics(s):
    """Remove all Vietnamese diacritics (tone + vowel marks, đ->d): 'phòng sạch' -> 'phong sach'."""
    return unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D")) if c not in _MARKS))


class Head(nn.Module):
    def __init__(self, d, out):
        super().__init__(); self.dense = nn.Linear(d, d); self.out = nn.Linear(d, out); self.drop = nn.Dropout(0.1)
    def forward(self, h):
        return self.out(self.drop(torch.tanh(self.dense(self.drop(h[:, 0])))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--task", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None, help="output dir override (default artifacts/research_diagnostics/mmbert_probe)")
    ap.add_argument("--eval-strip", action="store_true", help="also evaluate each epoch on the diacritic-stripped validation set")
    ap.add_argument("--eval-norm", action="store_true", help="also evaluate each epoch on the teencode-normalized validation set")
    ap.add_argument("--save-preds", action="store_true", help="save validation predictions (clean and stripped) at the best clean epoch")
    ap.add_argument("--tokenizer", default=None, help="tokenizer repo override (e.g. FacebookAI/xlm-roberta-large for uitnlp/CafeBERT, whose repo lacks tokenizer.json; same sentencepiece model)")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device); out_dir = Path(args.out) if args.out else OUT; out_dir.mkdir(parents=True, exist_ok=True)
    bundle = tr.load_task_bundle(args.task, ROOT, include_test=False)
    A = tr.task_num_aspects(bundle); counts = tr.task_aspect_class_counts(bundle) if A else None
    num_labels = tr.task_num_classes(bundle)
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    enc = AutoModel.from_pretrained(args.model)
    model = nn.ModuleDict({"enc": enc, "head": Head(enc.config.hidden_size, num_labels)}).to(device)
    tr_split = tr.split_field(bundle, "train"); va_split = tr.split_field(bundle, "validation")
    tr_texts = list(tr.split_texts(tr_split)); va_texts = list(tr.split_texts(va_split)); va_strip = [strip_diacritics(t) for t in va_texts]; va_norm = [normalize_teencode(t) for t in va_texts]
    Ytr = torch.tensor(tr.model_labels(bundle, "train", tr_split)); gold_va = tr.model_labels(bundle, "validation", va_split)
    accum = 32 // args.micro; steps_per_epoch = math.ceil(len(tr_texts) / args.micro / accum)
    opt = torch.optim.AdamW([p for p in model.parameters()], lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, round(0.1 * steps_per_epoch * args.epochs), steps_per_epoch * args.epochs)
    best = (-1.0, 0); curves = []; t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); perm = torch.randperm(len(tr_texts)); opt.zero_grad()
        for bi, i in enumerate(range(0, len(tr_texts), args.micro)):
            idx = perm[i:i + args.micro].tolist()
            e = tok([tr_texts[j] for j in idx], padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model["enc"](input_ids=e["input_ids"].to(device), attention_mask=e["attention_mask"].to(device)).last_hidden_state
                logits = model["head"](h.float())
            if A: logits = logits.view(-1, A, counts[0]) if len(set(counts)) == 1 else logits
            loss = tr.compute_loss(logits.float(), Ytr[idx].to(device), num_aspects=A, aspect_class_counts=counts) / accum
            loss.backward()
            if (bi + 1) % accum == 0 or i + args.micro >= len(tr_texts):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad()
        model.eval()
        def evaluate(texts, keep=None):
            preds = []
            with torch.no_grad():
                for i in range(0, len(texts), 64):
                    e = tok(texts[i:i + 64], padding=True, truncation=True, max_length=256, return_tensors="pt")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        h = model["enc"](input_ids=e["input_ids"].to(device), attention_mask=e["attention_mask"].to(device)).last_hidden_state
                        lg = model["head"](h.float()).float().cpu()
                    if A and len(set(counts)) == 1: lg = lg.view(-1, A, counts[0])
                    preds += tr._predictions(lg, aspect_class_counts=counts or ())
            if keep is not None: keep.append(preds)
            return float(tr.evaluate_predictions(args.task, bundle, gold_va, preds, split="validation")["selection_f1"])
        kept = []
        f1 = evaluate(va_texts, kept); rec = {"epoch": ep, "f1": round(100 * f1, 3), "elapsed": round(time.time() - t0)}
        if args.eval_strip: rec["f1_strip"] = round(100 * evaluate(va_strip, kept), 3)
        if args.eval_norm: rec["f1_norm"] = round(100 * evaluate(va_norm), 3)
        curves.append(rec)
        if f1 > best[0]:
            best = (f1, ep)
            if args.save_preds:
                json.dump({"epoch": ep, "gold": [list(map(int, g)) if hasattr(g, "__len__") else int(g) for g in gold_va], "pred_clean": [list(map(int, x)) if hasattr(x, "__len__") else int(x) for x in kept[0]],
                           "pred_strip": [list(map(int, x)) if hasattr(x, "__len__") else int(x) for x in kept[1]] if len(kept) > 1 else None, "texts": va_texts},
                          open(out_dir / f"{args.task}_{args.model.rstrip('/').split('/')[-1]}_lr{args.lr:g}_s{args.seed}{args.tag}.preds.json", "w"))
        extra = (f" strip {rec['f1_strip']:.2f}" if args.eval_strip else "") + (f" norm {rec['f1_norm']:.2f}" if args.eval_norm else "")
        print(f"{args.task} {args.model} lr={args.lr} epoch {ep} f1 {100*f1:.2f}{extra} best {100*best[0]:.2f}@{best[1]} ({round(time.time()-t0)}s)", flush=True)
        summary = {"args": vars(args), "best_f1": round(100 * best[0], 3), "best_epoch": best[1], "curves": curves}
        if args.eval_strip: summary["strip_f1_at_best"] = curves[best[1] - 1]["f1_strip"]
        if args.eval_norm: summary["norm_f1_at_best"] = curves[best[1] - 1]["f1_norm"]
        json.dump(summary, open(out_dir / f"{args.task}_{args.model.rstrip('/').split('/')[-1]}_lr{args.lr:g}_s{args.seed}{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
