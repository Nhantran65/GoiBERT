"""Mechanism battery for candidate C1 (soft-multiplicity-normalised attention), inference only.

No training, no optimizer step, no test split, no modification of E1-E17 artifacts.
Checkpoints: E8 seed 42 (student) and E14 selected LR (BamiBERT) for the three ABSA tasks.

0a  multiplicity structure of pretrained keys in the target layers (train rows only) +
    exact-duplicate-text probe (does a verbatim repeated clause look like a duplicate in key space?)
0b  duplication probe on validation single-aspect rows: A + B(+B+B): how often does A's predicted
    polarity flip to B's polarity as B is repeated, under baseline / C1 (rho=0.5,1) / permuted control.
parity: manual FP32 attention path with rho=0 vs native SDPA path on full validation (F1 and logits).
"""
from __future__ import annotations

import argparse, hashlib, json, math, random, sys, time, types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from goibert import train as tr  # noqa: E402
from goibert import data as data_mod  # noqa: E402
from goibert_e2.modeling import build_e2_student  # noqa: E402
from goibert_e14.modeling import build_bamibert_classifier  # noqa: E402

OUT = ROOT / "artifacts" / "independent_review" / "battery"
TASKS = ("visfd", "absa_restaurant", "absa_hotel")
MODELS = {
    "student_e8": {"dir": ROOT / "models" / "goiturn-s-gd", "layers": (7, 8, 9),
                   "ckpt": lambda t: ROOT / "artifacts" / "e8" / t / "seed_42" / "best_checkpoint.pt"},
    "bami_e14": {"dir": ROOT / "models" / "bamibert-base-original", "layers": (8, 9, 10, 11),
                 "ckpt": lambda t: ROOT / "artifacts" / "e14" / "trials" / t / "lr_5e-05" / "seed_42" / "best_checkpoint.pt"},
}
TAUS = (4.0, 8.0)
MAX_LEN = 256


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------
# C1 attention wrapper (inference): manual FP32 attention with optional key-multiplicity bias
# ----------------------------------------------------------------------------------------------
class Battery:
    def __init__(self, model: Any, layers: tuple[int, ...], special_ids: list[int]):
        self.model = model
        self.layers = layers
        self.special_ids = special_ids
        self.rho = 0.0
        self.tau = 8.0
        self.center = True
        self.perm = False            # permuted-multiplicity control
        self.collect = False         # collect m~ / key stats
        self.stats: list[dict[str, Any]] = []
        self.mask2d: torch.Tensor | None = None
        self.input_ids: torch.Tensor | None = None
        self.gen = torch.Generator(device="cpu").manual_seed(12345)
        self._originals = {}
        for idx in layers:
            sa = model.encoder.encoder.layer[idx].attention.self
            self._originals[idx] = sa.forward
            sa.forward = types.MethodType(self._make_forward(idx), sa)

    def restore(self):
        for idx, fwd in self._originals.items():
            self.model.encoder.encoder.layer[idx].attention.self.forward = fwd

    def _make_forward(self, idx: int):
        battery = self

        def forward(sa, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                    encoder_attention_mask=None, past_key_value=None, output_attentions=False):
            B, N, _ = hidden_states.shape
            H, d = sa.num_attention_heads, sa.attention_head_size
            with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                x = hidden_states.float()
                q = sa.query(x).view(B, N, H, d).transpose(1, 2)
                k = sa.key(x).view(B, N, H, d).transpose(1, 2)
                v = sa.value(x).view(B, N, H, d).transpose(1, 2)
                valid = battery.mask2d.to(hidden_states.device).bool()          # B,N
                vf = valid.float()
                scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)       # B,H,N,N
                bias = None
                if battery.rho != 0.0 or battery.collect:
                    kk = k
                    if battery.center:
                        mean = (kk * vf[:, None, :, None]).sum(2, keepdim=True) / vf.sum(1)[:, None, None, None].clamp(min=1)
                        kk = kk - mean
                    kn = kk / (kk.norm(dim=-1, keepdim=True) + 1e-6)
                    C = torch.matmul(kn, kn.transpose(-1, -2))                     # B,H,N,N cosine
                    eye = torch.eye(N, device=C.device, dtype=torch.bool)[None, None]
                    ker = torch.exp(battery.tau * (C - 1.0)) * vf[:, None, None, :]
                    ker = ker.masked_fill(eye, 0.0)
                    m = 1.0 + ker.sum(-1)                                           # B,H,N soft multiplicity
                    if battery.collect:
                        battery.stats.append({"layer": idx, "m": m.detach().cpu(), "valid": valid.cpu(),
                                              "ids": battery.input_ids.cpu(), "kn": kn.detach().cpu()})
                    if battery.rho != 0.0:
                        logm = torch.log(m)
                        if battery.perm:
                            # permute m~ among valid positions of each row (same values, wrong tokens)
                            rnd = torch.rand(B, N, generator=battery.gen).to(logm.device)
                            rnd = rnd.masked_fill(~valid, 2.0)                       # invalid last
                            order = rnd.argsort(dim=1)                             # B,N: valid positions first (random order)
                            nvalid = valid.sum(1)
                            # positions to fill: the sorted valid positions; values: logm at a shuffled valid set
                            logm_p = logm.clone()
                            for b in range(B):
                                pos = order[b, : nvalid[b]]
                                src = pos[torch.randperm(int(nvalid[b]), generator=battery.gen).to(pos.device)]
                                logm_p[b, :, pos] = logm[b, :, src]
                            logm = logm_p
                        bias = -battery.rho * logm                                  # B,H,N (per key)
                        scores = scores + bias[:, :, None, :]
                scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
                A = torch.softmax(scores, dim=-1)
                ctx = torch.matmul(A, v)                                            # B,H,N,d
                ctx = ctx.transpose(1, 2).reshape(B, N, H * d)
            return (ctx.to(hidden_states.dtype),)
        return forward


# ----------------------------------------------------------------------------------------------
def load_model(kind: str, task: str, device: torch.device):
    spec = MODELS[kind]
    bundle = tr.load_task_bundle(task, ROOT, include_test=False)
    num_aspects = tr.task_num_aspects(bundle)
    counts = tr.task_aspect_class_counts(bundle) if num_aspects else None
    if kind == "student_e8":
        model, _ = build_e2_student(spec["dir"], num_classes=tr.task_num_classes(bundle), num_aspects=num_aspects,
                                    aspect_class_counts=counts, attention_implementation="sdpa")
    else:
        model, _ = build_bamibert_classifier(spec["dir"], num_classes=tr.task_num_classes(bundle), num_aspects=num_aspects,
                                             aspect_class_counts=counts, attention_implementation="sdpa")
    ckpt = spec["ckpt"](task)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(spec["dir"], local_files_only=True, use_fast=True)
    return model, tok, bundle, counts, ckpt


@torch.inference_mode()
def forward_texts(model, tok, texts: list[str], device, battery: Battery | None, batch: int = 32):
    outs = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        ids = enc["input_ids"].to(device); am = enc["attention_mask"].to(device)
        if battery is not None:
            battery.mask2d = am; battery.input_ids = ids
        with tr.make_autocast("bf16", device):
            logits = model(ids, am)
        outs.append(logits.detach().float().cpu())
    return torch.cat(outs, 0)


def predictions(logits: torch.Tensor, counts) -> list[list[int]]:
    return tr._predictions(logits, aspect_class_counts=counts or ())


# ----------------------------------------------------------------------------------------------
def scored_aspects(task: str, bundle) -> list[int]:
    K = tr.task_num_aspects(bundle)
    return [i for i in range(K) if not (task == "visfd" and i == data_mod.VISFD_OTHERS_ASPECT_INDEX)]


def run_parity(model, tok, bundle, counts, task, device, battery: Battery, saved_pred_path: Path) -> dict[str, Any]:
    split = tr.split_field(bundle, "validation")
    texts = list(tr.split_texts(split)); ids = list(tr.split_ids(split))
    gold_model = tr.model_labels(bundle, "validation", split)
    battery.rho = 0.0
    lg_manual = forward_texts(model, tok, texts, device, battery)
    battery.restore()
    lg_native = forward_texts(model, tok, texts, device, None)
    # re-install
    battery.__init__(model, battery.layers, battery.special_ids)
    pm = predictions(lg_manual, counts); pn = predictions(lg_native, counts)
    f1_m = tr.evaluate_predictions(task, bundle, gold_model, pm, split="validation")["selection_f1"]
    f1_n = tr.evaluate_predictions(task, bundle, gold_model, pn, split="validation")["selection_f1"]
    saved = {r["id"]: r for r in json.load(open(saved_pred_path))["rows"]}
    # compare native predictions with saved artifact predictions (model codes)
    agree = sum(1 for i, p in zip(ids, pn) if list(saved[i]["prediction_model"] if "prediction_model" in saved[i] else saved[i]["prediction"]) == list(p)) if task != "visfd" else None
    return {"f1_manual_rho0": float(f1_m), "f1_native_sdpa": float(f1_n),
            "max_abs_logit_diff": float((lg_manual - lg_native).abs().max()),
            "mean_abs_logit_diff": float((lg_manual - lg_native).abs().mean()),
            "pred_rows_differ_manual_vs_native": int(sum(1 for a, b in zip(pm, pn) if list(a) != list(b))),
            "native_pred_rows_equal_saved_artifact": agree, "n_rows": len(texts)}


def run_0a(model, tok, bundle, task, device, battery: Battery, n_rows: int, seed: int) -> dict[str, Any]:
    split = tr.split_field(bundle, "train")
    texts = list(tr.split_texts(split))
    rng = random.Random(seed); idxs = rng.sample(range(len(texts)), min(n_rows, len(texts)))
    sub = [texts[i] for i in idxs]
    special = torch.tensor(battery.special_ids)
    result = {}
    for tau in TAUS:
        battery.tau = tau; battery.rho = 0.0; battery.collect = True; battery.stats = []
        forward_texts(model, tok, sub, device, battery)
        battery.collect = False
        per_layer = {}
        for idx in battery.layers:
            logs = []  # per head list of tensors
            for s in battery.stats:
                if s["layer"] != idx: continue
                content = s["valid"] & ~torch.isin(s["ids"], special)
                m = s["m"]                                        # B,H,N
                for h in range(m.shape[1]):
                    vals = m[:, h][content]
                    logs.append((h, torch.log(vals)))
            H = max(h for h, _ in logs) + 1
            heads = {}
            for h in range(H):
                x = torch.cat([t for hh, t in logs if hh == h])
                q = torch.quantile(x, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
                heads[h] = {"median_m": float(math.exp(q[2])), "iqr_logm": float(q[3] - q[1]),
                            "p10_m": float(math.exp(q[0])), "p90_m": float(math.exp(q[4])), "frac_m_ge_1.5": float((x >= math.log(1.5)).float().mean())}
            per_layer[idx] = heads
        result[f"tau{tau:g}"] = per_layer
        battery.stats = []
    # exact-duplicate probe: text + ' ' + text ; cosine between key of token t in copy1 and copy2
    dup_texts = [t + " " + t for t in sub[:64]]
    battery.tau = 8.0; battery.collect = True; battery.stats = []
    forward_texts(model, tok, dup_texts, device, battery, batch=16)
    battery.collect = False
    dup = {}
    for idx in battery.layers:
        per_head_cos = []; per_head_m = []
        for s in battery.stats:
            if s["layer"] != idx: continue
            ids = s["ids"]; kn = s["kn"]; m = s["m"]; valid = s["valid"]
            for b in range(ids.shape[0]):
                seq = ids[b][valid[b]].tolist()
                body = seq[1:-1]
                L = len(body)
                # byte-level BPE may tokenise the boundary token of the second copy differently
                # (leading-space variant); compare the copies excluding each copy's first token.
                half = None
                for cand in (L // 2, (L + 1) // 2, L // 2 - 1):
                    if cand >= 2 and 2 * cand <= L + 1 and body[1:cand] == body[cand + 1:2 * cand]:
                        half = cand; break
                if half is None: continue
                p1 = torch.arange(2, 1 + half); p2 = p1 + half   # positions in full sequence (offset 1 for <s>), skip first token of copy
                cos = (kn[b, :, p1, :] * kn[b, :, p2, :]).sum(-1)          # H,half
                per_head_cos.append(cos); per_head_m.append(m[b][:, p1])
        if per_head_cos:
            c = torch.cat(per_head_cos, 1); mm = torch.cat(per_head_m, 1)
            dup[idx] = {"n_tokens": int(c.shape[1]), "mean_cos_copy1_copy2_per_head": [float(x) for x in c.mean(1)],
                        "median_m_copy1_per_head_tau8": [float(x) for x in mm.median(1).values]}
    battery.stats = []
    return {"n_rows": len(sub), "multiplicity": result, "duplicate_text_probe": dup}


def build_pairs(task, bundle, scored, tok, n_pairs, seed, max_tok=48):
    split = tr.split_field(bundle, "validation")
    texts = list(tr.split_texts(split)); ids = list(tr.split_ids(split)); labels = list(tr.split_labels(split))
    singles = []
    for i, (t, lab) in enumerate(zip(texts, labels)):
        present = [a for a in scored if lab[a] > 0]
        if len(present) != 1: continue
        a = present[0]
        if lab[a] not in (1, 2): continue          # positive / negative only for the A/B roles
        if task == "visfd" and lab[data_mod.VISFD_OTHERS_ASPECT_INDEX] > 0: continue
        n = len(tok(t, truncation=True, max_length=MAX_LEN)["input_ids"])
        if n > max_tok: continue
        singles.append({"row": i, "id": ids[i], "text": t, "aspect": a, "pol": int(lab[a])})
    rng = random.Random(seed)
    pairs = []
    tries = 0
    while len(pairs) < n_pairs and tries < 20000:
        tries += 1
        A, B = rng.sample(singles, 2)
        if A["aspect"] == B["aspect"] or A["pol"] == B["pol"]: continue
        pairs.append((A, B))
    return singles, pairs


def run_0b(model, tok, counts, task, scored, device, battery: Battery, pairs, conditions) -> dict[str, Any]:
    reps = (0, 1, 2, 3)   # number of B copies
    orders = ("A_first", "B_first")
    texts = []; meta = []
    for pi, (A, B) in enumerate(pairs):
        for order in orders:
            for r in reps:
                parts = [A["text"]] + [B["text"]] * r if order == "A_first" else [B["text"]] * r + [A["text"]]
                texts.append(" ".join(parts)); meta.append((pi, order, r))
    out = {}
    for name, cfg in conditions.items():
        battery.rho = cfg["rho"]; battery.tau = cfg.get("tau", 8.0); battery.perm = cfg.get("perm", False); battery.center = True
        battery.gen.manual_seed(777)
        lg = forward_texts(model, tok, texts, device, battery, batch=32)
        preds = predictions(lg, counts)
        agg = {}
        for (pi, order, r), p in zip(meta, preds):
            A, B = pairs[pi]
            key = (order, r)
            d = agg.setdefault(key, {"n": 0, "A_correct": 0, "A_flip_to_B": 0, "A_other_pol": 0, "A_missed": 0, "B_present_correct": 0, "B_present_wrong": 0})
            d["n"] += 1
            pa = p[A["aspect"]]
            if pa == A["pol"]: d["A_correct"] += 1
            elif pa == 0: d["A_missed"] += 1
            elif pa == B["pol"]: d["A_flip_to_B"] += 1
            else: d["A_other_pol"] += 1
            if r > 0:
                pb = p[B["aspect"]]
                if pb == B["pol"]: d["B_present_correct"] += 1
                elif pb > 0: d["B_present_wrong"] += 1
        out[name] = {f"{o}|B_copies={r}": v for (o, r), v in sorted(agg.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--n-rows-0a", type=int, default=512)
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    conditions = {"baseline_rho0": {"rho": 0.0}, "c1_rho0.5_tau8": {"rho": 0.5}, "c1_rho1_tau8": {"rho": 1.0},
                  "c1_rho1_tau4": {"rho": 1.0, "tau": 4.0}, "perm_rho1_tau8": {"rho": 1.0, "perm": True}}
    for kind in args.models:
        for task in args.tasks:
            t0 = time.time()
            model, tok, bundle, counts, ckpt = load_model(kind, task, device)
            scored = scored_aspects(task, bundle)
            special = list(tok.all_special_ids)
            battery = Battery(model, MODELS[kind]["layers"], special)
            saved_pred = (ROOT / "artifacts" / "e8" / task / "seed_42" / "validation_predictions.json") if kind == "student_e8" else (ROOT / "artifacts" / "e14" / "trials" / task / "lr_5e-05" / "seed_42" / "validation_predictions.json")
            report: dict[str, Any] = {"model": kind, "task": task, "checkpoint": str(ckpt.relative_to(ROOT)), "checkpoint_sha256": sha256(ckpt),
                                      "layers": list(MODELS[kind]["layers"]), "device": str(device), "seed": args.seed, "include_test": False}
            report["parity"] = run_parity(model, tok, bundle, counts, task, device, battery, saved_pred)
            report["step0a"] = run_0a(model, tok, bundle, task, device, battery, args.n_rows_0a, args.seed)
            singles, pairs = build_pairs(task, bundle, scored, tok, args.n_pairs, args.seed)
            report["step0b_pairs"] = {"n_single_aspect_rows_eligible": len(singles), "n_pairs": len(pairs),
                                      "pair_ids": [(A["id"], B["id"]) for A, B in pairs]}
            report["step0b"] = run_0b(model, tok, counts, task, scored, device, battery, pairs, conditions)
            report["elapsed_s"] = time.time() - t0
            battery.restore()
            path = OUT / f"{kind}__{task}.json"
            json.dump(report, open(path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
            print(f"[done] {kind} {task} parity={report['parity']} elapsed={report['elapsed_s']:.0f}s -> {path}", flush=True)
            del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
