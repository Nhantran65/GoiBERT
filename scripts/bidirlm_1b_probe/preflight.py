#!/usr/bin/env python3
"""BidirLM-1B preflight: strict load, native attention, tokenizer, capacity, one epoch."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import finetune as FT  # noqa: E402
from goibert import train  # noqa: E402

PHASE2 = C.ROOT / "artifacts" / "bidirlm_1b_probe" / "phase2"
OUT = PHASE2 / "preflight.json"


def save(report):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n")


def profile_capacity(model, tokenizer, device):
    """Find largest micro batch divisor of 32 with a full first AdamW step under 40 GB."""
    seed_ids = tokenizer.encode("Nhà hàng phục vụ món ăn ngon và nhân viên thân thiện. ", add_special_tokens=False)
    ids = (seed_ids * (256 // len(seed_ids) + 1))[:256]
    profiles, selected = [], None
    model.train().to(device)
    for micro in (32, 16, 8, 4, 2, 1):
        x = mask = opt = logits = loss = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            x = torch.tensor([ids] * micro, dtype=torch.long, device=device)
            mask = torch.ones_like(x)
            opt = train.make_optimizer(model, 1e-5)
            opt.zero_grad(set_to_none=True)
            with train.make_autocast("bf16", device):
                logits = model(x, mask)
                loss = logits.float().square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            peak = int(torch.cuda.max_memory_allocated(device))
            profiles.append({"micro_batch": micro, "peak_allocated_bytes": peak, "finite": bool(torch.isfinite(loss).item()),
                             "under_40gb": peak <= 40_000_000_000})
            print(f"[capacity] micro {micro}: {peak/1e9:.3f} GB", flush=True)
            if profiles[-1]["finite"] and profiles[-1]["under_40gb"]:
                selected = micro
                break
        except torch.cuda.OutOfMemoryError as error:
            profiles.append({"micro_batch": micro, "error": "CUDA OOM", "detail": str(error)[:300]})
            print(f"[capacity] micro {micro}: CUDA OOM", flush=True)
        finally:
            model.zero_grad(set_to_none=True)
            loss = logits = opt = mask = x = None
            torch.cuda.empty_cache()
    return selected, profiles


def main():
    start = time.time()
    gate_path = PHASE2.parent / "phase1" / "gate.json"
    gate = json.loads(gate_path.read_text())
    if gate["decision"] != "PROCEED":
        raise RuntimeError("phase1 gate is not PROCEED")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible GPU required")
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    if free < 20 * (1 << 30):
        raise RuntimeError(f"phase2 needs >=20 GiB free, found {free/(1<<30):.3f}")
    report = {"protocol": "bidirlm-1b-probe-preflight-v1", "started_at": C.now(),
              "gpu_original_id": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_name": torch.cuda.get_device_name(device),
              "free_bytes_at_launch": free, "total_bytes": total,
              "phase1_gate_sha256": C.sha256_file(gate_path), "checks": []}

    def check(name, condition, detail):
        report["checks"].append({"name": name, "pass": bool(condition), "detail": detail})
        print(f"[preflight] {name}: {'PASS' if condition else 'FAIL'} {detail}", flush=True)

    try:
        report["model_manifest"] = C.model_manifest("bidirlm_1b")
        tokenizer = C.load_tokenizer("B1B")
        train.seed_everything(C.SEED)
        restaurant = train.load_task_bundle("absa_restaurant", C.ROOT, include_test=False)
        model, spec = C.build_model("B1B", num_classes=train.task_num_classes(restaurant),
                                   num_aspects=train.task_num_aspects(restaurant),
                                   aspect_class_counts=train.task_aspect_class_counts(restaurant))
        report["model_spec"] = spec
        check("strict_model_load", not spec["dropped_keys_on_load"],
              {"missing": [], "unexpected": spec["dropped_keys_on_load"], "mismatched": []})
        check("encoder_at_most_1b", spec["encoder_parameters"] <= 1_000_000_000,
              {k: spec[k] for k in ("encoder_parameters", "embedding_parameters", "encoder_non_embedding_parameters", "total_parameters_with_head")})
        report["head_init_sha256"] = FT._hash_params(model.head)
        check("head_hash", len(report["head_init_sha256"]) == 64, report["head_init_sha256"])

        example = "Tôi thích món ăn Việt Nam."
        special = tokenizer(example, add_special_tokens=True)["input_ids"]
        plain = tokenizer(example, add_special_tokens=False)["input_ids"]
        report["tokenizer"] = {"class": type(tokenizer).__name__, "length": len(tokenizer),
                               "bos_id": tokenizer.bos_token_id, "eos_id": tokenizer.eos_token_id,
                               "pad_id": tokenizer.pad_token_id, "with_special": special, "without_special": plain,
                               "model_prepends_bos_id": C.BOS_TOKEN_ID}
        check("tokenizer_bos_eos", tokenizer.bos_token_id == 2 and tokenizer.eos_token_id == 1
              and tokenizer.pad_token_id == 0 and special == plain and max(plain) < model.encoder.config.vocab_size,
              report["tokenizer"])

        lines = ["Nhà hàng này có món ăn rất ngon, giá hợp lý và nhân viên phục vụ chu đáo vào buổi tối.", "Quán khá ổn."]
        batch = tokenizer(lines, padding=True, add_special_tokens=True, return_tensors="pt")
        x, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        lengths = mask.sum(1).tolist()
        check("padded_batch", lengths[0] > lengths[1] and bool((mask[1] == 0).any()), lengths)
        model.to(device).eval()
        alt = tokenizer.encode("khác", add_special_tokens=False)[0]
        last = int(lengths[0])-1
        a, b, c = x.clone(), x.clone(), x.clone()
        a[0, last] = alt if int(x[0, last]) != alt else 123
        b[0, 0] = alt if int(x[0, 0]) != alt else 123
        c[mask == 0] = alt
        with torch.inference_mode():
            h0 = model.encoder(x, mask).last_hidden_state.clone()
            ha = model.encoder(a, mask).last_hidden_state
            hb = model.encoder(b, mask).last_hidden_state
            hc = model.encoder(c, mask).last_hidden_state
        backward = float((h0[0, 0]-ha[0, 0]).abs().max().item())
        forward = float((h0[0, last]-hb[0, last]).abs().max().item())
        pad_delta = float((h0[mask.bool()]-hc[mask.bool()]).abs().max().item())
        report["direction"] = {"last_to_first": backward, "first_to_last": forward, "pad_to_real": pad_delta}
        check("bidirectional_backward", backward > 1e-5, backward)
        check("bidirectional_forward", forward > 1e-5, forward)
        check("padding_invariance_fp32", pad_delta <= 1e-4, pad_delta)

        touched = C.set_attention_dropout(model.encoder, C.ATTENTION_DROPOUT)
        model.encoder.train()
        with torch.no_grad():
            train1 = model.encoder(x, mask).last_hidden_state
            train2 = model.encoder(x, mask).last_hidden_state
        model.encoder.eval()
        with torch.inference_mode():
            eval1 = model.encoder(x, mask).last_hidden_state
            eval2 = model.encoder(x, mask).last_hidden_state
        td = float((train1-train2).abs().max().item())
        ed = float((eval1-eval2).abs().max().item())
        report["attention_dropout"] = {"modules": touched, "configured": C.ATTENTION_DROPOUT,
                                       "train_max_abs": td, "eval_max_abs": ed}
        check("attention_dropout_train_eval", touched == model.encoder.config.num_hidden_layers and td > 1e-5 and ed <= 1e-4,
              report["attention_dropout"])

        lengths_by_task = {}
        for task in C.TASKS:
            bundle = restaurant if task == "absa_restaurant" else train.load_task_bundle(task, C.ROOT, include_test=False)
            texts = list(train.split_texts(train.split_field(bundle, "train")))
            lengths_by_task[task] = train.inspect_train_lengths(tokenizer, texts).to_dict()
        report["train_token_lengths"] = lengths_by_task
        check("train_truncation_rates", all(0 <= v["truncation_rate_256"] <= 1 for v in lengths_by_task.values()),
              {k: v["truncation_rate_256"] for k, v in lengths_by_task.items()})
        if not all(c["pass"] for c in report["checks"]):
            report["passed"] = False
            save(report)
            return 1
        del x, mask, a, b, c, h0, ha, hb, hc, train1, train2, eval1, eval2
        micro, profiles = profile_capacity(model, tokenizer, device)
        report["capacity_profiles"] = profiles
        report["selected_micro_batch"] = micro
        check("micro_batch_capacity", micro is not None, profiles)
        del model
        torch.cuda.empty_cache()
        if micro is None:
            report["passed"] = False
            save(report)
            return 1

        smoke_dir = PHASE2 / "preflight_smoke"
        log_path = PHASE2 / "logs" / "preflight_smoke.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(HERE / "finetune.py"), "--arm", "B1B", "--task", "absa_restaurant",
               "--lr", "1e-5", "--seed", "42", "--device", "cuda:0", "--micro-batch", str(micro),
               "--out-dir", str(smoke_dir), "--smoke-epoch"]
        with log_path.open("w") as log:
            result = subprocess.run(cmd, cwd=C.ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=False)
        report["smoke"] = {"command": cmd, "exit_code": result.returncode, "log": str(log_path)}
        if (smoke_dir / "metadata.json").exists():
            meta = json.loads((smoke_dir / "metadata.json").read_text())
            curve = json.loads((smoke_dir / "epoch_curves.json").read_text())["epochs"]
            report["smoke"].update({"status": meta["status"], "epochs": len(curve),
                                    "train_loss": curve[0]["train_loss"], "postrun_checks": meta["postrun_checks"]})
        check("restaurant_one_epoch_smoke", result.returncode == 0 and report["smoke"].get("status") == "completed"
              and report["smoke"].get("epochs") == 1 and np.isfinite(report["smoke"].get("train_loss", float("nan"))),
              report["smoke"])
        report["passed"] = all(c["pass"] for c in report["checks"])
        report["elapsed_seconds"] = time.time()-start
        report["finished_at"] = C.now()
        save(report)
        print(f"[preflight] {'PASS' if report['passed'] else 'FAIL'}", flush=True)
        return 0 if report["passed"] else 1
    except Exception as error:
        report.update({"passed": False, "exception": repr(error), "traceback": traceback.format_exc(),
                       "elapsed_seconds": time.time()-start})
        save(report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
