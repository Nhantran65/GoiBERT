#!/usr/bin/env python3
"""XLM-R-large baseline preflight: pinned manifest (sha256), clean loading info, parameter count, and a
smoke epoch on Restaurant.  Writes artifacts/xlmr_large_baseline/manifest.json + preflight.json."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke-batches", type=int, default=20)
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()
    started = time.time()
    report: dict = {"protocol": common.PROTOCOL + "-preflight", "created_at": common.now(), "checks": []}
    checks = report["checks"]

    def check(name: str, ok: bool, detail) -> bool:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"[preflight] {'OK  ' if ok else 'FAIL'} {name}: {detail}", flush=True)
        return bool(ok)

    import torch

    manifest = common.model_manifest("xlmr_large")
    common.ART.mkdir(parents=True, exist_ok=True)
    (common.ART / "manifest.json").write_text(json.dumps(
        {"protocol": common.PROTOCOL, "created_at": common.now(), "model": "xlmr_large", **manifest}, indent=1))
    report["manifest"] = manifest
    check("weights_sha256_recorded", bool(manifest["weights_sha256"]), manifest["weights_sha256"])
    report["environment"] = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                             "device": torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else "cpu",
                             "venv": sys.prefix}

    from goibert import train as e1_train

    bundle = e1_train.load_task_bundle("absa_restaurant", common.ROOT, include_test=False)
    model, spec = common.build_model("xlmr_large", num_classes=e1_train.task_num_classes(bundle),
                                     num_aspects=e1_train.task_num_aspects(bundle),
                                     aspect_class_counts=e1_train.task_aspect_class_counts(bundle))
    report["model_spec"] = spec
    dropped = list(spec["dropped_keys_on_load"])
    allowed_prefixes = ("lm_head", "decoder", "head")
    unexpected = [k for k in dropped if not (k.startswith(allowed_prefixes) or "pooler" in k)]
    check("loading_info_clean", not unexpected,
          {"dropped_keys_on_load": dropped, "unexpected_beyond_lm_head_and_pooler": unexpected})
    check("architecture_24L_hidden1024", spec["num_hidden_layers"] == 24 and spec["hidden_size"] == 1024,
          {"num_hidden_layers": spec["num_hidden_layers"], "hidden_size": spec["hidden_size"]})
    check("parameters_recorded", spec["encoder_parameters"] > 0,
          {"encoder": spec["encoder_parameters"], "non_embedding": spec["encoder_non_embedding_parameters"],
           "embedding": spec["embedding_parameters"], "total_with_head": spec["total_parameters_with_head"], "vocab": spec["vocab_size"]})
    head_params = sum(p.numel() for p in model.head.parameters())
    check("shared_e14_head_4_layers_fresh", head_params > 0, {"head_parameters": head_params})
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not args.skip_smoke:
        out_dir = common.ART / "smoke"
        command = [sys.executable, str(Path(__file__).resolve().parent / "finetune.py"), "--model", "xlmr_large",
                   "--task", "absa_restaurant", "--lr", "2e-5", "--seed", "42", "--device", args.device,
                   "--micro-batch", "32", "--smoke", str(args.smoke_batches), "--out-dir", str(out_dir)]
        log = common.ART / "logs" / "preflight_smoke.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as fh:
            completed = subprocess.run(command, stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ))
        meta = json.loads((out_dir / "metadata.json").read_text()) if (out_dir / "metadata.json").exists() else {}
        report["smoke"] = {"returncode": completed.returncode, "status": meta.get("status"), "log": str(log),
                           "best_selection_f1": meta.get("best_selection_f1"),
                           "peak_gpu_memory_allocated_bytes": meta.get("peak_gpu_memory_allocated_bytes"),
                           "reload_matches_best": meta.get("postrun_checks", {}).get("reload_matches_best")}
        check("smoke_1_epoch_completed", completed.returncode == 0 and str(meta.get("status", "")).startswith("completed"),
              report["smoke"])
    report["elapsed_seconds"] = time.time() - started
    report["passed"] = all(c["ok"] for c in checks)
    (common.ART / "preflight.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    print(f"[preflight] {'PASSED' if report['passed'] else 'FAILED'} ({sum(c['ok'] for c in checks)}/{len(checks)} checks)", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
