"""D1 adapter over scripts/cdbk_pilot: adds arms TD (CafeBERT-DAPT teacher) and KD_D (BamiBERT student distilled from TD)
without editing the M1 sources.  `install()` patches the M1 modules in-process:
  * ARMS gains TD (init cafebert_dapt, CE, lr 2e-5, head 1024) and KD_D (init bamibert via the E14 path, CE+KD, lr 5e-5).
  * common.build_model / load_tokenizer handle init "cafebert_dapt" (encoder from artifacts/cdbk_dapt_pilot/dapt/hf, same
    head class / RNG order as T) and delegate everything else to M1.
  * common.run_dir / CACHE_ROOT / CHECK_ROOT / source_hash and finetune.PROTOCOL point at the D1 artifact tree.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "cdbk_pilot"))
import common as m1  # noqa: E402
from common import ARMS, ATTENTION_IMPLEMENTATION, ROOT as M1_ROOT, SEED, TASK_ID, e1_train  # noqa: E402
from goibert_e14.modeling import BamiBERTClassifier  # noqa: E402
from transformers import XLMRobertaModel  # noqa: E402

assert M1_ROOT == ROOT
PROTOCOL_D1 = "CDBK-DAPT-pilot-D1-v1"
D1_ROOT = ROOT / "artifacts" / "cdbk_dapt_pilot"
DAPT_DIR = D1_ROOT / "dapt"
DAPT_HF_DIR = DAPT_DIR / "hf"
ARMS_D1: dict[str, dict[str, Any]] = {
    "TD": {"init": "cafebert_dapt", "learning_rate": 2e-5, "kd": False, "head_hidden": 1024},
    "KD_D": {"init": "bamibert", "learning_rate": 5e-5, "kd": True, "head_hidden": 768},
}

_m1_build_model = m1.build_model
_m1_load_tokenizer = m1.load_tokenizer
_m1_source_hash = m1.source_hash


def dapt_summary() -> dict[str, Any]:
    return m1.read_json(DAPT_DIR / "summary.json")


def build_model(init: str, contract: Mapping[str, Any]):
    if init != "cafebert_dapt":
        return _m1_build_model(init, contract)
    summary = dapt_summary()
    if summary.get("status") != "completed":
        raise RuntimeError("DAPT run is not completed; refusing to build TD from a partial export")
    weights = DAPT_HF_DIR / "model.safetensors"
    weight_hash = m1.sha256_file(weights)
    if weight_hash != summary["export"]["model_safetensors_sha256"]:
        raise RuntimeError("DAPT export sha256 differs from summary.json")
    encoder, info = XLMRobertaModel.from_pretrained(
        DAPT_HF_DIR, local_files_only=True, add_pooling_layer=False, attn_implementation=ATTENTION_IMPLEMENTATION, output_loading_info=True)
    unexpected = [k for k in info.get("unexpected_keys", []) if not k.startswith("lm_head")]
    if info.get("missing_keys") or unexpected or info.get("mismatched_keys"):
        raise RuntimeError(f"DAPT encoder did not load cleanly: {info}")
    config = encoder.config
    expected = {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096}
    actual = {name: int(getattr(config, name)) for name in expected}
    if actual != expected:
        raise RuntimeError(f"DAPT encoder architecture mismatch: {actual}")
    model = BamiBERTClassifier(
        encoder, num_labels=int(contract["num_classes"]), num_aspects=int(contract["num_aspects"]),
        aspect_class_counts=tuple(contract["aspect_class_counts"]), head_hidden_size=int(config.hidden_size))
    spec = model.spec().to_dict()
    spec.update({
        "backbone": "uitnlp/CafeBERT + D1 DAPT (ViSoBERT social corpus, 300M tokens)", "base_hf_commit": m1.TEACHER_HF_COMMIT,
        "dapt_hf_dir": str(DAPT_HF_DIR.relative_to(ROOT)), "weight_file_sha256": weight_hash, "encoder_class": type(encoder).__name__,
        "dapt_tokens_processed": summary.get("tokens_processed"), "dapt_holdout_loss": summary.get("holdout", {}).get("final"),
        "dropped_keys_on_load": [k for k in info.get("unexpected_keys", []) if k.startswith("lm_head")],
        "note": "fresh task head (goibert_e14 ClassificationHead, hidden 1024), same construction order as arm T; MLM head not carried",
    })
    return model, spec


def load_tokenizer(init: str):
    if init == "cafebert_dapt":
        tokenizer, manifest = _m1_load_tokenizer("cafebert")
        manifest = dict(manifest)
        manifest["note_d1"] = "same pinned XLM-R-large tokenizer as T; DAPT did not change vocab/embedding layout"
        return tokenizer, manifest
    return _m1_load_tokenizer(init)


def run_dir(arm: str) -> Path:
    return D1_ROOT / arm


def source_hash() -> str:
    digest = hashlib.sha256(_m1_source_hash().encode("ascii"))
    for path in sorted((ROOT / "scripts" / "cdbk_dapt_pilot").glob("*.py")):
        digest.update(path.name.encode()); digest.update(m1.sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def install() -> None:
    for arm, spec in ARMS_D1.items():
        ARMS[arm] = spec
    m1.build_model = build_model
    m1.load_tokenizer = load_tokenizer
    m1.run_dir = run_dir
    m1.source_hash = source_hash
    m1.CACHE_ROOT = D1_ROOT / "teacher_cache"
    m1.CHECK_ROOT = D1_ROOT / "checks"
    m1.PROTOCOL = PROTOCOL_D1
    m1.RUN_ROOT = D1_ROOT
