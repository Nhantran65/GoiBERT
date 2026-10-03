#!/usr/bin/env python3
"""Check the per-task in-memory Cafe trim results produced by export_logits.py."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/two_backbone_ensemble/trim"


def main():
    for task in ("absa_restaurant", "visfd", "vsfc_topic"):
        result = json.loads((BASE / f"{task}.json").read_text())
        if not result["tokenizer_cafe_xlmr_identical"]:
            raise RuntimeError(f"{task}: Cafe/XLM-R tokenizer mismatch")
        if result["parameters_trim"] >= result["parameters_original"]:
            raise RuntimeError(f"{task}: vocab was not trimmed")
        print(task, result["token_ids_identical_rate"], result["max_abs_logits_delta"],
              result["predictions_identical"], result["parameters_trim"])


if __name__ == "__main__":
    main()
