import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "releases" / "bidirlm_vi_mntp_joi_intent8_2026-09-22"
sys.path.insert(0, str(PACKAGE))
import predict

sys.path.insert(0, str(ROOT / "intent_benchmark" / "training"))
import train_intent as original


def main():
    tokenizer, model, labels, device = predict.load_classifier(PACKAGE, "cuda:1")
    rows = original.build_dataset_rows(original.TEST_PATH)
    previous = {
        row["sample_id"]: row
        for row in original.load_jsonl(ROOT / "intent_benchmark/extensions/bidirlm_vi_mntp/test_predictions.jsonl")
    }
    exact_ids = 0
    exact_predictions = 0
    for row in rows:
        messages = row["model_input"]["messages"]
        expected_ids = original.serialize_and_encode(tokenizer, {**row, "label_id": labels.index(row["label"])}, 512).input_ids
        actual_ids = predict.encode_message(tokenizer, messages[-1]["text"], [m["text"] for m in messages[:-1]])["input_ids"][0].tolist()
        exact_ids += expected_ids == actual_ids
        prediction = predict.predict(tokenizer, model, labels, device, messages[-1]["text"], [m["text"] for m in messages[:-1]])
        exact_predictions += prediction["intent"] == previous[row["sample_id"]]["pred_label"]
    print(json.dumps({"samples": len(rows), "exact_token_ids": exact_ids, "matching_predictions": exact_predictions}))
    assert exact_ids == exact_predictions == len(rows)


if __name__ == "__main__":
    main()
