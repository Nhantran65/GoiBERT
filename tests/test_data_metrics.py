from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from goibert.data import (
    EXPECTED_COUNTS,
    VISFD_OTHERS_CODE,
    load_task,
    visfd_from_model_labels,
    visfd_to_model_labels,
)
from goibert.metrics import absa_metrics, classification_metrics, evaluate_task


REPO_ROOT = Path(__file__).resolve().parents[1]


class ClassificationMetricTests(unittest.TestCase):
    def test_fixed_macro_f1_and_accuracy(self) -> None:
        result = classification_metrics(
            [0, 1, 2, 2],
            [0, 2, 2, 1],
            labels=[0, 1, 2],
        )
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], 0.5)
        self.assertEqual(result["per_class"]["0"]["support"], 1)
        self.assertEqual(result["per_class"]["1"]["tp"], 0)

    def test_fixed_label_set_rejects_unlisted_prediction(self) -> None:
        with self.assertRaises(ValueError):
            classification_metrics([0], [3], labels=[0, 1, 2])


class AbsaMetricTests(unittest.TestCase):
    def test_detection_and_pair_micro_macro_are_distinct(self) -> None:
        gold = [[0, 1, 2], [1, 0, 3]]
        predicted = [[0, 2, 3], [1, 3, 0]]
        result = absa_metrics(gold, predicted, ["a", "b", "c"])

        # Presence: TP=3, predicted=4, gold=4 -> F1=6/8=0.75.
        self.assertAlmostEqual(result["detection_micro_f1"], 0.75)
        # Exact aspect-polarity pairs: TP=1, predicted=4, gold=4 -> F1=0.25.
        self.assertAlmostEqual(result["pair_micro_f1"], 0.25)
        # Per-aspect pair F1 values are 1, 0, 0.
        self.assertAlmostEqual(result["pair_macro_f1"], 1 / 3)
        self.assertNotEqual(result["detection_micro_f1"], result["pair_micro_f1"])
        self.assertNotIn("macro_f1", result)

    def test_others_is_detection_only_when_masked(self) -> None:
        result = absa_metrics(
            [[1, VISFD_OTHERS_CODE], [0, VISFD_OTHERS_CODE]],
            [[1, 0], [0, VISFD_OTHERS_CODE]],
            ["BATTERY", "OTHERS"],
            pair_scored_aspects=[True, False],
        )
        # Detection: TP=2, predicted=2, gold=3 -> 4/5.
        self.assertAlmostEqual(result["detection_micro_f1"], 4 / 5)
        # Only BATTERY is scored for sentiment pairs, and it is exact.
        self.assertAlmostEqual(result["pair_micro_f1"], 1.0)
        self.assertIsNone(result["per_aspect"]["OTHERS"]["pair"])

    def test_runner_evaluator_maps_visfd_model_others(self) -> None:
        gold = [[1, VISFD_OTHERS_CODE], [0, VISFD_OTHERS_CODE]]
        model_predictions = [[1, 1], [0, 1]]
        result = evaluate_task(
            task_id="visfd",
            gold=gold,
            predictions=model_predictions,
            aspect_names=["BATTERY", "OTHERS"],
            split="validation",
        )
        self.assertEqual(result["selection_metric"], "end_to_end_pair_micro_f1")
        self.assertEqual(result["selection_f1"], 1.0)


class LoaderTests(unittest.TestCase):
    def test_all_task_counts_and_stable_ids(self) -> None:
        for task_id, expected in EXPECTED_COUNTS.items():
            with self.subTest(task_id=task_id):
                task = load_task(task_id, REPO_ROOT, include_test=True)
                self.assertEqual(
                    {name: len(split) for name, split in task.splits.items()}, expected
                )
                self.assertEqual(task.train[0].id, f"{task_id}:train:000000")
                self.assertTrue(task.train[0].text)
                self.assertEqual(task.train.ids, [
                    f"{task_id}:train:{row:06d}" for row in range(len(task.train))
                ])

    def test_visfd_others_is_not_positive(self) -> None:
        task = load_task("visfd", REPO_ROOT, include_test=True)
        self.assertEqual(task.aspect_names[5], "OTHERS")
        self.assertFalse(task.pair_scored_aspects[5])
        self.assertEqual(task.train[1].label[5], VISFD_OTHERS_CODE)
        self.assertEqual(task.slot_label_names[5], ("absent", "others"))
        self.assertEqual(task.aspect_class_counts[5], 2)
        self.assertEqual(task.num_labels, 42)
        self.assertEqual(task.labels_for_model("train")[1][5], 1)

    def test_visfd_others_model_adapter_is_explicit(self) -> None:
        raw = [(0, 1, 2, 3, 0, VISFD_OTHERS_CODE, 0, 0, 0, 0, 0)]
        model = visfd_to_model_labels(raw)
        self.assertEqual(model[0][5], 1)
        self.assertEqual(visfd_from_model_labels(model), raw)

    def test_source_text_is_not_normalized(self) -> None:
        task = load_task("vsfc_sentiment", REPO_ROOT, include_test=False)
        raw = (
            REPO_ROOT / "data" / "raw" / "uit-vsfc" / "train" / "sents.txt"
        ).read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(task.train[0].text, raw)

    def test_screening_loader_does_not_load_test(self) -> None:
        original_open = Path.open

        def reject_test_open(path: Path, *args, **kwargs):
            if path.name in {"Test.csv", "test"}:
                raise AssertionError(f"screening unexpectedly opened test path: {path}")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", reject_test_open):
            task = load_task("vsfc_sentiment", REPO_ROOT, include_test=False)
        self.assertFalse(task.has_test)
        with self.assertRaises(RuntimeError):
            _ = task.test


if __name__ == "__main__":
    unittest.main()
