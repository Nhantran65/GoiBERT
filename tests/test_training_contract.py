from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from goibert.modeling import GoiBERTClassifier
from goibert.run_grid import _screening_specs, lr_slug
from goibert.train import (
    LEARNING_RATES,
    compute_loss,
    inspect_train_lengths,
    make_optimizer,
    selection_score,
)


class FakeEncoder(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, input_ids, attention_mask, return_dict=True):
        batch, length = input_ids.shape
        hidden = self.weight.view(1, 1, -1).expand(batch, length, -1)
        return SimpleNamespace(last_hidden_state=hidden)


class FakeTokenizer:
    def __init__(self, lengths: list[int]) -> None:
        self.lengths = lengths
        self.calls = 0

    def __call__(self, text, **kwargs):
        value = self.lengths[self.calls]
        self.calls += 1
        return {"input_ids": list(range(value))}


class TrainingContractTests(unittest.TestCase):
    def test_head_and_variable_absa_slots_are_trainable(self) -> None:
        model = GoiBERTClassifier(
            FakeEncoder(),
            num_labels=6,
            num_aspects=2,
            aspect_class_counts=(4, 2),
        )
        logits = model(torch.ones(3, 5, dtype=torch.long), torch.ones(3, 5, dtype=torch.long))
        self.assertEqual(tuple(logits.shape), (3, 6))
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        loss = compute_loss(
            logits,
            torch.tensor([[0, 1], [1, 0], [3, 1]]),
            num_aspects=2,
            aspect_class_counts=(4, 2),
        )
        loss.backward()
        self.assertIsNotNone(model.head.output.weight.grad)

    def test_optimizer_excludes_bias_and_layernorm_weights(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        optimizer = make_optimizer(model, 2e-5)
        decay_by_id = {
            id(parameter): group["weight_decay"]
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertAlmostEqual(decay_by_id[id(model[0].weight)], 0.01)
        self.assertAlmostEqual(decay_by_id[id(model[0].bias)], 0.0)
        self.assertAlmostEqual(decay_by_id[id(model[1].weight)], 0.0)
        self.assertAlmostEqual(decay_by_id[id(model[1].bias)], 0.0)

    def test_length_policy_uses_train_only_and_hard_threshold(self) -> None:
        tokenizer = FakeTokenizer([100, 200, 257, 512])
        stats = inspect_train_lengths(tokenizer, ["a", "b", "c", "d"])
        self.assertEqual(stats.max_length, 512)
        self.assertAlmostEqual(stats.truncation_rate_256, 0.5)
        self.assertAlmostEqual(stats.truncation_rate_512, 0.0)
        self.assertFalse(stats.warning_over_512)

    def test_selection_metric_is_explicit(self) -> None:
        self.assertAlmostEqual(selection_score({"selection_f1": 0.7, "macro_f1": 0.99}), 0.7)
        with self.assertRaises(KeyError):
            selection_score({"macro_f1": 0.99})

    def test_screening_grid_has_exact_locked_lrs_and_seed(self) -> None:
        specs = _screening_specs(["vsfc_sentiment"], output_root=Path("artifacts/e1"))
        self.assertEqual([spec.learning_rate for spec in specs], list(LEARNING_RATES))
        self.assertEqual({spec.seed for spec in specs}, {42})
        self.assertEqual(lr_slug(1e-5), "1e-05")


if __name__ == "__main__":
    unittest.main()
