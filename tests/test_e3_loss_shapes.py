from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from goibert_e3.losses import combined_polarity_loss, conditional_polarity_cross_entropy


class E3LossShapeTests(unittest.TestCase):
    def test_pools_cells_when_slots_have_unequal_support(self) -> None:
        # Slot 0 contributes two active cells while slot 1 contributes one.
        # Pooling all cells must differ from averaging the two slot means.
        labels = torch.tensor([[1, 2], [2, 0], [0, 0]])
        logits = torch.tensor(
            [
                [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
            ],
            requires_grad=True,
        )
        value, active = conditional_polarity_cross_entropy(
            logits,
            labels,
            num_aspects=2,
            aspect_class_counts=(4, 4),
            polarity_slots=(0, 1),
        )
        expected = F.cross_entropy(
            torch.cat((logits[0:2, 1:4], logits[0:1, 5:8]), dim=0),
            torch.tensor([0, 1, 1]),
        )
        per_slot_average = 0.5 * (
            F.cross_entropy(logits[0:2, 1:4], torch.tensor([0, 1]))
            + F.cross_entropy(logits[0:1, 5:8], torch.tensor([1]))
        )
        self.assertEqual(active, 3)
        self.assertTrue(torch.allclose(value, expected))
        self.assertFalse(torch.allclose(value, per_slot_average))

    def test_uniform_three_dimensional_restaurant_logits(self) -> None:
        torch.manual_seed(11)
        logits = torch.randn(2, 12, 4, requires_grad=True)
        labels = torch.tensor(
            [
                [0, 1, 2, 3, 0, 0, 1, 0, 2, 0, 3, 0],
                [3, 0, 0, 1, 2, 0, 0, 3, 0, 1, 0, 2],
            ]
        )
        result = combined_polarity_loss(
            logits,
            labels,
            num_aspects=12,
            aspect_class_counts=(4,) * 12,
            polarity_slots=tuple(range(12)),
        )
        self.assertEqual(result.active_slot_count, 12)
        self.assertTrue(torch.isfinite(result.total))
        result.total.backward()
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
