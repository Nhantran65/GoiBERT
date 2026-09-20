from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from goibert.train import compute_loss
from goibert_e3.losses import (
    POLARITY_WEIGHT,
    combined_polarity_loss,
    conditional_polarity_cross_entropy,
)


class E3LossTests(unittest.TestCase):
    def test_pools_present_slots_and_removes_absent_logit(self) -> None:
        counts = (4, 4)
        labels = torch.tensor([[1, 0], [0, 3]])
        logits = torch.tensor(
            [
                [10.0, 1.0, 2.0, 3.0, 50.0, 60.0, 70.0, 80.0],
                [40.0, 41.0, 42.0, 43.0, 4.0, 5.0, 6.0, 7.0],
            ],
            requires_grad=True,
        )
        value, active = conditional_polarity_cross_entropy(
            logits,
            labels,
            num_aspects=2,
            aspect_class_counts=counts,
            polarity_slots=(0, 1),
        )
        expected = F.cross_entropy(
            torch.cat((logits[0:1, 1:4], logits[1:2, 5:8]), dim=0),
            torch.tensor([0, 2]),
        )
        self.assertEqual(active, 2)
        self.assertTrue(torch.allclose(value, expected))
        value.backward()
        self.assertEqual(float(logits.grad[0, 0]), 0.0)
        self.assertEqual(float(logits.grad[1, 4]), 0.0)

    def test_all_absent_is_a_differentiable_zero(self) -> None:
        logits = torch.randn(2, 8, requires_grad=True)
        labels = torch.zeros(2, 2, dtype=torch.long)
        value, active = conditional_polarity_cross_entropy(
            logits,
            labels,
            num_aspects=2,
            aspect_class_counts=(4, 4),
        )
        self.assertEqual(active, 0)
        self.assertEqual(float(value), 0.0)
        value.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits.grad)))

    def test_visfd_others_binary_slot_is_excluded(self) -> None:
        counts = (4, 2, 4)
        labels = torch.tensor([[1, 1, 3]])
        logits = torch.randn(1, sum(counts), requires_grad=True)
        value, active = conditional_polarity_cross_entropy(
            logits,
            labels,
            num_aspects=3,
            aspect_class_counts=counts,
            polarity_slots=(0, 2),
        )
        altered = logits.detach().clone()
        altered[:, 4:6] = torch.tensor([[10000.0, -10000.0]])
        altered_value, altered_active = conditional_polarity_cross_entropy(
            altered,
            labels,
            num_aspects=3,
            aspect_class_counts=counts,
            polarity_slots=(0, 2),
        )
        self.assertEqual(active, 2)
        self.assertEqual(altered_active, active)
        self.assertTrue(torch.allclose(value, altered_value))
        value.backward()
        self.assertTrue(torch.equal(logits.grad[:, 4:6], torch.zeros_like(logits.grad[:, 4:6])))

    def test_combined_weight_zero_is_exactly_e1_ce(self) -> None:
        torch.manual_seed(7)
        counts = (4, 2, 4)
        logits = torch.randn(3, sum(counts), requires_grad=True)
        labels = torch.tensor([[0, 0, 1], [3, 1, 0], [2, 0, 3]])
        components = combined_polarity_loss(
            logits,
            labels,
            num_aspects=3,
            aspect_class_counts=counts,
            polarity_weight=0.0,
            polarity_slots=(0, 2),
        )
        baseline = compute_loss(
            logits,
            labels,
            num_aspects=3,
            aspect_class_counts=counts,
        )
        self.assertTrue(torch.allclose(components.base_ce, baseline))
        self.assertTrue(torch.allclose(components.total, baseline))

    def test_declared_default_weight_is_half(self) -> None:
        self.assertEqual(POLARITY_WEIGHT, 0.5)


if __name__ == "__main__":
    unittest.main()
