import unittest

import numpy as np
import torch
from torch import nn

from goibert.finetune import fold_assignment, kd_term
from goibert.optim import build_layerwise_optimizer


class FoldTests(unittest.TestCase):
    def test_folds_are_balanced_and_deterministic(self):
        a, b = fold_assignment(103), fold_assignment(103)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(sorted(np.bincount(a).tolist()), [20, 20, 21, 21, 21])


class KdTests(unittest.TestCase):
    def test_kd_is_zero_when_student_matches_teacher(self):
        logits = torch.randn(4, 7)
        self.assertAlmostEqual(float(kd_term(logits, logits.clone(), [(0, 3), (3, 7)])), 0.0, places=6)

    def test_kd_averages_slots(self):
        s, t = torch.randn(5, 8), torch.randn(5, 8)
        joint = kd_term(s, t, [(0, 4), (4, 8)])
        separate = (kd_term(s[:, :4], t[:, :4], [(0, 4)]) + kd_term(s[:, 4:], t[:, 4:], [(0, 4)])) / 2
        self.assertTrue(torch.allclose(joint, separate))


class _Tiny(nn.Module):
    def __init__(self, layers=3):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.embeddings = nn.Module()
        self.encoder.embeddings.word = nn.Linear(2, 2)
        self.encoder.embeddings.LayerNorm = nn.LayerNorm(2)
        self.encoder.encoder = nn.Module()
        self.encoder.encoder.layer = nn.ModuleList(nn.Linear(2, 2) for _ in range(layers))
        self.head = nn.Linear(2, 2)


class LlrdTests(unittest.TestCase):
    def test_learning_rates_decay_by_depth(self):
        optimizer = build_layerwise_optimizer(_Tiny(), 1e-4, num_layers=3, gamma=0.8)
        lr = {g["region"]: g["lr"] for g in optimizer.param_groups}
        self.assertAlmostEqual(lr["head"], 1e-4)
        self.assertAlmostEqual(lr["block_02"], 1e-4)
        self.assertAlmostEqual(lr["block_00"], 1e-4 * 0.8**2)
        self.assertAlmostEqual(lr["embeddings"], 1e-4 * 0.8**3)
        no_decay = {g["group_name"]: g["weight_decay"] for g in optimizer.param_groups}
        self.assertEqual(no_decay["embeddings_no_decay"], 0.0)


if __name__ == "__main__":
    unittest.main()
