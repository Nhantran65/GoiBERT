"""Focused tests for TinyBERT layer mapping and masked alignment losses."""

from __future__ import annotations

import unittest

import torch

from distill import layer_mapping, masked_attention_mse, masked_hidden_mse


class DistillationLossTests(unittest.TestCase):
    def test_layer_mapping_includes_embeddings_and_final_teacher_layer(self) -> None:
        self.assertEqual(layer_mapping(6), [0, 2, 4, 6, 8, 10, 12])
        self.assertEqual(layer_mapping(10), [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12])

    def test_hidden_loss_ignores_padding_tokens(self) -> None:
        student = torch.zeros(1, 3, 2)
        teacher = torch.zeros(1, 3, 2)
        teacher[:, 2] = 100.0
        mask = torch.tensor([[1, 1, 0]])
        self.assertEqual(float(masked_hidden_mse(student, teacher, mask)), 0.0)

    def test_attention_loss_aligns_different_head_counts(self) -> None:
        student = torch.ones(1, 4, 3, 3)
        teacher = torch.ones(1, 12, 3, 3)
        mask = torch.tensor([[1, 1, 0]])
        loss = masked_attention_mse(student, teacher, mask)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
