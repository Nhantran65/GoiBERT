from __future__ import annotations

import random
import unittest

import torch

from goibert_e2.losses import (
    combined_kd_noise_loss,
    logits_kd_loss,
    noise_consistency_loss,
    supervised_cross_entropy,
)
from goibert_e2.noise import (
    NoiseBatch,
    make_rng,
    noisy_batch,
    noisy_view,
    strip_diacritics,
    teencode,
)


class E2LossTests(unittest.TestCase):
    def test_kd_direction_and_temperature_squared(self) -> None:
        student = torch.tensor([[0.3, -0.4]], requires_grad=True)
        teacher = torch.tensor([[1.2, -0.1]])
        temperature = 2.0
        value = logits_kd_loss(student, teacher, temperature=temperature)
        teacher_log_probs = torch.log_softmax(teacher / temperature, dim=-1)
        student_log_probs = torch.log_softmax(student / temperature, dim=-1)
        expected = (
            teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)
        ).sum(dim=-1).mean() * (temperature**2)
        self.assertTrue(torch.allclose(value, expected, atol=1e-7))
        reverse = (
            student_log_probs.exp() * (student_log_probs - teacher_log_probs)
        ).sum(dim=-1).mean() * (temperature**2)
        self.assertFalse(torch.allclose(value, reverse, atol=1e-7))
        value.backward()
        self.assertIsNotNone(student.grad)

    def test_padded_invalid_classes_are_masked(self) -> None:
        counts = (4, 2)
        labels = torch.tensor([[0, 0], [1, 1]])
        student = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.2, 0.7, 1000.0, -1000.0]],
                [[0.0, 1.0, 0.0, 0.0], [0.5, 0.3, -1000.0, 1000.0]],
            ],
            requires_grad=True,
        )
        teacher = torch.tensor(
            [
                [[0.9, 0.1, 0.0, 0.0], [0.4, 0.6, 1000.0, -1000.0]],
                [[0.2, 0.8, 0.0, 0.0], [0.8, 0.2, -1000.0, 1000.0]],
            ]
        )
        masked = logits_kd_loss(student, teacher, slot_class_counts=counts)
        student_alt = student.detach().clone()
        teacher_alt = teacher.clone()
        student_alt[:, 1, 2:] = torch.tensor([[10_000.0, -10_000.0], [-10_000.0, 10_000.0]])
        teacher_alt[:, 1, 2:] = torch.tensor([[-10_000.0, 10_000.0], [10_000.0, -10_000.0]])
        self.assertTrue(
            torch.allclose(
                masked,
                logits_kd_loss(student_alt, teacher_alt, slot_class_counts=counts),
                atol=1e-6,
            )
        )
        self.assertTrue(torch.isfinite(supervised_cross_entropy(student, labels, slot_class_counts=counts)))

    def test_absent_ce_is_an_ordinary_target(self) -> None:
        logits = torch.zeros(3, 3, requires_grad=True)
        labels = torch.zeros(3, dtype=torch.long)
        # E1's shared task helper represents categorical tasks with an empty
        # slot schema; E2 accepts that form as categorical logits.
        value = supervised_cross_entropy(logits, labels, slot_class_counts=())
        self.assertAlmostEqual(float(value), float(torch.log(torch.tensor(3.0))), places=6)
        value.backward()
        self.assertIsNotNone(logits.grad)

    def test_visfd_ten_four_class_slots_plus_binary_others(self) -> None:
        # Current E1 aspect order keeps OTHERS at index 5.
        counts = (4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4)
        logits = torch.zeros(2, sum(counts), requires_grad=True)
        labels = torch.zeros(2, 11, dtype=torch.long)
        labels[:, 5] = torch.tensor([0, 1])
        ce = supervised_cross_entropy(logits, labels, slot_class_counts=counts)
        self.assertTrue(torch.isfinite(ce))
        teacher = torch.randn_like(logits)
        kd = logits_kd_loss(logits, teacher, slot_class_counts=counts)
        total = combined_kd_noise_loss(
            logits,
            logits + 0.01,
            teacher,
            labels,
            slot_class_counts=counts,
            alpha=0.5,
            noise_weight=0.2,
        )
        self.assertTrue(torch.isfinite(kd))
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(logits.grad)

    def test_noise_consistency_has_both_gradients_and_no_temperature_scale(self) -> None:
        clean = torch.tensor([[1.0, 0.0]], requires_grad=True)
        noisy = torch.tensor([[0.0, 1.0]], requires_grad=True)
        value = noise_consistency_loss(clean, noisy)
        expected = 0.5 * (
            (torch.softmax(clean, -1) * (torch.log_softmax(clean, -1) - torch.log_softmax(noisy, -1))).sum(-1)
            + (torch.softmax(noisy, -1) * (torch.log_softmax(noisy, -1) - torch.log_softmax(clean, -1))).sum(-1)
        ).mean()
        self.assertTrue(torch.allclose(value, expected, atol=1e-7))
        value.backward()
        self.assertIsNotNone(clean.grad)
        self.assertIsNotNone(noisy.grad)


class E2NoiseTests(unittest.TestCase):
    def test_same_seed_epoch_id_is_deterministic_and_ids_are_independent(self) -> None:
        text = "Bạn ơi, giá là 12.000đ, không tệ!"
        first = noisy_view(text, seed=42, epoch=3, stable_id="row-a")
        second = noisy_view(text, seed=42, epoch=3, stable_id="row-a")
        other_id = noisy_view(text, seed=42, epoch=3, stable_id="row-b")
        self.assertEqual(first, second)
        # Different IDs use separate streams.  A collision is astronomically
        # unlikely, and this fixture confirms the ID participates in the seed.
        self.assertNotEqual(first, other_id)

    def test_noise_does_not_mutate_global_rng(self) -> None:
        random.seed(12345)
        expected = random.random()
        random.seed(12345)
        _ = noisy_view("Sản phẩm tốt.", seed=42, epoch=1, stable_id="x")
        self.assertEqual(random.random(), expected)

    def test_batch_preserves_raw_text_and_reports_actual_change_rate(self) -> None:
        texts = ["Bạn có 2 món!", "Không được xóa dấu câu: 3.14.", ""]
        original = list(texts)
        result = noisy_batch(texts, ["a", "b", "c"], seed=42, epoch=1)
        self.assertIsInstance(result, NoiseBatch)
        self.assertEqual(texts, original)
        self.assertEqual(len(result.texts), len(texts))
        self.assertGreaterEqual(result.change_rate, 0.0)
        self.assertLessEqual(result.change_rate, 1.0)
        self.assertEqual(result.to_dict()["count"], 3)
        for before, after in zip(texts, result.texts):
            self.assertEqual(sum(character.isdigit() for character in before), sum(character.isdigit() for character in after))
            punctuation_before = [character for character in before if not character.isalnum() and not character.isspace()]
            punctuation_after = [character for character in after if not character.isalnum() and not character.isspace()]
            self.assertEqual(punctuation_before, punctuation_after)

    def test_accent_and_teencode_preserve_negation_and_boundaries(self) -> None:
        self.assertEqual(strip_diacritics("không 12!"), "khong 12!")
        value = teencode("Không được. khôngđược được!", make_rng(1, 1, "fixture"), p_word=1.0)
        self.assertTrue(value.startswith("Không dc."))
        self.assertIn("khôngđược", value)
        self.assertIn("dc!", value)


if __name__ == "__main__":
    unittest.main()
