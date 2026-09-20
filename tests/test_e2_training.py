from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from goibert_e2.cache import build_teacher_logit_cache, load_teacher_logit_cache
from goibert_e2.losses import logits_kd_loss, symmetric_kl_consistency_loss
from goibert_e2.modeling import E2TeacherClassifier
from goibert_e2.noise import noisy_batch
from goibert_e2.train import (
    EXPECTED_E1_SOURCE_HASH,
    _load_loss_api,
    _validated_source_hashes,
    build_arg_parser,
    source_hashes,
)


class FakeEncoder(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.weight = nn.Parameter(torch.arange(hidden_size, dtype=torch.float32))

    def forward(self, input_ids, attention_mask, return_dict=True):
        batch, length = input_ids.shape
        hidden = self.weight.view(1, 1, -1).expand(batch, length, -1)
        return type("Output", (), {"last_hidden_state": hidden})()


class FakeTeacher(nn.Module):
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # The first token is enough to make row alignment observable.
        first = input_ids[:, :1].float()
        return torch.cat((first, first + 1.0, first + 2.0), dim=1)


def collate(rows):
    return {
        "input_ids": torch.tensor([row["input_ids"] for row in rows], dtype=torch.long),
        "attention_mask": torch.ones(len(rows), len(rows[0]["input_ids"]), dtype=torch.long),
        "ids": [row["id"] for row in rows],
        # Cache code must ignore these fields in the serialized payload.
        "labels": torch.tensor([row["label"] for row in rows], dtype=torch.long),
    }


class E2TrainingTests(unittest.TestCase):
    def test_teacher_head_is_768_wide_and_trainable(self) -> None:
        model = E2TeacherClassifier(FakeEncoder(8), num_labels=3)
        self.assertEqual(model.head.dense.out_features, 8)
        logits = model(
            torch.ones(2, 4, dtype=torch.long), torch.ones(2, 4, dtype=torch.long)
        )
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_cache_is_aligned_fp32_train_only_and_rng_stable(self) -> None:
        rows = [
            {"input_ids": [1, 0], "id": "task:train:0", "label": 0},
            {"input_ids": [2, 0], "id": "task:train:1", "label": 1},
        ]
        loader = DataLoader(rows, batch_size=2, shuffle=False, collate_fn=collate)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher_logits.pt"
            random.seed(7)
            np.random.seed(7)
            torch.manual_seed(7)
            before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            cache = build_teacher_logit_cache(
                path,
                teacher=FakeTeacher(),
                loader=loader,
                device=torch.device("cpu"),
                precision="fp32",
                task_id="task",
                source_ids=["task:train:0", "task:train:1"],
                source_texts=["one", "two"],
                tokenizer_hashes={"tokenizer.json": "hash"},
                max_length=256,
                teacher_hash="teacher-hash",
                aspect_class_counts=(),
                num_aspects=0,
                num_labels=3,
            )
            after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            self.assertEqual(before[0], after[0])
            self.assertTrue(np.array_equal(before[1][1], after[1][1]))
            self.assertTrue(torch.equal(before[2], after[2]))
            self.assertEqual(cache.logits.dtype, torch.float32)
            self.assertTrue(torch.equal(cache.lookup(["task:train:1"]), torch.tensor([[2.0, 3.0, 4.0]])))
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertNotIn("labels", payload)
            self.assertNotIn("labels", payload["metadata"])
            reloaded = load_teacher_logit_cache(path)
            self.assertEqual(reloaded.metadata["source_ids_hash"], cache.metadata["source_ids_hash"])

    def test_existing_cache_contract_cannot_be_overwritten(self) -> None:
        rows = [{"input_ids": [1, 0], "id": "task:train:0", "label": 0}]
        loader = DataLoader(rows, batch_size=1, shuffle=False, collate_fn=collate)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher_logits.pt"
            kwargs = dict(
                teacher=FakeTeacher(),
                loader=loader,
                device=torch.device("cpu"),
                precision="fp32",
                task_id="task",
                source_ids=["task:train:0"],
                source_texts=["one"],
                tokenizer_hashes={"tokenizer.json": "hash"},
                max_length=256,
                teacher_hash="teacher-hash",
                aspect_class_counts=(),
                num_aspects=0,
                num_labels=3,
            )
            build_teacher_logit_cache(path, **kwargs)
            with self.assertRaises(FileExistsError):
                build_teacher_logit_cache(path, **{**kwargs, "teacher_hash": "different"})

    def test_cli_has_no_test_mode(self) -> None:
        parser = build_arg_parser()
        self.assertNotIn("test", parser._option_string_actions)
        self.assertEqual(EXPECTED_E1_SOURCE_HASH, "456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873")

    def test_cache_launch_checks_all_expected_hashes(self) -> None:
        root = Path.cwd()
        current = source_hashes(root)
        self.assertEqual(
            _validated_source_hashes(
                root,
                expected_e2_code_hash=current["e2"],
                expected_e2_config_hash=current["e2_config"],
                expected_orchestrator_hash=current["orchestrator"],
            ),
            current,
        )
        for field, argument in (
            ("e2", "expected_e2_code_hash"),
            ("e2_config", "expected_e2_config_hash"),
            ("orchestrator", "expected_orchestrator_hash"),
        ):
            kwargs = {
                "expected_e2_code_hash": current["e2"],
                "expected_e2_config_hash": current["e2_config"],
                "expected_orchestrator_hash": current["orchestrator"],
            }
            kwargs[argument] = "stale-" + field
            with self.assertRaises(RuntimeError):
                _validated_source_hashes(root, **kwargs)

    def test_kd_runner_contract_is_t2_once_and_slotwise(self) -> None:
        student = torch.tensor([[1.0, 0.0, -1.0, 2.0, 0.0, -2.0]])
        teacher = torch.tensor([[0.5, 1.5, -0.5, 1.0, -1.0, 0.0]])
        kd = logits_kd_loss(student, teacher, temperature=2.0, slot_class_counts=(3, 3))
        self.assertGreater(float(kd), 0.0)
        # A padded/invalid class in another slot must not influence the first
        # slot's KL.  The explicit schema is the ViSFD protection against a
        # single 42-way softmax.
        teacher_changed = teacher.clone()
        teacher_changed[0, 5] += 100.0
        first_slot_a = logits_kd_loss(student[:, :3], teacher[:, :3], temperature=2.0)
        first_slot_b = logits_kd_loss(student[:, :3], teacher_changed[:, :3], temperature=2.0)
        self.assertTrue(torch.allclose(first_slot_a, first_slot_b))

    def test_runner_categorical_kd_and_nc_accept_empty_slot_schema(self) -> None:
        kd_loss, nc_loss = _load_loss_api()
        clean = torch.tensor([[0.8, -0.1, 0.3]], requires_grad=True)
        teacher = torch.tensor([[0.2, 0.9, -0.4]])
        noisy = torch.tensor([[0.7, 0.0, 0.4]], requires_grad=True)
        kd = kd_loss(
            clean,
            teacher,
            temperature=2.0,
            slot_class_counts=(),
        )
        nc = nc_loss(
            clean,
            noisy,
            temperature=1.0,
            slot_class_counts=(),
        )
        self.assertTrue(torch.isfinite(kd))
        self.assertTrue(torch.isfinite(nc))
        (kd + nc).backward()
        self.assertIsNotNone(clean.grad)
        self.assertIsNotNone(noisy.grad)

    def test_noise_is_stable_per_id_when_batch_order_changes(self) -> None:
        texts = ["Bạn thích dịch vụ này", "Phòng rất sạch"]
        ids = ["task:train:0", "task:train:1"]
        first = noisy_batch(texts, ids, seed=42, epoch=3)
        reordered = noisy_batch(list(reversed(texts)), list(reversed(ids)), seed=42, epoch=3)
        by_id = dict(zip(ids, first.texts))
        reordered_by_id = dict(zip(reversed(ids), reordered.texts))
        self.assertEqual(by_id, reordered_by_id)


if __name__ == "__main__":
    unittest.main()
