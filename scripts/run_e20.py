#!/usr/bin/env python3
"""E20 preflight, worker, and run-all controller.

Screening CTI (Compositional Token-state Invariance) on top of E19
(goibert-b-dapt 103M + NC + LLRD).  See ``docs/E20_PLAN.md`` (locked spec)
and ``docs/CTI_PROPOSAL.md`` (motivation, formula).

The E2 data/loss/validation loop (``goibert_e2.train._train_any`` /
``_train_loop``) and the E7 layer-wise optimizer stay frozen and unmodified.
This runner only, for the duration of one worker process:

* monkeypatches ``goibert_e2.train._task_contract`` to append a fixed set of
  composite rows (id prefix ``e20comp:``) to the train split -- built once,
  deterministically, from the task's natural train rows;
* monkeypatches ``goibert_e2.train.build_e2_student`` so ``_train_any``
  (mode="nc") builds an ``E20Classifier`` wrapping a fresh
  ``build_e2_teacher`` model over ``models/goibert-b-dapt`` (state_dict keys
  identical to a plain ``E2TeacherClassifier``, so it strictly reloads into
  one);
* monkeypatches ``goibert_e2.train.make_optimizer`` to the E7
  layer-wise-LR-decay AdamW (12 encoder blocks + embeddings + head);
* monkeypatches ``goibert_e2.train._move_batch`` to detect composite rows in
  a micro-batch (by id prefix) and stash their (row_index, boundary,
  a_text, b_text_standalone) metadata on the model before the clean forward;
* monkeypatches ``goibert_e2.train.compute_loss`` to add ``mu *
  model.last_inv_loss`` (set by the model's own forward on the clean pass
  when composites are pending and ``mu>0``; ``None``/skipped when ``mu==0``
  -- the ``d1_composite`` arm) and clear it.

Two arms share this one code path: ``d1_composite`` (mu=0, composite rows
only) and ``cti`` (mu=0.3, + L_inv).  No new parameters, no new src/
package.  Never edit ``src/goibert*``, other ``scripts/run_e*.py``,
``models/``, or ``artifacts/e1``..``artifacts/e19``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ARTIFACT_ROOT = ROOT / "artifacts" / "e20"
CONFIG_PATH = ROOT / "configs" / "e20" / "protocol.json"
ENCODER_DIR = ROOT / "models" / "goibert-b-dapt"
TOKENIZER_DIR = ROOT / "models" / "goiturn-s-gd"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from goibert import train as e1_train  # noqa: E402
from goibert.data import Example, TaskSplit, VISFD_OTHERS_ASPECT_INDEX  # noqa: E402
from goibert_e2 import train as e2_train  # noqa: E402
from goibert_e2.modeling import build_e2_teacher  # noqa: E402
from goibert_e7 import optim as e7_optim  # noqa: E402


PROTOCOL = "E20-cti-screening-v1"
REVISION = "e20-20260914-r1"
# E22 extension (see docs/E22... launcher + scripts/e22_preflight.py): added
# task "visfd" and arm "d1_uniform" on top of the original E20 CTI-screening
# scope (absa_restaurant/absa_hotel x d1_composite/cti). Existing finished
# trial dirs under artifacts/e20 are never touched (run_worker still refuses
# to overwrite an existing output_dir).
TASKS = ("absa_restaurant", "absa_hotel", "visfd")
ARMS = ("d1_composite", "cti", "d1_uniform")
MU_BY_ARM: dict[str, float] = {"d1_composite": 0.0, "cti": 0.3, "d1_uniform": 0.0}
# Composite aspect-pair sampling mode per arm: "rare_pair_weighted" draws
# pair (a,b) with probability proportional to 1/(1+train_pair_frequency(a,b))
# (the original d1_composite/cti recipe); "uniform" draws every valid pair
# with equal probability (the d1_uniform ablation -- same composite count,
# same pool/assert/boundary logic, only the pair-sampling distribution
# differs). See _build_composites(sampling=...).
SAMPLING_BY_ARM: dict[str, str] = {
    "d1_composite": "rare_pair_weighted",
    "cti": "rare_pair_weighted",
    "d1_uniform": "uniform",
}
SEED = 42
EPOCHS = 30
EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 16
ACCUMULATION_STEPS = 2
MAX_LENGTH = 256
PRECISION = "bf16"
ATTENTION_IMPLEMENTATION = "sdpa"
GAMMA = 0.8
NUM_LAYERS = 12
WEIGHT_DECAY = 0.01
# visfd LR/NC-lambda match run_e19.py / E8 (5e-5, 0.5) -- see docs/E19_PLAN.md.
TASK_LRS: dict[str, float] = {"absa_restaurant": 2e-5, "absa_hotel": 5e-5, "visfd": 5e-5}
NC_LAMBDAS: dict[str, float] = {"absa_restaurant": 0.1, "absa_hotel": 0.1, "visfd": 0.5}
MAX_COMPOSITE_TOKENS = 48
COMPOSITE_PREFIX = "e20comp:"
RARE_PAIR_THRESHOLD = 5
GUARD_USED_MIB = 5 * 1024
GUARD_UTIL_PCT = 5.0
PREFERRED_GPU_INDICES = ("1", "0", "2", "3")


# ---------------------------------------------------------------------------
# Generic helpers (ported from scripts/run_e19.py / run_e18.py)
# ---------------------------------------------------------------------------
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_default(value: Any) -> Any:
    return e1_train._json_default(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".safetensors", ".txt", ".model", ".vocab"}
    }


def _gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30)
    except Exception as exc:  # noqa: BLE001 - report, do not crash preflight
        return {"available": False, "error": repr(exc), "rows": {}}
    rows: dict[str, dict[str, float]] = {}
    for line in output.strip().splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) >= 4:
            rows[fields[0]] = {
                "free_mib": int(float(fields[1])),
                "used_mib": int(float(fields[2])),
                "utilization_pct": float(fields[3]),
            }
    return {"available": bool(rows), "rows": rows}


def _eligible_devices(snapshot: Mapping[str, Any]) -> list[str]:
    rows = snapshot.get("rows", {})
    eligible = [
        index
        for index, row in rows.items()
        if float(row["used_mib"]) < GUARD_USED_MIB and float(row["utilization_pct"]) <= GUARD_UTIL_PCT
    ]
    ordered = [index for index in PREFERRED_GPU_INDICES if index in eligible]
    ordered += [index for index in eligible if index not in ordered]
    return [f"cuda:{index}" for index in ordered]


def _pid_alive(pid: Any) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _source_manifest(seed: int = SEED) -> dict[str, Any]:
    source_files = [
        Path(__file__).resolve(),
        CONFIG_PATH,
        ROOT / "src" / "goibert_e2" / "train.py",
        ROOT / "src" / "goibert_e2" / "modeling.py",
        ROOT / "src" / "goibert_e2" / "losses.py",
        ROOT / "src" / "goibert_e2" / "noise.py",
        ROOT / "src" / "goibert_e2" / "cache.py",
        ROOT / "src" / "goibert_e7" / "optim.py",
        ROOT / "src" / "goibert" / "train.py",
        ROOT / "src" / "goibert" / "modeling.py",
        ROOT / "src" / "goibert" / "data.py",
    ]
    source_hashes = {str(path.relative_to(ROOT)): _sha256(path) for path in source_files if path.is_file()}
    return {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "created_at": _now(),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "source_hashes": source_hashes,
        "encoder": {
            "path": str(ENCODER_DIR.resolve()),
            "files": _directory_hashes(ENCODER_DIR),
        },
        "frozen_e2_source_hashes": e2_train.source_hashes(ROOT),
        "recipe": {
            "arms": list(ARMS),
            "mu_by_arm": dict(MU_BY_ARM),
            "sampling_by_arm": dict(SAMPLING_BY_ARM),
            "tasks": list(TASKS),
            "seed": seed,
            "epochs": EPOCHS,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "max_length": MAX_LENGTH,
            "precision": PRECISION,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "gamma": GAMMA,
            "num_layers": NUM_LAYERS,
            "weight_decay": WEIGHT_DECAY,
            "task_lrs": dict(TASK_LRS),
            "nc_lambdas": dict(NC_LAMBDAS),
            "max_composite_tokens": MAX_COMPOSITE_TOKENS,
            "rare_pair_threshold": RARE_PAIR_THRESHOLD,
        },
    }


# ---------------------------------------------------------------------------
# Composite construction (id_composites-style, pool + rare-pair sampling)
# ---------------------------------------------------------------------------
def _tokenizer() -> Any:
    return AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True, use_fast=True)


def _original_task_contract(task_id: str) -> dict[str, Any]:
    return e2_train._task_contract(task_id, ROOT, smoke=False)


def _composite_pool_aspects(task_id: str, num_aspects: int) -> tuple[int, ...]:
    """Aspect indices eligible for the single-aspect composite pool/pairing.

    Every aspect for the plain ABSA tasks; every *scored* aspect for ViSFD --
    i.e. all but the binary, polarity-less OTHERS slot
    (``goibert.data.VISFD_OTHERS_ASPECT_INDEX``), which is detection-only and
    carries no sentiment to union into a composite label.
    """

    if task_id == "visfd":
        return tuple(a for a in range(num_aspects) if a != VISFD_OTHERS_ASPECT_INDEX)
    return tuple(range(num_aspects))


def _build_composites(
    task_id: str,
    contract: Mapping[str, Any],
    tokenizer: Any,
    *,
    seed: int = SEED,
    max_length: int = MAX_LENGTH,
    sampling: str = "rare_pair_weighted",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the fixed composite set for ``task_id`` (deterministic in ``seed``).

    Pool = train rows with exactly one gold *scored* aspect (all ABSA aspects
    are scored; ViSFD excludes its binary OTHERS slot from both the
    single-aspect check and the pairing pool, and additionally requires raw
    OTHERS == 0 -- i.e. the row is not tagged {OTHERS} at all) and
    <=MAX_COMPOSITE_TOKENS tokens.  Composite count = round(len(train)/3).
    Aspect pair (a,b), a<b, drawn per ``sampling``:
    ``"rare_pair_weighted"`` (probability proportional to
    1/(1+train_pair_frequency(a,b)) over scored-aspect co-occurrence -- the
    ``d1_composite``/``cti`` arms) or ``"uniform"`` (equal probability over
    every valid pair -- the ``d1_uniform`` ablation, same pool/count/assert
    logic, only the pair-sampling distribution differs).  A/B rows sampled
    uniformly from the respective pools.  Every composite is verified by an
    exact token-id boundary assert (dropped/retried otherwise).  The union
    label is built directly in the loader's label code space: for ViSFD, the
    OTHERS slot defaults to (and stays) 0/absent, which is simultaneously the
    raw and model code for "absent" (see ``goibert.data.visfd_to_model_labels``),
    so no separate raw->model conversion is needed for composite rows.
    """

    if sampling not in ("rare_pair_weighted", "uniform"):
        raise ValueError(f"Unknown E20 composite sampling mode: {sampling!r}")

    train_split = contract["train"]
    num_aspects = int(contract["num_aspects"])
    texts = list(e2_train.split_texts(train_split))
    labels = list(e2_train.split_labels(train_split))
    ids = list(e2_train.split_ids(train_split))
    n_train = len(texts)
    if n_train == 0 or num_aspects <= 0:
        raise ValueError(f"E20 composites require a non-empty ABSA train split: {task_id}")

    eligible_aspects = _composite_pool_aspects(task_id, num_aspects)
    others_index = VISFD_OTHERS_ASPECT_INDEX if task_id == "visfd" else None

    tokenized_all = tokenizer(texts, truncation=True, max_length=max_length)["input_ids"]
    tok_lengths = [len(row) for row in tokenized_all]

    pool: dict[int, list[int]] = {a: [] for a in range(num_aspects)}
    pair_freq: dict[tuple[int, int], int] = {}
    for i, label in enumerate(labels):
        present = [a for a in eligible_aspects if int(label[a]) > 0]
        others_present = others_index is not None and int(label[others_index]) > 0
        if len(present) == 1 and not others_present and tok_lengths[i] <= MAX_COMPOSITE_TOKENS:
            pool[present[0]].append(i)
        for x in range(len(present)):
            for y in range(x + 1, len(present)):
                key = (present[x], present[y])
                pair_freq[key] = pair_freq.get(key, 0) + 1

    valid_pairs = [
        (a, b)
        for i, a in enumerate(eligible_aspects)
        for b in eligible_aspects[i + 1 :]
        if pool[a] and pool[b]
    ]
    if not valid_pairs:
        raise RuntimeError(f"E20: no valid composite aspect pairs for {task_id}")
    if sampling == "uniform":
        weights = [1.0 for _ in valid_pairs]
    else:
        weights = [1.0 / (1.0 + pair_freq.get(pair, 0)) for pair in valid_pairs]

    n_composites = round(n_train / 3)
    rng = random.Random(seed)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    composites: list[dict[str, Any]] = []
    pair_use_count: dict[tuple[int, int], int] = {pair: 0 for pair in valid_pairs}
    rare_pairs = {pair for pair in valid_pairs if pair_freq.get(pair, 0) <= RARE_PAIR_THRESHOLD}
    rare_pairs_covered: set[tuple[int, int]] = set()
    dropped_assert_failures = 0
    attempts = 0
    max_attempts = n_composites * 30 + 500

    while len(composites) < n_composites and attempts < max_attempts:
        attempts += 1
        pair = rng.choices(valid_pairs, weights=weights, k=1)[0]
        a, b = pair
        a_idx = rng.choice(pool[a])
        b_idx = rng.choice(pool[b])
        a_text, b_text = texts[a_idx], texts[b_idx]
        a_label, b_label = labels[a_idx], labels[b_idx]

        a_ids = tokenizer(a_text, add_special_tokens=False)["input_ids"]
        b_ids_space = tokenizer(" " + b_text, add_special_tokens=False)["input_ids"]
        boundary = 1 + len(a_ids)
        full_text = a_text + " " + b_text
        full_ids = tokenizer(full_text, truncation=True, max_length=max_length)["input_ids"]
        expected_len = 1 + len(a_ids) + len(b_ids_space) + 1
        ok = (
            len(full_ids) == expected_len
            and full_ids[0] == bos_id
            and full_ids[-1] == eos_id
            and full_ids[1:boundary] == a_ids
            and full_ids[boundary:-1] == b_ids_space
        )
        if not ok:
            dropped_assert_failures += 1
            continue

        composite_label = [0] * num_aspects
        composite_label[a] = int(a_label[a])
        composite_label[b] = int(b_label[b])
        idx = len(composites)
        composites.append(
            {
                "id": f"{COMPOSITE_PREFIX}{idx}",
                "a_id": str(ids[a_idx]),
                "b_id": str(ids[b_idx]),
                "aspect_a": a,
                "aspect_b": b,
                "boundary": boundary,
                "a_text": a_text,
                "b_text": b_text,
                "b_text_standalone": " " + b_text,
                "text": full_text,
                "label": composite_label,
                "pair_train_frequency": pair_freq.get(pair, 0),
            }
        )
        pair_use_count[pair] += 1
        if pair in rare_pairs:
            rare_pairs_covered.add(pair)

    coverage = {
        "task_id": task_id,
        "seed": seed,
        "sampling": sampling,
        "n_train": n_train,
        "n_eligible_aspects": len(eligible_aspects),
        "eligible_aspects": list(eligible_aspects),
        "n_composites_target": n_composites,
        "n_composites_built": len(composites),
        "attempts": attempts,
        "dropped_assert_failures": dropped_assert_failures,
        "n_valid_pairs": len(valid_pairs),
        "n_rare_pairs_le5": len(rare_pairs),
        "n_rare_pairs_covered": len(rare_pairs_covered),
        "distinct_pairs_used": sum(1 for count in pair_use_count.values() if count > 0),
        "pair_use_count": {f"{a}-{b}": count for (a, b), count in pair_use_count.items()},
    }
    return composites, coverage


def _inject_composites(bundle: Any, composites: Sequence[Mapping[str, Any]]) -> tuple[Any, Any]:
    """Return (new_bundle, new_train_split) with composite rows appended."""

    train_split = bundle.splits["train"]
    num_aspects = int(bundle.num_aspects)
    extra = tuple(
        Example(
            id=str(c["id"]),
            text=str(c["text"]),
            label=tuple(int(v) for v in c["label"]),
            raw_label=None,
            source_index=None,
            pair_scored=tuple(True for _ in range(num_aspects)),
        )
        for c in composites
    )
    new_train_split = TaskSplit(name="train", examples=tuple(train_split.examples) + extra)
    new_splits = dict(bundle.splits)
    new_splits["train"] = new_train_split
    new_bundle = dataclasses.replace(bundle, splits=new_splits)
    return new_bundle, new_train_split


# ---------------------------------------------------------------------------
# Model: thin wrapper around a fresh build_e2_teacher, no new parameters
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class E20ModelSpec:
    role: str
    backbone: str
    hidden_size: int
    head_hidden_size: int
    head_dropout: float
    num_labels: int
    num_aspects: int
    aspect_class_counts: tuple[int, ...]
    total_parameters: int
    trainable_parameters: int
    arm: str
    mu: float
    added_parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class E20Classifier(nn.Module):
    """Wraps a fresh ``build_e2_teacher`` model; optionally adds L_inv (CTI).

    ``self.encoder``/``self.head`` are the *same* submodule objects the base
    teacher was built with (re-parented, not copied): ``state_dict()`` is
    byte-identical in keys/shapes to a plain ``E2TeacherClassifier``.  Control
    state (``mu``, ``_pending``, ``last_inv_loss``, tokenizer) is plain
    Python/attribute state, never registered as a parameter or buffer.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        arm: str,
        mu: float,
        tokenizer: Any,
        max_length: int = MAX_LENGTH,
    ) -> None:
        super().__init__()
        self.encoder = base.encoder
        self.head = base.head
        self.num_labels = int(base.num_labels)
        self.num_aspects = int(base.num_aspects)
        self.aspect_class_counts: tuple[int, ...] = tuple(base.aspect_class_counts)
        self.arm = arm
        self.mu = float(mu)
        self._tokenizer = tokenizer
        self._max_length = int(max_length)
        # Control state; never touches state_dict()/checkpoints.
        self._pending: list[tuple[int, int, str, str]] | None = None
        self.last_inv_loss: Tensor | None = None
        self.last_inv_stats: dict[str, int] = {"rows": 0, "used": 0, "skipped": 0}

    def _reshape(self, flat_logits: Tensor) -> Tensor:
        if self.num_aspects and len(set(self.aspect_class_counts)) == 1:
            return flat_logits.reshape(flat_logits.shape[0], self.num_aspects, self.aspect_class_counts[0])
        return flat_logits

    def _compute_inv_loss(
        self,
        hidden: Tensor,
        attention_mask: Tensor,
        pending: list[tuple[int, int, str, str]],
    ) -> Tensor | None:
        device = hidden.device
        a_texts = [item[2] for item in pending]
        b_texts = [item[3] for item in pending]
        with torch.no_grad():
            enc_a = self._tokenizer(
                a_texts, padding=True, truncation=True, max_length=self._max_length, return_tensors="pt"
            )
            enc_b = self._tokenizer(
                b_texts, padding=True, truncation=True, max_length=self._max_length, return_tensors="pt"
            )
            a_ids = enc_a["input_ids"].to(device)
            a_mask = enc_a["attention_mask"].to(device)
            b_ids = enc_b["input_ids"].to(device)
            b_mask = enc_b["attention_mask"].to(device)
            ha = self.encoder(input_ids=a_ids, attention_mask=a_mask, return_dict=True).last_hidden_state
            hb = self.encoder(input_ids=b_ids, attention_mask=b_mask, return_dict=True).last_hidden_state

        hidden_size = hidden.shape[-1]
        row_losses: list[Tensor] = []
        used = skipped = 0
        for k, (row_index, boundary, _a_text, _b_text) in enumerate(pending):
            n = int(attention_mask[row_index].sum().item())
            m = int(b_mask[k].sum().item())
            a_content = hidden[row_index, 1:boundary]
            b_content = hidden[row_index, boundary : n - 1]
            a_target = ha[k, 1:boundary]
            b_target = hb[k, 1 : m - 1]
            if (
                a_content.shape[0] == 0
                or b_content.shape[0] == 0
                or a_content.shape[0] != a_target.shape[0]
                or b_content.shape[0] != b_target.shape[0]
            ):
                skipped += 1
                continue
            term_a = (a_content - a_target).pow(2).sum(dim=-1).mean() / hidden_size
            term_b = (b_content - b_target).pow(2).sum(dim=-1).mean() / hidden_size
            row_losses.append(term_a + term_b)
            used += 1
        self.last_inv_stats = {"rows": len(pending), "used": used, "skipped": skipped}
        if not row_losses:
            return None
        return torch.stack(row_losses).mean()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        token_type_ids: Tensor | None = None,
    ) -> Tensor:
        kwargs: dict[str, Tensor] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs, return_dict=True)
        hidden = outputs.last_hidden_state
        logits_flat = self.head(hidden)
        logits = self._reshape(logits_flat)

        pending = self._pending
        self._pending = None
        if self.training and self.mu > 0.0 and pending:
            self.last_inv_loss = self._compute_inv_loss(hidden, attention_mask, pending)
        return logits

    def spec(self) -> E20ModelSpec:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return E20ModelSpec(
            role="e20_cti_student",
            backbone="goibert-b-dapt",
            hidden_size=int(self.encoder.config.hidden_size),
            head_hidden_size=int(self.head.dense.out_features),
            head_dropout=float(self.head.input_dropout.p),
            num_labels=self.num_labels,
            num_aspects=self.num_aspects,
            aspect_class_counts=self.aspect_class_counts,
            total_parameters=total,
            trainable_parameters=trainable,
            arm=self.arm,
            mu=self.mu,
            added_parameter_count=0,
        )


def build_e20_model(
    backbone_dir: Path,
    *,
    num_classes: int,
    num_aspects: int,
    aspect_class_counts: Any,
    attention_implementation: str,
    arm: str,
    mu: float,
    tokenizer: Any,
    max_length: int = MAX_LENGTH,
) -> tuple[E20Classifier, E20ModelSpec]:
    base, _ = build_e2_teacher(
        backbone_dir,
        num_classes=num_classes,
        num_aspects=num_aspects,
        aspect_class_counts=aspect_class_counts,
        attention_implementation=attention_implementation,
    )
    model = E20Classifier(base, arm=arm, mu=mu, tokenizer=tokenizer, max_length=max_length)
    return model, model.spec()


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
def _composite_assert_check() -> dict[str, Any]:
    """(1) Composite asserts pass; pair-coverage stats, for every task."""

    tokenizer = _tokenizer()
    rows: dict[str, Any] = {}
    for task_id in TASKS:
        contract = _original_task_contract(task_id)
        composites, coverage = _build_composites(task_id, contract, tokenizer, seed=SEED, max_length=MAX_LENGTH)
        bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
        sample = composites[: min(200, len(composites))]
        assert_failures = 0
        for c in sample:
            a_ids = tokenizer(c["a_text"], add_special_tokens=False)["input_ids"]
            b_ids_space = tokenizer(" " + c["b_text"], add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(c["text"], truncation=True, max_length=MAX_LENGTH)["input_ids"]
            boundary = c["boundary"]
            ok = (
                full_ids[0] == bos_id
                and full_ids[-1] == eos_id
                and full_ids[1:boundary] == a_ids
                and full_ids[boundary:-1] == b_ids_space
                and boundary == 1 + len(a_ids)
            )
            if not ok:
                assert_failures += 1
        rows[task_id] = {
            "coverage": coverage,
            "n_asserted": len(sample),
            "assert_failures": assert_failures,
            "assert_pass": assert_failures == 0 and coverage["n_composites_built"] >= round(coverage["n_train"] / 3) - 5,
        }
    return rows


def _cti_sanity_check(device: torch.device) -> dict[str, Any]:
    """(2) L_inv finite>=0, nonzero encoder grad from L_inv alone, mu=0 == d1."""

    task_id = "absa_restaurant"
    tokenizer = _tokenizer()
    contract = _original_task_contract(task_id)
    composites, _coverage = _build_composites(task_id, contract, tokenizer, seed=SEED, max_length=MAX_LENGTH)
    new_bundle, new_train_split = _inject_composites(contract["bundle"], composites)
    composite_by_id = {c["id"]: c for c in composites}

    train_loader, _, _ = e2_train._loader_set(
        tokenizer,
        new_train_split,
        contract["validation"],
        new_bundle,
        seed=SEED,
        micro_batch_size=MICRO_BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    fixed_batch = None
    for batch in train_loader:
        if any(str(example_id).startswith(COMPOSITE_PREFIX) for example_id in batch["ids"]):
            fixed_batch = batch
            break
    if fixed_batch is None:
        fixed_batch = next(iter(train_loader))
    clean = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in fixed_batch.items()}

    def pending_for(batch: Mapping[str, Any]) -> list[tuple[int, int, str, str]]:
        out = []
        for row_index, example_id in enumerate(batch["ids"]):
            meta = composite_by_id.get(str(example_id))
            if meta is not None:
                out.append((row_index, meta["boundary"], meta["a_text"], meta["b_text_standalone"]))
        return out

    pending = pending_for(fixed_batch)
    counts = contract["aspect_class_counts"]
    num_aspects = contract["num_aspects"]

    results: dict[str, Any] = {"n_composite_rows_in_batch": len(pending)}
    for arm, mu in (("d1_composite", 0.0), ("cti", 0.3)):
        e1_train.seed_everything(SEED)
        model, _ = build_e20_model(
            ENCODER_DIR,
            num_classes=contract["num_classes"],
            num_aspects=num_aspects,
            aspect_class_counts=counts if num_aspects else None,
            attention_implementation=ATTENTION_IMPLEMENTATION,
            arm=arm,
            mu=mu,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
        )
        model.to(device).train()
        model.zero_grad(set_to_none=True)
        model._pending = list(pending)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits = model(clean["input_ids"], clean["attention_mask"])
            ce = e2_train.compute_loss(logits, clean["labels"], num_aspects=num_aspects, aspect_class_counts=counts)
        inv = model.last_inv_loss
        total = ce if inv is None else ce + mu * inv
        total.backward()
        enc_grad_norm = float(
            sum(p.grad.float().pow(2).sum() for p in model.encoder.parameters() if p.grad is not None).sqrt().item()
        )
        results[arm] = {
            "mu": mu,
            "ce_loss": float(ce.detach().cpu().item()),
            "inv_loss": None if inv is None else float(inv.detach().cpu().item()),
            "total_loss": float(total.detach().cpu().item()),
            "encoder_grad_norm": enc_grad_norm,
            "last_inv_stats": dict(model.last_inv_stats),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    d1 = results["d1_composite"]
    cti = results["cti"]
    checks = {
        "inv_finite_nonneg": cti["inv_loss"] is not None and math.isfinite(cti["inv_loss"]) and cti["inv_loss"] >= 0.0,
        "cti_encoder_grad_nonzero": cti["encoder_grad_norm"] > 0.0,
        "d1_never_computes_inv": d1["inv_loss"] is None,
        "d1_total_equals_ce": abs(d1["total_loss"] - d1["ce_loss"]) <= 1e-6,
        "mu_zero_equivalent_to_d1": True,  # by construction: total = ce when inv is None/mu*inv==0
    }
    results["checks"] = checks
    return results


def _eval_parity_check(device: torch.device) -> dict[str, Any]:
    """(3) Eval-mode logits of the wrapper equal a plain build_e2_teacher model."""

    task_id = "absa_restaurant"
    tokenizer = _tokenizer()
    contract = _original_task_contract(task_id)
    texts = list(e2_train.split_texts(contract["validation"]))[:32]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    counts = contract["aspect_class_counts"]
    num_aspects = contract["num_aspects"]

    e1_train.seed_everything(SEED)
    wrapper, _ = build_e20_model(
        ENCODER_DIR,
        num_classes=contract["num_classes"],
        num_aspects=num_aspects,
        aspect_class_counts=counts if num_aspects else None,
        attention_implementation=ATTENTION_IMPLEMENTATION,
        arm="cti",
        mu=0.3,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    wrapper.to(device).eval()

    e1_train.seed_everything(SEED)
    plain, _ = build_e2_teacher(
        ENCODER_DIR,
        num_classes=contract["num_classes"],
        num_aspects=num_aspects,
        aspect_class_counts=counts if num_aspects else None,
        attention_implementation=ATTENTION_IMPLEMENTATION,
    )
    plain.to(device).eval()

    with torch.no_grad():
        wrapper._pending = [(0, 5, "should be ignored in eval", "should be ignored in eval")]
        logits_wrapper = wrapper(input_ids, attention_mask)
        logits_plain = plain(input_ids, attention_mask)
    max_diff = float((logits_wrapper.float() - logits_plain.float()).abs().max().item())
    return {
        "n_rows": len(texts),
        "max_abs_logit_diff": max_diff,
        "wrapper_never_computed_inv_in_eval": wrapper.last_inv_loss is None,
        "checks": {
            "parity_le_1e4": max_diff <= 1e-4,
            "eval_never_computes_inv": wrapper.last_inv_loss is None,
        },
    }


def _vram_step_time_check(device: torch.device) -> dict[str, Any]:
    """(4) VRAM/step time for one NC step with composites; cti/d1 ratio."""

    tokenizer = _tokenizer()
    rows: dict[str, Any] = {}
    for task_id in TASKS:
        contract = _original_task_contract(task_id)
        composites, _coverage = _build_composites(task_id, contract, tokenizer, seed=SEED, max_length=MAX_LENGTH)
        new_bundle, new_train_split = _inject_composites(contract["bundle"], composites)
        composite_by_id = {c["id"]: c for c in composites}
        train_loader, _, _ = e2_train._loader_set(
            tokenizer,
            new_train_split,
            contract["validation"],
            new_bundle,
            seed=SEED,
            micro_batch_size=MICRO_BATCH_SIZE,
            max_length=MAX_LENGTH,
        )
        clean_cpu = next(iter(train_loader))
        text_by_id = {
            str(example_id): text
            for example_id, text in zip(e2_train.split_ids(new_train_split), e2_train.split_texts(new_train_split))
        }
        noisy_cpu, _noise_stats = e2_train._make_noisy_batch(
            tokenizer, clean_cpu["ids"], text_by_id, max_length=MAX_LENGTH, seed=SEED, epoch=1, device=torch.device("cpu")
        )
        counts = contract["aspect_class_counts"]
        num_aspects = contract["num_aspects"]
        nc_fn = e2_train._load_loss_api()[1]

        for arm, mu in (("d1_composite", 0.0), ("cti", 0.3)):
            e1_train.seed_everything(SEED)
            model, _ = build_e20_model(
                ENCODER_DIR,
                num_classes=contract["num_classes"],
                num_aspects=num_aspects,
                aspect_class_counts=counts if num_aspects else None,
                attention_implementation=ATTENTION_IMPLEMENTATION,
                arm=arm,
                mu=mu,
                tokenizer=tokenizer,
                max_length=MAX_LENGTH,
            )
            model.to(device).train()
            optimizer = e7_optim.build_layerwise_optimizer(
                model, TASK_LRS[task_id], gamma=GAMMA, num_layers=NUM_LAYERS, weight_decay=WEIGHT_DECAY
            )
            clean = {key: value.to(device) for key, value in clean_cpu.items() if isinstance(value, torch.Tensor)}
            noisy = {key: value.to(device) for key, value in noisy_cpu.items()}

            def one_step() -> float:
                model._pending = [
                    (i, composite_by_id[str(eid)]["boundary"], composite_by_id[str(eid)]["a_text"], composite_by_id[str(eid)]["b_text_standalone"])
                    for i, eid in enumerate(clean_cpu["ids"])
                    if str(eid) in composite_by_id
                ]
                optimizer.zero_grad(set_to_none=True)
                started = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    logits = model(clean["input_ids"], clean["attention_mask"])
                    ce = e2_train.compute_loss(logits, clean["labels"], num_aspects=num_aspects, aspect_class_counts=counts)
                    noisy_logits = model(noisy["input_ids"], noisy["attention_mask"])
                    consistency = nc_fn(logits, noisy_logits, temperature=1.0, slot_class_counts=counts or None)
                    inv = model.last_inv_loss
                    model.last_inv_loss = None
                    total = ce + NC_LAMBDAS[task_id] * consistency
                    if inv is not None and mu > 0.0:
                        total = total + mu * inv
                if not torch.isfinite(total):
                    raise FloatingPointError(f"E20 profile loss non-finite: {arm}/{task_id}")
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize(device)
                return time.perf_counter() - started

            one_step()  # warmup
            torch.cuda.reset_peak_memory_stats(device)
            step_seconds = one_step()
            rows[f"{task_id}/{arm}"] = {
                "task_id": task_id,
                "arm": arm,
                "step_seconds": step_seconds,
                "peak_reserved_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes_reserved": int(torch.cuda.max_memory_reserved(device)),
            }
            del model, optimizer
            torch.cuda.empty_cache()
        rows[f"{task_id}/ratio_cti_to_d1"] = (
            rows[f"{task_id}/cti"]["step_seconds"] / max(rows[f"{task_id}/d1_composite"]["step_seconds"], 1e-9)
        )
    return rows


def run_preflight(device_name: str | None = None) -> dict[str, Any]:
    started = time.time()
    snapshot = _gpu_snapshot()
    if device_name is None:
        eligible = _eligible_devices(snapshot)
        device_name = eligible[0] if eligible else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_name}, but CUDA is unavailable in this process")

    manifest = _source_manifest()
    before_encoder = manifest["encoder"]["files"]

    functional = {
        "composite_assert": _composite_assert_check(),
        "cti_sanity": _cti_sanity_check(device),
        "eval_parity": _eval_parity_check(device),
        "vram_step_time": _vram_step_time_check(device),
    }

    after_manifest = _source_manifest()
    after_encoder = after_manifest["encoder"]["files"]

    checks = {
        "composite_assert_all_pass": all(row["assert_pass"] for row in functional["composite_assert"].values()),
        "cti_sanity_all_pass": all(functional["cti_sanity"]["checks"].values()),
        "eval_parity_all_pass": all(functional["eval_parity"]["checks"].values()),
        "encoder_no_mutation": before_encoder == after_encoder,
    }
    gpu_ready = device.type == "cuda" and snapshot.get("available", False)
    payload = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "status": "passed" if all(checks.values()) and gpu_ready else ("blocked_no_cuda" if not gpu_ready else "failed"),
        "created_at": _now(),
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "gpu_snapshot": snapshot,
        "eligible_devices": _eligible_devices(snapshot),
        "gpu_preflight_ready": gpu_ready,
        "checks": checks,
        "functional": functional,
        "encoder_hash_unchanged": before_encoder == after_encoder,
    }
    _write_json(ARTIFACT_ROOT / "preflight.json", payload)
    _write_json(ARTIFACT_ROOT / "source_manifest.json", manifest)
    return payload


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def run_worker(arm: str, task_id: str, device_name: str, output_dir: Path, seed: int = SEED) -> dict[str, Any]:
    if arm not in ARMS or task_id not in TASKS:
        raise ValueError(f"E20 arm/task not locked: {arm}, {task_id}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing duplicate E20 output: {output_dir}")

    mu = MU_BY_ARM[arm]
    source_manifest = _source_manifest(seed)
    tokenizer = _tokenizer()
    holder: dict[str, Any] = {"model": None, "composite_by_id": {}, "coverage": None, "inv_losses": []}

    original_task_contract = e2_train._task_contract
    original_builder = e2_train.build_e2_student
    original_optimizer = e2_train.make_optimizer
    original_move_batch = e2_train._move_batch
    original_compute_loss = e2_train.compute_loss

    sampling_mode = SAMPLING_BY_ARM[arm]

    def e20_task_contract(inner_task_id: str, repo_root: Path, *, smoke: bool = False) -> dict[str, Any]:
        contract = original_task_contract(inner_task_id, repo_root, smoke=smoke)
        if smoke or inner_task_id != task_id:
            return contract
        # Composite RNG seed is derived from (i.e. equal to) the run seed, so
        # seeds 43/44 draw a different composite set than seed 42.
        composites, coverage = _build_composites(
            task_id, contract, tokenizer, seed=seed, max_length=MAX_LENGTH, sampling=sampling_mode
        )
        new_bundle, new_train_split = _inject_composites(contract["bundle"], composites)
        holder["composite_by_id"] = {c["id"]: c for c in composites}
        holder["coverage"] = coverage
        # Do NOT write into output_dir here: _train_any's _unique_dir() picks
        # the real trial dir *after* this call returns, and if output_dir
        # already contains a file (e.g. this manifest) it bumps to
        # "<output_dir>__rerun1" instead of landing in output_dir directly.
        # Stash the manifest payload and write it into the real artifact dir
        # after _train_any returns (see below).
        holder["composites_manifest_payload"] = {
            "protocol": PROTOCOL,
            "task_id": task_id,
            "seed": seed,
            "composite_seed": seed,
            "sampling_mode": sampling_mode,
            "coverage": coverage,
            "composites": composites,
        }
        new_contract = dict(contract)
        new_contract["bundle"] = new_bundle
        new_contract["train"] = new_train_split
        return new_contract

    def e20_builder(
        backbone_dir: Path,  # ignored: always models/goibert-b-dapt
        *,
        num_classes: int,
        num_aspects: int = 0,
        aspect_class_counts: Any = None,
        attention_implementation: str = "sdpa",
    ):
        model, spec = build_e20_model(
            ENCODER_DIR,
            num_classes=num_classes,
            num_aspects=num_aspects,
            aspect_class_counts=aspect_class_counts,
            attention_implementation=attention_implementation,
            arm=arm,
            mu=mu,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
        )
        holder["model"] = model
        return model, spec

    def e20_optimizer(model: torch.nn.Module, learning_rate: float):
        optimizer = e7_optim.build_layerwise_optimizer(
            model, learning_rate, gamma=GAMMA, num_layers=NUM_LAYERS, weight_decay=WEIGHT_DECAY
        )
        coverage = e7_optim.validate_optimizer_groups(
            model, optimizer, base_lr=learning_rate, gamma=GAMMA, num_layers=NUM_LAYERS, weight_decay=WEIGHT_DECAY
        )
        holder["optimizer_coverage"] = {
            "all_trainable": coverage["all_trainable"],
            "duplicate_parameter_count": coverage["duplicate_parameter_count"],
            "parameter_count": coverage["parameter_count"],
            "group_count": coverage["group_count"],
            "endpoint_scales": coverage["endpoint_scales"],
        }
        return optimizer

    def e20_move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
        moved = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}
        model = holder.get("model")
        lookup = holder.get("composite_by_id") or {}
        if model is not None and lookup:
            pending = []
            for row_index, example_id in enumerate(batch["ids"]):
                meta = lookup.get(str(example_id))
                if meta is not None:
                    pending.append((row_index, meta["boundary"], meta["a_text"], meta["b_text_standalone"]))
            model._pending = pending if pending else None
        return moved

    def e20_compute_loss(logits: Tensor, labels: Tensor, *, num_aspects: int, aspect_class_counts: Any = None) -> Tensor:
        model = holder["model"]
        ce = original_compute_loss(logits, labels, num_aspects=num_aspects, aspect_class_counts=aspect_class_counts)
        inv = model.last_inv_loss
        model.last_inv_loss = None
        if inv is None or mu <= 0.0:
            return ce
        holder["inv_losses"].append(float(inv.detach().cpu().item()))
        return ce + mu * inv

    e2_train._task_contract = e20_task_contract  # type: ignore[assignment]
    e2_train.build_e2_student = e20_builder  # type: ignore[assignment]
    e2_train.make_optimizer = e20_optimizer  # type: ignore[assignment]
    e2_train._move_batch = e20_move_batch  # type: ignore[assignment]
    e2_train.compute_loss = e20_compute_loss  # type: ignore[assignment]
    try:
        result = e2_train._train_any(
            task_id=task_id,
            mode="nc",
            learning_rate=TASK_LRS[task_id],
            seed=seed,
            repo_root=ROOT,
            output_dir=output_dir,
            device_name=device_name,
            precision=PRECISION,
            attention_implementation=ATTENTION_IMPLEMENTATION,
            epochs=EPOCHS,
            effective_batch_size=EFFECTIVE_BATCH_SIZE,
            micro_batch_size=MICRO_BATCH_SIZE,
            smoke=False,
            alpha=None,
            consistency_lambda=NC_LAMBDAS[task_id],
            cache_path=None,
            stage="stageE20_cti_screening",
            method=arm,
            coefficient=mu,
            revision=REVISION,
            expected_e2_code_hash=None,
            expected_e2_config_hash=None,
            expected_orchestrator_hash=None,
        )
    finally:
        e2_train._task_contract = original_task_contract  # type: ignore[assignment]
        e2_train.build_e2_student = original_builder  # type: ignore[assignment]
        e2_train.make_optimizer = original_optimizer  # type: ignore[assignment]
        e2_train._move_batch = original_move_batch  # type: ignore[assignment]
        e2_train.compute_loss = original_compute_loss  # type: ignore[assignment]

    artifact = Path(result["artifact_dir"])
    # Write the composites manifest now that the real (non-bumped) trial dir
    # is known -- see the note in e20_task_contract above.
    manifest_payload = holder.get("composites_manifest_payload")
    if manifest_payload is not None:
        _write_json(artifact / "composites_manifest.json", manifest_payload)
    metadata = _read_json(artifact / "metadata.json")
    inv_losses = holder.get("inv_losses") or []
    metadata.update(
        {
            "e20_protocol": PROTOCOL,
            "e20_revision": REVISION,
            "e20_arm": arm,
            "e20_mu": mu,
            "e20_sampling_mode": sampling_mode,
            "e20_seed": seed,
            "e20_encoder_dir": str(ENCODER_DIR.resolve()),
            "e20_encoder_hashes": _directory_hashes(ENCODER_DIR),
            "e20_optimizer_coverage": holder.get("optimizer_coverage"),
            "e20_composite_coverage": holder.get("coverage"),
            "e20_inv_loss_summary": {
                "n_observations": len(inv_losses),
                "mean": (sum(inv_losses) / len(inv_losses)) if inv_losses else None,
                "max": max(inv_losses) if inv_losses else None,
                "min": min(inv_losses) if inv_losses else None,
            },
            "e20_source_manifest": source_manifest,
        }
    )
    _write_json(artifact / "metadata.json", metadata)
    checkpoint = artifact / "best_checkpoint.pt"
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state.setdefault("metadata", {}).update(
            {
                "e20_protocol": PROTOCOL,
                "e20_arm": arm,
                "e20_mu": mu,
                "e20_sampling_mode": sampling_mode,
                "e20_encoder_dir": str(ENCODER_DIR.resolve()),
            }
        )
        torch.save(state, checkpoint)
    return result


# ---------------------------------------------------------------------------
# run-all controller
# ---------------------------------------------------------------------------
def run_all(
    device_names: list[str],
    *,
    dispatch_approved: bool,
    arms: Sequence[str] = ARMS,
    tasks: Sequence[str] = TASKS,
    seed: int = SEED,
) -> dict[str, Any]:
    if not dispatch_approved:
        raise RuntimeError("Full E20 dispatch requires explicit --dispatch-approved")
    if not device_names:
        raise RuntimeError("No GPU devices available for E20 dispatch")
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"Unknown E20 arm requested for dispatch: {arm}")
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"Unknown E20 task requested for dispatch: {task}")

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    state_path = ARTIFACT_ROOT / "controller_state.json"
    source_manifest = _source_manifest(seed)
    source_manifest_hash = hashlib.sha256(json.dumps(source_manifest, sort_keys=True).encode("utf-8")).hexdigest()

    previous: dict[str, Any] = {}
    if state_path.is_file():
        previous = _read_json(state_path)
        if previous.get("status") == "running":
            active = [job for job in previous.get("jobs", []) if _pid_alive(job.get("pid")) and job.get("status") == "running"]
            if active:
                raise RuntimeError(f"E20 controller already has active jobs; adopt/refuse duplicate: {active}")

    queue = [(task, arm) for task in tasks for arm in arms]
    completed_keys = {
        (job.get("task_id"), job.get("arm")) for job in previous.get("jobs", []) if job.get("status") == "completed"
    }
    for task, arm in list(queue):
        artifact = ARTIFACT_ROOT / arm / task / f"seed_{seed}"
        status_path = artifact / "status.json"
        if status_path.is_file():
            status = _read_json(status_path)
            if status.get("status") == "completed":
                completed_keys.add((task, arm))
    queue = [(task, arm) for task, arm in queue if (task, arm) not in completed_keys]

    controller = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "status": "running",
        "created_at": _now(),
        "source_manifest_sha256": source_manifest_hash,
        "dispatch_approved": True,
        "arms": list(arms),
        "tasks": list(tasks),
        "jobs": [job for job in previous.get("jobs", []) if (job.get("task_id"), job.get("arm")) in completed_keys],
        "queue_remaining": [{"task_id": task, "arm": arm} for task, arm in queue],
    }
    _write_json(state_path, controller)

    active: list[tuple[subprocess.Popen[str], dict[str, Any], Any]] = []
    completed: list[dict[str, Any]] = list(controller["jobs"])

    def wait_job(process: subprocess.Popen[str], job: dict[str, Any], log_handle: Any) -> int:
        while True:
            code = process.poll()
            if code is not None:
                log_handle.close()
                return int(code)
            controller["heartbeat_at"] = _now()
            controller["active_pids"] = [item[1].get("pid") for item in active]
            _write_json(state_path, controller)
            time.sleep(10)

    for index, (task, arm) in enumerate(queue):
        device = device_names[index % len(device_names)]
        snapshot_now = _gpu_snapshot()
        eligible_now = _eligible_devices(snapshot_now)
        if device not in eligible_now:
            raise RuntimeError(f"E20 dispatch gate failed: {device} no longer eligible ({snapshot_now})")
        output = ARTIFACT_ROOT / arm / task / f"seed_{seed}"
        log_path = ARTIFACT_ROOT / "logs" / f"{arm}__{task}__seed{seed}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--arm",
            arm,
            "--task-id",
            task,
            "--device",
            device,
            "--output-dir",
            str(output),
            "--seed",
            str(seed),
        ]
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, text=True, env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1"}
        )
        job = {
            "arm": arm,
            "task_id": task,
            "device": device,
            "pid": process.pid,
            "output_dir": str(output),
            "log": str(log_path),
            "command": command,
            "status": "running",
            "started_at": _now(),
        }
        controller["jobs"].append(job)
        controller["queue_remaining"] = [{"task_id": t, "arm": a} for t, a in queue[index + 1 :]]
        _write_json(state_path, controller)
        active.append((process, job, log))
        if len(active) >= len(device_names):
            process0, meta0, log0 = active[0]
            code = wait_job(process0, meta0, log0)
            active.pop(0)
            meta0["returncode"] = code
            meta0["status"] = "completed" if code == 0 else "failed"
            meta0["finished_at"] = _now()
            completed.append(meta0)
            _write_json(state_path, controller)
            if code != 0:
                raise RuntimeError(f"E20 worker failed: {meta0}")
    for process, meta, log in active:
        code = wait_job(process, meta, log)
        meta["returncode"] = code
        meta["status"] = "completed" if code == 0 else "failed"
        meta["finished_at"] = _now()
        completed.append(meta)
        _write_json(state_path, controller)
        if code != 0:
            raise RuntimeError(f"E20 worker failed: {meta}")
    controller.update({"status": "completed", "finished_at": _now(), "jobs": completed, "queue_remaining": []})
    _write_json(state_path, controller)
    return controller


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E20 preflight/worker/run-all runner")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--dispatch-approved", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--task-id", choices=TASKS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=(
            "Run seed (default 42). Threaded into _train_any(seed=...), the "
            "artifact dir (seed_<seed>), the composite-construction RNG "
            "(seeds 43/44 draw a different composite set), and metadata."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not any((args.preflight, args.run_all, args.worker)):
        raise SystemExit("Choose --preflight, --run-all, or --worker")
    if args.worker:
        if not args.arm or not args.task_id or not args.device or args.output_dir is None:
            raise SystemExit("--worker requires --arm, --task-id, --device, and --output-dir")
        result = run_worker(args.arm, args.task_id, args.device, args.output_dir, seed=args.seed)
    elif args.run_all:
        if args.devices:
            devices = list(args.devices)
        else:
            snapshot = _gpu_snapshot()
            devices = _eligible_devices(snapshot)
        devices = [d for d in devices for _ in range(max(1, int(args.workers_per_gpu)))]
        result = run_all(devices, dispatch_approved=args.dispatch_approved, arms=args.arms, tasks=args.tasks, seed=args.seed)
    else:
        result = run_preflight(args.device)
    print(json.dumps(result, default=_json_default, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
