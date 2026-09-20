"""Shared contract for the CafeBERT aspect-pair pilot (docs/CAFEBERT_ASPECT_PAIR_PLAN.md).

Two arms on absa_restaurant, seed 42, train/validation only, both initialised from the *pretrained*
CafeBERT encoder (uitnlp/CafeBERT, XLM-R-large backbone) with a fresh shared 4-way head:

* A: encoder reads the pair (review verbatim, "mục NN") for NN = 01..12 (aspect identifier only).
* B: encoder reads the pair (review verbatim, Vietnamese aspect description) for the same 12 slots.

Every review is expanded to exactly 12 pairs (absent slots included); the target of pair (id, slot) is
gold[slot] as stored by the frozen loader (integer codes 0..3, never recoded).  Per-pair argmax predictions
are regrouped by (id, slot) into the 12-slot vector and scored with the unchanged E1 evaluator
(pair micro-F1), so A/B are directly comparable to the CafeBERT 48-logit reference run
artifacts/cdbk_pilot/runs/T_absa_restaurant_seed42 (validation pair micro-F1 0.7771971496).

Recipe (frozen from the plan): LR 2e-5, 30 epochs, effective batch 32 reviews = 384 pairs, AdamW
(0.9, 0.999, eps 1e-8, wd 0.01, no decay on bias/LayerNorm), 10 % warmup + linear decay, clip 1.0,
FP32 master weights + BF16 autocast, SDPA, CE over the 4 classes, best epoch by validation pair micro-F1.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
CDBK_SCRIPTS = ROOT / "scripts" / "cdbk_pilot"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import XLMRobertaModel  # noqa: E402


def _load_cdbk_common():
    """scripts/cdbk_pilot/common.py (tokenizer + CafeBERT source contract), loaded by path to avoid the name clash."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("cdbk_pilot_common", CDBK_SCRIPTS / "common.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cdbk_pilot_common"] = module
    spec.loader.exec_module(module)
    return module


cdbk = _load_cdbk_common()
from goibert import train as e1_train  # noqa: E402
from goibert.data import RESTAURANT_ASPECT_NAMES  # noqa: E402
from goibert_e14.modeling import HEAD_DROPOUT, ClassificationHead  # noqa: E402

PROTOCOL = "CafeBERT-aspect-pair-restaurant-v1"
ARTIFACT_ROOT = ROOT / "artifacts" / "cafebert_aspect_pair"
RUN_ROOT = ARTIFACT_ROOT / "runs"
CHECK_ROOT = ARTIFACT_ROOT / "checks"
REFERENCE_RUN = ROOT / "artifacts" / "cdbk_pilot" / "runs" / "T_absa_restaurant_seed42"

TASK_ID = "absa_restaurant"
SEED = 42
EPOCHS = 30
LEARNING_RATE = 2e-5
EFFECTIVE_BATCH_REVIEWS = 32
MAX_LENGTH = 256
TRUNCATION = "only_first"
PRECISION = "bf16"
ATTENTION_IMPLEMENTATION = "sdpa"
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
GRADIENT_CLIP_NORM = 1.0
NUM_CLASSES = 4
NUM_ASPECTS = 12
HEAD_HIDDEN = 1024

TEACHER_HF_REPO = cdbk.TEACHER_HF_REPO
TEACHER_HF_COMMIT = cdbk.TEACHER_HF_COMMIT

# Locked before training (plan §3).  Order == RESTAURANT_ASPECT_NAMES of the loader.
ASPECT_TEXTS: dict[str, tuple[str, ...]] = {
    "A": tuple(f"mục {slot + 1:02d}" for slot in range(NUM_ASPECTS)),
    "B": (
        "không gian nhà hàng",
        "giá đồ uống",
        "chất lượng đồ uống",
        "cách trình bày và sự lựa chọn đồ uống",
        "giá món ăn",
        "chất lượng món ăn",
        "cách trình bày và sự lựa chọn món ăn",
        "vị trí nhà hàng",
        "đánh giá chung về nhà hàng",
        "các khía cạnh khác của nhà hàng",
        "mức giá chung của nhà hàng",
        "dịch vụ phục vụ",
    ),
}
ARMS = tuple(ASPECT_TEXTS)
# Integer codes are taken verbatim from the loader's gold vectors.  Names are for display only.
LABEL_NAMES = {0: "absent", 1: "code1", 2: "code2", 3: "code3"}


def mapping_table(arm: str) -> list[dict[str, Any]]:
    texts = ASPECT_TEXTS[arm]
    if len(texts) != NUM_ASPECTS or len(RESTAURANT_ASPECT_NAMES) != NUM_ASPECTS:
        raise RuntimeError("aspect mapping must have exactly 12 entries")
    return [{"slot": slot, "aspect": RESTAURANT_ASPECT_NAMES[slot], "text": texts[slot]} for slot in range(NUM_ASPECTS)]


now = cdbk.now
write_json = cdbk.write_json
read_json = cdbk.read_json
sha256_file = cdbk.sha256_file
runtime = cdbk.runtime
gpu_snapshot = cdbk.gpu_snapshot
load_tokenizer = lambda: cdbk.load_tokenizer("cafebert")  # noqa: E731


def source_hash() -> str:
    paths = [
        *sorted((ROOT / "scripts" / "cafebert_aspect_pair").glob("*.py")),
        CDBK_SCRIPTS / "common.py",
        SRC / "goibert" / "train.py",
        SRC / "goibert" / "data.py",
        SRC / "goibert" / "metrics.py",
        SRC / "goibert_e14" / "modeling.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(str(path.resolve().relative_to(ROOT)).encode("utf-8"))
            digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def state_dict_hash(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------- data


def task_contract() -> dict[str, Any]:
    contract = cdbk.task_contract()  # loads train + validation only, asserts 12 slots x 4 classes
    bundle = contract["bundle"]
    if tuple(bundle.aspect_names) != tuple(RESTAURANT_ASPECT_NAMES):
        raise RuntimeError("loader aspect order differs from RESTAURANT_ASPECT_NAMES")
    return contract


@dataclass(frozen=True)
class Review:
    index: int          # position in the split (validation order for regrouping)
    id: str
    text: str
    gold: tuple[int, ...]


def reviews_of(split: Any) -> list[Review]:
    ids = [str(v) for v in e1_train.split_ids(split)]
    texts = list(e1_train.split_texts(split))
    labels = [tuple(int(x) for x in row) for row in e1_train.split_labels(split)]
    if not (len(ids) == len(texts) == len(labels)):
        raise RuntimeError("split fields have inconsistent lengths")
    if len(set(ids)) != len(ids):
        raise RuntimeError("review IDs are not unique")
    for row in labels:
        if len(row) != NUM_ASPECTS or any(v < 0 or v >= NUM_CLASSES for v in row):
            raise RuntimeError(f"gold vector outside the 12 x {{0..3}} contract: {row}")
    return [Review(i, ids[i], texts[i], labels[i]) for i in range(len(ids))]


@dataclass(frozen=True)
class Pair:
    review_index: int
    id: str
    slot: int
    label: int
    input_ids: tuple[int, ...]
    truncated: bool


def encode_pairs(tokenizer: Any, reviews: Sequence[Review], arm: str) -> list[list[Pair]]:
    """Expand each review into its 12 (review, aspect_text) pairs using the official pair API.

    Returns one list of 12 Pair objects per review, slot order 0..11.  ``truncated`` marks pairs whose
    untruncated pair encoding exceeds MAX_LENGTH (the review side is what ``only_first`` would cut).
    """

    texts = ASPECT_TEXTS[arm]
    review_texts = [r.text for r in reviews]
    per_review: list[list[Pair]] = [[] for _ in reviews]
    for slot in range(NUM_ASPECTS):
        full = tokenizer(review_texts, [texts[slot]] * len(reviews), add_special_tokens=True, truncation=False,
                         padding=False, return_attention_mask=False)["input_ids"]
        cut = tokenizer(review_texts, [texts[slot]] * len(reviews), add_special_tokens=True, truncation=TRUNCATION,
                        max_length=MAX_LENGTH, padding=False, return_attention_mask=False)["input_ids"]
        for i, review in enumerate(reviews):
            ids = list(cut[i])
            if len(ids) > MAX_LENGTH:
                raise RuntimeError("tokenizer returned a pair longer than MAX_LENGTH")
            per_review[i].append(Pair(review.index, review.id, slot, review.gold[slot], tuple(ids), len(full[i]) > MAX_LENGTH))
    for i, review in enumerate(reviews):
        if [p.slot for p in per_review[i]] != list(range(NUM_ASPECTS)):
            raise RuntimeError("slot expansion order broken")
        if tuple(p.label for p in per_review[i]) != review.gold:
            raise RuntimeError("expanded labels do not equal the gold vector")
    return per_review


def regroup(reviews: Sequence[Review], pair_predictions: Mapping[tuple[str, int], int]) -> list[list[int]]:
    """Regroup per-pair predictions keyed by (id, slot) into 12-slot vectors in review order (fail closed)."""

    expected = {(r.id, slot) for r in reviews for slot in range(NUM_ASPECTS)}
    keys = set(pair_predictions)
    if keys != expected:
        missing = sorted(expected - keys)[:5]
        extra = sorted(keys - expected)[:5]
        raise RuntimeError(f"pair predictions do not cover every (id, slot) exactly once: missing={missing} extra={extra}")
    return [[int(pair_predictions[(r.id, slot)]) for slot in range(NUM_ASPECTS)] for r in reviews]


class ReviewPairDataset(Dataset[list[Pair]]):
    """One item == one review == its 12 encoded pairs; shuffling happens at review level."""

    def __init__(self, per_review: Sequence[Sequence[Pair]]):
        self.per_review = [list(p) for p in per_review]

    def __len__(self) -> int:
        return len(self.per_review)

    def __getitem__(self, index: int) -> list[Pair]:
        return self.per_review[index]


def _collate_reviews(groups: list[list[Pair]]) -> list[Pair]:
    return [pair for group in groups for pair in group]


def review_loader(per_review: Sequence[Sequence[Pair]], *, reviews_per_group: int, shuffle: bool, seed: int) -> DataLoader:
    """Same sampler construction as the reference run (RandomSampler + torch.Generator(seed)) so the review
    permutation per epoch equals the baseline's, only the batch now expands to <= 12 * reviews_per_group pairs."""

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ReviewPairDataset(per_review), batch_size=reviews_per_group, shuffle=shuffle,
                      generator=generator if shuffle else None, num_workers=0, collate_fn=_collate_reviews)


def pad_microbatch(tokenizer: Any, pairs: Sequence[Pair], device: torch.device) -> dict[str, Any]:
    encoded = tokenizer.pad([{"input_ids": list(p.input_ids)} for p in pairs], padding=True, return_tensors="pt")
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "labels": torch.tensor([p.label for p in pairs], dtype=torch.long, device=device),
        "keys": [(p.id, p.slot) for p in pairs],
    }


def iter_microbatches(pairs: Sequence[Pair], micro: int) -> Iterator[Sequence[Pair]]:
    for start in range(0, len(pairs), micro):
        yield pairs[start : start + micro]


# --------------------------------------------------------------------------- model


class AspectPairClassifier(nn.Module):
    """CafeBERT encoder + fresh shared head: CLS -> dropout .1 -> dense 1024 -> tanh -> dropout .1 -> 4 logits."""

    def __init__(self, encoder: XLMRobertaModel) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = ClassificationHead(int(encoder.config.hidden_size), NUM_CLASSES, dropout=HEAD_DROPOUT, hidden_size=HEAD_HIDDEN)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return self.head(outputs.last_hidden_state)  # [n_pairs, 4]


def build_model() -> tuple[AspectPairClassifier, dict[str, Any]]:
    encoder = XLMRobertaModel.from_pretrained(
        TEACHER_HF_REPO, revision=TEACHER_HF_COMMIT, local_files_only=True, add_pooling_layer=False,
        attn_implementation=ATTENTION_IMPLEMENTATION,
    )
    config = encoder.config
    expected = {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096}
    actual = {name: int(getattr(config, name)) for name in expected}
    if actual != expected:
        raise RuntimeError(f"CafeBERT encoder architecture mismatch: expected {expected}, got {actual}")
    model = AspectPairClassifier(encoder)
    snapshot = cdbk._snapshot_dir(TEACHER_HF_REPO, TEACHER_HF_COMMIT)
    head_state = {k: v for k, v in model.state_dict().items() if k.startswith("head.")}
    spec = {
        "backbone": TEACHER_HF_REPO, "hf_commit": TEACHER_HF_COMMIT, "encoder_class": type(encoder).__name__,
        "weight_file_sha256": sha256_file((snapshot / "model.safetensors").resolve()),
        "config_name_or_path": getattr(config, "_name_or_path", None),
        **actual,
        "head": f"CLS -> dropout{HEAD_DROPOUT} -> dense{HEAD_HIDDEN} -> tanh -> dropout{HEAD_DROPOUT} -> {NUM_CLASSES} logits (shared across 12 aspects)",
        "head_class": "goibert_e14.modeling.ClassificationHead",
        "head_parameters": sum(v.numel() for v in head_state.values()),
        "head_init_sha256": state_dict_hash(head_state),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "note": "pretrained CafeBERT encoder (not the fine-tuned T checkpoint); pooler dropped; MLM head dropped",
    }
    return model, spec


def run_dir(arm: str) -> Path:
    return RUN_ROOT / f"{arm}_{TASK_ID}_seed{SEED}"


def save_model_checkpoint(path: Path, model: nn.Module, *, epoch: int, global_step: int, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": "cafebert-aspect-pair-model-only-v1",
        "model": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "metadata": {k: metadata[k] for k in ("protocol", "arm", "task_id", "seed", "learning_rate", "model", "source_hash", "aspect_mapping") if k in metadata},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


load_model_checkpoint = cdbk.load_model_checkpoint


# --------------------------------------------------------------------------- validation


@torch.inference_mode()
def predict_pairs(model: nn.Module, tokenizer: Any, per_review: Sequence[Sequence[Pair]], device: torch.device,
                  precision: str, *, micro: int) -> dict[tuple[str, int], int]:
    model.eval()
    predictions: dict[tuple[str, int], int] = {}
    flat = [pair for group in per_review for pair in group]
    for chunk in iter_microbatches(flat, micro):
        moved = pad_microbatch(tokenizer, chunk, device)
        with e1_train.make_autocast(precision, device):
            logits = model(moved["input_ids"], moved["attention_mask"])
        if logits.shape != (len(chunk), NUM_CLASSES):
            raise RuntimeError(f"unexpected logits shape {tuple(logits.shape)}")
        for key, value in zip(moved["keys"], logits.argmax(dim=-1).tolist()):
            if key in predictions:
                raise RuntimeError(f"duplicate pair prediction for {key}")
            predictions[key] = int(value)
    return predictions


def run_validation(model: nn.Module, tokenizer: Any, reviews: Sequence[Review], per_review: Sequence[Sequence[Pair]],
                   bundle: Any, device: torch.device, precision: str, *, micro: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_predictions = predict_pairs(model, tokenizer, per_review, device, precision, micro=micro)
    predicted = regroup(reviews, pair_predictions)
    gold = [list(r.gold) for r in reviews]
    metrics = e1_train.evaluate_predictions(TASK_ID, bundle, gold, predicted, split="validation")
    rows = [{"id": r.id, "gold": list(r.gold), "prediction": pred} for r, pred in zip(reviews, predicted)]
    return metrics, rows


def rows_f1(bundle: Any, rows: Sequence[Mapping[str, Any]]) -> float:
    metrics = e1_train.evaluate_predictions(TASK_ID, bundle, [r["gold"] for r in rows], [r["prediction"] for r in rows], split="validation")
    return e1_train.selection_score(metrics)
