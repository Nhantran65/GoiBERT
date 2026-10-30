"""Stage 1: general distillation from BamiBERT.

The student is a RoBERTa encoder that shares BamiBERT's tokenizer.  Its word
embeddings are initialised from a PCA projection of BamiBERT's embeddings;
all Transformer blocks start from random weights.  On unlabeled text, the
student learns to match, layer by layer:

* BamiBERT's hidden states, through one learned linear projection per
  student layer (mean-squared error over non-padding tokens);
* BamiBERT's head-averaged attention maps (mean-squared error over
  non-padding query/key pairs).

The corpus is a text file with one paragraph per line and a blank line
between documents.  Lines are tokenised in order and packed, separated by
``</s>``, into 256-token sequences.
"""

from __future__ import annotations

import json
import random
import shutil
import time
from itertools import islice
from pathlib import Path
from typing import Iterator

import torch
import transformers
from torch import nn
from transformers import AutoModel, AutoTokenizer, RobertaConfig, RobertaModel, get_linear_schedule_with_warmup

from .training import sha256_file

VARIANTS = {
    "base": {"num_hidden_layers": 10, "hidden_size": 384, "num_attention_heads": 6, "intermediate_size": 1536},
    "large": {"num_hidden_layers": 12, "hidden_size": 512, "num_attention_heads": 8, "intermediate_size": 2048},
}
# Student layer i (0 = embeddings) learns from teacher layer LAYER_MAP[variant][i].
LAYER_MAP = {
    "base": [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12],
    "large": list(range(13)),
}
TEACHER_HIDDEN = 768
SEED = 42
BATCH_SIZE = 32
MAX_LENGTH = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 1000


def student_config(variant: str) -> RobertaConfig:
    config = RobertaConfig(
        vocab_size=20481, type_vocab_size=1, pad_token_id=1, bos_token_id=0, eos_token_id=2,
        position_embedding_type="absolute", hidden_act="gelu", hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1, layer_norm_eps=1e-5, initializer_range=0.02, use_cache=False,
        max_position_embeddings=2050, **VARIANTS[variant],
    )
    config._attn_implementation = "eager"  # attention maps are needed for the loss
    return config


def pca_initialize(student: RobertaModel, teacher: nn.Module) -> None:
    """Project the teacher's word embeddings onto their leading principal components.

    The projection is standardised per dimension, then rescaled to the
    per-dimension scale of the student's random initialisation.
    """

    teacher_weight = teacher.get_input_embeddings().weight.detach().float().cpu()
    student_weight = student.get_input_embeddings().weight
    if teacher_weight.shape[0] != student_weight.shape[0]:
        raise ValueError(f"Vocabulary mismatch: {tuple(teacher_weight.shape)} vs {tuple(student_weight.shape)}")
    centered = teacher_weight - teacher_weight.mean(0, keepdim=True)
    _, _, components = torch.pca_lowrank(centered, q=student.config.hidden_size, center=False, niter=2)
    projected = centered @ components
    projected = projected / projected.std(0, unbiased=False).clamp_min(1e-12)
    initialized = projected * student_weight.detach().float().std(0, unbiased=False).cpu()
    with torch.no_grad():
        student_weight.copy_(initialized.to(dtype=student_weight.dtype, device=student_weight.device))


def iter_packed_sequences(corpus: Path, tokenizer, max_length: int = MAX_LENGTH) -> Iterator[list[int]]:
    """Endless stream of packed sequences; restarts from the top of the file."""

    payload_length = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    separator = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    while True:
        produced, buffer = 0, []
        with corpus.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                token_ids = tokenizer.encode(line, add_special_tokens=False)
                if not token_ids:
                    continue
                if buffer and separator is not None:
                    buffer.append(separator)
                buffer.extend(token_ids)
                while len(buffer) >= payload_length:
                    payload = buffer[:payload_length]
                    del buffer[:payload_length]
                    yield tokenizer.build_inputs_with_special_tokens(payload)
                    produced += 1
        if buffer:
            yield tokenizer.build_inputs_with_special_tokens(buffer)
            produced += 1
        if produced == 0:
            raise ValueError("The corpus has no tokenizable lines")


def iter_packed_batches(corpus: Path, tokenizer, batch_size: int = BATCH_SIZE, max_length: int = MAX_LENGTH, skip_sequences: int = 0):
    iterator = iter_packed_sequences(corpus, tokenizer, max_length)
    for _ in islice(iterator, skip_sequences):
        pass
    pad_id = tokenizer.pad_token_id
    while True:
        sequences = [next(iterator) for _ in range(batch_size)]
        ids = torch.full((batch_size, max(map(len, sequences))), pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, sequence in enumerate(sequences):
            ids[i, : len(sequence)] = torch.tensor(sequence)
            mask[i, : len(sequence)] = 1
        yield {"input_ids": ids, "attention_mask": mask}


def hidden_mse(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = (student.float() - teacher.detach().float()).pow(2).mean(-1)
    mask = mask.to(losses.dtype)
    return (losses * mask).sum() / mask.sum().clamp_min(1)


def attention_mse(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    s, t = student.float().mean(1), teacher.detach().float().mean(1)
    pair = mask.float()[:, :, None] * mask.float()[:, None, :]
    return ((s - t).pow(2) * pair).sum() / pair.sum().clamp_min(1)


class DistillationLoss(nn.Module):
    """Hidden-state loss (through projections) and attention-map loss."""

    def __init__(self, hidden_size: int, layer_map: list[int]):
        super().__init__()
        self.layer_map = layer_map
        self.projections = nn.ModuleList(nn.Linear(hidden_size, TEACHER_HIDDEN) for _ in layer_map)

    def forward(self, student_out, teacher_out, mask):
        hidden = [
            hidden_mse(self.projections[i](student_out.hidden_states[i].float()), teacher_out.hidden_states[j], mask)
            for i, j in enumerate(self.layer_map)
        ]
        attention = [
            attention_mse(student_out.attentions[i], teacher_out.attentions[j - 1], mask)
            for i, j in enumerate(self.layer_map[1:])
        ]
        return torch.stack(hidden).mean(), torch.stack(attention).mean()


def _latest_checkpoint(out: Path) -> Path | None:
    candidates = [p for p in (out / "checkpoints").glob("checkpoint-*") if (p / "training_state.pt").is_file()]
    return sorted(candidates)[-1] if candidates else None


def _save_checkpoint(out: Path, step: int, student, criterion, optimizer, scheduler, elapsed: float) -> None:
    base = out / "checkpoints"
    base.mkdir(parents=True, exist_ok=True)
    tmp, final = base / f".checkpoint-{step:08d}.tmp", base / f"checkpoint-{step:08d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    student.save_pretrained(tmp, safe_serialization=True)
    torch.save(
        {
            "step": step, "projections": criterion.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "elapsed_seconds": elapsed, "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        },
        tmp / "training_state.pt",
    )
    tmp.rename(final)
    for old in sorted(base.glob("checkpoint-*"))[:-2]:
        shutil.rmtree(old)


def distill(variant: str, corpus: Path, teacher_name: str, out: Path, *, steps: int = 120_000, resume: bool = False,
            checkpoint_every: int = 2000, device_name: str = "cuda:0", local_files_only: bool = False) -> Path:
    """Run general distillation and write the backbone to ``out/final``.

    ``steps=0`` writes the initialised student without any distillation.
    """

    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {list(VARIANTS)}")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device(device_name)
    teacher = AutoModel.from_pretrained(teacher_name, attn_implementation="eager", add_pooling_layer=False, local_files_only=local_files_only).to(device).eval()
    if teacher.config.num_hidden_layers != 12 or teacher.config.hidden_size != TEACHER_HIDDEN:
        raise ValueError("The teacher must be BamiBERT (12 layers, hidden size 768)")
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, local_files_only=local_files_only)

    resume_from = _latest_checkpoint(out) if resume else None
    if resume_from:
        student = RobertaModel.from_pretrained(resume_from, attn_implementation="eager", add_pooling_layer=False)
    else:
        student = RobertaModel(student_config(variant), add_pooling_layer=False)
        pca_initialize(student, teacher)
    final = out / "final"
    if steps == 0:
        final.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(final, safe_serialization=True)
        tokenizer.save_pretrained(final)
        return final

    student.to(device)
    criterion = DistillationLoss(student.config.hidden_size, LAYER_MAP[variant]).to(device)
    params = list(student.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=steps)
    first_step, elapsed_before = 0, 0.0
    if resume_from:
        state = torch.load(resume_from / "training_state.pt", map_location="cpu", weights_only=False)
        first_step, elapsed_before = state["step"], state["elapsed_seconds"]
        criterion.load_state_dict(state["projections"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    batches = iter_packed_batches(corpus, tokenizer, BATCH_SIZE, MAX_LENGTH, skip_sequences=first_step * BATCH_SIZE)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    tail = []
    with (out / "loss_curve.jsonl").open("a", encoding="utf-8") as log:
        for step in range(first_step + 1, steps + 1):
            batch = next(batches)
            ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_out = teacher(input_ids=ids, attention_mask=mask, output_hidden_states=True, output_attentions=True, return_dict=True)
            student.train()
            criterion.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                student_out = student(input_ids=ids, attention_mask=mask, output_hidden_states=True, output_attentions=True, return_dict=True)
            with torch.autocast("cuda", enabled=False):
                hidden, attention = criterion(student_out, teacher_out, mask)
                loss = hidden + attention
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            values = {"step": step, "total": float(loss.detach()), "hidden": float(hidden.detach()), "attention": float(attention.detach())}
            tail = (tail + [values])[-20:]
            if step == 1 or step % 100 == 0 or step == steps:
                log.write(json.dumps({**values, "elapsed_seconds": elapsed_before + time.perf_counter() - start}) + "\n")
                log.flush()
            if step % checkpoint_every == 0 or step == steps:
                _save_checkpoint(out, step, student, criterion, optimizer, scheduler, elapsed_before + time.perf_counter() - start)

    final.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(final, safe_serialization=True)
    tokenizer.save_pretrained(final)
    manifest = {
        "variant": variant, "steps": steps, "teacher": str(teacher_name), "corpus_sha256": sha256_file(corpus),
        "layer_map": LAYER_MAP[variant], "architecture": VARIANTS[variant], "seed": SEED, "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "warmup_steps": WARMUP_STEPS,
        "parameters": sum(p.numel() for p in student.parameters()), "loss_tail": tail,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "elapsed_seconds": elapsed_before + time.perf_counter() - start,
    }
    (final / "distill_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return final
