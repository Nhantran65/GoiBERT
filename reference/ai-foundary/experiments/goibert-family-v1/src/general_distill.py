"""Run resumable TinyBERT general distillation for GoiTurn-S or GoiTurn-XS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = DEFAULT_REPO_ROOT / "data" / "corpus_v1" / "mlm_mix.txt"
DEFAULT_TEACHER = EXPERIMENT_ROOT / "results" / "pretrained" / "goibert_b_dapt"
MAX_LENGTH = 256
STUDENTS = ("goiturn_s", "goiturn_xs")


def _amp_enabled(torch_module: Any, device: Any) -> bool:
    amp_environment = os.environ.get("GOITURN_AMP")
    is_rocm = getattr(torch_module.version, "hip", None) is not None
    enabled = device.type == "cuda" and (
        amp_environment == "1" if amp_environment is not None else not is_rocm
    )
    if device.type == "cuda":
        print(f"AMP fp16: {enabled} (rocm={is_rocm})", flush=True)
    return enabled


def _latest_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        if (path / "config.json").is_file() and (path / "training_state.pt").is_file():
            candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pca_initialize(student: Any, teacher: Any, torch_module: Any) -> None:
    """PCA-project the selected teacher's actual word embeddings into the student."""
    teacher_words = teacher.get_input_embeddings().weight.detach().float().cpu()
    student_words = student.get_input_embeddings().weight
    if teacher_words.shape[0] != student_words.shape[0]:
        raise ValueError(
            "Teacher and student vocabularies differ: "
            f"{teacher_words.shape[0]} vs {student_words.shape[0]}"
        )
    hidden_size = student_words.shape[1]
    if hidden_size > teacher_words.shape[1]:
        raise ValueError("Student hidden size exceeds teacher embedding width")
    centered = teacher_words - teacher_words.mean(dim=0, keepdim=True)
    _, _, components = torch_module.pca_lowrank(
        centered, q=hidden_size, center=False, niter=2
    )
    projected = centered @ components
    projected = projected / projected.std(dim=0, unbiased=False).clamp_min(1e-12)
    target_std = student_words.detach().float().std(dim=0, unbiased=False).cpu()
    initialized = projected * target_std
    with torch_module.no_grad():
        student_words.copy_(
            initialized.to(dtype=student_words.dtype, device=student_words.device)
        )
    print(
        f"PCA-initialized {hidden_size}-wide student embeddings from selected teacher",
        flush=True,
    )


def _save_model_robust(model: Any, out_dir: Path) -> None:
    # safetensors serialization mmaps the target file, which fails with OS error 12 on
    # Windows-mounted Docker volumes; fall back to the classic pytorch_model.bin format.
    try:
        model.save_pretrained(out_dir)
    except Exception as exc:
        print(f"safetensors save failed ({exc}); retrying with safe_serialization=False", flush=True)
        model.save_pretrained(out_dir, safe_serialization=False)


def _save_checkpoint(
    output_dir: Path,
    step: int,
    student: Any,
    tokenizer: Any,
    aligner: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    use_amp: bool,
    loss_tail: list[dict[str, float]],
    torch_module: Any,
) -> Path:
    checkpoint_dir = output_dir / f"checkpoint-{step:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _save_model_robust(student, checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    state = {
        "step": step,
        "aligner": aligner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss_curve_tail": loss_tail[-20:],
    }
    if use_amp:
        state["scaler"] = scaler.state_dict()
    torch_module.save(state, checkpoint_dir / "training_state.pt")
    print(f"Saved checkpoint {checkpoint_dir}", flush=True)
    return checkpoint_dir


def train(args: argparse.Namespace) -> None:
    import torch
    import transformers
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoTokenizer,
        RobertaModel,
        get_linear_schedule_with_warmup,
    )

    from distill import DistillationAligner, layer_mapping
    from mlm_pretrain import iter_packed_batches
    from modeling import load_architectures, make_config

    repo_root = args.repo_root.resolve()
    corpus_path = args.corpus.resolve()
    stats_path = corpus_path.parent / "stats.json"
    if not corpus_path.is_file() or not stats_path.is_file():
        raise FileNotFoundError(
            f"Corpus and stats are required: {corpus_path}, {stats_path}"
        )
    requested_teacher = args.teacher_dir.resolve()
    teacher_dir = requested_teacher
    if not (teacher_dir / "config.json").is_file():
        teacher_dir = repo_root / "models" / "bamibert"
        print(
            f"DAPT teacher not found at {requested_teacher}; falling back to {teacher_dir}",
            flush=True,
        )
    else:
        print(f"Using DAPT teacher {teacher_dir}", flush=True)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else EXPERIMENT_ROOT / "results" / "pretrained" / f"{args.student}_gd"
    )
    steps = min(args.steps, 20) if args.smoke else args.steps
    batch_size = min(args.batch_size, 2) if args.smoke else args.batch_size
    checkpoint_every = min(args.checkpoint_every, 10) if args.smoke else args.checkpoint_every
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    attention_implementation = os.environ.get("GOITURN_ATTN") or "eager"
    print(f"Attention implementation: {attention_implementation}", flush=True)

    teacher_config = AutoConfig.from_pretrained(teacher_dir, local_files_only=True)
    teacher_config._attn_implementation = attention_implementation
    teacher = AutoModel.from_pretrained(
        teacher_dir, config=teacher_config, local_files_only=True
    )
    if teacher.config.num_hidden_layers != 12 or teacher.config.hidden_size != 768:
        raise ValueError(
            "General distillation requires a 12-layer, 768-wide GoiBERT/BamiBERT teacher"
        )
    tokenizer = AutoTokenizer.from_pretrained(teacher_dir, local_files_only=True)
    architectures = load_architectures(EXPERIMENT_ROOT)
    architecture = architectures[args.student]
    resume_dir = _latest_checkpoint(output_dir) if args.resume else None
    if resume_dir is not None:
        student_config = AutoConfig.from_pretrained(resume_dir, local_files_only=True)
        student_config._attn_implementation = attention_implementation
        student = AutoModel.from_pretrained(
            resume_dir, config=student_config, local_files_only=True
        )
    else:
        student_config = make_config(architecture, require_attentions=True)
        student_config._attn_implementation = attention_implementation
        student = RobertaModel(student_config, add_pooling_layer=False)
        _pca_initialize(student, teacher, torch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student.to(device)
    aligner = DistillationAligner(
        architecture["num_hidden_layers"], architecture["hidden_size"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(aligner.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, steps),
        num_training_steps=steps,
    )
    use_amp = _amp_enabled(torch, device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_step = 0
    loss_tail: list[dict[str, float]] = []
    if resume_dir is not None:
        state = torch.load(resume_dir / "training_state.pt", map_location="cpu")
        start_step = int(state["step"])
        if start_step > steps:
            raise ValueError(
                f"Checkpoint step {start_step} exceeds requested --steps {steps}"
            )
        aligner.load_state_dict(state["aligner"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        loss_tail = [
            {key: float(value) for key, value in item.items()}
            for item in state.get("loss_curve_tail", [])
        ]
        if use_amp and "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        print(f"Resuming from {resume_dir} at step={start_step}", flush=True)
    elif args.resume:
        print("No valid student checkpoint found; starting from teacher PCA init", flush=True)

    batches = iter_packed_batches(corpus_path, tokenizer, batch_size, MAX_LENGTH)
    last_checkpoint_step = start_step
    # Intent: transfer the DAPT teacher's general representations without task labels or logits.
    # Status: mapped hidden projections and head-mean attention MSE are jointly optimized and resumable.
    # Next: initialize task distillation from the final HF student encoder.
    # Blocker: task-stage checkpoint hooks require scoped train/distill API changes outside this file.
    # Confidence: high; the loss reuses the Stage A mapping and masked alignment implementation.
    for step in range(start_step + 1, steps + 1):
        batch = next(batches)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.inference_mode():
            teacher_output = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
        student.train()
        aligner.train()
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            student_output = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
            hidden_loss, attention_loss = aligner(
                SimpleNamespace(
                    hidden_states=tuple(student_output.hidden_states or ()),
                    attentions=tuple(student_output.attentions or ()),
                ),
                SimpleNamespace(
                    hidden_states=tuple(teacher_output.hidden_states or ()),
                    attentions=tuple(teacher_output.attentions or ()),
                ),
                attention_mask,
            )
            loss = hidden_loss + attention_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite general-distillation loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(student.parameters()) + list(aligner.parameters()), args.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses = {
            "total": float(loss.detach().cpu()),
            "hidden": float(hidden_loss.detach().cpu()),
            "attention": float(attention_loss.detach().cpu()),
        }
        loss_tail.append(losses)
        loss_tail = loss_tail[-20:]
        if step == 1 or step % args.log_every == 0 or step == steps:
            print(
                f"step={step} loss={losses['total']:.6f} "
                f"hidden={losses['hidden']:.6f} attention={losses['attention']:.6f}",
                flush=True,
            )
        if step % checkpoint_every == 0:
            _save_checkpoint(
                output_dir,
                step,
                student,
                tokenizer,
                aligner,
                optimizer,
                scheduler,
                scaler,
                use_amp,
                loss_tail,
                torch,
            )
            last_checkpoint_step = step
    if steps > last_checkpoint_step:
        _save_checkpoint(
            output_dir,
            steps,
            student,
            tokenizer,
            aligner,
            optimizer,
            scheduler,
            scaler,
            use_amp,
            loss_tail,
            torch,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_model_robust(student, output_dir)
    tokenizer.save_pretrained(output_dir)
    manifest = {
        "model": f"{args.student}_gd",
        "student": args.student,
        "completed_steps": steps,
        "teacher_checkpoint": str(teacher_dir),
        "teacher_fallback_used": teacher_dir != requested_teacher,
        "corpus": str(corpus_path),
        "corpus_stats_sha256": _sha256(stats_path),
        "loss_curve_tail": loss_tail,
        "seed": args.seed,
        "architecture": architecture,
        "max_length": MAX_LENGTH,
        "batch_size": batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": min(args.warmup_steps, steps),
        "grad_clip": args.grad_clip,
        "checkpoint_every": checkpoint_every,
        "loss": "mapped hidden-state MSE + head-mean attention MSE; no labels or logits",
        "layer_mapping": layer_mapping(architecture["num_hidden_layers"]),
        "pca_initialization": "selected teacher word embeddings",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor() or "CPU"
        ),
        "attention_implementation": attention_implementation,
        "amp_fp16": use_amp,
        "smoke": args.smoke,
    }
    manifest_path = output_dir / "distill_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Saved final HF student and manifest under {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", choices=STUDENTS, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in ("steps", "batch_size", "warmup_steps", "checkpoint_every", "log_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    steps = min(args.steps, 20) if args.smoke else args.steps
    batch_size = min(args.batch_size, 2) if args.smoke else args.batch_size
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else EXPERIMENT_ROOT / "results" / "pretrained" / f"{args.student}_gd"
    )
    if args.dry_run:
        print(
            f"Dry run: student={args.student} steps={steps} batch_size={batch_size} "
            f"max_length={MAX_LENGTH} resume={args.resume}"
        )
        print(f"Teacher preference: {args.teacher_dir.resolve()}")
        print(f"Teacher fallback: {args.repo_root.resolve() / 'models' / 'bamibert'}")
        print(f"Corpus: {args.corpus.resolve()}")
        print(f"Output: {output_dir}")
        print("Dry run complete: torch and transformers were not imported.")
        return
    train(args)


if __name__ == "__main__":
    main()
