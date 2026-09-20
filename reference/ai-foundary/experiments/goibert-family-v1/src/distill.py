"""TinyBERT-style task distillation for GoiTurn-XS and GoiTurn-S."""

from __future__ import annotations

import math
import platform
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
import transformers
from torch import nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data import DatasetSplits, load_dataset
from modeling import ClassifierOutput, build_model
from train import (
    EXPERIMENT_ROOT,
    NC_WEIGHT,
    amp_enabled,
    configure_output_head,
    evaluate,
    lr_slug,
    make_loaders,
    seed_everything,
    seed_suffix,
    selection_score,
    supervised_loss,
    symmetric_kl,
    tokenize_corrupted_view,
    write_json,
)


TEMPERATURE = 2.0
HARD_LABEL_WEIGHT = 0.5
LOGIT_WEIGHT = 0.5
HIDDEN_WEIGHT = 0.1
ATTENTION_WEIGHT = 0.1
TEACHER_LAYERS = 12
TEACHER_HIDDEN_SIZE = 768


def layer_mapping(student_layers: int) -> list[int]:
    """Map embeddings and each 1-based student layer to BamiBERT layers."""
    if student_layers < 1:
        raise ValueError("student_layers must be positive")
    return [0] + [round(index * TEACHER_LAYERS / student_layers) for index in range(1, student_layers + 1)]


def masked_hidden_mse(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Average hidden-state MSE over dimensions and non-padding tokens."""
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            f"Aligned hidden shapes differ: {tuple(student_hidden.shape)} vs {tuple(teacher_hidden.shape)}"
        )
    token_loss = (student_hidden.float() - teacher_hidden.detach().float()).pow(2).mean(dim=-1)
    mask = attention_mask.to(dtype=token_loss.dtype)
    return (token_loss * mask).sum() / mask.sum().clamp_min(1.0)


def masked_attention_mse(
    student_attention: torch.Tensor,
    teacher_attention: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Align head-mean attention probabilities over valid token pairs."""
    if student_attention.ndim != 4 or teacher_attention.ndim != 4:
        raise ValueError("Attention tensors must have shape [batch, heads, query, key]")
    student_mean = student_attention.float().mean(dim=1)
    teacher_mean = teacher_attention.detach().float().mean(dim=1)
    if student_mean.shape != teacher_mean.shape:
        raise ValueError(
            f"Head-mean attention shapes differ: {tuple(student_mean.shape)} vs {tuple(teacher_mean.shape)}"
        )
    token_mask = attention_mask.to(dtype=student_mean.dtype)
    pair_mask = token_mask.unsqueeze(1) * token_mask.unsqueeze(2)
    pair_loss = (student_mean - teacher_mean).pow(2)
    return (pair_loss * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)


def logit_distillation_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
    """Compute KL(student/T || teacher/T), as required by the experiment spec."""
    student_log_probabilities = functional.log_softmax(
        student_logits.float() / TEMPERATURE, dim=-1
    )
    teacher_log_probabilities = functional.log_softmax(
        teacher_logits.detach().float() / TEMPERATURE, dim=-1
    )
    per_class = student_log_probabilities.exp() * (
        student_log_probabilities - teacher_log_probabilities
    )
    return per_class.sum(dim=-1).mean()


class DistillationAligner(nn.Module):
    """Learn student-to-teacher hidden projections and compute mapped losses."""

    def __init__(self, student_layers: int, student_hidden_size: int) -> None:
        super().__init__()
        self.mapping = layer_mapping(student_layers)
        self.projections = nn.ModuleList(
            nn.Linear(student_hidden_size, TEACHER_HIDDEN_SIZE)
            for _ in range(student_layers + 1)
        )

    def forward(
        self,
        student_output: ClassifierOutput,
        teacher_output: ClassifierOutput,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(student_output.hidden_states) != len(self.mapping):
            raise ValueError("Student hidden-state count does not match its layer mapping")
        if len(student_output.attentions) != len(self.mapping) - 1:
            raise ValueError(
                "Student attention probabilities are unavailable; use eager attention or unset GOITURN_ATTN"
            )
        hidden_losses = []
        for student_index, teacher_index in enumerate(self.mapping):
            projected = self.projections[student_index](
                student_output.hidden_states[student_index]
            )
            hidden_losses.append(
                masked_hidden_mse(
                    projected,
                    teacher_output.hidden_states[teacher_index],
                    attention_mask,
                )
            )
        attention_losses = []
        for student_index, teacher_layer in enumerate(self.mapping[1:]):
            student_attention = student_output.attentions[student_index]
            teacher_attention = teacher_output.attentions[teacher_layer - 1]
            if not isinstance(student_attention, torch.Tensor) or not isinstance(
                teacher_attention, torch.Tensor
            ):
                raise ValueError("The selected attention implementation did not return probabilities")
            attention_losses.append(
                masked_attention_mse(student_attention, teacher_attention, attention_mask)
            )
        return torch.stack(hidden_losses).mean(), torch.stack(attention_losses).mean()


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _load_teacher(
    checkpoint_path: Path,
    splits: DatasetSplits,
    repo_root: Path,
    device: torch.device,
) -> nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Validation-selected teacher checkpoint is missing: {checkpoint_path}. Run --stage baseline first."
        )
    teacher, _, teacher_counts = build_model(
        "goibert_b",
        len(splits.aspect_names),
        repo_root,
        EXPERIMENT_ROOT,
        pretrained=True,
        require_attentions=True,
    )
    teacher, _ = configure_output_head(teacher, teacher_counts, splits)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("aspect_names", []) != splits.aspect_names:
        raise ValueError("Teacher checkpoint aspect order does not match the current dataset")
    teacher.load_state_dict(checkpoint["model_state"])
    teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def train_student_trial(
    dataset: str,
    model_name: str,
    learning_rate: float,
    seed: int,
    smoke: bool,
    repo_root: Path,
    teacher_checkpoint: Path,
    init_dir: Path | None = None,
    noise_consistency: bool = False,
) -> dict[str, Any]:
    """Train one student LR trial and stop before test evaluation."""
    if model_name not in {"goiturn_xs", "goiturn_s"}:
        raise ValueError(f"Unsupported student {model_name!r}")
    base_run_name = f"{model_name}_gd" if init_dir is not None else model_name
    run_name = f"{base_run_name}_nc" if noise_consistency else base_run_name
    seed_everything(seed)
    splits = load_dataset(dataset, repo_root, smoke=smoke)
    tokenizer = AutoTokenizer.from_pretrained(
        repo_root / "models" / "bamibert", local_files_only=True
    )
    train_loader, valid_loader, _ = make_loaders(splits, tokenizer, seed, smoke)
    student, architecture, model_counts = build_model(
        model_name,
        len(splits.aspect_names),
        repo_root,
        EXPERIMENT_ROOT,
        pretrained=False,
        init_dir=init_dir,
        pca_init=init_dir is None,
        require_attentions=True,
    )
    student, model_counts = configure_output_head(student, model_counts, splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device)
    teacher = _load_teacher(
        teacher_checkpoint,
        splits,
        repo_root,
        device,
    )
    aligner = DistillationAligner(
        architecture["num_hidden_layers"], architecture["hidden_size"]
    ).to(device)
    projection_count = sum(parameter.numel() for parameter in aligner.parameters())
    param_counts = {
        **model_counts,
        "distillation_projections": projection_count,
        "training_total": model_counts["model_total"] + projection_count,
    }
    print(
        f"Distillation projections={projection_count:,}; training total={param_counts['training_total']:,}",
        flush=True,
    )
    epochs = 1 if smoke else 15
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(aligner.parameters()),
        lr=learning_rate,
        weight_decay=0.01,
    )
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(total_steps * 0.06),
        num_training_steps=total_steps,
    )
    use_amp = amp_enabled(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_score = -1.0
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        student.train()
        aligner.train()
        totals = {"total": 0.0, "hard": 0.0, "logit": 0.0, "hidden": 0.0, "attention": 0.0}
        if noise_consistency:
            totals["consistency"] = 0.0
        for batch_index, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.inference_mode():
                teacher_output = teacher(
                    input_ids,
                    attention_mask,
                    output_hidden_states=True,
                    output_attentions=True,
                )
            if not isinstance(teacher_output, ClassifierOutput):
                raise TypeError("Teacher did not return distillation details")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                student_output = student(
                    input_ids,
                    attention_mask,
                    output_hidden_states=True,
                    output_attentions=True,
                )
                if not isinstance(student_output, ClassifierOutput):
                    raise TypeError("Student did not return distillation details")
                hard_loss = supervised_loss(student_output.logits, labels)
                logit_loss = logit_distillation_loss(
                    student_output.logits, teacher_output.logits
                )
                hidden_loss, attention_loss = aligner(
                    student_output, teacher_output, attention_mask
                )
                loss = (
                    HARD_LABEL_WEIGHT * hard_loss
                    + LOGIT_WEIGHT * TEMPERATURE**2 * logit_loss
                    + HIDDEN_WEIGHT * hidden_loss
                    + ATTENTION_WEIGHT * attention_loss
                )
                consistency_loss = None
                if noise_consistency:
                    corrupted_input_ids, corrupted_attention_mask = tokenize_corrupted_view(
                        batch["texts"], tokenizer, seed, epoch, batch_index, device
                    )
                    corrupted_logits = student(
                        corrupted_input_ids, corrupted_attention_mask
                    )
                    if not isinstance(corrupted_logits, torch.Tensor):
                        raise TypeError("Noise-consistency training expected classifier logits")
                    consistency_loss = symmetric_kl(
                        student_output.logits, corrupted_logits
                    )
                    loss = loss + NC_WEIGHT * consistency_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            for key, value in {
                "total": loss,
                "hard": hard_loss,
                "logit": logit_loss,
                "hidden": hidden_loss,
                "attention": attention_loss,
            }.items():
                totals[key] += float(value.detach().cpu())
            if consistency_loss is not None:
                totals["consistency"] += float(consistency_loss.detach().cpu())
        metrics = evaluate(
            student, valid_loader, device, splits.is_absa, splits.class_names
        )
        score = selection_score(metrics, splits.is_absa)
        if not math.isfinite(score):
            raise RuntimeError(f"Non-finite validation score at epoch {epoch}")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics
            best_state = _clone_state(student)
        loss_components = {key: value / len(train_loader) for key, value in totals.items()}
        history.append({"epoch": epoch, "losses": loss_components, "metrics": metrics})
        print(
            f"{run_name} lr={learning_rate:g} epoch={epoch}/{epochs} "
            f"loss={loss_components['total']:.4f} val_score={score:.4f}",
            flush=True,
        )
    candidate_path = (
        EXPERIMENT_ROOT
        / "results"
        / "checkpoints"
        / ".candidates"
        / f"{dataset}__{run_name}__lr_{lr_slug(learning_rate)}{seed_suffix(seed)}.pt"
    )
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "config": architecture,
            "aspect_names": splits.aspect_names,
            "best_epoch": best_epoch,
            "lr": learning_rate,
            "seed": seed,
        },
        candidate_path,
    )
    result_path = (
        EXPERIMENT_ROOT
        / "results"
        / f"{dataset}__{run_name}__lr_{lr_slug(learning_rate)}{seed_suffix(seed)}.json"
    )
    description = (
        "General-distilled student, TinyBERT-style task distillation"
        if init_dir is not None
        else "PCA-initialized student, TinyBERT-style task distillation"
    )
    if noise_consistency:
        description += ", noise consistency"
    result: dict[str, Any] = {
        "dataset": dataset,
        "model": run_name,
        "description": description,
        "mode": "distill",
        "smoke": smoke,
        "seed": seed,
        "config": architecture,
        "aspect_names": splits.aspect_names,
        "lr": learning_rate,
        "epochs_run": epochs,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "best_validation_metrics": best_metrics,
        "per_epoch_validation": history,
        "test_metrics": None,
        "test_evaluated": False,
        "param_counts": param_counts,
        "distillation": {
            "temperature": TEMPERATURE,
            "hard_label_weight": HARD_LABEL_WEIGHT,
            "logit_weight": LOGIT_WEIGHT,
            "hidden_weight": HIDDEN_WEIGHT,
            "attention_weight": ATTENTION_WEIGHT,
            "layer_mapping": layer_mapping(architecture["num_hidden_layers"]),
            "attention_head_alignment": "mean probabilities across heads before masked MSE",
            "teacher_checkpoint": str(teacher_checkpoint),
        },
        "wall_clock_train_seconds": time.perf_counter() - started,
        "candidate_checkpoint": str(candidate_path),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor() or "CPU"
        ),
    }
    if noise_consistency:
        result["noise_consistency"] = {"enabled": True, "weight": NC_WEIGHT}
    if init_dir is not None:
        result["architecture_name"] = model_name
        result["initialization_checkpoint"] = str(init_dir)
    write_json(result_path, result)
    result["result_path"] = str(result_path)
    print(f"Wrote validation-only trial {result_path}", flush=True)
    return result


def finalize_student_trial(result: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Evaluate one validation-selected student once on test and save it."""
    seed_everything(int(result["seed"]))
    splits = load_dataset(result["dataset"], repo_root, smoke=bool(result["smoke"]))
    tokenizer = AutoTokenizer.from_pretrained(
        repo_root / "models" / "bamibert", local_files_only=True
    )
    _, _, test_loader = make_loaders(splits, tokenizer, int(result["seed"]), bool(result["smoke"]))
    architecture_name = result.get("architecture_name", result["model"].removesuffix("_nc"))
    student, _, model_counts = build_model(
        architecture_name,
        len(splits.aspect_names),
        repo_root,
        EXPERIMENT_ROOT,
        pretrained=False,
    )
    student, model_counts = configure_output_head(student, model_counts, splits)
    candidate_path = Path(result["candidate_checkpoint"])
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    student.load_state_dict(candidate["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device)
    result["test_metrics"] = evaluate(
        student, test_loader, device, splits.is_absa, splits.class_names
    )
    result["test_evaluated"] = True
    result["selected_by"] = "best validation score across the LR grid"
    result["param_counts"].update(model_counts)
    final_checkpoint = (
        EXPERIMENT_ROOT
        / "results"
        / "checkpoints"
        / f"{result['dataset']}__{result['model']}{seed_suffix(int(result['seed']))}.pt"
    )
    final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": candidate["model_state"],
            "config": result["config"],
            "aspect_names": splits.aspect_names,
            "best_epoch": result["best_epoch"],
            "lr": result["lr"],
            "seed": result["seed"],
        },
        final_checkpoint,
    )
    result["checkpoint"] = str(final_checkpoint)
    result.pop("candidate_checkpoint", None)
    result_path = Path(result.pop("result_path"))
    write_json(result_path, result)
    best_path = (
        EXPERIMENT_ROOT
        / "results"
        / f"{result['dataset']}__{result['model']}__best{seed_suffix(int(result['seed']))}.json"
    )
    write_json(best_path, result)
    candidate_path.unlink()
    print(f"Saved validation-selected student checkpoint {final_checkpoint}", flush=True)
    return result
