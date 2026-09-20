"""Run resumable continued masked-language-model pretraining for GoiBERT-B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = DEFAULT_REPO_ROOT / "data" / "corpus_v1" / "mlm_mix.txt"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "results" / "pretrained" / "goibert_b_dapt"
MAX_LENGTH = 256
MASK_PROBABILITY = 0.30


def iter_packed_sequences(
    corpus_path: Path,
    tokenizer: Any,
    max_length: int = MAX_LENGTH,
) -> Iterator[list[int]]:
    """Cycle a corpus and concatenate short lines into near-full token sequences."""
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    payload_length = max_length - special_count
    if payload_length < 1:
        raise ValueError("max_length is too small for tokenizer special tokens")
    separator_id = tokenizer.sep_token_id
    if separator_id is None:
        separator_id = tokenizer.eos_token_id
    while True:
        produced = 0
        buffer: list[int] = []
        with corpus_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                token_ids = tokenizer.encode(line, add_special_tokens=False)
                if not token_ids:
                    continue
                if buffer and separator_id is not None:
                    buffer.append(separator_id)
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
            raise ValueError(f"Corpus has no tokenizable lines: {corpus_path}")


def iter_packed_batches(
    corpus_path: Path,
    tokenizer: Any,
    batch_size: int,
    max_length: int = MAX_LENGTH,
) -> Iterator[dict[str, Any]]:
    """Yield dynamically padded Torch batches from packed corpus sequences."""
    import torch

    sequence_iterator = iter_packed_sequences(corpus_path, tokenizer, max_length)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define pad_token_id")
    while True:
        sequences = [next(sequence_iterator) for _ in range(batch_size)]
        longest = max(len(sequence) for sequence in sequences)
        input_ids = torch.full((batch_size, longest), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, longest), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            length = len(sequence)
            input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[row, :length] = 1
        yield {"input_ids": input_ids, "attention_mask": attention_mask}


def mask_tokens(
    input_ids: Any,
    attention_mask: Any,
    tokenizer: Any,
    *,
    probability: float = MASK_PROBABILITY,
    generator: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Apply 30% dynamic MLM masking with the standard 80/10/10 replacement rule."""
    import torch

    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer must define mask_token_id")
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, probability, dtype=torch.float32)
    special_masks = [
        tokenizer.get_special_tokens_mask(row.tolist(), already_has_special_tokens=True)
        for row in labels
    ]
    special_mask = torch.tensor(special_masks, dtype=torch.bool)
    probability_matrix.masked_fill_(special_mask | attention_mask.eq(0), 0.0)
    selected = torch.bernoulli(probability_matrix, generator=generator).bool()
    labels[~selected] = -100

    replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.80), generator=generator).bool()
        & selected
    )
    masked_inputs = input_ids.clone()
    masked_inputs[replaced] = tokenizer.mask_token_id
    random_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.50), generator=generator).bool()
        & selected
        & ~replaced
    )
    random_words = torch.randint(
        len(tokenizer), labels.shape, dtype=torch.long, generator=generator
    )
    masked_inputs[random_replaced] = random_words[random_replaced]
    return masked_inputs, labels, selected


def masking_self_test(torch_module: Any) -> None:
    """Prove deterministically that the masking selector is approximately 30%."""
    class ToyTokenizer:
        mask_token_id = 4

        def __len__(self) -> int:
            return 100

        @staticmethod
        def get_special_tokens_mask(
            values: list[int], *, already_has_special_tokens: bool
        ) -> list[int]:
            del already_has_special_tokens
            return [int(value in {0, 1, 2}) for value in values]

    generator = torch_module.Generator().manual_seed(42)
    inputs = torch_module.randint(5, 100, (128, 256), generator=generator)
    attention = torch_module.ones_like(inputs)
    _, _, selected = mask_tokens(
        inputs, attention, ToyTokenizer(), generator=generator
    )
    fraction = float(selected.float().mean())
    if not 0.285 <= fraction <= 0.315:
        raise AssertionError(f"Expected approximately 30% masking, observed {fraction:.4f}")
    print(f"Masking self-test passed: selected_fraction={fraction:.4f}", flush=True)


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


def _save_checkpoint(
    output_dir: Path,
    step: int,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    use_amp: bool,
    loss_tail: list[float],
    initial_missing_keys: list[str],
    torch_module: Any,
) -> Path:
    return _save_checkpoint_impl(
        output_dir, step, model, tokenizer, optimizer, scheduler, scaler, use_amp,
        loss_tail, initial_missing_keys, torch_module,
    )


def _save_model_robust(model: Any, out_dir: Path) -> None:
    # safetensors serialization mmaps the target file, which fails with OS error 12 on
    # Windows-mounted Docker volumes; fall back to the classic pytorch_model.bin format.
    try:
        model.save_pretrained(out_dir)
    except Exception as exc:
        print(f"safetensors save failed ({exc}); retrying with safe_serialization=False", flush=True)
        model.save_pretrained(out_dir, safe_serialization=False)


def _save_checkpoint_impl(
    output_dir: Path,
    step: int,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    use_amp: bool,
    loss_tail: list[float],
    initial_missing_keys: list[str],
    torch_module: Any,
) -> Path:
    checkpoint_dir = output_dir / f"checkpoint-{step:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _save_model_robust(model, checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss_curve_tail": loss_tail[-20:],
        "initial_missing_keys": initial_missing_keys,
    }
    if use_amp:
        state["scaler"] = scaler.state_dict()
    torch_module.save(state, checkpoint_dir / "training_state.pt")
    print(f"Saved checkpoint {checkpoint_dir}", flush=True)
    return checkpoint_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(args: argparse.Namespace) -> None:
    import torch
    import transformers
    from transformers import (
        AutoConfig,
        AutoModelForMaskedLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    repo_root = args.repo_root.resolve()
    corpus_path = args.corpus.resolve()
    output_dir = args.output_dir.resolve()
    stats_path = corpus_path.parent / "stats.json"
    if not corpus_path.is_file() or not stats_path.is_file():
        raise FileNotFoundError(
            f"Corpus and stats are required: {corpus_path}, {stats_path}"
        )
    steps = min(args.steps, 20) if args.smoke else args.steps
    batch_size = min(args.batch_size, 2) if args.smoke else args.batch_size
    checkpoint_every = min(args.checkpoint_every, 10) if args.smoke else args.checkpoint_every
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    masking_self_test(torch)

    resume_dir = _latest_checkpoint(output_dir) if args.resume else None
    model_source = resume_dir or (repo_root / "models" / "bamibert")
    config = AutoConfig.from_pretrained(model_source, local_files_only=True)
    attention_implementation = os.environ.get("GOITURN_ATTN") or (
        "eager" if getattr(torch.version, "hip", None) is not None else "sdpa"
    )
    config._attn_implementation = attention_implementation
    print(f"Attention implementation: {attention_implementation}", flush=True)
    loaded = AutoModelForMaskedLM.from_pretrained(
        model_source,
        config=config,
        local_files_only=True,
        output_loading_info=True,
    )
    model, loading_info = loaded
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
    initial_missing_keys = list(loading_info.get("missing_keys", []))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, steps),
        num_training_steps=steps,
    )
    use_amp = _amp_enabled(torch, device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_step = 0
    loss_tail: list[float] = []
    if resume_dir is not None:
        state = torch.load(resume_dir / "training_state.pt", map_location="cpu")
        start_step = int(state["step"])
        if start_step > steps:
            raise ValueError(
                f"Checkpoint step {start_step} exceeds requested --steps {steps}"
            )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        loss_tail = [float(value) for value in state.get("loss_curve_tail", [])]
        initial_missing_keys = list(state.get("initial_missing_keys", initial_missing_keys))
        if use_amp and "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        print(f"Resuming from {resume_dir} at step={start_step}", flush=True)
    elif args.resume:
        print("No valid checkpoint found; starting from local BamiBERT", flush=True)

    batches = iter_packed_batches(corpus_path, tokenizer, batch_size, MAX_LENGTH)
    mask_generator = torch.Generator().manual_seed(args.seed + start_step)
    last_checkpoint_step = start_step
    # Intent: make GoiBERT-B genuinely domain-adapted while surviving LUMI preemption.
    # Status: every checkpoint includes HF weights plus optimizer, scheduler, scaler, and loss state.
    # Next: use the final root directory as the general-distillation teacher.
    # Blocker: training needs the generated corpus and an importable Torch/Transformers stack.
    # Confidence: high; dynamic masking is guarded by a deterministic 30% self-test.
    for step in range(start_step + 1, steps + 1):
        batch = next(batches)
        # Vietnamese informal-text pretraining follows the ViSoBERT-reported 30% mask rate;
        # replacement remains the standard 80% mask / 10% random / 10% unchanged split.
        masked_inputs, labels, _ = mask_tokens(
            batch["input_ids"],
            batch["attention_mask"],
            tokenizer,
            generator=mask_generator,
        )
        input_ids = masked_inputs.to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            ).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite MLM loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        loss_value = float(loss.detach().cpu())
        loss_tail.append(loss_value)
        loss_tail = loss_tail[-20:]
        if step == 1 or step % args.log_every == 0 or step == steps:
            print(f"step={step} loss={loss_value:.6f}", flush=True)
        if step % checkpoint_every == 0:
            _save_checkpoint(
                output_dir,
                step,
                model,
                tokenizer,
                optimizer,
                scheduler,
                scaler,
                use_amp,
                loss_tail,
                initial_missing_keys,
                torch,
            )
            last_checkpoint_step = step
    if steps > last_checkpoint_step:
        _save_checkpoint(
            output_dir,
            steps,
            model,
            tokenizer,
            optimizer,
            scheduler,
            scaler,
            use_amp,
            loss_tail,
            initial_missing_keys,
            torch,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_model_robust(model, output_dir)
    tokenizer.save_pretrained(output_dir)
    mlm_head_missing = [key for key in initial_missing_keys if "lm_head" in key]
    manifest = {
        "model": "goibert_b_dapt",
        "completed_steps": steps,
        "corpus": str(corpus_path),
        "corpus_stats_sha256": _sha256(stats_path),
        "loss_curve_tail": loss_tail,
        "seed": args.seed,
        "max_length": MAX_LENGTH,
        "batch_size": batch_size,
        "mask_probability": MASK_PROBABILITY,
        "mask_replacement": "80% mask / 10% random / 10% unchanged",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": min(args.warmup_steps, steps),
        "grad_clip": args.grad_clip,
        "checkpoint_every": checkpoint_every,
        "initial_checkpoint": str(repo_root / "models" / "bamibert"),
        "initial_missing_keys": initial_missing_keys,
        "mlm_head_initialization": (
            "Transformers reported missing MLM-head weights; those weights were randomly initialized."
            if mlm_head_missing
            else "Transformers reported no missing MLM-head weights."
        ),
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
    manifest_path = output_dir / "pretrain_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Saved final HF model and manifest under {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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
    if args.dry_run:
        print(
            f"Dry run: steps={steps} batch_size={batch_size} max_length={MAX_LENGTH} "
            f"mask_probability={MASK_PROBABILITY:.2f} resume={args.resume}"
        )
        print(f"Initial model: {args.repo_root.resolve() / 'models' / 'bamibert'}")
        print(f"Corpus: {args.corpus.resolve()}")
        print(f"Output: {args.output_dir.resolve()}")
        print("Dry run complete: torch and transformers were not imported.")
        return
    train(args)


if __name__ == "__main__":
    main()
