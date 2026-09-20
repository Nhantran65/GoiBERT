"""Evaluate saved GoiBERT-family task checkpoints on deterministic noisy test slices."""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Any

from data import DATASET_NAMES, DatasetSplits, load_dataset
from noise import corrupt, strip_diacritics, teencode


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
CHECKPOINT_DIR = EXPERIMENT_ROOT / "results" / "checkpoints"
MODEL_NAMES = (
    "baseline",
    "baseline_nc",
    "goibert_b_dapt",
    "goibert_b_dapt_nc",
    "goiturn_s_gd",
    "goiturn_s_gd_nc",
    "goiturn_xs_gd",
    "goiturn_xs_gd_nc",
)
ARCHITECTURE_NAMES = {
    "baseline": "goibert_b",
    "goibert_b_dapt": "goibert_b",
    "goiturn_s_gd": "goiturn_s",
    "goiturn_xs_gd": "goiturn_xs",
}
ARCHITECTURE_KEYS = (
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "intermediate_size",
    "max_position_embeddings",
)
ALL_SLICES = ("clean", "no_diacritics", "teencode", "combined")
ABSA_CLASS_NAMES = ("0", "1", "2", "3")


def seed_suffix(seed: int) -> str:
    """Match the task-checkpoint seed suffix used by the family harness."""
    return "" if seed == 42 else f"__seed{seed}"


def parse_models(value: str) -> tuple[str, ...]:
    """Parse and validate a comma-separated checkpoint model filter."""
    models = tuple(part.strip() for part in value.split(",") if part.strip())
    if not models:
        raise argparse.ArgumentTypeError("--models must contain at least one model name")
    unknown = sorted(set(models).difference(MODEL_NAMES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown model(s) {unknown}; choose from {list(MODEL_NAMES)}"
        )
    return models


def planned_slices(selection: str) -> tuple[str, ...]:
    """Return the clean control plus the requested corruption slices."""
    if selection == "all":
        return ALL_SLICES
    return ("clean", "no_diacritics" if selection == "diacritics" else selection)


def discover_checkpoints(
    dataset: str, seed: int, models: tuple[str, ...] | None
) -> list[tuple[str, Path]]:
    """Discover exact dataset/model/seed checkpoint names in stable family order."""
    requested = set(models) if models is not None else set(MODEL_NAMES)
    suffix = seed_suffix(seed)
    discovered = []
    for model_name in MODEL_NAMES:
        if model_name not in requested:
            continue
        path = CHECKPOINT_DIR / f"{dataset}__{model_name}{suffix}.pt"
        if path.is_file():
            discovered.append((model_name, path))
    return discovered


def build_slice_texts(
    clean_texts: list[str], slices: tuple[str, ...], seed: int
) -> dict[str, list[str]]:
    """Build each selected slice from clean text with slice-local deterministic RNGs."""
    output = {"clean": list(clean_texts)}
    if "no_diacritics" in slices:
        output["no_diacritics"] = [strip_diacritics(text) for text in clean_texts]
    if "teencode" in slices:
        rng = random.Random(seed)
        output["teencode"] = [teencode(text, rng, p_word=0.6) for text in clean_texts]
    if "combined" in slices:
        rng = random.Random(seed)
        output["combined"] = [corrupt(text, rng) for text in clean_texts]
    return output


def primary_macro_f1(metrics: dict[str, Any]) -> float:
    """Use classification macro F1 for both flat classification and ABSA."""
    key = "macro_f1" if "macro_f1" in metrics else "classification_macro_f1"
    return float(metrics[key])


def add_absa_breakdown(
    metrics: dict[str, Any],
    gold: list[list[int]],
    predicted: list[list[int]],
    aspect_names: list[str],
    classification_metrics: Any,
) -> dict[str, Any]:
    """Add reusable flat-class and per-aspect metrics to an ABSA result."""
    if len(gold) != len(predicted):
        raise ValueError("ABSA gold and prediction row counts differ")
    flat_gold = [value for row in gold for value in row]
    flat_predicted = [value for row in predicted for value in row]
    slot_metrics = classification_metrics(
        flat_gold, flat_predicted, ABSA_CLASS_NAMES
    )
    metrics["accuracy"] = slot_metrics["accuracy"]
    metrics["per_class"] = slot_metrics["per_class"]
    metrics["per_aspect"] = {
        aspect_name: classification_metrics(
            [row[index] for row in gold],
            [row[index] for row in predicted],
            ABSA_CLASS_NAMES,
        )
        for index, aspect_name in enumerate(aspect_names)
    }
    return metrics


def rebuild_model(
    checkpoint: dict[str, Any],
    model_name: str,
    splits: DatasetSplits,
) -> Any:
    """Rebuild one architecture from stored config before strict state loading."""
    from modeling import (
        RobertaClassifier,
        full_architecture,
        load_architectures,
        make_config,
    )
    from transformers import RobertaModel

    stored_config = checkpoint.get("config")
    if not isinstance(stored_config, dict):
        raise ValueError(f"{model_name} checkpoint has no config dictionary")
    try:
        architecture = {key: int(stored_config[key]) for key in ARCHITECTURE_KEYS}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{model_name} checkpoint config is incomplete") from error
    expected_stored_config = full_architecture(architecture)
    if stored_config != expected_stored_config:
        raise ValueError(
            f"{model_name} checkpoint config differs from the saved family format"
        )

    base_name = model_name.removesuffix("_nc")
    architecture_name = ARCHITECTURE_NAMES[base_name]
    catalog_architecture = load_architectures(EXPERIMENT_ROOT)[architecture_name]
    if architecture != catalog_architecture:
        raise ValueError(
            f"{model_name} stored config does not match {architecture_name}"
        )

    encoder = RobertaModel(make_config(architecture), add_pooling_layer=False)
    model = RobertaClassifier(
        encoder,
        output_size=splits.output_size,
        n_aspects=len(splits.aspect_names),
    )
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError(f"{model_name} checkpoint has no model_state dictionary")
    model.load_state_dict(model_state, strict=True)
    return model


def write_outputs(
    dataset: str,
    seed: int,
    slices: tuple[str, ...],
    smoke: bool,
    results: dict[str, dict[str, Any]],
) -> tuple[Path, Path]:
    """Write the compact robustness table and full evaluator output."""
    output_stem = EXPERIMENT_ROOT / "results" / f"noisy_eval_{dataset}{seed_suffix(seed)}"
    markdown_path = output_stem.with_suffix(".md")
    json_path = output_stem.with_suffix(".json")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    headers = ["model", *(name.replace("_", " ") for name in slices), "clean - combined"]
    lines = [
        f"# Noisy-slice evaluation — {dataset} — seed {seed}",
        "",
        "Values are macro F1. ABSA values use classification macro F1.",
        "",
        "| " + " | ".join(headers) + " |",
        "|---|" + "---:|" * (len(headers) - 1),
    ]
    payload: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "smoke": smoke,
        "slices": list(slices),
        "models": {},
    }
    for model_name, model_result in results.items():
        slice_metrics = model_result["slices"]
        combined_delta = None
        if "combined" in slice_metrics:
            combined_delta = primary_macro_f1(slice_metrics["clean"]) - primary_macro_f1(
                slice_metrics["combined"]
            )
        values = [f"{primary_macro_f1(slice_metrics[name]):.4f}" for name in slices]
        values.append("—" if combined_delta is None else f"{combined_delta:+.4f}")
        lines.append(f"| {model_name} | " + " | ".join(values) + " |")
        payload["models"][model_name] = {
            "checkpoint": model_result["checkpoint"],
            "clean_minus_combined_macro_f1": combined_delta,
            "slices": slice_metrics,
        }

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote {markdown_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)
    return markdown_path, json_path


def dry_run(args: argparse.Namespace) -> None:
    """Print the plan without importing Torch or reading checkpoint payloads."""
    slices = planned_slices(args.slices)
    checkpoints = discover_checkpoints(args.dataset, args.seed, args.models)
    print(
        f"Dry run: dataset={args.dataset} seed={args.seed} smoke={args.smoke}",
        flush=True,
    )
    print(f"Planned slices: {', '.join(slices)}", flush=True)
    print("Discovered checkpoints:", flush=True)
    if checkpoints:
        for model_name, path in checkpoints:
            print(f"  {model_name}: {path}", flush=True)
    else:
        print("  (none)", flush=True)
    print("Dry run complete: torch and transformers were not imported.", flush=True)


def execute(args: argparse.Namespace) -> None:
    """Load all selected checkpoints serially and evaluate every planned slice."""
    try:
        import torch
        from torch.utils.data import DataLoader
        from train import (
            EncodedDataset,
            FULL_BATCH_SIZE,
            classification_metrics,
            collate_pad,
            evaluate,
        )
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Noisy evaluation requires importable torch, transformers, and sklearn; "
            "use --dry-run on this host"
        ) from error

    slices = planned_slices(args.slices)
    checkpoints = discover_checkpoints(args.dataset, args.seed, args.models)
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found for dataset={args.dataset!r}, seed={args.seed}"
        )

    print(f"Loading clean {args.dataset} test split", flush=True)
    splits = load_dataset(args.dataset, REPO_ROOT, smoke=False)
    test_texts = splits.test_texts[:200] if args.smoke else splits.test_texts
    test_labels = splits.test_labels[:200] if args.smoke else splits.test_labels
    print(f"Test rows: {len(test_texts)}", flush=True)
    texts_by_slice = build_slice_texts(test_texts, slices, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        REPO_ROOT / "models" / "bamibert", local_files_only=True
    )
    batch_size = 8 if args.smoke else FULL_BATCH_SIZE
    loaders = {
        name: DataLoader(
            EncodedDataset(texts, test_labels, tokenizer),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_pad,
        )
        for name, texts in texts_by_slice.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    results: dict[str, dict[str, Any]] = {}
    for index, (model_name, checkpoint_path) in enumerate(checkpoints, start=1):
        print(
            f"[{index}/{len(checkpoints)}] Loading {model_name}: {checkpoint_path}",
            flush=True,
        )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        if int(checkpoint.get("seed", -1)) != args.seed:
            raise ValueError(f"{model_name} checkpoint seed does not match --seed")
        if checkpoint.get("aspect_names", []) != splits.aspect_names:
            raise ValueError(
                f"{model_name} checkpoint aspect order does not match the dataset"
            )
        model = rebuild_model(checkpoint, model_name, splits).to(device)
        slice_metrics: dict[str, Any] = {}
        for slice_name in slices:
            print(f"[{model_name}] Evaluating {slice_name}", flush=True)
            predicted: list[list[int]] = []

            def capture_predictions(
                _module: Any, _inputs: tuple[Any, ...], output: Any
            ) -> None:
                if not isinstance(output, torch.Tensor):
                    raise TypeError("Prediction capture expected classifier logits")
                predicted.extend(output.argmax(dim=-1).cpu().tolist())

            hook = model.register_forward_hook(capture_predictions) if splits.is_absa else None
            try:
                metrics = evaluate(
                    model,
                    loaders[slice_name],
                    device,
                    splits.is_absa,
                    splits.class_names,
                )
            finally:
                if hook is not None:
                    hook.remove()
            if splits.is_absa:
                metrics = add_absa_breakdown(
                    metrics,
                    test_labels,
                    predicted,
                    splits.aspect_names,
                    classification_metrics,
                )
            slice_metrics[slice_name] = metrics
        results[model_name] = {
            "checkpoint": str(checkpoint_path.relative_to(EXPERIMENT_ROOT)),
            "slices": slice_metrics,
        }
        del model, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[{model_name}] Freed model", flush=True)

    write_outputs(args.dataset, args.seed, slices, args.smoke, results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--models", type=parse_models)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--slices",
        choices=("all", "diacritics", "teencode", "combined"),
        default="all",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run(args)
    else:
        execute(args)


if __name__ == "__main__":
    main()
