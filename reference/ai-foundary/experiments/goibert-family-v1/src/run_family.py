"""Run the fair BamiBERT/GoiBERT family LR grids and write a test-honest summary."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "family.json"
DATASETS = ("vsfc", "vlsp_restaurant", "visfd", "phoatis")
BASELINE_LRS = (1e-5, 2e-5, 5e-5)
STUDENT_LRS = (5e-5, 1e-4, 2e-4)
STUDENTS = ("goiturn_s", "goiturn_xs")
VOCAB_SIZE = 20_481
TYPE_VOCAB_SIZE = 1

EXPECTED_CONFIGS = {
    "goiturn_xs": {
        "num_hidden_layers": 6,
        "hidden_size": 256,
        "num_attention_heads": 4,
        "intermediate_size": 1024,
        "max_position_embeddings": 2050,
    },
    "goiturn_s": {
        "num_hidden_layers": 10,
        "hidden_size": 384,
        "num_attention_heads": 6,
        "intermediate_size": 1536,
        "max_position_embeddings": 2050,
    },
    "goibert_b": {
        "num_hidden_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 2050,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required result is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_and_validate_configs() -> dict[str, dict[str, int]]:
    configs = _read_json(CONFIG_PATH)
    if configs != EXPECTED_CONFIGS:
        raise ValueError(f"{CONFIG_PATH} does not match the exact family specification")
    for name, config in configs.items():
        if config["hidden_size"] % config["num_attention_heads"]:
            raise ValueError(f"{name} hidden size is not divisible by its head count")
    return configs


def arithmetic_counts(config: dict[str, int], output_size: int) -> dict[str, int]:
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    embeddings = (
        VOCAB_SIZE * hidden
        + config["max_position_embeddings"] * hidden
        + TYPE_VOCAB_SIZE * hidden
        + 2 * hidden
    )
    per_layer = 4 * hidden * hidden + 2 * hidden * intermediate + 9 * hidden + intermediate
    encoder = embeddings + config["num_hidden_layers"] * per_layer
    head = hidden * hidden + hidden + hidden * output_size + output_size
    projections = (config["num_hidden_layers"] + 1) * (hidden * 768 + 768)
    return {
        "encoder_and_embeddings": encoder,
        "head": head,
        "model_total": encoder + head,
        "distillation_projections": projections,
    }


def _planned_lrs(smoke: bool) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if smoke:
        return (2e-5,), (1e-4,)
    return BASELINE_LRS, STUDENT_LRS


def _seed_suffix(seed: int) -> str:
    return "" if seed == 42 else f"__seed{seed}"


def _noise_run_name(run_name: str, enabled: bool) -> str:
    return f"{run_name}_nc" if enabled else run_name


def _print_planned_artifacts(
    dataset: str,
    run_name: str,
    learning_rate: float,
    seed: int,
) -> None:
    suffix = _seed_suffix(seed)
    stem = f"{dataset}__{run_name}__lr_{learning_rate:.0e}{suffix}"
    print(f"    result={stem}.json candidate={stem}.pt")


def _print_selected_artifacts(dataset: str, run_name: str, seed: int) -> None:
    suffix = _seed_suffix(seed)
    print(
        f"    best={dataset}__{run_name}__best{suffix}.json "
        f"checkpoint={dataset}__{run_name}{suffix}.pt"
    )


def dry_run(args: argparse.Namespace) -> None:
    configs = load_and_validate_configs()
    from data import load_dataset

    splits = load_dataset(args.dataset, args.repo_root.resolve(), smoke=True)
    output_size = splits.output_size
    noise_text = " noise_consistency=True" if args.noise_consistency else ""
    print(
        f"Dry run: dataset={args.dataset} stage={args.stage} "
        f"seed={args.seed} output_size={output_size}{noise_text}"
    )
    print("Exact arithmetic parameter counts:")
    for name in ("goiturn_xs", "goiturn_s", "goibert_b"):
        counts = arithmetic_counts(configs[name], output_size)
        projection_text = (
            f" distill_projections={counts['distillation_projections']:,}"
            if name in STUDENTS
            else ""
        )
        print(
            f"  {name}: encoder={counts['encoder_and_embeddings']:,} "
            f"head={counts['head']:,} model_total={counts['model_total']:,}{projection_text}"
        )
    baseline_lrs, student_lrs = _planned_lrs(args.smoke)
    print("Planned run matrix (test only after validation-grid selection):")
    if args.stage in {"baseline", "all"}:
        baseline_name = _noise_run_name("baseline", args.noise_consistency)
        for learning_rate in baseline_lrs:
            print(f"  baseline finetune lr={learning_rate:g}")
            _print_planned_artifacts(args.dataset, baseline_name, learning_rate, args.seed)
        _print_selected_artifacts(args.dataset, baseline_name, args.seed)
        if args.init_from_b is not None:
            dapt_name = _noise_run_name("goibert_b_dapt", args.noise_consistency)
            for learning_rate in baseline_lrs:
                print(
                    f"  goibert_b_dapt finetune lr={learning_rate:g} "
                    f"init={args.init_from_b}"
                )
                _print_planned_artifacts(
                    args.dataset, dapt_name, learning_rate, args.seed
                )
            _print_selected_artifacts(args.dataset, dapt_name, args.seed)
        else:
            print("  goibert_b alias -> validation-selected baseline (no duplicate training)")
            print(
                f"    best={args.dataset}__"
                f"{_noise_run_name('goibert_b', args.noise_consistency)}__best"
                f"{_seed_suffix(args.seed)}.json"
            )
    else:
        baseline_name = _noise_run_name("baseline", args.noise_consistency)
        print(
            f"  baseline input={args.dataset}__{baseline_name}__best"
            f"{_seed_suffix(args.seed)}.json"
        )
        if args.init_from_b is not None:
            dapt_name = _noise_run_name("goibert_b_dapt", args.noise_consistency)
            print(
                f"  goibert_b_dapt input={args.dataset}__{dapt_name}__best"
                f"{_seed_suffix(args.seed)}.json"
            )
        else:
            print("  goibert_b alias -> validation-selected baseline (no duplicate training)")
            print(
                f"    best={args.dataset}__"
                f"{_noise_run_name('goibert_b', args.noise_consistency)}__best"
                f"{_seed_suffix(args.seed)}.json"
            )
    if args.stage in {"students", "all"}:
        for model_name in STUDENTS:
            base_run_name = (
                f"{model_name}_gd" if args.init_from_students is not None else model_name
            )
            run_name = _noise_run_name(base_run_name, args.noise_consistency)
            for learning_rate in student_lrs:
                if args.init_from_students is not None:
                    print(
                        f"  {run_name} distill lr={learning_rate:g} "
                        f"init={args.init_from_students / base_run_name}"
                    )
                else:
                    print(f"  {run_name} distill lr={learning_rate:g}")
                _print_planned_artifacts(args.dataset, run_name, learning_rate, args.seed)
            _print_selected_artifacts(args.dataset, run_name, args.seed)
    artifact_suffix = (
        "__stage_p"
        if args.init_from_b is not None or args.init_from_students is not None
        else ""
    )
    noise_suffix = "_nc" if args.noise_consistency else ""
    print(
        f"  summary=summary_{args.dataset}{artifact_suffix}{noise_suffix}"
        f"{_seed_suffix(args.seed)}.md"
    )
    print("Dry run complete: torch and transformers were not imported.")


def _discard_losing_candidates(trials: list[dict[str, Any]], winner: dict[str, Any]) -> None:
    for trial in trials:
        if trial is winner:
            continue
        candidate_path = Path(trial.pop("candidate_checkpoint"))
        result_path = Path(trial.pop("result_path"))
        candidate_path.unlink()
        trial["checkpoint_retained"] = False
        _write_json(result_path, trial)


def _alias_goibert_b(
    baseline: dict[str, Any], noise_consistency: bool
) -> dict[str, Any]:
    alias = copy.deepcopy(baseline)
    alias["model"] = _noise_run_name("goibert_b", noise_consistency)
    alias["description"] = "BamiBERT init, fine-tuned (exact alias of winning baseline run)"
    alias["alias_of"] = _noise_run_name("baseline", noise_consistency)
    alias_path = (
        EXPERIMENT_ROOT
        / "results"
        / f"{baseline['dataset']}__{alias['model']}__best"
        f"{_seed_suffix(int(baseline['seed']))}.json"
    )
    _write_json(alias_path, alias)
    return alias


def _primary_metrics(result: dict[str, Any]) -> tuple[str, float, float, float]:
    metrics = result["test_metrics"]
    if "accuracy" in metrics:
        return (
            f"{metrics['accuracy']:.4f}",
            float(metrics["macro_precision"]),
            float(metrics["macro_recall"]),
            float(metrics["macro_f1"]),
        )
    return (
        "—",
        float(metrics["classification_macro_precision"]),
        float(metrics["classification_macro_recall"]),
        float(metrics["classification_macro_f1"]),
    )


def write_summary(
    dataset: str,
    rows: list[dict[str, Any]],
    smoke: bool,
    seed: int,
    artifact_suffix: str = "",
    noise_consistency: bool = False,
) -> Path:
    baseline = next(
        row
        for row in rows
        if row["model"] == _noise_run_name("baseline", noise_consistency)
    )
    _, _, _, baseline_f1 = _primary_metrics(baseline)
    lines = [
        f"# GoiBERT family v1 — {dataset}{' smoke' if smoke else ''} — seed {seed}",
        "",
        "LR and epoch selection use validation metrics only. Each selected run is evaluated once on test.",
        "For ABSA datasets, P/R/F1 are classification macro metrics and accuracy is not applicable.",
        "",
        "| model | description | params | lr | accuracy | macro P | macro R | macro F1 | gap% F1 vs baseline |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in rows:
        accuracy, precision, recall, macro_f1 = _primary_metrics(result)
        gap = 0.0 if baseline_f1 == 0 else (baseline_f1 - macro_f1) / baseline_f1 * 100
        lines.append(
            f"| {result['model']} | {result['description']} | "
            f"{result['param_counts']['model_total']:,} | {result['lr']:.0e} | {accuracy} | "
            f"{precision:.4f} | {recall:.4f} | {macro_f1:.4f} | {gap:.2f}% |"
        )
    output_path = (
        EXPERIMENT_ROOT
        / "results"
        / f"summary_{dataset}{artifact_suffix}"
        f"{'_nc' if noise_consistency else ''}{_seed_suffix(seed)}.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}", flush=True)
    return output_path


def execute(args: argparse.Namespace) -> None:
    load_and_validate_configs()
    try:
        from distill import finalize_student_trial, train_student_trial
        from train import finalize_baseline_trial, train_baseline_trial
    except ImportError as error:
        raise RuntimeError(
            "Training requires importable torch, transformers, and sklearn; use --dry-run on this host"
        ) from error

    repo_root = args.repo_root.resolve()
    baseline_lrs, student_lrs = _planned_lrs(args.smoke)
    baseline: dict[str, Any]

    # Intent: preserve a strict validation-to-test boundary across LR trials.
    # Status: every grid trial persists validation evidence; only the winner touches test.
    # Next: full LUMI runs can consume the same paths after smoke verification.
    # Blockers: the Windows host cannot import Torch, so end-to-end execution belongs on Linux.
    # Confidence: high; candidate lifecycle and summary inputs are explicit below.
    if args.stage in {"baseline", "all"}:
        baseline_trials = [
            train_baseline_trial(
                args.dataset,
                learning_rate,
                args.seed,
                args.smoke,
                repo_root,
                noise_consistency=args.noise_consistency,
            )
            for learning_rate in baseline_lrs
        ]
        baseline_winner = max(baseline_trials, key=lambda trial: trial["best_validation_score"])
        _discard_losing_candidates(baseline_trials, baseline_winner)
        baseline = finalize_baseline_trial(baseline_winner, repo_root)
    else:
        baseline_name = _noise_run_name("baseline", args.noise_consistency)
        baseline_path = (
            EXPERIMENT_ROOT
            / "results"
            / f"{args.dataset}__{baseline_name}__best{_seed_suffix(args.seed)}.json"
        )
        baseline = _read_json(baseline_path)
        if bool(baseline.get("smoke")) != args.smoke:
            mode = "smoke" if args.smoke else "full"
            raise ValueError(f"--stage students requires a validation-selected {mode} baseline")
    if args.init_from_b is not None:
        if args.stage in {"baseline", "all"}:
            dapt_trials = [
                train_baseline_trial(
                    args.dataset,
                    learning_rate,
                    args.seed,
                    args.smoke,
                    repo_root,
                    args.init_from_b.resolve(),
                    args.noise_consistency,
                )
                for learning_rate in baseline_lrs
            ]
            dapt_winner = max(
                dapt_trials, key=lambda trial: trial["best_validation_score"]
            )
            _discard_losing_candidates(dapt_trials, dapt_winner)
            goibert_b = finalize_baseline_trial(dapt_winner, repo_root)
        else:
            dapt_name = _noise_run_name("goibert_b_dapt", args.noise_consistency)
            dapt_path = (
                EXPERIMENT_ROOT
                / "results"
                / f"{args.dataset}__{dapt_name}__best{_seed_suffix(args.seed)}.json"
            )
            goibert_b = _read_json(dapt_path)
            if bool(goibert_b.get("smoke")) != args.smoke:
                mode = "smoke" if args.smoke else "full"
                raise ValueError(
                    f"--stage students requires a validation-selected {mode} goibert_b_dapt"
                )
        teacher_checkpoint = Path(goibert_b["checkpoint"])
    else:
        goibert_b = _alias_goibert_b(baseline, args.noise_consistency)
        teacher_checkpoint = Path(baseline["checkpoint"])
    rows = [baseline, goibert_b]

    if args.stage in {"students", "all"}:
        for model_name in STUDENTS:
            student_init_dir = (
                args.init_from_students.resolve() / f"{model_name}_gd"
                if args.init_from_students is not None
                else None
            )
            student_trials = [
                train_student_trial(
                    args.dataset,
                    model_name,
                    learning_rate,
                    args.seed,
                    args.smoke,
                    repo_root,
                    teacher_checkpoint,
                    student_init_dir,
                    args.noise_consistency,
                )
                for learning_rate in student_lrs
            ]
            student_winner = max(
                student_trials, key=lambda trial: trial["best_validation_score"]
            )
            _discard_losing_candidates(student_trials, student_winner)
            rows.append(finalize_student_trial(student_winner, repo_root))
    artifact_suffix = (
        "__stage_p"
        if args.init_from_b is not None or args.init_from_students is not None
        else ""
    )
    write_summary(
        args.dataset,
        rows,
        args.smoke,
        args.seed,
        artifact_suffix,
        args.noise_consistency,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--stage", choices=("baseline", "students", "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--noise-consistency", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--init-from-b", type=Path)
    parser.add_argument("--init-from-students", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run(args)
    else:
        execute(args)


if __name__ == "__main__":
    main()
