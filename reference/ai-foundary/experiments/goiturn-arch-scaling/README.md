# GoiTurn Architecture Scaling Experiment

Type: experiment README. Audience: ML engineers. Owner: Line. Status: implementation-complete. Parent policy: repository `AGENTS.md`.

This experiment compares BamiBERT with two smaller RoBERTa encoders. It runs Vietnamese sentiment classification and two aspect-based sentiment classification tasks. It serves the GoiTurn-S model-size feasibility decision; it does not change a production model.

## Requirements

Use Linux with Python 3.11 or newer. The environment must contain PyTorch 2.x, Transformers, pandas, scikit-learn, safetensors, and NumPy. Runtime is offline. The scripts load `models/bamibert/` with `local_files_only=True`.

## Run

From `experiments/goiturn-arch-scaling/`, run a capped end-to-end check:

```bash
python src/run_all.py --smoke
```

Run the full 15-run experiment and latency benchmark:

```bash
python src/run_all.py
```

Run only one dataset:

```bash
python src/run_all.py --only vsfc
python src/run_all.py --only vlsp_restaurant
python src/run_all.py --only visfd
```

The orchestrator always runs the baseline first. The baseline writes the checkpoint and ordered train-set logits required by both distillation runs. A standalone distillation command stops with a clear error when its teacher logits are missing.

## Outputs

The scripts write only under `results/`:

- One JSON file per training run, named `<dataset>__<model>__<mode>.json`.
- Baseline checkpoints under `results/checkpoints/`.
- Teacher logits under `results/teacher_logits/`.
- Reusable PCA projections under `results/pca/`.
- CPU measurements in `results/latency.json`.
- The comparison tables in `results/summary.md`.

Smoke mode uses 200/100/100 rows, one epoch, batch size 8, two latency warmups, and five timed iterations. Full mode uses the experiment settings in the task specification, including 30 latency warmups and 300 timed iterations.

## Verify

The Windows host cannot load Torch DLLs. Syntax-check each source file there with `python -m py_compile`. Run `python src/run_all.py --smoke` in the Linux environment before using full-mode results.

Done looks like: all 15 run JSON files, `latency.json`, and `summary.md` exist with no failed subprocess.
