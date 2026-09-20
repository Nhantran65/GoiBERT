# GoiBERT Family v1

Type: experiment README. Audience: ML engineers. Owner: Line. Status: Stage A, Stage B, and Stage P implementation-complete. Parent policy: repository `AGENTS.md`.

Stage A compares BamiBERT with GoiBERT-B and two PCA-initialized, task-distilled students on VSFC, VLSP Restaurant, and ViSFD. Stage B adds PhoATIS intent classification and optional noise-consistency training. Stage P adds domain-adaptive MLM and TinyBERT general distillation. These experiment stages do not change a production model.

Build the existing project image and run the capped Docker smoke check from the repository root:

```bash
docker build -t goibert-v1 -f experiments/goiturn-arch-scaling/docker/Dockerfile .
docker run --rm -v "$PWD:/workspace" goibert-v1 python experiments/goibert-family-v1/src/run_family.py --dataset vsfc --smoke
```

On LUMI, activate the project environment from the repository root, then run one stage or the full protocol:

```bash
python experiments/goibert-family-v1/src/run_family.py --dataset vsfc --stage baseline
python experiments/goibert-family-v1/src/run_family.py --dataset vsfc --stage students
python experiments/goibert-family-v1/src/run_family.py --dataset vsfc --stage all
```

Each LR trial writes JSON under `results/`. Only validation-selected trials are evaluated on test. Best checkpoints use `results/checkpoints/`; the comparison is `results/summary_<dataset>.md`.

GoiTurn attention alignment averages probabilities across heads before masked MSE because student and teacher head counts differ.

Verify syntax with `python -m py_compile experiments/goibert-family-v1/src/*.py`. In Docker, run `python experiments/goibert-family-v1/src/test_distill.py`, then run the smoke command. If imports fail, use `--dry-run` to check configs and arithmetic counts without Torch.

## Stage P: continued pretraining and general distillation

Stage P builds a domain-weighted corpus, continues BamiBERT MLM training, and distills the DAPT teacher into both student encoders. Run the LUMI jobs in this exact order. `--wait` prevents a later stage from starting before its input checkpoint exists.

```bash
sbatch --wait lumi_job_pretrain.sh corpus --general-mb 1500 --domain-repeat 20 --seed 42
sbatch --wait lumi_job_pretrain.sh mlm --steps 40000 --resume
sbatch --wait lumi_job_pretrain.sh gd-s --steps 30000 --resume
sbatch --wait lumi_job_pretrain.sh gd-xs --steps 30000 --resume
sbatch --wait lumi_job_pretrain.sh family --dataset vsfc --stage all --init-from-b experiments/goibert-family-v1/results/pretrained/goibert_b_dapt --init-from-students experiments/goibert-family-v1/results/pretrained
```

The first four commands run `build_corpus.py`, `mlm_pretrain.py`, and `general_distill.py`. The final command fine-tunes `goibert_b_dapt`, selects its best validation checkpoint as the task-distillation teacher, and initializes the two students from their general-distillation checkpoints.

The flagged run writes `goibert_b_dapt`, `goiturn_s_gd`, and `goiturn_xs_gd` result and checkpoint names. It writes the comparison to `results/summary_<dataset>__stage_p.md`. These names preserve every Stage A JSON, checkpoint, and summary file. Without the flags, `run_family.py` keeps the Stage A run matrix and artifact names.

The corpus output is `data/corpus_v1/mlm_mix.txt`. DAPT output is `results/pretrained/goibert_b_dapt/` under this experiment. General-distillation outputs are `results/pretrained/goiturn_s_gd/` and `results/pretrained/goiturn_xs_gd/`.

Use dry runs to inspect paths and settings without importing Torch or contacting CC100:

```bash
python experiments/goibert-family-v1/src/build_corpus.py --dry-run
python experiments/goibert-family-v1/src/mlm_pretrain.py --dry-run
python experiments/goibert-family-v1/src/general_distill.py --student goiturn_s --dry-run
python experiments/goibert-family-v1/src/general_distill.py --student goiturn_xs --dry-run
python experiments/goibert-family-v1/src/run_family.py --dataset vsfc --dry-run --init-from-b experiments/goibert-family-v1/results/pretrained/goibert_b_dapt --init-from-students experiments/goibert-family-v1/results/pretrained
```

For smoke runs, execute the same four stages with `--smoke`. The corpus stage requests a 2 MiB compressed prefix and emits at most 1 MiB of general text. The training stages cap themselves at 20 steps and batch size 2.

## Stage B: PhoATIS intent classification

PhoATIS is the fourth benchmark. The loader reads utterances from `data/raw/phoatis/<split>/seq.in` and intent labels from `data/raw/phoatis/<split>/label`. It validates matching row counts and the 28-intent label vocabulary across the official splits.

Run the same family protocol with the `phoatis` dataset key:

```bash
python experiments/goibert-family-v1/src/run_family.py --dataset phoatis --stage all
python experiments/goibert-family-v1/src/run_family.py --dataset phoatis --dry-run
```

PhoATIS results report accuracy, macro precision, macro recall, macro F1, and per-class precision and recall. The published PhoBERT intent-accuracy result is about 97% in the JointIDSF paper. Use it as a sanity anchor, not as a hard assertion for BamiBERT.

## Stage B: noise-consistency ablation

Pass `--noise-consistency` to train on clean text plus a deterministic noisy view. The noisy view can remove diacritics, replace words with Vietnamese teencode, and apply low-rate character edits. Training adds symmetric KL divergence between the clean and noisy logits with weight `1.0`.

```bash
python experiments/goibert-family-v1/src/run_family.py --dataset vsfc --stage all --noise-consistency
python experiments/goibert-family-v1/src/run_family.py --dataset vsfc --dry-run --noise-consistency
python experiments/goibert-family-v1/src/noise.py
```

The flag is off by default. Flagged model names and artifacts use `_nc` before an optional `__seed<k>` suffix, such as `vsfc__baseline_nc__best__seed7.json` and `summary_vsfc_nc__seed7.md`. Runs without the flag keep the existing names and training path.

## Noisy-slice evaluation

Run one LUMI process per dataset. Each process discovers every matching task checkpoint, then evaluates the clean, no-diacritics, teencode, and combined test slices.

```bash
python experiments/goibert-family-v1/src/eval_noisy.py --dataset vsfc
python experiments/goibert-family-v1/src/eval_noisy.py --dataset vlsp_restaurant
python experiments/goibert-family-v1/src/eval_noisy.py --dataset visfd
python experiments/goibert-family-v1/src/eval_noisy.py --dataset phoatis
```

Use `--models baseline,baseline_nc` to filter checkpoints. Use `--seed <k>` for seeded runs. Use `--slices diacritics`, `--slices teencode`, or `--slices combined` to run one corruption with the clean control. `--smoke` caps the clean test split at 200 rows.

The script writes `results/noisy_eval_<dataset>[__seed<k>].md` and a JSON file with the same stem. The Markdown table reports macro F1 and clean-minus-combined degradation. The JSON file keeps the full evaluator metrics. For ABSA datasets, it also includes slot accuracy, per-class metrics for labels 0-3, and a full breakdown for each aspect.
