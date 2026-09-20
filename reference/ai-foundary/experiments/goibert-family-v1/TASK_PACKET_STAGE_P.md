# TASK PACKET — GoiBERT family, Stage P: pretraining thật cho dòng Goi

GOAL: Add three scripts to `experiments/goibert-family-v1/` that make the family genuinely
trained by us: (1) corpus builder, (2) continued-MLM pretraining for GoiBERT-B, (3) TinyBERT
general distillation for GoiTurn-S/XS — plus an `--init-from` hook in run_family.py so the
task stage can start from these pretrained checkpoints.

BUSINESS REASON: Line's directive — GoiBERT-B must not be a pure copy of BamiBERT; the Goi
series must be trained by us. Orthodox route per proposal §13.5 Stage 1-2 (domain-weighted
corpus + continued MLM at 30% masking) and TinyBERT two-stage distillation (general then
task) — the measured 8.94% gap for GoiTurn-S is expected to shrink mainly via these.

SUCCESS CRITERIA:
1. `python src/build_corpus.py --general-mb 1500 --domain-repeat 20 --seed 42` produces
   `data/corpus_v1/mlm_mix.txt` + `stats.json` (line counts per source, total bytes).
2. `python src/mlm_pretrain.py --steps 40000 --resume` continued-pretrains BamiBERT with
   MLM, checkpoints periodically, and is resumable after job kill.
3. `python src/general_distill.py --student goiturn_s --steps 30000 --resume` trains the
   student against the DAPT teacher on unlabeled corpus text; same for goiturn_xs.
4. `run_family.py --dataset vsfc --stage all --init-from-b <dir> --init-from-students <dir>`
   fine-tunes/distills starting FROM those checkpoints, and labels rows honestly
   (e.g. `goibert_b_dapt`, `goiturn_s_gd`) so v2 numbers never overwrite Stage A rows.
5. `--smoke` on every new script (tiny corpus, tens of steps, CPU-capable).

FILES (repo root D:\01_Workspaces\TwoHearts\ml-models):
- CREATE `experiments/goibert-family-v1/src/build_corpus.py`
- CREATE `experiments/goibert-family-v1/src/mlm_pretrain.py`
- CREATE `experiments/goibert-family-v1/src/general_distill.py`
- MODIFY `experiments/goibert-family-v1/src/run_family.py` (add init-from options + row
  naming; do not change existing behavior when flags absent)
- MODIFY `experiments/goibert-family-v1/README.md` (append Stage P commands)
- CREATE `lumi_job_pretrain.sh` at repo root (sbatch: small-g, 1 GPU, --time=24:00:00,
  same module/venv pattern as lumi_job_family.sh; args: script stage passthrough)
- Corpus outputs under `data/corpus_v1/` (add to .gitignore)
- Pretrained checkpoints under `experiments/goibert-family-v1/results/pretrained/<name>/`

FACTS PRE-FETCHED (embed, do not re-verify):
- CC100 Vietnamese: `https://data.statmt.org/cc-100/vi.txt.xz` — VERIFIED 200 OK,
  Content-Length 29,508,806,944, `Accept-Ranges: bytes`. One text line per line, blank lines
  between docs. Download ONLY a range prefix (e.g. first ~500MB-2GB compressed) via HTTP
  Range header (urllib Request with `Range: bytes=0-N`), stream-decompress with Python
  `lzma.LZMADecompressor` and CATCH the truncation error at the end (EOFError/LZMAError) —
  the decompressed prefix up to the last complete block is valid text. Cap output at
  `--general-mb` megabytes of UTF-8 text. Filter: drop lines < 20 chars, drop non-Vietnamese
  lines cheaply (keep lines where >=60% chars are in [a-zA-Z0-9À-ỹ\s\p{{P}}]-ish; simple
  heuristic ok, no external libs).
- Domain text sources (all already local): text columns via the existing loaders in
  `src/data.py` (VSFC train+valid, ViSFD, VLSP CSVs) and `data/raw/phoatis/*/seq.in`
  (plain text lines). Dedup exact lines. Repeat the whole domain block `--domain-repeat`
  times in the mix (default 20), then shuffle lines with random.Random(seed).
- BamiBERT has a proper MLM head available via
  `AutoModelForMaskedLM.from_pretrained("models/bamibert")` (RoBERTa-style checkpoint —
  if the head weights are missing transformers will warn and random-init the LM head; that
  is acceptable, note it in the JSON).

MLM PRETRAIN SPEC (`mlm_pretrain.py`) — orthodox continued pretraining:
- Data: stream `data/corpus_v1/mlm_mix.txt`, pack lines into sequences of max_length 256
  (concatenate short lines with tokenizer sep, no padding waste), batch 32, dynamic device.
- Masking: 30% of tokens (ViSoBERT finding for Vietnamese informal text; cite in comment),
  80/10/10 mask/random/keep convention.
- Optimizer: AdamW lr 5e-5, linear warmup 1000 steps then linear decay to 0 at --steps;
  weight_decay 0.01; grad clip 1.0. Seed 42.
- Checkpoint every 2000 steps to `results/pretrained/goibert_b_dapt/` (model + optimizer +
  step count); `--resume` picks up the latest. Log loss every 100 steps (`step=... loss=...`
  flush=True). Keep ROCm guards (GOITURN_AMP/GOITURN_ATTN pattern from existing code).
- Final save: HF save_pretrained format (so run_family can `from_pretrained` it) + a
  `pretrain_manifest.json` (steps, corpus stats hash, loss curve tail, versions).

GENERAL DISTILLATION SPEC (`general_distill.py`) — TinyBERT stage 1:
- Teacher: the DAPT GoiBERT-B (`--teacher-dir`, default the dapt output; fall back to
  models/bamibert if absent). Frozen, eval, output_hidden_states+attentions.
- Student: goiturn_s or goiturn_xs config from configs/family.json, PCA-initialized
  embeddings (reuse existing helper against the TEACHER's embeddings).
- Loss (no labels, no logits): hidden-state MSE via learned projections with the same layer
  map as distill.py + attention MSE (head-mean, same as distill.py). Reuse/refactor the
  shared pieces from distill.py rather than duplicating (small shared module is fine).
- Same streaming/packing as mlm_pretrain; lr 1e-4 warmup 1000, linear decay; batch 32;
  checkpoint+resume identical pattern; save HF format + manifest to
  `results/pretrained/<student>_gd/`.

RUN_FAMILY HOOKS:
- `--init-from-b <dir>`: baseline row stays the pure BamiBERT fine-tune (NEVER changes —
  it is the measuring stick); ADD a new trained row `goibert_b_dapt` fine-tuned from the
  DAPT checkpoint with the same lr grid. Teacher for students becomes the best of
  goibert_b_dapt (by val) when this flag is present.
- `--init-from-students <dir>`: students initialize from their GD checkpoints (instead of
  PCA init) before task distillation; row names become `goiturn_s_gd`, `goiturn_xs_gd`.
- Summary table then contains: baseline, goibert_b_dapt, goiturn_s_gd, goiturn_xs_gd, with
  gap% computed vs baseline as before. Old rows/files from Stage A must not be overwritten
  (new result filenames carry the suffix).

CONSTRAINTS:
- Write only the files listed. Do not modify distill.py's existing training behavior, the
  other experiments, tracker/plan files.
- stdlib + torch + transformers + sklearn only.
- All long-running loops must print progress with flush=True and be resumable.

NON-GOALS: No noise-consistency training yet (separate stage). No PhoATIS task head. No
messenger/production data. No modern block. No tokenizer training.

DONE WHEN:
- py_compile passes on all touched files.
- `build_corpus.py --smoke` (a few MB range download; if network blocked in your sandbox,
  code complete + say so) runs; `mlm_pretrain.py --smoke` and `general_distill.py --smoke`
  verified to at least construct batches and compute one loss IF torch importable — torch is
  NOT importable on this host, so provide `--dry-run` printing the planned config instead,
  and state exactly which checks ran.
- README updated with the exact LUMI sequence: build corpus -> mlm_pretrain -> general_distill
  (s, xs) -> run_family with init flags.

VALIDATE:
- `python -m py_compile` each touched file
- `python experiments/goibert-family-v1/src/build_corpus.py --dry-run`
- `python experiments/goibert-family-v1/src/mlm_pretrain.py --dry-run`
- `python experiments/goibert-family-v1/src/general_distill.py --student goiturn_s --dry-run`
- `python experiments/goibert-family-v1/src/run_family.py --dataset vsfc --dry-run` (must
  still work unchanged without new flags)

STOP RULES:
- Missing prerequisite -> implement, verify what you can, report exactly what is blocked.
- Ambiguous requirement -> stop and report the gap. Only a hack remains -> stop and report.
- Unsure a loss/masking implementation is solid -> add a toy self-test proving it (masking
  fraction ~30%, loss decreases over 20 steps on random data) before reporting done.

ROLLBACK: new files additive; run_family changes are flag-gated (no flags = old behavior).
