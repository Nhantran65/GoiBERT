# TASK PACKET — family v1 Stage B: noise-consistency training + PhoATIS intent dataset

GOAL: Two additions to `experiments/goibert-family-v1/`: (1) PhoATIS intent classification
as the 4th benchmark dataset; (2) noise-augmented consistency training (proposal §13.5
Stage 5) as a flag-gated ablation — the first differentiator BamiBERT does not have.

BUSINESS REASON: Robustness to noisy Vietnamese (no diacritics, teencode, typos) is the
axis where Vietnamese models actually differ (ViSoBERT ablations); customer chat is noisy.
PhoATIS (transactional dialogue) is the 4th dataset Line requested.

FILES (ONLY inside experiments/goibert-family-v1/): `src/data.py`, `src/train.py`,
`src/distill.py`, `src/run_family.py`, NEW `src/noise.py`, `README.md`. Nothing else.

PART 1 — PhoATIS intent (single-label classification, 28 intents):
- Data already on disk: `data/raw/phoatis/{train,dev,test}/seq.in` (one utterance/line)
  and `{split}/label` (one intent label/line, same line count). ~4478/500/893 lines.
- Add loader in data.py; dataset key `phoatis` in run_family (`--dataset phoatis`).
- Metrics: accuracy + macro F1 + per-class P/R (same convention as vsfc). Published
  anchor for sanity (JointIDSF paper): intent accuracy ~97% with PhoBERT — ours with
  BamiBERT should land near that; do not hard-assert it.

PART 2 — noise-augmented consistency (flag `--noise-consistency`, default OFF):
- NEW `src/noise.py`, pure functions, deterministic under a passed random.Random:
  * `strip_diacritics(text)`: NFD decompose, drop combining marks, also đ→d, Đ→D.
  * `teencode(text)`: word-level map applied with per-word probability (không/ko, được/dc,
    với/vs, gì/j, mình/mk, biết/bit, rồi/r, bạn/b, ơi/oi, vâng/vg, shop/shp, anh/a, chị/c,
    em/e, quá/wa, yêu/iu, buồn/bun, tiền/xiền, thích/thik, đi/dj, chờ/cho, giờ/h).
  * `char_noise(text)`: random adjacent swap / duplicate / drop, low per-char probability.
  * `corrupt(text, rng)`: apply diacritic-strip with p=0.5, teencode p_word=0.3,
    char_noise p_char=0.02; always returns a non-empty string; cap length growth.
  * Self-test block (run as __main__): fixed seed → fixed output; corruption changes at
    least one of 10 sample sentences; never returns empty.
- Training integration (train.py AND distill.py), when the flag is on:
  * For each batch, tokenize a corrupted copy of the same texts (corruption in text space
    BEFORE tokenization, deterministic rng seeded from (seed, epoch, batch_index)).
  * Add consistency loss: symmetric KL between softmax logits of clean and corrupted views
    (R-Drop-style), weight constant NC_WEIGHT = 1.0, added to the existing loss.
  * This requires carrying raw texts through the dataset objects — extend minimally.
- run_family: `--noise-consistency` passthrough; when ON, model row names and ALL output
  filenames get suffix `_nc` (compose correctly with the existing `__seed<k>` suffix).
  Flag OFF → byte-identical behavior and filenames as today.

CONSTRAINTS: no new dependencies (unicodedata is stdlib). Keep existing style. Another
worker may run later in goiturn-synthetic-v0/ — never touch that directory.

DONE WHEN:
- py_compile all touched files; `python src/noise.py` self-test passes.
- `run_family.py --dataset phoatis --dry-run` prints the run matrix.
- `run_family.py --dataset vsfc --dry-run --noise-consistency` shows `_nc` suffixed
  filenames; without the flag, today's output is unchanged.
- README documents both features.

VALIDATE (run, report):
- py_compile each touched file
- `python experiments/goibert-family-v1/src/noise.py`
- the three dry-runs above

STOP RULES: standard — missing prerequisite/ambiguity/hack-only → report; unsure about the
consistency-loss wiring → add a 20-step toy self-test proving loss is finite and decreases.
ROLLBACK: flag-gated diffs + one new file; delete noise.py and revert to undo.
