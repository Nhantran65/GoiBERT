# Cavity key algebra audit

This report records a synthetic CPU Torch audit of the literal and factorized formulas in `docs/CAVITY_KEY_ALGEBRA_SPEC.md`. No dataset, checkpoint, optimizer, training, or GPU path was used.

Seed: `20260912`. Torch: `2.6.0+cu124`. Device: `cpu`.

## Float64 checks

- Forward parity: **pass** across 12 random shape cases (`N` in 1/2/5/9 and `H` in 1/2/3). Tolerances were abs `1e-8`, rel `1e-6`.
- Backward parity: **pass** for gradients of H0, Q, V, Wo, bo, Wk, bk, gamma, betaLN, and A-logits.
- Token permutation equivariance: **pass**.
- Partial-path exclusion: **pass**; the full-context control changed as expected for the selected receiver.
- No-LN limiting identity: **pass**.
- Centered score rank control: cavity rank `8` and full-context rank `2` at threshold `1e-8`; this is a numerical control only.

The factorized implementation reports logical intermediate shapes and the derived largest logical element count. This is not a peak VRAM measurement.

## Constructed diagonal-mask witness

For the supplied N=4, H=1, d=p=3 construction, the saved values show full-context `j1 > j2` and cavity `j1 < j2` while receiver 0 and receiver 3 retain self edges in A. The hard A witness does not run a gradient check.
The ranking reversal check (including the all-diagonal-masked additive variant) is **pass**. Values are in `results.json`.

## Float32 diagnostics

Random scale, near-removed-message, and constant-centered-row cases are reported as arithmetic diagnostics. Their raw absolute/relative mismatches, negative/near-zero variance counts, and finite checks are retained. Any failure against the strict Float64 thresholds is recorded as a failure; no tolerance was widened to make a case pass.

| case | max abs | max rel | strict Float64 threshold | negative vnorm | near-zero vnorm |
|---|---:|---:|:---:|---:|---:|
| random_scale_0.001 | 6.82121e-13 | 4.52959e-05 | fail | 0 | 0 |
| random_scale_1 | 9.53674e-07 | 5.41762e-06 | fail | 0 | 0 |
| random_scale_1000 | 768 | 9.2331e-07 | fail | 0 | 0 |
| near_removed_message | 0.000106812 | 7.51279e-05 | fail | 0 | 0 |
| scaled_near_removed_message | 2.79648 | 0.632684 | fail | 0 | 0 |
| constant_centered_rows | 0.000259876 | 0.000117736 | fail | 0 | 16 |

The mandatory formula/parity conclusions above are separated from Float32 arithmetic behavior. See `results.json` for all case shapes, errors, witnesses, and intermediate metadata.
