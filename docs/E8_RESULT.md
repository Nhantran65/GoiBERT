# E8 Result: Noise Consistency + Layer-wise Learning-rate Decay

Five original GoiTurn-S-GD students were trained once each with seed 42, 30 epochs, effective and micro batch size 32, max length 256, BF16, and SDPA. E2 noise consistency used the fixed task lambdas; E7 AdamW layer-wise decay used gamma 0.8.

Validation selection F1 is recomputed from the saved prediction rows. The test split was never loaded.

| Task | E1 | E2 NC | E7 LLRD | E8 NC+LLRD | Best epoch | Δ E8 vs better component |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vsfc_sentiment | 0.8451 | 0.8477 | 0.8406 | 0.8516 | 21 | +0.0039 |
| vsfc_topic | 0.7977 | 0.7985 | 0.7995 | 0.8037 | 25 | +0.0041 |
| visfd | 0.8007 | 0.8109 | 0.8149 | 0.8244 | 28 | +0.0095 |
| absa_restaurant | 0.6902 | 0.6906 | 0.7152 | 0.7128 | 25 | -0.0024 |
| absa_hotel | 0.7204 | 0.7358 | 0.7329 | 0.7305 | 26 | -0.0053 |

The combination adds value where its E8 score exceeds both fixed components. Differences below 0.3 percentage points are marked as weak signals; a negative delta means the combination did not beat the better component on this seed. Over-regularization is a possible explanation, not a demonstrated cause.

Source hashes: E8 `089726d5d0294d2835ff6b4212bf30e6e47f246e84f50213536c0cbe4ccfd437`, E7 dependency `4f586eba100e66574081aac2647018b237f4754642723bf69b35eadae9e930f5`, E2 dependency `1a9d9b1d22f038f8231e5194a947515df1361c1b5603ed262e55f69e9e851ca6`, E1 `456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873`.

No coefficient retuning, multi-seed confirmation, extra tests, architecture changes, or encoder probes were run.

## Root handoff

Implementation/execution agent: `gpt-5.6-luna`, reasoning `xhigh`, explicitly selected with no parent-model inheritance. Project-wide future implementation-agent policy is saved in [AGENTS.md](../AGENTS.md).

Giữ E8 làm ứng viên validation mới cho Sentiment (85.1604) và ViSFD (82.4421). Topic E8 (80.3678) vượt NC/LLRD nhưng vẫn dưới E2 KD (80.7842); Restaurant giữ E7 (71.5175); Hotel giữ E2 NC (73.5776). Tất cả là cùng seed42, chưa có xác nhận đa seed hoặc test. Không tự thay cấu hình chung cho cả5task.
