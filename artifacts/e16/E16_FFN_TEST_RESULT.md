# E16 authorized test results

Scope: baseline and ffn_input_gate, seed 42, all five tasks. attention_update_gate was excluded. Validation lock and source history were preserved.

| Task | Baseline test selection F1 | FFN input gate test selection F1 | FFN − baseline (pp) |
|---|---:|---:|---:|
| vsfc_sentiment | 0.81915931 | 0.81148229 | -0.7677 |
| vsfc_topic | 0.78965197 | 0.79429877 | +0.4647 |
| visfd | 0.84601874 | 0.84218978 | -0.3829 |
| absa_restaurant | 0.74450910 | 0.74422969 | -0.0279 |
| absa_hotel | 0.74214619 | 0.73817229 | -0.3974 |

All values are evaluator fractions in [0,1]; displayed deltas are percentage points.

## Checkpoint and data facts

- Authorized checkpoint inferences: 10.
- Test counts: vsfc_sentiment=3166, vsfc_topic=3166, visfd=2224, absa_restaurant=1938, absa_hotel=2030.
- ABSA metadata overlay: 0 absent, 1 positive, 2 negative, 3 neutral; integer gold labels were preserved.
- ViSFD OTHERS: raw code 4/model code 1 conversion for detection; pair metric masked.
- Test access was authorized in the separate manifest before the first test bundle load.
