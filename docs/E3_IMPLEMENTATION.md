# E3 implementation record

E3 is a two trial, validation-only screening experiment for the hypothesis
that the GoiBERT student benefits from a separate polarity signal while its
architecture stays fixed.

The implementation is isolated in `src/goibert_e3` and
`scripts/execute_e3_validation.py`.  It imports the frozen E1 data loader,
classifier, optimizer, scheduler, tokenization, validation evaluator, and
checkpoint writer.  No E1 or E2 source/config file is modified.

For every ABSA batch the objective is

```text
L = CE_E1(all source classes) + 0.5 * CE(present polarity slots)
```

The auxiliary term pools all gold-present slots in the batch.  For a selected
four-class slot it drops class 0 (absent), keeps logits 1..3 in source order,
and changes the target from `gold` to `gold - 1`.  A batch with no gold-present
slot returns a zero connected to the logits graph.  ViSFD slot index 5
(`OTHERS`) is binary and has no polarity, so it is excluded explicitly;
Restaurant uses all twelve slots.

The two locked trials are seed 42, learning rate `1e-4`, 30 epochs, batch size
32, max length 256, BF16, SDPA, fresh `models/goiturn-s-gd`, and the original
E1 classifier head.  Validation pair micro F1 selects the checkpoint.  The
test split is never loaded by this runner.

`tests/test_e3_loss.py` covers pooled polarity CE, absent-logit removal,
differentiable zero, OTHERS exclusion, and exact recovery of the E1 CE when
the auxiliary weight is zero.  Full training is launched only through the
durable orchestrator after its implementation review gate.
