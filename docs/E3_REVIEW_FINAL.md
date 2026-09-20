# E3 independent review

Status: PASS at the implementation gate and post-run validation audit.

The two authorized E3 trials completed successfully. The independent audit
found exact validation row alignment, matching metrics, matching checkpoint
and source hashes, completed 30-epoch artifacts, and no test access.

## Gate evidence

Targeted tests covered pooled unequal support, uniform 3D logits, all-absent
differentiable zero, absent-logit removal, and ViSFD OTHERS exclusion. The
runner preserved the frozen E1 model, head, optimizer, scheduler, loader,
seed, and validation selection. The launch hash was
`a17ec51f8386162e34ed35a7d4695feb06bb813d21c8c812b221b1084b1b2777`.

## Independent post-run audit

The audit recomputed validation metrics from both E3 prediction files using
the frozen evaluator, verified IDs and gold labels against train-free
validation splits, and checked checkpoint hashes and source hashes.

ViSFD E3 pair F1 was 80.223 versus E1 80.072 and NC 81.088; detection F1
was 85.954 versus E1 86.909. Missed/spurious aspects were 498/505 versus
460/476, and wrong polarity fell from 318 to 295. Restaurant E3 pair F1 was
70.130 versus E1 69.017 and NC 69.060; detection F1 was 81.962 versus E1
83.686. Missed/spurious aspects were 201/174 versus 163/184, and wrong
polarity fell from 156 to 123.

The polarity loss therefore helped Restaurant pair F1 through a precision
trade-off but missed the 0.3-point screening threshold on ViSFD. It is not a
common improvement across both tasks.
