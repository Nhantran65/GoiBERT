# E8 test evaluation — user authorization

User requested: “Chạy test hết đi. Rồi so sánh lại với BamiBERT”. Scope interpreted from the preceding decision: E8 NC+LLRD recipe on all5tasks, seed42. No mixing task-specific champions from earlier rounds.

Root coordinates and interprets. Implementator explicitly uses gpt-5.6-luna with reasoning xhigh, fork_turns=none, per AGENTS.md.

Locked inputs are artifacts/e8/<task>/seed_42/best_checkpoint.pt. Selected epochs before test: Sentiment21,Topic25,ViSFD28,Restaurant25,Hotel26. Create checkpoint/source/data hash manifest before inference. No new training, seeds, coefficient changes or test-based reselection.

Test inference: original CLS classifier, strict weight load, eval/inference_mode, CUDA BF16/SDPA,max256,batch32. No NC noise, dropout training, teacher or optimizer at inference. Existing E1/E2 test results may be reused as same-seed42 baselines, not rerun or replaced by3/5seed aggregates.

Expected rows:3166/3166/2224/1938/2030. Recompute metrics from predictions, verify ID/gold alignment and checkpoint hashes unchanged. Compare all10metrics with benchmarks/paper_targets.csv. F1 equivalence with paper/ABSA lineage limitations remain; they do not block this explicitly authorized test evaluation. Single-seed point differences are not significance claims.

Outputs: E8test manifest/readout/CSV/accesslog, percheckpoint e8_test_* files, docs/E8_TEST_RESULT.md. Complete all5 before final handoff.
