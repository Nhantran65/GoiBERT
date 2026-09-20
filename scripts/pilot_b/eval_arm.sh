#!/bin/bash
# Fine-tune probes for one pilot-B arm: 5 tasks (+ seed 43 on ABSA/ViSFD), clean + diacritic-stripped validation.
# usage: scripts/pilot_b/eval_arm.sh <arm> "<gpu list>"    — one serial queue per GPU (micro 8 -> 8-12 GB per job,
# which fits next to a 20 GB DDP rank); finished or currently-running (task, seed) pairs are skipped.
arm=$1; gpus=(${2:-0 1 2 3})
cd /home/kiettqa/GoiBERT
model=artifacts/pilot_b/cpt/$arm/hf; out=artifacts/pilot_b/probe/$arm; mkdir -p $out/logs $out/queue; rm -f $out/queue/*
tasks=(absa_restaurant absa_hotel visfd); seeds=(42 43)
pending=()
for t in "${tasks[@]}"; do for s in "${seeds[@]}"; do pending+=("$t:$s"); done; done
pending+=("vsfc_sentiment:42" "vsfc_topic:42")
n=${#gpus[@]}; i=0
for j in "${pending[@]}"; do
  t=${j%%:*}; s=${j##*:}
  [ -f $out/${t}_hf_lr2e-05_s$s.json ] && grep -q '"epoch": 20' $out/${t}_hf_lr2e-05_s$s.json && continue
  pgrep -f "cpt/$arm/hf .*--task $t .*--seed $s " > /dev/null && continue
  g=${gpus[$((i % n))]}; i=$((i+1)); echo "$t $s" >> $out/queue/gpu$g.txt
done
for q in $out/queue/gpu*.txt; do
  g=${q##*gpu}; g=${g%.txt}
  ( while read -r t s; do
      CUDA_VISIBLE_DEVICES=$g CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv-e0-cuda/bin/python scripts/research_checks/probe_mmbert_finetune.py --model $model --tokenizer FacebookAI/xlm-roberta-large --device cuda:0 --task $t --lr 2e-5 --epochs 20 --micro 8 --seed $s --eval-strip --out $out > $out/logs/${t}_s$s.log 2>&1
    done < $q ) &
done
wait
echo "eval $arm done $(date)" >> $out/logs/done.txt
