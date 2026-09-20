#!/bin/bash
# Social-media probes: tasks x arms, one serial queue per GPU. usage: run_social.sh "<tasks>" "<arms>" "<gpus>"
cd /home/kiettqa/GoiBERT; tasks=($1); arms=($2); gpus=($3); out=artifacts/pilot_b/social; mkdir -p $out/logs $out/queue; rm -f $out/queue/*
i=0; for t in "${tasks[@]}"; do for a in "${arms[@]}"; do g=${gpus[$((i % ${#gpus[@]}))]}; i=$((i+1)); echo "$t $a" >> $out/queue/gpu$g.txt; done; done
for q in $out/queue/gpu*.txt; do g=${q##*gpu}; g=${g%.txt}
  ( while read -r t a; do
      if [ "$a" = "cafebert_0tok" ]; then m=uitnlp/CafeBERT; else m=artifacts/pilot_b/cpt/$a/hf; fi
      CUDA_VISIBLE_DEVICES=$g CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv-e0-cuda/bin/python scripts/pilot_b/probe_csv_finetune.py --model $m --tokenizer FacebookAI/xlm-roberta-large --device cuda:0 --task $t --lr 2e-5 --epochs 10 --micro 8 --seed 42 --eval-strip --out $out --tag _$a > $out/logs/${t}_$a.log 2>&1
    done < $q ) &
done; wait; echo "social done $(date)" >> $out/logs/done.txt
