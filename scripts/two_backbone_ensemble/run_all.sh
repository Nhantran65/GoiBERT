#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
for model in cafe cafe_s43 xlmr_large p_vi o b1b; do
  for task in absa_restaurant visfd vsfc_topic; do
    if [[ "$model" == cafe_s43 && "$task" != absa_restaurant ]]; then continue; fi
    if [[ -f "artifacts/two_backbone_ensemble/manifest_parts/$model/$task.json" ]]; then continue; fi
    if [[ "$model" == cafe || "$model" == cafe_s43 || "$model" == xlmr_large ]]; then
      venv=.venv-e0-cuda
    else
      venv=.venv-qwen-probe
    fi
    echo "START $model $task $venv" >&2
    if "$venv/bin/python" scripts/two_backbone_ensemble/export_logits.py "$model" "$task" --device cuda:0 > "artifacts/two_backbone_ensemble/logs/${model}_${task}.log" 2>&1; then
      echo "OK $model $task" >&2
    else
      echo "FAIL $model $task (see log)" >&2
    fi
  done
done
