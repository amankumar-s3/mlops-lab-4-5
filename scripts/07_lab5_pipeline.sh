#!/usr/bin/env bash
# B7: DVC pipeline = prepare -> train. (dvc.yaml is created by these commands.)
set -euo pipefail
cd "$(dirname "$0")/.."

dvc stage add -n prepare \
  -d src/prepare.py -d src/common.py -d data/dataset.csv \
  -p configs/config.yaml:data.test_size,data.val_size,data.random_state \
  -o data/processed \
  python src/prepare.py

dvc stage add -n train \
  -d src/train.py -d src/common.py -d data/processed \
  -p configs/config.yaml:candidates.rf_large \
  -o models/model.joblib -M reports/metrics.json \
  python src/train.py --candidate rf_large --use-processed --save-local --purpose pipeline --run-name pipeline_rf_large

cat dvc.yaml
dvc dag
dvc repro
cat reports/metrics.json
dvc metrics show
echo; echo "== second repro: nothing changed, so nothing reruns =="
dvc repro
for p in dvc.yaml dvc.lock data/.gitignore models/.gitignore reports/metrics.json; do [ -e "$p" ] && git add "$p"; done
git commit -m "Add DVC pipeline (prepare -> train)"
dvc push
