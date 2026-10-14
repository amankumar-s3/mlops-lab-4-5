#!/usr/bin/env bash
# Part C: retrain on the dataset version in the current checkout, log Git commit + dataset hash,
# register the model as a new version, and trace it back to code and data.
set -euo pipefail
cd "$(dirname "$0")/.."

for p in src scripts configs docs README.md requirements.txt; do [ -e "$p" ] && git add "$p"; done
git diff --cached --quiet || git commit -m "Update code for integration run"
dvc pull
python src/dataset_tools.py verify

TAG=$(git describe --tags --match 'dataset-v*' --abbrev=0)
python src/train.py --candidate gbr --run-name "gbr_on_${TAG}" --purpose integration --register

python src/trace_model.py --latest
echo; echo "== and the current production model =="
python src/trace_model.py --alias production
python src/lifecycle.py show
