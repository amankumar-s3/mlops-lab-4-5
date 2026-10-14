#!/usr/bin/env bash
# B5 (steps 22-24): create dataset v2 (and optionally v3) with controlled changes.
#   V3=1 bash scripts/05_lab5_dataset_versions.sh     -> also creates the optional v3
set -euo pipefail
cd "$(dirname "$0")/.."

version() {  # $1=tag $2=make-target $3=commit msg $4=note
  python src/dataset_tools.py "$2"
  dvc status
  dvc add data/dataset.csv
  git add data/dataset.csv.dvc
  git commit -m "$3"
  git tag -f "$1"
  dvc push
  python src/dataset_tools.py info --tag "$1" --note "$4"
}

version dataset-v2 make-v2 "Update dataset version (v2: drop rows with missing values)" \
        "Dropped rows with missing numeric features"
if [ "${V3:-0}" = "1" ]; then
  version dataset-v3 make-v3 "Update dataset version (v3: drop capped targets)" \
          "Also dropped rows with capped (max) target value"
fi
echo; git log --oneline --decorate -- data/dataset.csv.dvc
echo; python src/dataset_tools.py table
