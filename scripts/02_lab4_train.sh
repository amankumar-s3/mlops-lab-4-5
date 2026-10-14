#!/usr/bin/env bash
# A3 + A4: track experiments (4 candidates), autologging demo, model signature demo.
set -euo pipefail
cd "$(dirname "$0")/.."

for p in src scripts configs docs README.md .gitignore requirements.txt data/.gitkeep models/.gitkeep reports/figures/.gitkeep; do
  [ -e "$p" ] && git add "$p"   # never `git add -A`: data/dataset.csv must be tracked by DVC, not Git
done
git diff --cached --quiet || git commit -m "Add MLflow training code (Lab 4)"

python src/train.py             # 4 runs -> reports/candidate_comparison.csv
python src/autolog_demo.py      # -> reports/autolog_inspection.md
python src/signature_demo.py    # -> reports/signature_demo.md

echo
echo "NEXT: open the MLflow UI in another terminal and take screenshots S1-S6:"
echo "   mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000"
