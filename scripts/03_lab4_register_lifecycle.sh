#!/usr/bin/env bash
# A5 + A6: register v1-v3, then run the lifecycle (candidate -> production -> archived).
set -euo pipefail
cd "$(dirname "$0")/.."

python src/register_models.py
echo; echo "=== STEP 1: first production release (v1) ==="
python src/lifecycle.py candidate --version 1
python src/lifecycle.py validate  --version 1
python src/lifecycle.py promote   --version 1
python src/lifecycle.py show

echo; echo "=== STEP 2: better model (v3) replaces it; v1 is archived automatically ==="
python src/lifecycle.py candidate --version 3
python src/lifecycle.py validate  --version 3
python src/lifecycle.py promote   --version 3
python src/lifecycle.py show

echo; echo "=== STEP 3: how a teammate retrieves the production model ==="
python src/lifecycle.py retrieve
python src/trace_model.py --alias production
echo; echo "Tip: take MLflow UI screenshots (Models -> HousePriceModel) after STEP 1 and after STEP 2."
