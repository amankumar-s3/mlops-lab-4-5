# MLOps Lab 4-5: Advanced MLflow + Data Versioning with DVC

House-price regression on the Kaggle **California Housing Prices** dataset
(`camnugent/california-housing-prices`, file `housing.csv`, target `median_house_value`).

Core question: *Can another engineer identify the model that was deployed and recover the code and
dataset version used to create it?*  Git (code) + DVC (data) + MLflow (experiments/models) answer it together.

## Repository layout
```
.dvc/                      DVC config (committed)
data/dataset.csv.dvc       DVC metadata of the dataset (committed); dataset.csv itself lives in DVC cache/remote
src/common.py              shared helpers (config, splits, pipeline, git/dvc info)
src/train.py               trains candidates, logs params/metrics/artifacts/model+signature to MLflow
src/evaluate.py            evaluates a model URI on the held-out test split
src/autolog_demo.py        sklearn autologging + inspection of what it recorded
src/signature_demo.py      model-signature inspection and malformed-input tests
src/register_models.py     registers the 3 best runs as versions v1-v3
src/lifecycle.py           candidate -> production -> archived (aliases + tags), retrieval demo
src/dataset_tools.py       dataset info / controlled changes (v2, v3) / verify against .dvc md5
src/prepare.py             DVC pipeline stage 1
src/trace_model.py         registered model -> source run -> Git commit + dataset hash + recovery commands
configs/config.yaml        dataset columns, candidates, acceptance rule
scripts/                   numbered scripts that run the whole lab
reports/                   generated tables, figures, evaluation reports (paste into Report)
```

## Reproduce (Linux)
```bash
bash scripts/00_setup_env.sh          # venv + installs + git init  (then: source .venv/bin/activate)
source .venv/bin/activate
bash scripts/01_get_data.sh           # Kaggle download -> data/dataset.csv   (needs ~/.kaggle/kaggle.json)

# ---- Lab 4: MLflow
bash scripts/02_lab4_train.sh         # 4 runs, autolog demo, signature demo
mlflow ui --backend-store-uri sqlite:///mlflow.db   # second terminal -> http://127.0.0.1:5000 (screenshots)
bash scripts/03_lab4_register_lifecycle.sh          # v1-v3, candidate/production/archived

# ---- Lab 5: DVC
bash scripts/04_lab5_dvc_track.sh     # dvc init, track dataset v1, local remote, push
V3=1 bash scripts/05_lab5_dataset_versions.sh       # dataset v2 (+ optional v3), push
bash scripts/06_lab5_recover.sh       # restore v1 on a temp branch, verify, return to latest
bash scripts/07_lab5_pipeline.sh      # dvc stage add / repro / dag

# ---- Part C: integration
bash scripts/08_integration.sh        # retrain on latest data, register v4, trace to Git + DVC
```
Every script prints the tables/outputs you need for the report; generated markdown/CSV tables are in `reports/`.

## Notes
* MLflow registry workflow used: **aliases + tags** (stages are deprecated in MLflow 2.9+/3.x).
* Local DVC remote (`../dvc-storage`) is for learning only - it is not a shared backup.
* Never commit `~/.kaggle/kaggle.json` or any credentials. Check the dataset license before publishing data.
* `mlflow.db`, `mlartifacts/` and `.venv/` are git-ignored; the dataset is tracked by DVC, not Git.
