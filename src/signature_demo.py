"""A4: inspect a logged model's signature and test what it does / does not catch.

    python src/signature_demo.py                       # best run by val_rmse
    python src/signature_demo.py --model-uri models:/HousePriceModel@production
Writes reports/signature_demo.md
"""
import argparse

import mlflow
import numpy as np
import pandas as pd

from common import ROOT, load_config, load_splits, setup_mlflow

ap = argparse.ArgumentParser()
ap.add_argument("--model-uri", default=None)
args = ap.parse_args()

cfg = load_config()
setup_mlflow(cfg)
client = mlflow.MlflowClient()
uri = args.model_uri
if uri is None:
    exp = client.get_experiment_by_name(cfg["mlflow"]["experiment_name"])
    best = client.search_runs([exp.experiment_id], "tags.purpose = 'candidate_training'",
                              order_by=["metrics.val_rmse ASC"], max_results=1)[0]
    uri = f"runs:/{best.info.run_id}/model"

info = mlflow.models.get_model_info(uri)
sig = info.signature
out = [f"# Signature demo for `{uri}`", "", f"MLflow {mlflow.__version__}", "",
       "## Input schema", "", "| column | type |", "|---|---|"]
out += [f"| {c.name} | {c.type} |" for c in sig.inputs.inputs]
out += ["", "## Output schema", "", f"`{sig.outputs}`", ""]

model = mlflow.pyfunc.load_model(uri)
X_good = load_splits(cfg)["test"][0].dropna().head(3).reset_index(drop=True)


def attempt(title, frame):
    try:
        pred = model.predict(frame)
        res = f"ACCEPTED -> predictions {np.round(np.asarray(pred).ravel(), 0).tolist()}"
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        msg = msg.split("Error:")[-1].strip().strip('"') if "Error:" in msg else msg.splitlines()[0]
        res = f"REJECTED -> {type(e).__name__}: {msg[:200]}"
    out.append(f"- **{title}**: {res}")
    print(f"{title}: {res}")


out += ["## Experiments", ""]
attempt("1. Valid request", X_good)
attempt("2. Missing column (median_income dropped)", X_good.drop(columns=["median_income"]))
bad = X_good.copy(); bad["median_income"] = ["high", "low", "mid"]
attempt("3. Wrong type (text in numeric column)", bad)
attempt("4. Integer column where the signature says double (int64 -> float64)", X_good.assign(housing_median_age=X_good["housing_median_age"].astype(int)))
attempt("5. Extra unexpected column", X_good.assign(unexpected_col=1))
attempt("6. Columns in a different order", X_good[X_good.columns[::-1]])
nonsense = X_good.copy(); nonsense["median_income"] = -50.0; nonsense["housing_median_age"] = 9999.0
attempt("7. Right schema, absurd values (negative income, age 9999)", nonsense)
unseen = X_good.copy(); unseen["ocean_proximity"] = "MARS"
attempt("8. Right schema, unseen category value", unseen)
out += ["", "## Interpretation", "",
        "A signature validates **structure** (names, types, optionality). Tests 2-3 show malformed requests "
        "being stopped before they reach the model. Tests 7-8 show what it cannot guarantee: sensible value "
        "ranges, unseen categories, distribution drift or prediction quality. Add input validation/monitoring for those."]
(ROOT / "reports" / "signature_demo.md").write_text("\n".join(out))
print("\nSaved reports/signature_demo.md")
