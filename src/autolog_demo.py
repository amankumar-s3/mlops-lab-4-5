"""A3: explore sklearn autologging and inspect what it really recorded.

    python src/autolog_demo.py
Writes reports/autolog_inspection.md (paste into the report).
"""
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error

from common import ROOT, build_pipeline, load_config, load_splits, setup_mlflow

cfg = load_config()
setup_mlflow(cfg)
splits = load_splits(cfg)
(X_tr, y_tr), (X_val, y_val) = splits["train"], splits["val"]

mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True, log_models=True)
cand = cfg["candidates"]["rf_small"]
pipe = build_pipeline(cfg, cand["model"], cand["params"])

with mlflow.start_run(run_name="autolog_demo_rf_small") as run:
    mlflow.set_tag("purpose", "autolog_demo")
    pipe.fit(X_tr, y_tr)                       # autolog: params, training metrics, model
    pred = pipe.predict(X_val)                 # post-training metrics need an explicit metric call
    mean_squared_error(y_val, pred)            # may or may not be captured: inspect the output!
    pipe.score(X_val, y_val)                   # and this one (score_X_val)
mlflow.sklearn.autolog(disable=True)

client = mlflow.MlflowClient()
r = client.get_run(run.info.run_id)
arts = [a.path for a in client.list_artifacts(run.info.run_id)]
lines = [f"# Autologging inspection (run {run.info.run_id})", "",
         f"MLflow version: {mlflow.__version__}", "",
         f"## Parameters logged automatically ({len(r.data.params)})", ""]
lines += [f"- `{k}` = {v}" for k, v in sorted(r.data.params.items())][:40]
lines += ["", f"## Metrics logged automatically ({len(r.data.metrics)})", ""]
lines += [f"- `{k}` = {v:.4f}" for k, v in sorted(r.data.metrics.items())]
lines += ["", "## Run-level artifacts", ""] + [f"- `{a}`" for a in arts]
user_tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
lines += ["", "## Tags (non-system)", ""] + [f"- `{k}` = {v}" for k, v in user_tags.items()]
lines += ["", "## What autolog did NOT record (verify yourself)", "",
          "- Git commit id: " + ("present" if "git_commit" in r.data.tags else "NOT logged"),
          "- Dataset hash/version (DVC): " + ("present" if "dataset_md5" in r.data.tags else "NOT logged"),
          "- Validation metrics for the split I defined (val_rmse): "
          + ("present" if "val_rmse" in r.data.metrics else "NOT logged unless a metric function is called"),
          "- Custom plots (residuals, feature importance): NOT logged",
          "- Acceptance decision / business context: NOT logged"]
out = ROOT / "reports" / "autolog_inspection.md"
out.write_text("\n".join(lines))
print("\n".join(lines))
print(f"\nSaved {out}")
