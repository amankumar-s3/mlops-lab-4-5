"""Shared helpers: config, data loading/splitting, pipeline building, Git/DVC info, MLflow setup."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


# ----------------------------------------------------------------- config
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------- hashing / git / dvc
def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def git_info() -> dict:
    """Commit id, branch, dirty flag (tracked files only) and latest dataset tag."""
    commit = _run(["git", "rev-parse", "HEAD"]) or "no-git-commit"
    return {
        "git_commit": commit,
        "git_commit_short": commit[:8],
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "unknown",
        "git_dirty": str(bool(_run(["git", "status", "--porcelain", "--untracked-files=no"]))),
        "dataset_version": _run(["git", "describe", "--tags", "--match", "dataset-v*", "--abbrev=0"])
        or "untagged",
    }


def dvc_file(cfg: dict) -> Path:
    return ROOT / (cfg["data"]["path"] + ".dvc")


def dvc_md5(cfg: dict) -> str | None:
    """md5 recorded in the .dvc metadata file of the CURRENT Git checkout."""
    p = dvc_file(cfg)
    if not p.exists():
        return None
    meta = yaml.safe_load(p.read_text())
    return meta["outs"][0]["md5"]


def dataset_summary(cfg: dict) -> dict:
    path = ROOT / cfg["data"]["path"]
    df = pd.read_csv(path)
    return {
        "dataset_path": cfg["data"]["path"],
        "dataset_rows": len(df),
        "dataset_cols": df.shape[1],
        "dataset_size_bytes": path.stat().st_size,
        "dataset_md5": file_md5(path),
        "dataset_nulls": int(df.isna().sum().sum()),
    }


# ----------------------------------------------------------------- data
def load_data(cfg: dict, path: Path | None = None) -> pd.DataFrame:
    d = cfg["data"]
    df = pd.read_csv(path or ROOT / d["path"])
    df = df.dropna(subset=[d["target"]]).reset_index(drop=True)
    # Force numeric features to float64 so the model signature stays stable
    # (an int column that later receives NaN would otherwise change type).
    for c in d["numeric_features"]:
        df[c] = df[c].astype("float64")
    return df


def feature_cols(cfg: dict) -> list[str]:
    return cfg["data"]["numeric_features"] + cfg["data"]["categorical_features"]


def split_data(df: pd.DataFrame, cfg: dict):
    d = cfg["data"]
    train_val, test = train_test_split(df, test_size=d["test_size"], random_state=d["random_state"])
    val_frac = d["val_size"] / (1 - d["test_size"])
    train, val = train_test_split(train_val, test_size=val_frac, random_state=d["random_state"])
    return train, val, test


def load_splits(cfg: dict, use_processed: bool = False):
    """Return dict name -> (X, y). Splits are deterministic (fixed random_state)."""
    d = cfg["data"]
    if use_processed:
        pdir = ROOT / d["processed_dir"]
        parts = [load_data(cfg, pdir / f"{n}.csv") for n in ("train", "val", "test")]
    else:
        parts = split_data(load_data(cfg), cfg)
    cols = feature_cols(cfg)
    return {n: (p[cols], p[d["target"]]) for n, p in zip(("train", "val", "test"), parts)}


# ----------------------------------------------------------------- model
def build_pipeline(cfg: dict, kind: str, params: dict) -> Pipeline:
    d = cfg["data"]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
             d["numeric_features"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), d["categorical_features"]),
        ]
    )
    models = {"ridge": Ridge, "random_forest": RandomForestRegressor,
              "gradient_boosting": GradientBoostingRegressor}
    return Pipeline([("pre", pre), ("model", models[kind](**params))])


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def feature_importance(pipe: Pipeline) -> pd.Series:
    names = pipe.named_steps["pre"].get_feature_names_out()
    m = pipe.named_steps["model"]
    vals = m.feature_importances_ if hasattr(m, "feature_importances_") else np.abs(m.coef_)
    return pd.Series(vals, index=names).sort_values(ascending=False)


def save_plots(pipe, y_true, y_pred, name: str) -> list[Path]:
    out = ROOT / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    resid = np.asarray(y_true) - np.asarray(y_pred)

    p1 = out / f"{name}_residuals.png"
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, resid, s=5, alpha=0.4)
    plt.axhline(0, color="red", lw=1)
    plt.xlabel("Predicted value")
    plt.ylabel("Residual (actual - predicted)")
    plt.title(f"Residual plot (validation) - {name}")
    plt.tight_layout()
    plt.savefig(p1, dpi=120)
    plt.close()

    p2 = out / f"{name}_feature_importance.png"
    imp = feature_importance(pipe).head(12)[::-1]
    plt.figure(figsize=(6, 4))
    plt.barh(imp.index, imp.values)
    plt.title(f"Top features - {name}")
    plt.tight_layout()
    plt.savefig(p2, dpi=120)
    plt.close()
    return [p1, p2]


# ----------------------------------------------------------------- mlflow
def setup_mlflow(cfg: dict):
    import mlflow

    m = cfg["mlflow"]
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / m['db_file']}")
    if mlflow.get_experiment_by_name(m["experiment_name"]) is None:
        mlflow.create_experiment(m["experiment_name"],
                                 artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(m["experiment_name"])
    return mlflow


def md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:  # tabulate not installed
        return df.to_string(index=False)
