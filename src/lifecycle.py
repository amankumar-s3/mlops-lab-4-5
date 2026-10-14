"""A6: model lifecycle with registry ALIASES + TAGS (MLflow >= 2.9 / 3.x).

Mapping to the classic stages:
    Staging    -> alias `candidate`  + tag lifecycle_state=candidate
    Production -> alias `production` + tag lifecycle_state=production
    Archived   -> NO alias           + tag lifecycle_state=archived

Commands
    python src/lifecycle.py candidate --version 1
    python src/lifecycle.py validate  --version 1      # applies acceptance rule on the TEST split
    python src/lifecycle.py promote   --version 1      # needs validation_status=passed
    python src/lifecycle.py archive   --version 1 --reason "superseded by v3"
    python src/lifecycle.py show                       # table of versions/aliases/state
    python src/lifecycle.py retrieve                   # what a teammate does: load @production
"""
import argparse
import csv
from datetime import datetime, timezone

import mlflow
import pandas as pd

from common import ROOT, load_config, md_table, setup_mlflow
from evaluate import evaluate_model

cfg = load_config()
setup_mlflow(cfg)
client = mlflow.MlflowClient()
NAME = cfg["mlflow"]["registered_model_name"]
LOG = ROOT / "reports" / "lifecycle_log.csv"


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log_action(version, state, action, evidence):
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "model_version", "lifecycle_state", "action", "evidence_reason"])
        w.writerow([now(), f"v{version}", state, action, evidence])
    print(f"[{state}] v{version}: {action} | {evidence}")


def append_description(version, text):
    mv = client.get_model_version(NAME, str(version))
    client.update_model_version(NAME, str(version), (mv.description or "") + f"\n[{now()}] {text}")


def alias_version(alias):
    try:
        return str(client.get_model_version_by_alias(NAME, alias).version)
    except mlflow.exceptions.MlflowException:
        return None


def rule_text():
    a = cfg["acceptance"]
    return (f"test RMSE <= {a['max_test_rmse']} AND test R2 >= {a['min_test_r2']}"
            + (" AND signature present" if a.get("require_signature") else ""))


def cmd_candidate(v):
    client.set_registered_model_alias(NAME, "candidate", str(v))
    client.set_model_version_tag(NAME, str(v), "lifecycle_state", "candidate")
    log_action(v, "Staging/Candidate", f"alias 'candidate' -> v{v}", f"awaiting validation; rule: {rule_text()}")


def cmd_validate(v):
    a = cfg["acceptance"]
    metrics, _, _ = evaluate_model(f"models:/{NAME}/{v}", cfg)
    has_sig = mlflow.models.get_model_info(f"models:/{NAME}/{v}").signature is not None
    ok = metrics["rmse"] <= a["max_test_rmse"] and metrics["r2"] >= a["min_test_r2"] \
        and (has_sig or not a.get("require_signature"))
    status = "passed" if ok else "failed"
    for k, val in {"validation_status": status, "validation_test_rmse": f"{metrics['rmse']:.2f}",
                   "validation_test_r2": f"{metrics['r2']:.4f}", "validation_test_mae": f"{metrics['mae']:.2f}",
                   "validation_rule": rule_text(), "validated_at": now()}.items():
        client.set_model_version_tag(NAME, str(v), k, val)
    append_description(v, f"Validation {status.upper()}: test RMSE={metrics['rmse']:.0f}, R2={metrics['r2']:.3f}, "
                          f"signature={has_sig}; rule: {rule_text()}")
    log_action(v, "Staging/Candidate", f"validation {status}",
               f"test RMSE={metrics['rmse']:.0f}, R2={metrics['r2']:.3f}, signature={has_sig}")
    return ok


def cmd_archive(v, reason):
    if alias_version("production") == str(v):
        client.delete_registered_model_alias(NAME, "production")
    if alias_version("candidate") == str(v):
        client.delete_registered_model_alias(NAME, "candidate")
    for k, val in {"lifecycle_state": "archived", "archived_at": now(), "archived_reason": reason}.items():
        client.set_model_version_tag(NAME, str(v), k, val)
    append_description(v, f"ARCHIVED/RETIRED: {reason}")
    log_action(v, "Archived/Retired", "aliases removed, tag lifecycle_state=archived", reason)


def cmd_promote(v):
    tags = client.get_model_version(NAME, str(v)).tags
    if tags.get("validation_status") != "passed":
        raise SystemExit(f"v{v} has not passed validation (status={tags.get('validation_status')}). "
                         f"Run: python src/lifecycle.py validate --version {v}")
    previous = alias_version("production")
    client.set_registered_model_alias(NAME, "production", str(v))
    client.set_model_version_tag(NAME, str(v), "lifecycle_state", "production")
    client.set_model_version_tag(NAME, str(v), "promoted_at", now())
    if alias_version("candidate") == str(v):
        client.delete_registered_model_alias(NAME, "candidate")
    append_description(v, f"PROMOTED to production (alias 'production'). Evidence: validation passed, "
                          f"test RMSE={tags['validation_test_rmse']}, R2={tags['validation_test_r2']}.")
    log_action(v, "Production", f"alias 'production' -> v{v}",
               f"validation passed (test RMSE={tags['validation_test_rmse']}, R2={tags['validation_test_r2']})")
    if previous and previous != str(v):
        cmd_archive(previous, f"superseded by v{v} (lower test RMSE)")


def cmd_show():
    rows = []
    for mv in sorted(client.search_model_versions(f"name='{NAME}'"), key=lambda m: int(m.version)):
        mv = client.get_model_version(NAME, mv.version)  # search results may omit aliases
        t = mv.tags
        rows.append({"version": f"v{mv.version}", "state": t.get("lifecycle_state", "-"),
                     "aliases": ",".join(mv.aliases) or "-", "candidate": t.get("candidate", "-"),
                     "val_rmse": t.get("val_rmse", "-"), "validation": t.get("validation_status", "-"),
                     "test_rmse": t.get("validation_test_rmse", "-")})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    (ROOT / "reports" / "lifecycle_state_table.md").write_text(md_table(df))


def cmd_retrieve():
    mv = client.get_model_version_by_alias(NAME, "production")
    print(f"Alias 'production' of {NAME} -> version {mv.version}")
    print(f"  source run : {mv.run_id}")
    print(f"  tags       : { {k: v for k, v in mv.tags.items() if k in ('candidate','validation_status','git_commit','dataset_md5')} }")
    model = mlflow.pyfunc.load_model(f"models:/{NAME}@production")
    from common import load_splits
    X, _ = load_splits(cfg)["test"]
    print("  3 predictions:", [round(float(p)) for p in model.predict(X.head(3))])
    print(f"\nLoad it anywhere with:  mlflow.pyfunc.load_model('models:/{NAME}@production')")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["candidate", "validate", "promote", "archive", "show", "retrieve"])
    ap.add_argument("--version", type=int)
    ap.add_argument("--reason", default="manually archived")
    a = ap.parse_args()
    if a.action in ("candidate", "validate", "promote", "archive") and a.version is None:
        ap.error("--version is required")
    {"candidate": lambda: cmd_candidate(a.version), "validate": lambda: cmd_validate(a.version),
     "promote": lambda: cmd_promote(a.version), "archive": lambda: cmd_archive(a.version, a.reason),
     "show": cmd_show, "retrieve": cmd_retrieve}[a.action]()
