"""s3-t1: check the rebuilt 10-day UCS dataset against commit a0ae7bc's ucs_windows.parquet.

Loads data-engineering/data/ucs/ucs_windows.parquet (the rebuild) and a0ae7bc's
ucs_windows.parquet (read with `git show`, not copied) and checks:
  1. 4,545 windows and 10 distinct source_day values
  2. identical window_start_utc order and label_attack_type
  3. numeric columns = exactly the 406 features in ml1/configs/mdn_v1_feature_schema.txt
     plus label_binary and future_attack_label, nothing else
  4. all 406 feature values equal a0ae7bc's (np.allclose, equal_nan=True)
Every check is evaluated (no early exit). Writes data-engineering/data/ucs/REBUILD_CHECK.json.
Run from the repository root or anywhere: python data-engineering/src/check_10day_rebuild.py
"""
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REBUILT = REPO / "data-engineering/data/ucs/ucs_windows.parquet"
REF_REV = "a0ae7bc"
REF_PATH = "data-engineering/data/ucs/ucs_windows.parquet"
SCHEMA = REPO / "ml1/configs/mdn_v1_feature_schema.txt"
OUT = REPO / "data-engineering/data/ucs/REBUILD_CHECK.json"


def main():
    new = pd.read_parquet(REBUILT)
    blob = subprocess.run(["git", "show", f"{REF_REV}:{REF_PATH}"], cwd=REPO, capture_output=True, check=True).stdout
    ref = pd.read_parquet(io.BytesIO(blob))
    features = [c for c in SCHEMA.read_text().splitlines()[1].strip().split(",") if c]
    checks = {}

    days = sorted(new["source_day"].astype(str).unique())
    checks["1_windows_and_days"] = {
        "passed": bool(len(new) == 4545 and len(days) == 10),
        "rebuilt_windows": int(len(new)), "rebuilt_distinct_source_days": len(days), "source_days": days,
        "reference_windows": int(len(ref)),
    }

    same_len = len(new) == len(ref)
    ts_equal = bool(same_len and (pd.to_datetime(new["window_start_utc"], utc=True).values
                                   == pd.to_datetime(ref["window_start_utc"], utc=True).values).all())
    lab_equal = bool(same_len and (new["label_attack_type"].astype(str).values
                                    == ref["label_attack_type"].astype(str).values).all())
    checks["2_order_and_labels"] = {"passed": ts_equal and lab_equal,
                                    "window_start_utc_identical": ts_equal, "label_attack_type_identical": lab_equal}

    numeric = [c for c in new.columns if pd.api.types.is_numeric_dtype(new[c]) and not pd.api.types.is_bool_dtype(new[c])]
    expected = set(features) | {"label_binary", "future_attack_label"}
    checks["3_numeric_columns"] = {
        "passed": bool(set(numeric) == expected and len(features) == 406),
        "schema_features": len(features), "rebuilt_numeric_columns": len(numeric),
        "unexpected": sorted(set(numeric) - expected), "missing": sorted(expected - set(numeric)),
    }

    missing_ref = [c for c in features if c not in ref.columns]
    missing_new = [c for c in features if c not in new.columns]
    if same_len and not missing_ref and not missing_new:
        a = new[features].to_numpy(dtype=np.float64)
        b = ref[features].to_numpy(dtype=np.float64)
        close = np.isclose(a, b, equal_nan=True)
        bad_cols = [f for f, ok in zip(features, close.all(axis=0)) if not ok]
        diff = np.abs(np.nan_to_num(a) - np.nan_to_num(b))
        suffix = lambda c: c.rsplit("_", 1)[-1] if c.rsplit("_", 1)[-1] in ("mean", "std", "sum", "min", "max") else "other"
        by_suffix = {}
        for c, ok in zip(features, close.all(axis=0)):
            s = by_suffix.setdefault(suffix(c), {"columns": 0, "not_close": 0})
            s["columns"] += 1
            s["not_close"] += int(not ok)
        checks["4_feature_values"] = {
            "passed": bool(close.all()),
            "columns_not_close": len(bad_cols), "first_columns_not_close": bad_cols[:20],
            "not_close_by_aggregation": by_suffix,
            "rows_with_any_difference": int((~close).any(axis=1).sum()),
            "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        }
    else:
        checks["4_feature_values"] = {"passed": False, "reason": "row count differs or feature columns missing",
                                      "missing_in_reference": missing_ref, "missing_in_rebuilt": missing_new}

    result = {"generated_by": "data-engineering/src/check_10day_rebuild.py", "reference": f"{REF_REV}:{REF_PATH}",
              "all_passed": all(c["passed"] for c in checks.values()), "checks": checks}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v["passed"] for k, v in checks.items()} | {"all_passed": result["all_passed"]}, indent=2))


if __name__ == "__main__":
    main()
