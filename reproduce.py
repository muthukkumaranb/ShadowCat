"""Reproduce ShadowCat's results in one command.

    python reproduce.py            # quick check (a few minutes): tests, pipeline, audit chain, headline numbers
    python reproduce.py --full     # also retrains models for the robustness, ATT&CK and probabilistic results (~30 min, CPU)

Every step recomputes a result and compares it with the committed artifact. Committed files are never
modified: anything a step rewrites is backed up first and restored afterwards. A summary table is printed,
and a report is written to runtime/REPRODUCE_REPORT.md (gitignored). The exit code is 0 only if every step passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "data-engineering/data/ucs/ucs_windows_models_v1.parquet"
DATA_SHA256 = "ce8fb65625f945a41fd066079f947229d4bed964ca9c358dad5b13b1abb8beee"
PY = sys.executable
RESULTS: list[dict] = []


# ----------------------------------------------------------------------------- helpers
def step(name):
    def deco(fn):
        def run(*a, **k):
            t0 = time.time()
            print(f"\n=== {name} ===", flush=True)
            try:
                ok, detail = fn(*a, **k)
            except Exception as e:  # a crashed step is a failed step, never a silent pass
                ok, detail = False, f"error: {type(e).__name__}: {e}"
            RESULTS.append({"step": name, "ok": ok, "detail": detail, "seconds": round(time.time() - t0)})
            print(f"--> {'PASS' if ok else 'FAIL'}: {detail}", flush=True)
            return ok
        return run
    return deco


def sh(cmd, env=None, timeout=3600):
    # UTF-8 both ways, so child output (e.g. em dashes) is not garbled on a Windows console code page
    p = subprocess.run(cmd, cwd=REPO, env={**os.environ, "PYTHONIOENCODING": "utf-8", **(env or {})},
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-4:])
    print(tail, flush=True)
    return p.returncode, p.stdout + p.stderr


class Preserve:
    """Back up files a script rewrites and restore them afterwards, so committed artifacts never change."""

    def __init__(self, *paths):
        self.paths = [REPO / p for p in paths]
        self.tmp = Path(tempfile.mkdtemp(prefix="shadowcat_preserve_"))

    def __enter__(self):
        for i, p in enumerate(self.paths):
            if p.is_dir():
                shutil.copytree(p, self.tmp / str(i))
            elif p.exists():
                shutil.copy2(p, self.tmp / str(i))
        return self

    def __exit__(self, *exc):
        for i, p in enumerate(self.paths):
            src = self.tmp / str(i)
            if not src.exists():  # did not exist before the step: remove whatever the step created
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.exists():
                    p.unlink()
                continue
            if src.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                shutil.copytree(src, p)
            elif src.exists():
                shutil.copy2(src, p)
        shutil.rmtree(self.tmp, ignore_errors=True)


def close(a, b, tol):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def committed_json(rel):
    return json.loads((REPO / rel).read_text())


# ----------------------------------------------------------------------------- quick steps
@step("1. Environment")
def check_env():
    print(f"Interpreter: {sys.executable}")
    if sys.version_info < (3, 11):
        return False, f"Python {sys.version.split()[0]}; 3.11+ required (scikit-learn 1.9)"
    mods = ("torch", "sklearn", "pandas", "numpy", "pyarrow", "yaml", "cryptography", "streamlit", "plotly", "scipy", "pytest")
    failed = {}
    for mod in mods:
        # Each package is first imported on its own in a fresh interpreter, so a broken install is told apart from
        # a clash between two packages (on Windows, typically two different C++ runtime DLLs).
        r = subprocess.run([PY, "-c", f"import {mod}"], capture_output=True, text=True)
        if r.returncode != 0:
            failed[mod] = (r.stderr.strip().splitlines() or ["(no message)"])[-1]
    if failed:
        for mod, err in failed.items():
            print(f"  {mod}: {err}")
        hint = ""
        if any("DLL" in e for e in failed.values()):
            hint = " A 'DLL load failed' error on Windows is usually fixed by installing the latest Microsoft Visual C++ Redistributable (x64)."
        return False, f"cannot import: {', '.join(failed)} (details above; run: pip install -r requirements.txt).{hint}"
    # All packages import on their own; now import them together, in the order the pipeline uses them.
    for mod in mods:
        try:
            __import__(mod)
        except Exception as e:  # noqa: BLE001  (DLL clashes surface as ImportError or OSError)
            print(f"  {mod} fails after importing {', '.join(mods[:mods.index(mod)])}: {type(e).__name__}: {e}")
            return False, (f"{mod} imports on its own but not after the other packages (a DLL clash). On Windows, install "
                           f"the latest Microsoft Visual C++ Redistributable (x64), then reopen the terminal.")
    return True, f"Python {sys.version.split()[0]}, all required packages present"


@step("2. Dataset integrity")
def check_data():
    if not DATA.exists():
        return False, f"{DATA.relative_to(REPO)} not found"
    h = hashlib.sha256(DATA.read_bytes()).hexdigest()
    return h == DATA_SHA256, f"sha256 {h[:16]}… ({'matches' if h == DATA_SHA256 else 'DOES NOT match'} the dataset the models were trained on)"


@step("3. Automated test suite")
def check_tests():
    code, out = sh([PY, "-m", "pytest", "backend/tests", "frontend/tests", "evaluation/benchmark", "tests/prob_forecast", "tests/pcap", "tests/test_merkle.py", "-q"])
    summary = next((l for l in reversed(out.splitlines()) if " passed" in l or " failed" in l), "no summary")
    return code == 0, summary.strip("= ")


@step("4. End-to-end inference (predict smoke test)")
def check_smoke():
    with tempfile.TemporaryDirectory() as rt:
        code, out = sh([PY, "backend/smoke_test.py"], env={"SHADOWCAT_RUNTIME_DIR": rt})
    return code == 0 and "SUCCESSFULLY" in out, "predict() ran on real windows" if code == 0 else "smoke test failed"


@step("5. Tamper-evident audit chain (build + verify)")
def check_chain():
    with tempfile.TemporaryDirectory() as rt:
        env = {"SHADOWCAT_RUNTIME_DIR": rt}
        c1, _ = sh([PY, "backend/build_audit_chain.py"], env=env)
        c2, out = sh([PY, "backend/verify_audit_chain.py"], env=env)
    ok = c1 == 0 and c2 == 0 and "AUDIT CHAIN VALID" in out
    return ok, "chain built and verified, Ed25519 signatures intact" if ok else "chain verification failed"


@step("6. Headline benchmark (37 LOEO folds, committed checkpoints)")
def check_benchmark():
    ref = committed_json("evaluation/benchmark/stacked_benchmark_results.json")
    files = ["evaluation/benchmark/stacked_benchmark_results.json",
             "ml1/artifacts/lstm/lstm_stacked/lstm_detection_loeo_test_predictions.csv",
             "ml1/artifacts/lstm/lstm_stacked/lstm_onset_loeo_test_predictions.csv"]
    with Preserve(*files):
        code, _ = sh([PY, "evaluation/benchmark/reproduce_stacked_benchmark.py", "--data", str(DATA.relative_to(REPO))])
        new = committed_json("evaluation/benchmark/stacked_benchmark_results.json") if code == 0 else None
    if new is None:
        return False, "benchmark script failed (it refuses to run unless all 37 folds reproduce their saved F1)"
    msgs, ok = [], True
    for task in ("detection", "onset"):
        a, b = new[task]["matched_fpr_5pct"]["stacked_ensemble"], ref[task]["matched_fpr_5pct"]["stacked_ensemble"]
        good = close(a["roc_auc"], b["roc_auc"], 1e-6) and close(a["f1"], b["f1"], 1e-6)
        ok &= good
        msgs.append(f"{task} ROC-AUC {a['roc_auc']:.3f} F1@5%FPR {a['f1']:.3f}")
    return ok, "all 37 folds reproduce; " + "; ".join(msgs)


@step("7. Robustness: thresholds chosen without test labels + final-dataset LR baseline")
def check_threshold():
    sys.path.insert(0, str(REPO / "evaluation/leakage_checks"))
    import pandas as pd
    import run_leakage_checks as L
    df = pd.read_parquet(DATA).sort_values("window_start_utc").reset_index(drop=True)
    feats = committed_json("ml1/artifacts/lr_set_a/set_a_features.json")["ordered_feature_names"]
    man = committed_json("ml1/artifacts/loeo/corrected_37fold_manifest.json")
    new = L.check_a(df, feats, man)
    ref = committed_json("evaluation/leakage_checks/leakage_results.json")["check_a_threshold_without_test_labels_37fold"]
    msgs, ok = [], True
    for task in ("detection", "onset"):
        for model in ("stacked_ensemble", "lr_baseline_refit_on_same_dataset"):
            good = close(new[task][model]["f1"], ref[task][model]["f1"], 1e-3) and close(new[task][model]["fpr"], ref[task][model]["fpr"], 1e-3)
            ok &= good
        msgs.append(f"{task} F1 {new[task]['stacked_ensemble']['f1']:.3f} (FPR {new[task]['stacked_ensemble']['fpr']:.3f}) vs LR {new[task]['lr_baseline_refit_on_same_dataset']['f1']:.3f}")
    return ok, "; ".join(msgs)


# ----------------------------------------------------------------------------- full steps (retraining)
@step("8. ATT&CK tactic classifier (retrained, 37 folds)")
def check_family():
    ref = committed_json("evaluation/family/family_results.json")["loeo_evaluation"]
    with Preserve("evaluation/family", "ml1/artifacts/family_classifier"):
        code, _ = sh([PY, "ml1/scripts/train_family_classifier.py"])
        new = committed_json("evaluation/family/family_results.json")["loeo_evaluation"] if code == 0 else None
    if new is None:
        return False, "training script failed"
    ok = close(new["macro_f1_tactic"], ref["macro_f1_tactic"], 1e-3) and close(new["macro_f1_family"], ref["macro_f1_family"], 1e-3)
    return ok, f"tactic macro-F1 {new['macro_f1_tactic']:.3f}, family macro-F1 {new['macro_f1_family']:.3f}"


@step("9. Leave-one-day-out stress test (models retrained, 6 folds)")
def check_lodo():
    sys.path.insert(0, str(REPO / "evaluation/leakage_checks"))
    import pandas as pd
    import run_leakage_checks as L
    df = pd.read_parquet(DATA).sort_values("window_start_utc").reset_index(drop=True)
    feats = committed_json("ml1/artifacts/lr_set_a/set_a_features.json")["ordered_feature_names"]
    new = L.check_b(df, feats, L.lodo_manifest(df))
    ref = committed_json("evaluation/leakage_checks/leakage_results.json")["check_b_leave_one_day_out"]
    ok = all(close(new[t]["pooled"]["stacked"]["roc_auc"], ref[t]["pooled"]["stacked"]["roc_auc"], 0.02) for t in ("detection", "onset"))
    return ok, "; ".join(f"{t} pooled ROC-AUC {new[t]['pooled']['stacked']['roc_auc']:.3f}" for t in ("detection", "onset"))


@step("10. Probabilistic K=1..5 forecast (world model retrained per fold)")
def check_prob():
    ref = committed_json("evaluation/prob_forecast/prob_results.json")
    files = [f"evaluation/prob_forecast/{f}" for f in ("prob_results.json", "binary_results.json", "PROB_FORECAST.md",
                                                       "per_window_scores.csv.gz", "per_window_binary.csv.gz", "pit_k1_k5.png")]
    with Preserve(*files, "scratch"):
        code, _ = sh([PY, "evaluation/prob_forecast/run_prob_forecast.py"], timeout=7200)
        new = committed_json("evaluation/prob_forecast/prob_results.json") if code == 0 else None
    if new is None:
        return False, "forecast script failed"
    f1 = lambda r: r["horizons"]["1"]["all"]["forecasters"]["F1_noised_persistence"]["crps"]
    ok = new["h_star_prob"] == ref["h_star_prob"] and close(f1(new), f1(ref), 1e-3)
    return ok, f"H*_prob = {new['h_star_prob']} (committed {ref['h_star_prob']})"


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="also retrain models for the robustness, ATT&CK and probabilistic results")
    args = ap.parse_args()
    os.chdir(REPO)
    t0 = time.time()

    if check_env():
        check_data()
        check_tests()
        check_smoke()
        check_chain()
        check_benchmark()
        check_threshold()
        if args.full:
            check_family()
            check_lodo()
            check_prob()

    width = max(len(r["step"]) for r in RESULTS)
    lines = ["", "SHADOWCAT REPRODUCTION SUMMARY", "=" * (width + 60)]
    for r in RESULTS:
        lines.append(f"{'PASS' if r['ok'] else 'FAIL'}  {r['step']:<{width}}  {r['detail']}  ({r['seconds']}s)")
    n_ok = sum(r["ok"] for r in RESULTS)
    lines += ["=" * (width + 60), f"{n_ok}/{len(RESULTS)} steps passed in {round(time.time() - t0)}s "
              f"({'full' if args.full else 'quick'} mode{'' if args.full else '; add --full to retrain models'})"]
    print("\n".join(lines))

    rt = REPO / "runtime"
    rt.mkdir(exist_ok=True)
    md = ["# Reproduction report", "", f"Mode: {'full' if args.full else 'quick'}", "", "| Result | Step | Detail | Time (s) |", "|---|---|---|---|"]
    md += [f"| {'PASS' if r['ok'] else 'FAIL'} | {r['step']} | {r['detail']} | {r['seconds']} |" for r in RESULTS]
    (rt / "REPRODUCE_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
