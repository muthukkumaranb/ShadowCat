"""
Spot-check live demo inference path across all attack groups:
- Botnet (02-03-2018)
- SSH-Bruteforce (14-02-2018)
- DDOS-LOIC-UDP (21-02-2018)
"""
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "backend"))

import pandas as pd
from backend.predict import predict

def run_spot_checks():
    data_path = repo_root / "data-engineering" / "data" / "ucs" / "ucs_windows.parquet"
    windows = pd.read_parquet(data_path).sort_values("window_start_utc").reset_index(drop=True)
    
    samples = {
        "Botnet": windows[windows["source_day"] == "02-03-2018"].head(35),
        "SSH-Bruteforce": windows[windows["source_day"] == "14-02-2018"].head(35),
        "DDOS-LOIC-UDP": windows[windows["source_day"] == "21-02-2018"].head(35),
    }

    print("=" * 75)
    print("LIVE DEMO PATH SPOT-CHECK ACROSS ALL ATTACK GROUPS")
    print("=" * 75)

    for group_name, df_sample in samples.items():
        print(f"\n[*] Evaluating sample from: {group_name} ({len(df_sample)} windows)")
        res = predict(df_sample, source_type="flows")
        fc = res["forecast_trajectory"]
        
        print(f"    - Window ID          : {res['window_id']}")
        print(f"    - Risk Trajectory    : {fc['risk']}")
        print(f"    - Stage Trajectory   : {fc['stage']}")
        print(f"    - Calibrated Thresh  : {fc['calibrated_threshold']}")
        print(f"    - Hazard Alert       : {fc['hazard_alert']}")
        print(f"    - Notarized Via      : {res.get('notarized_via')}")
        print(f"    - Lead Times         : {fc['lead_time']}")

        assert fc["calibrated_threshold"] in (0.15, 0.22), f"Stale threshold detected: {fc['calibrated_threshold']}"
        assert len(fc["risk"]) == 4
        print(f"    [PASS] {group_name} live demo inference verified.")

    print("\n" + "=" * 75)
    print("ALL ATTACK GROUPS CONFIRMED RUNNING WITH HONEST CALIBRATED THRESHOLDS")
    print("=" * 75)

if __name__ == "__main__":
    run_spot_checks()
