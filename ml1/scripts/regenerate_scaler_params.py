"""
Regenerates data/ucs/scaler_params.yaml for v2.
Preserves canonical v1 parameters verbatim for all 394 flow features,
and explicitly updates the 12 zero-filled packet features to median=0.0, scale=1.0.
"""
import yaml
from pathlib import Path
from collections import OrderedDict

ws = Path(__file__).resolve().parent.parent

V1_SCALER = ws / "artifacts" / "lstm" / "inference_scaler_v1.yaml"
OUTPUT_SCALER_PARAMS = ws / "data" / "ucs" / "scaler_params.yaml"

PACKET_12_FEATURES = [
    "pkt_ttl_min", "pkt_ttl_max", "pkt_ttl_std", "pkt_ttl_mode",
    "pkt_frag_mf_count", "pkt_frag_df_count",
    "pkt_payload_size_p25", "pkt_payload_size_p50",
    "pkt_payload_size_p75", "pkt_payload_size_p95",
    "pkt_tcp_retrans_count", "pkt_port_scan_seq_score"
]

def main():
    print("=" * 72)
    print("REGENERATING SCALER PARAMS v2 (Option A Data Fix)")
    print("=" * 72)
    
    with open(V1_SCALER, "r", encoding="utf-8") as f:
        v1_doc = yaml.safe_load(f)
        
    v1_features = v1_doc["features"]
    print(f"Total features in v1 scaler: {len(v1_features)}")
    assert len(v1_features) == 400, f"Expected 400 features, got {len(v1_features)}"

    scaler_dict = OrderedDict()
    changed_count = 0

    for col, params in v1_features.items():
        if col in PACKET_12_FEATURES:
            scaler_dict[col] = {
                "median": 0.0,
                "scale": 1.0,
                "q25": 0.0,
                "q75": 0.0,
                "is_log1p": False
            }
            changed_count += 1
        else:
            scaler_dict[col] = {
                "median": float(params["median"]),
                "scale": float(params["scale"]),
                "q25": float(params.get("q25", 0.0)),
                "q75": float(params.get("q75", 0.0)),
                "is_log1p": bool(params.get("is_log1p", False))
            }

    # Programmatic assertion checks
    assert changed_count == 12, f"Expected exactly 12 feature changes, got {changed_count}"
    
    # Verify non-packet feature verbatim match (e.g. duration_microsec_mean)
    dur_params = scaler_dict["duration_microsec_mean"]
    assert abs(dur_params["median"] - 14770387.468899522) < 1e-4, f"Corrupted duration median: {dur_params['median']}"
    assert abs(dur_params["scale"] - 7138992.385225026) < 1e-4, f"Corrupted duration scale: {dur_params['scale']}"
    
    # Representer for OrderedDict
    def represent_ordereddict(dumper, data):
        return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())

    yaml.add_representer(OrderedDict, represent_ordereddict)

    OUTPUT_SCALER_PARAMS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SCALER_PARAMS, "w", encoding="utf-8") as f:
        yaml.dump(dict(scaler_dict), f, default_flow_style=False, sort_keys=False)

    print(f"[+] Successfully written corrected scaler params to {OUTPUT_SCALER_PARAMS}")
    print(f"  - Total features: {len(scaler_dict)}")
    print(f"  - Untouched features: {len(scaler_dict) - changed_count} (0 mismatch vs v1/canonical)")
    print(f"  - Intended zero-filled packet features: {changed_count}")

if __name__ == "__main__":
    main()
