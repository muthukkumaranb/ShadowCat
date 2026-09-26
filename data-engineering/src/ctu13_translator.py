"""
CTU-13 to Pipeline-Schema Translator
SIH 2026 - Unified Cyber State (UCS) Ingestion Pipeline

Provides an honest, auditable schema-compatibility translation layer converting
Argus bi-flow records (CTU-13 format) into the 80-column CICFlowMeter raw schema
contract expected by UCSExtractor.

CRITICAL METHODOLOGICAL CONSTRAINTS:
1. Argus captures flow-level summaries only (15 columns).
2. Packet-level distribution statistics (variance, TCP flags, subflow block rates)
   cannot be derived from CTU-13 and are explicitly filled with sentinel values (NaN),
   subsequently handled by UCSExtractor's frozen baseline medians.
3. This is a schema-compatibility tool, not a claim that CSE-CIC-IDS2018 model weights
   transfer full predictive accuracy to Argus biflow telemetry.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
import yaml
import logging


# Authoritative protocol mapping from IANA strings/numbers to integers
PROTO_MAP = {
    "tcp": 6,
    "udp": 17,
    "icmp": 1,
    "ipv6-icmp": 58,
    "igmp": 2,
    "gre": 47,
    "esp": 50,
    "ah": 51,
    "rsvp": 46,
}


@dataclass
class ColumnClassification:
    column_name: str
    category: str  # "DIRECT", "DERIVED", "UNAVAILABLE"
    source_field: str
    derivation_rule: str
    notes: str


@dataclass
class FidelityReport:
    total_expected_columns: int
    direct_count: int
    derived_count: int
    unavailable_count: int
    fidelity_score_pct: float
    direct_columns: List[str]
    derived_columns: List[str]
    unavailable_columns: List[str]
    notes: str


class CTU13Translator:
    """
    Translates CTU-13 Argus bi-flow CSVs into CICFlowMeter-style schema.
    """

    DIRECT_MAPPINGS: Dict[str, Tuple[str, str, str]] = {
        "Dst Port": ("Dport", "pd.to_numeric(Dport)", "Destination transport port"),
        "Protocol": ("Proto", "Proto mapped via PROTO_MAP", "IANA protocol number (tcp=6, udp=17, icmp=1)"),
        "Timestamp": ("StartTime", "pd.to_datetime(StartTime).strftime('%d/%m/%Y %H:%M:%S')", "Normalized datetime string"),
        "Flow Duration": ("Dur", "Dur * 1e6", "Flow duration converted from seconds to microseconds"),
        "TotLen Fwd Pkts": ("SrcBytes", "pd.to_numeric(SrcBytes)", "Direct measurement of forward bytes from source"),
        "Label": ("Label", "Label string preserved", "Flow classification label"),
    }

    DERIVED_MAPPINGS: Dict[str, Tuple[str, str, str]] = {
        "TotLen Bwd Pkts": ("TotBytes, SrcBytes", "max(0, TotBytes - SrcBytes)", "Backward byte count derived from total minus source bytes"),
        "Tot Fwd Pkts": ("TotPkts, Dir, SrcBytes", "Round(TotPkts * (SrcBytes / TotBytes))", "Forward packet count estimated from byte proportion and Dir"),
        "Tot Bwd Pkts": ("TotPkts, Tot Fwd Pkts", "max(0, TotPkts - Tot Fwd Pkts)", "Remaining packets attributed to backward direction"),
        "Flow Byts/s": ("TotBytes, Dur", "TotBytes / max(Dur, 1e-6)", "Average flow byte transfer rate"),
        "Flow Pkts/s": ("TotPkts, Dur", "TotPkts / max(Dur, 1e-6)", "Average flow packet transfer rate"),
        "Fwd Pkts/s": ("Tot Fwd Pkts, Dur", "Tot Fwd Pkts / max(Dur, 1e-6)", "Forward direction packet rate"),
        "Bwd Pkts/s": ("Tot Bwd Pkts, Dur", "Tot Bwd Pkts / max(Dur, 1e-6)", "Backward direction packet rate"),
        "Pkt Size Avg": ("TotBytes, TotPkts", "TotBytes / max(TotPkts, 1)", "Coarse global mean packet size across flow"),
        "Fwd Pkt Len Mean": ("TotLen Fwd Pkts, Tot Fwd Pkts", "TotLen Fwd Pkts / Tot Fwd Pkts", "Average forward packet size"),
        "Bwd Pkt Len Mean": ("TotLen Bwd Pkts, Tot Bwd Pkts", "TotLen Bwd Pkts / max(Tot Bwd Pkts, 1)", "Average backward packet size"),
        "Fwd Seg Size Avg": ("Fwd Pkt Len Mean", "Equal to Fwd Pkt Len Mean", "TCP segment size average approximation"),
        "Bwd Seg Size Avg": ("Bwd Pkt Len Mean", "Equal to Bwd Pkt Len Mean", "TCP segment size average approximation"),
        "Down/Up Ratio": ("Tot Bwd Pkts, Tot Fwd Pkts", "Tot Bwd Pkts / max(Tot Fwd Pkts, 1)", "Ratio of backward to forward packets"),
        "Subflow Fwd Pkts": ("Tot Fwd Pkts", "Tot Fwd Pkts", "Single-subflow assumption for Argus aggregated biflow"),
        "Subflow Fwd Byts": ("TotLen Fwd Pkts", "TotLen Fwd Pkts", "Single-subflow forward bytes"),
        "Subflow Bwd Pkts": ("Tot Bwd Pkts", "Tot Bwd Pkts", "Single-subflow backward packets"),
        "Subflow Bwd Byts": ("TotLen Bwd Pkts", "TotLen Bwd Pkts", "Single-subflow backward bytes"),
        "Fwd Act Data Pkts": ("Tot Fwd Pkts", "max(0, Tot Fwd Pkts - 1)", "Data packets assuming 1 handshake/control packet"),
        "Flow IAT Mean": ("Dur, TotPkts", "(Dur * 1e6) / max(TotPkts - 1, 1)", "Uniform inter-arrival time approximation"),
        "Fwd IAT Tot": ("Dur, Tot Fwd Pkts", "Dur * 1e6 if Tot Fwd Pkts > 1 else 0", "Total forward inter-arrival duration"),
        "Fwd IAT Mean": ("Dur, Tot Fwd Pkts", "(Dur * 1e6) / max(Tot Fwd Pkts - 1, 1)", "Mean forward inter-arrival time"),
        "Bwd IAT Tot": ("Dur, Tot Bwd Pkts", "Dur * 1e6 if Tot Bwd Pkts > 1 else 0", "Total backward inter-arrival duration"),
        "Bwd IAT Mean": ("Dur, Tot Bwd Pkts", "(Dur * 1e6) / max(Tot Bwd Pkts - 1, 1)", "Mean backward inter-arrival time"),
    }

    UNAVAILABLE_COLUMNS: Dict[str, Tuple[str, str]] = {
        # TCP Flags (12)
        "FIN Flag Cnt": ("TCP Flags", "Argus biflow summary does not capture per-packet TCP flags"),
        "SYN Flag Cnt": ("TCP Flags", "Argus biflow summary does not capture per-packet TCP flags"),
        "RST Flag Cnt": ("TCP Flags", "Argus biflow summary does not capture per-packet TCP flags"),
        "PSH Flag Cnt": ("TCP Flags", "Argus biflow summary does not capture per-packet TCP flags"),
        "ACK Flag Cnt": ("TCP Flags", "Argus biflow summary does not capture per-packet TCP flags"),
        "URG Flag Cnt": ("TCP Flags", "Argus biflow summary does not capture per-packet TCP flags"),
        "CWE Flag Count": ("TCP Flags", "Argus biflow summary does not capture per-packet TCP flags"),
        "ECE Flag Cnt": ("TCP Flags", "Argus biflow summary does not capture per-packet TCP flags"),
        "Fwd PSH Flags": ("TCP Flags", "Argus biflow summary does not capture per-packet TCP flags"),
        "Bwd PSH Flags": ("TCP Flags", "Argus biflow summary does not capture per-packet TCP flags"),
        "Fwd URG Flags": ("TCP Flags", "Argus biflow summary does not capture per-packet TCP flags"),
        "Bwd URG Flags": ("TCP Flags", "Argus biflow summary does not capture per-packet TCP flags"),
        # Packet Length Statistics & Variance (11)
        "Fwd Pkt Len Max": ("Packet Distributions", "Max/min packet lengths not recorded by Argus biflow"),
        "Fwd Pkt Len Min": ("Packet Distributions", "Max/min packet lengths not recorded by Argus biflow"),
        "Fwd Pkt Len Std": ("Packet Distributions", "Standard deviation of packet size requires packet-level traces"),
        "Bwd Pkt Len Max": ("Packet Distributions", "Max/min packet lengths not recorded by Argus biflow"),
        "Bwd Pkt Len Min": ("Packet Distributions", "Max/min packet lengths not recorded by Argus biflow"),
        "Bwd Pkt Len Std": ("Packet Distributions", "Standard deviation of packet size requires packet-level traces"),
        "Pkt Len Min": ("Packet Distributions", "Global min packet size not recorded in Argus summary"),
        "Pkt Len Max": ("Packet Distributions", "Global max packet size not recorded in Argus summary"),
        "Pkt Len Mean": ("Packet Distributions", "Global packet mean without direction split"),
        "Pkt Len Std": ("Packet Distributions", "Packet size standard deviation requires per-packet length stream"),
        "Pkt Len Var": ("Packet Distributions", "Packet size variance requires per-packet length stream"),
        "Fwd Seg Size Min": ("TCP Segment Details", "Minimum segment size not available in Argus"),
        # Timing Extrema & Variance (9)
        "Flow IAT Std": ("IAT Distributions", "Inter-arrival standard deviation requires individual packet timestamps"),
        "Flow IAT Max": ("IAT Distributions", "Max inter-arrival time requires individual packet timestamps"),
        "Flow IAT Min": ("IAT Distributions", "Min inter-arrival time requires individual packet timestamps"),
        "Fwd IAT Std": ("IAT Distributions", "Forward IAT variance requires individual packet timestamps"),
        "Fwd IAT Max": ("IAT Distributions", "Forward IAT max requires individual packet timestamps"),
        "Fwd IAT Min": ("IAT Distributions", "Forward IAT min requires individual packet timestamps"),
        "Bwd IAT Std": ("IAT Distributions", "Backward IAT variance requires individual packet timestamps"),
        "Bwd IAT Max": ("IAT Distributions", "Backward IAT max requires individual packet timestamps"),
        "Bwd IAT Min": ("IAT Distributions", "Backward IAT min requires individual packet timestamps"),
        # Headers & TCP Window Initialization (4)
        "Fwd Header Len": ("Header Stats", "Total forward IP/TCP header bytes not tracked in Argus biflow"),
        "Bwd Header Len": ("Header Stats", "Total backward IP/TCP header bytes not tracked in Argus biflow"),
        "Init Fwd Win Byts": ("TCP Window State", "Initial TCP window advertisement not parsed by Argus"),
        "Init Bwd Win Byts": ("TCP Window State", "Initial TCP window advertisement not parsed by Argus"),
        # Bulk Rates (6)
        "Fwd Byts/b Avg": ("Bulk Transfer Stats", "Bulk transfer rate requires packet train analysis"),
        "Fwd Pkts/b Avg": ("Bulk Transfer Stats", "Bulk transfer rate requires packet train analysis"),
        "Fwd Blk Rate Avg": ("Bulk Transfer Stats", "Bulk transfer rate requires packet train analysis"),
        "Bwd Byts/b Avg": ("Bulk Transfer Stats", "Bulk transfer rate requires packet train analysis"),
        "Bwd Pkts/b Avg": ("Bulk Transfer Stats", "Bulk transfer rate requires packet train analysis"),
        "Bwd Blk Rate Avg": ("Bulk Transfer Stats", "Bulk transfer rate requires packet train analysis"),
        # Active / Idle Subflow Timers (8)
        "Active Mean": ("Activity Timers", "Flow idle/active periodic timers not tracked by Argus"),
        "Active Std": ("Activity Timers", "Flow idle/active periodic timers not tracked by Argus"),
        "Active Max": ("Activity Timers", "Flow idle/active periodic timers not tracked by Argus"),
        "Active Min": ("Activity Timers", "Flow idle/active periodic timers not tracked by Argus"),
        "Idle Mean": ("Activity Timers", "Flow idle/active periodic timers not tracked by Argus"),
        "Idle Std": ("Activity Timers", "Flow idle/active periodic timers not tracked by Argus"),
        "Idle Max": ("Activity Timers", "Flow idle/active periodic timers not tracked by Argus"),
        "Idle Min": ("Activity Timers", "Flow idle/active periodic timers not tracked by Argus"),
    }

    def __init__(self):
        # Verify 80 columns total
        self.total_expected = len(self.DIRECT_MAPPINGS) + len(self.DERIVED_MAPPINGS) + len(self.UNAVAILABLE_COLUMNS)
        assert self.total_expected == 80, f"Expected exactly 80 columns, got {self.total_expected}"

    def get_column_classifications(self) -> List[ColumnClassification]:
        """Returns the complete, authoritative 80-column classification audit list."""
        records = []
        for col, (src, rule, notes) in sorted(self.DIRECT_MAPPINGS.items()):
            records.append(ColumnClassification(col, "DIRECT", src, rule, notes))
        for col, (src, rule, notes) in sorted(self.DERIVED_MAPPINGS.items()):
            records.append(ColumnClassification(col, "DERIVED", src, rule, notes))
        for col, (category, notes) in sorted(self.UNAVAILABLE_COLUMNS.items()):
            records.append(ColumnClassification(col, "UNAVAILABLE", "N/A", "Filled with np.nan sentinel", notes))
        return records

    def get_fidelity_report(self) -> FidelityReport:
        """Returns statistical fidelity breakdown across the 80 expected schema columns."""
        direct_cols = sorted(list(self.DIRECT_MAPPINGS.keys()))
        derived_cols = sorted(list(self.DERIVED_MAPPINGS.keys()))
        unavail_cols = sorted(list(self.UNAVAILABLE_COLUMNS.keys()))
        score = round((len(direct_cols) + len(derived_cols)) / self.total_expected * 100.0, 1)

        return FidelityReport(
            total_expected_columns=self.total_expected,
            direct_count=len(direct_cols),
            derived_count=len(derived_cols),
            unavailable_count=len(unavail_cols),
            fidelity_score_pct=score,
            direct_columns=direct_cols,
            derived_columns=derived_cols,
            unavailable_columns=unavail_cols,
            notes=(
                f"Fidelity breakdown: {len(direct_cols)} direct, {len(derived_cols)} derived, "
                f"{len(unavail_cols)} unavailable (sentinel-filled). "
                "This guarantees strict schema-compatibility for ingestion, but is NOT a claim "
                "of empirical feature equivalence or zero transfer-loss."
            ),
        )

    def translate(self, df_ctu13: pd.DataFrame) -> Tuple[pd.DataFrame, FidelityReport]:
        """
        Translates CTU-13 Argus bi-flow DataFrame into CICFlowMeter-compliant schema.
        Outputs both translated DataFrame and fidelity audit report.
        """
        if df_ctu13 is None or len(df_ctu13) == 0:
            raise ValueError("Input CTU-13 DataFrame is empty or None.")

        df = df_ctu13.copy()
        out = pd.DataFrame(index=df.index)

        # Detect format: Argus bi-flow vs partial pre-processed
        has_argus = any(c in df.columns for c in ("StartTime", "Dur", "Proto", "SrcAddr", "Sport"))

        if has_argus:
            # 1. Direct Mappings
            dport = df["Dport"] if "Dport" in df.columns else df.get("dport", 80)
            out["Dst Port"] = pd.to_numeric(dport, errors="coerce").fillna(80).astype(int)

            proto_series = df["Proto"] if "Proto" in df.columns else df.get("proto", "tcp")
            out["Protocol"] = proto_series.astype(str).str.lower().map(PROTO_MAP).fillna(6).astype(int)

            start_time = df["StartTime"] if "StartTime" in df.columns else df.get("starttime", "2011/08/10 09:46:53")
            ts = pd.to_datetime(start_time, errors="coerce")
            if ts.isna().all():
                ts = pd.date_range("2011-08-10 09:46:53", periods=len(df), freq="1s", tz="UTC")
            out["Timestamp"] = ts.dt.strftime("%d/%m/%Y %H:%M:%S")

            dur = pd.to_numeric(df["Dur"] if "Dur" in df.columns else df.get("dur", 1.0), errors="coerce").fillna(1.0)
            dur_us = (dur * 1e6).clip(lower=1.0)
            out["Flow Duration"] = dur_us.astype(int)

            src_bytes = pd.to_numeric(df["SrcBytes"] if "SrcBytes" in df.columns else df.get("srcbytes", 0), errors="coerce").fillna(0)
            out["TotLen Fwd Pkts"] = src_bytes.astype(int)

            label = df["Label"] if "Label" in df.columns else df.get("label", "Benign")
            out["Label"] = label.astype(str)

            # Preserve IP endpoints & Sport for graph traversal & endpoint attribution
            out["Src IP"] = df["SrcAddr"].astype(str) if "SrcAddr" in df.columns else df.get("srcaddr", "147.32.84.164")
            out["Dst IP"] = df["DstAddr"].astype(str) if "DstAddr" in df.columns else df.get("dstaddr", "147.32.80.9")
            sport = df["Sport"] if "Sport" in df.columns else df.get("sport", 1024)
            out["Src Port"] = pd.to_numeric(sport, errors="coerce").fillna(1024).astype(int)

            # 2. Derived Approximations
            tot_bytes = pd.to_numeric(df["TotBytes"] if "TotBytes" in df.columns else df.get("totbytes", src_bytes), errors="coerce").fillna(src_bytes)
            tot_pkts = pd.to_numeric(df["TotPkts"] if "TotPkts" in df.columns else df.get("totpkts", 1), errors="coerce").fillna(1)
            dur_s = dur.clip(lower=1e-6)

            out["TotLen Bwd Pkts"] = (tot_bytes - src_bytes).clip(lower=0).astype(int)

            # Directional packet splitting based on byte weight
            byte_ratio = (src_bytes / np.maximum(tot_bytes, 1)).clip(0.0, 1.0)
            fwd_pkts = np.maximum(1, np.round(tot_pkts * byte_ratio)).astype(int)
            out["Tot Fwd Pkts"] = fwd_pkts
            out["Tot Bwd Pkts"] = np.maximum(0, tot_pkts - fwd_pkts).astype(int)

            out["Flow Byts/s"] = tot_bytes / dur_s
            out["Flow Pkts/s"] = tot_pkts / dur_s
            out["Fwd Pkts/s"] = out["Tot Fwd Pkts"] / dur_s
            out["Bwd Pkts/s"] = out["Tot Bwd Pkts"] / dur_s
            out["Pkt Size Avg"] = tot_bytes / tot_pkts
            out["Fwd Pkt Len Mean"] = out["TotLen Fwd Pkts"] / out["Tot Fwd Pkts"]
            out["Bwd Pkt Len Mean"] = np.where(out["Tot Bwd Pkts"] > 0, out["TotLen Bwd Pkts"] / np.maximum(1, out["Tot Bwd Pkts"]), 0.0)
            out["Fwd Seg Size Avg"] = out["Fwd Pkt Len Mean"]
            out["Bwd Seg Size Avg"] = out["Bwd Pkt Len Mean"]
            out["Down/Up Ratio"] = out["Tot Bwd Pkts"] / out["Tot Fwd Pkts"]
            out["Subflow Fwd Pkts"] = out["Tot Fwd Pkts"]
            out["Subflow Fwd Byts"] = out["TotLen Fwd Pkts"]
            out["Subflow Bwd Pkts"] = out["Tot Bwd Pkts"]
            out["Subflow Bwd Byts"] = out["TotLen Bwd Pkts"]
            out["Fwd Act Data Pkts"] = np.maximum(0, out["Tot Fwd Pkts"] - 1)
            out["Flow IAT Mean"] = dur_us / np.maximum(tot_pkts - 1, 1)
            out["Fwd IAT Tot"] = np.where(out["Tot Fwd Pkts"] > 1, dur_us, 0.0)
            out["Fwd IAT Mean"] = np.where(out["Tot Fwd Pkts"] > 1, dur_us / np.maximum(out["Tot Fwd Pkts"] - 1, 1), 0.0)
            out["Bwd IAT Tot"] = np.where(out["Tot Bwd Pkts"] > 1, dur_us, 0.0)
            out["Bwd IAT Mean"] = np.where(out["Tot Bwd Pkts"] > 1, dur_us / np.maximum(out["Tot Bwd Pkts"] - 1, 1), 0.0)

        else:
            # Partial / already pre-extracted format (e.g. scratch/CTU13_Attack_Traffic.csv)
            for c in df.columns:
                clean_c = c.strip()
                if clean_c in self.DIRECT_MAPPINGS or clean_c in self.DERIVED_MAPPINGS:
                    out[clean_c] = df[c]

            # Supply critical missing columns if absent
            if "Dst Port" not in out.columns:
                out["Dst Port"] = 80
            if "Protocol" not in out.columns:
                out["Protocol"] = 6
            if "Timestamp" not in out.columns:
                out["Timestamp"] = pd.date_range("2011-08-10 09:46:53", periods=len(df), freq="1s").strftime("%d/%m/%Y %H:%M:%S")

        # 3. Unavailable columns -> filled with np.nan sentinel
        for col in self.UNAVAILABLE_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan

        fidelity_report = self.get_fidelity_report()
        logging.info(
            f"[CTU13Translator] Translated {len(out)} flows: "
            f"{fidelity_report.direct_count} direct, {fidelity_report.derived_count} derived, "
            f"{fidelity_report.unavailable_count} unavailable (sentinels applied)."
        )

        return out, fidelity_report


def translate_ctu13(df_ctu13: pd.DataFrame) -> Tuple[pd.DataFrame, FidelityReport]:
    """Convenience functional wrapper."""
    translator = CTU13Translator()
    return translator.translate(df_ctu13)


if __name__ == "__main__":
    t = CTU13Translator()
    classifs = t.get_column_classifications()
    print("=" * 80)
    print("CTU-13 SCHEMA TRANSLATOR: 80-COLUMN AUTHORITATIVE CLASSIFICATION TABLE")
    print("=" * 80)
    direct = [c for c in classifs if c.category == "DIRECT"]
    derived = [c for c in classifs if c.category == "DERIVED"]
    unavail = [c for c in classifs if c.category == "UNAVAILABLE"]

    print(f"\n[1] DIRECT MAPPINGS ({len(direct)} columns):")
    for c in direct:
        print(f"  - {c.column_name:<20} <= {c.source_field:<15} ({c.derivation_rule})")

    print(f"\n[2] REASONABLE DERIVED APPROXIMATIONS ({len(derived)} columns):")
    for c in derived:
        print(f"  - {c.column_name:<20} <= {c.source_field:<25} ({c.derivation_rule})")

    print(f"\n[3] UNAVAILABLE IN CTU-13 ({len(unavail)} columns filled with NaN sentinel):")
    for c in unavail:
        print(f"  - {c.column_name:<20} : {c.notes}")

    rep = t.get_fidelity_report()
    print("\n" + "=" * 80)
    print(f"FIDELITY BREAKDOWN: {rep.direct_count} Direct, {rep.derived_count} Derived, {rep.unavailable_count} Unavailable")
    print(f"Overall Mapping Fidelity: {rep.fidelity_score_pct}%")
    print("=" * 80)
