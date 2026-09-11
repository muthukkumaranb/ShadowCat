# Demo Compatibility Notes
**SIH26153 — Cyber World Model Architecture**

## Dual Input Contract for Packet-Level Features

To ensure the Phase 2 demo can successfully process both raw live traffic and cached historical windows, the `pcap_extractor.py` module exposes a dual-path contract:

### Path A: Live PCAP Extraction (Demo Mode)
For the live demonstration, the downstream inference runner can pass a raw `.pcap` file directly to the pipeline.
- **Function**: `extract_or_generate_packet_features(pcap_input_path="path/to/live.pcap")`
- **Behavior**: Utilizes `scapy.PcapReader` to stream packets in memory, dissecting IP/TCP layers to extract TTL statistics, IP fragmentation flags, payload size quantiles, and TCP port scan sequencing signatures. The output is a `DataFrame` of packet-level features grouped by 1-minute windows, ready for concatenation with the flow-level tensor.

### Path B: Cached / Deterministic Fallback (Batch Mode)
For batch training or environments without raw PCAP availability (e.g., standard CSE-CIC-IDS2018 CSV evaluation where PCAPs are prohibitively large).
- **Function**: `extract_or_generate_packet_features(pcap_input_path=None)`
- **Behavior**: Bypasses the Scapy processing loop. Instead, it generates a deterministic set of packet-level features that align with the flow statistics of the dataset, effectively falling back to a cached/synthetic representation for the designated `target_day` (defaults to Wednesday-14-02-2018).

### Usage Example
```python
from pcap_extractor import extract_or_generate_packet_features

# Option A: Live Demo Mode
packet_df = extract_or_generate_packet_features(
    pcap_input_path="data/raw/demo_capture.pcap"
)

# Option B: Batch Training / Fallback Mode
packet_df = extract_or_generate_packet_features(
    pcap_input_path=None,
    target_day="14-02-2018"
)
```

This design guarantees that the downstream model architecture will have the required packet-level tensor inputs (e.g., `pkt_ttl_std`, `pkt_port_scan_seq_score`) during the live evaluation phase without being blocked by the absence of multi-terabyte PCAPs during model training.
