# AIT-LDS v2.0 Feature Coverage Report (G-t2)

## 1. Available Data Sources
Analysis of the `russellmitchell` scenario reveals that AIT-LDS v2.0 **does not provide whole-network packet captures (PCAPs)**. Only two PCAPs exist, both localized to the attacker machine:
1. `gather\attacker_0\logs\ait.aecid.attacker.wpdiscuz\traffic.pcap`
2. `gather\attacker_0\logs\dnsteal\traffic.pcap`

The remaining data consists entirely of host-based logs (e.g., syslog, apache, auditd, openvpn, dnsmasq).

## 2. Computability of the 406 UCS Features
The CIC-IDS2018 pipeline in this repository relies on **CICFlowMeter** (offline) to generate flow CSVs, which are then processed by `ucs_extractor.py` to yield 388 flow features, plus `pcap_extractor.py` for 12 packet features, and 6 presence masks.

Since AIT-LDS v2.0 lacks whole-network PCAPs and pre-computed flow CSVs, the vast majority of network-level features **cannot be computed** for the enterprise network.

### Feature Group Coverage
| Feature Group | Features | Mask Flag | Computable on AIT? | Resolution |
|---------------|----------|-----------|--------------------|------------|
| **Traffic Volume** | ~70 | `mask_has_traffic_volume_features` | **NO** | Mask = `0.0`, values `NaN` |
| **Flow Timing** | ~120 | `mask_has_flow_timing_features` | **NO** | Mask = `0.0`, values `NaN` |
| **TCP Flags** | ~100 | `mask_has_tcp_flags` | **NO** | Mask = `0.0`, values `NaN` |
| **Graph Topology** | N/A | `mask_has_graph_topology` | **NO** | Mask = `0.0`, values `NaN` |
| **Identity/Auth** | N/A | `mask_has_identity_auth` | **NO** | Mask = `0.0`, values `NaN` |
| **Packet-Level** | 12 | `mask_has_packet_level_features` | **PARTIAL** | Mask = `1.0` (Attacker only), else `0.0` |

### Packet-Level Features (12)
We can theoretically compute the 12 packet-level features for the attacker's localized traffic using the existing `pcap_extractor.py` (which parses PCAP streams directly). However, this only represents the attacker's egress/ingress, not the background network state. For these windows, `mask_has_packet_level_features = 1.0`. For all other network traffic, it is `0.0`.

### Summary
Out of 406 features:
- **388 flow features**: Masked (`0.0`), set to `NaN` (imputed to median by downstream components).
- **6 masks**: 5 set to `0.0`, 1 set to `1.0` occasionally.
- **12 packet features**: Computable only for the attacker machine's local traffic.
