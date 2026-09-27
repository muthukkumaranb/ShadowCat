# Genuine PCAP Packet Telemetry Verification Report (v5 Cascade - Option A)

## 1. Multi-Day Extraction Summary

| Attack Day | Attack Type | PCAP Files | Windows | Populated Windows | Matched Packets |
|---|---|---|:---:|:---:|:---:|
| **14-02-2018** | SSH-Bruteforce / FTP-Bruteforce | 1 (`UCAP172.31.69.25.pcap`) | 543 | 496 (91.3%) | 4,718,327 |
| **02-03-2018** | Botnet (Ares) | 8 UCAP files | 455 | 453 (99.6%) | 60,309 |
| **21-02-2018** | DDOS-LOIC-UDP (Afternoon Attack + Endpoints) | 7 UCAP files | 170 | 170 (100.0%) | 5,349,504 |
| **Total** | | 16 files | **1168** | **1119** | **10,128,140** |

## 2. Feature Non-Triviality Verification Across All Three Headline Days

| Feature Name | 14-02-2018 Mean | 02-03-2018 Mean | 21-02-2018 Active Mean | Non-Zero Check |
|---|:---:|:---:|:---:|:---:|
| `pkt_ttl_min` | 46.3794 | 30.3846 | 32.0471 | PASS |
| `pkt_ttl_max` | 171.7109 | 231.5165 | 231.7412 | PASS |
| `pkt_ttl_std` | 32.9295 | 60.5132 | 57.3837 | PASS |
| `pkt_ttl_mode` | 70.5930 | 63.8154 | 64.6176 | PASS |
| `pkt_frag_df_count` | 8681.2247 | 65.7692 | 31419.6294 | PASS |
| `pkt_payload_size_p50` | 39.8416 | 36.1308 | 25.8059 | PASS |
| `pkt_payload_size_p95` | 214.7577 | 719.4623 | 558.1682 | PASS |
| `pkt_tcp_retrans_count` | 1710.8085 | 17.8396 | 9426.9529 | PASS |
| `pkt_port_scan_seq_score` | 0.1791 | 0.2880 | 0.3045 | PASS |

All features extracted strictly from binary packet headers with zero synthetic constants.
