# Genuine PCAP Packet Telemetry Verification Report (v5 Cascade - Option A)

## 1. Multi-Day Extraction Summary

| Attack Day | Attack Type | PCAP Files | Windows | Populated Windows | Matched Packets |
|---|---|---|:---:|:---:|:---:|
| **14-02-2018** | SSH-Bruteforce / FTP-Bruteforce | 1 (`UCAP172.31.69.25.pcap`) | 271 | 263 (97.0%) | 3,960,091 |
| **02-03-2018** | Botnet (Ares) | 8 UCAP files | 525 | 523 (99.6%) | 73,225 |
| **21-02-2018** | DDOS-LOIC-UDP (Afternoon Attack + Endpoints) | 7 UCAP files | 161 | 161 (100.0%) | 3,321,677 |
| **22-02-2018** | Web Attacks (XSS, SQLi, Brute Force) | 1 (`UCAP172.31.69.28_22022018.pcap`) | 549 | 546 (99.5%) | 47,714 |
| **23-02-2018** | Web Attacks | 1 (`UCAP172.31.69.28_23022018.pcap`) | 556 | 553 (99.5%) | 55,101 |
| **Total** | | 18 files | **2062** | **2046** | **7,457,808** |

## 2. Feature Non-Triviality Verification Across All Three Headline Days

| Feature Name | 14-02-2018 Mean | 02-03-2018 Mean | 21-02-2018 Active Mean | 22-02-2018 Mean | 23-02-2018 Mean | Non-Zero Check |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `pkt_ttl_min` | 48.5055 | 30.1714 | 32.2795 | 48.5902 | 48.4658 | PASS |
| `pkt_ttl_max` | 179.8561 | 231.4705 | 231.7143 | 175.9107 | 163.9550 | PASS |
| `pkt_ttl_std` | 20.2240 | 58.9545 | 58.8295 | 34.8519 | 30.9432 | PASS |
| `pkt_ttl_mode` | 70.9557 | 63.8400 | 64.6522 | 75.5264 | 78.2140 | PASS |
| `pkt_frag_df_count` | 14604.4465 | 73.0419 | 20583.7640 | 71.7413 | 84.7104 | PASS |
| `pkt_payload_size_p50` | 48.4244 | 35.7019 | 26.1304 | 428.3470 | 414.7482 | PASS |
| `pkt_payload_size_p95` | 333.7919 | 719.2228 | 535.9851 | 721.0075 | 799.5030 | PASS |
| `pkt_tcp_retrans_count` | 2844.1292 | 20.7352 | 6176.3478 | 16.7213 | 21.9119 | PASS |
| `pkt_port_scan_seq_score` | 0.2379 | 0.2882 | 0.2968 | 0.1335 | 0.1157 | PASS |

All features extracted strictly from binary packet headers with zero synthetic constants.
