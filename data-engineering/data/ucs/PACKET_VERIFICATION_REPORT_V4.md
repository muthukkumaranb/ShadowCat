# Genuine PCAP Packet Telemetry Verification Report (v4 Cascade)

## 1. Multi-Day Extraction Summary

| Attack Day | Attack Type | PCAP Files | Windows | Populated Windows | Matched Packets |
|---|---|---|:---:|:---:|:---:|
| **14-02-2018** | SSH-Bruteforce / FTP-Bruteforce | 1 (`UCAP172.31.69.25.pcap`) | 543 | 417 (76.8%) | 4,716,542 |
| **02-03-2018** | Botnet (Ares) | 8 UCAP files | 455 | 380 (83.5%) | 51,838 |
| **Total** | | 9 files | **998** | **797** | **4,768,380** |

## 2. Feature Non-Triviality Verification

| Feature Name | 14-02-2018 Mean | 02-03-2018 Mean | Non-Zero Check |
|---|:---:|:---:|:---:|
| `pkt_ttl_min` | 39.3573 | 25.2571 | PASS |
| `pkt_ttl_max` | 144.0368 | 194.7956 | PASS |
| `pkt_ttl_std` | 25.0434 | 50.9267 | PASS |
| `pkt_ttl_mode` | 59.1436 | 53.5473 | PASS |
| `pkt_frag_df_count` | 8679.1915 | 57.6462 | PASS |
| `pkt_payload_size_p50` | 34.9107 | 31.0670 | PASS |
| `pkt_payload_size_p95` | 195.4413 | 627.3454 | PASS |
| `pkt_tcp_retrans_count` | 1710.1989 | 15.6176 | PASS |
| `pkt_port_scan_seq_score` | 0.1592 | 0.2407 | PASS |

All features extracted strictly from binary packet headers with zero synthetic constants.
