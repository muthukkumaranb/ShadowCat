# AIT-LDS v2.0 Transition Stats & Label Distribution (G-t3)

## 1. State Transitions (1-minute windows)
| Transition | Count |
|------------|-------|
| attack->attack | 645 |
| attack->benign | 1 |
| attackA->attackB | 9 |
| benign->attack | 1 |
| benign->benign | 113 |

## 2. Label Distribution
| Label | Window Count |
|-------|--------------|
| vpn_disconnect | 552 |
| benign | 115 |
| crack_wphash | 37 |
| host_discover_dmz | 35 |
| host_discover_local | 20 |
| dnsteal_stop | 6 |
| service_scan | 1 |
| wpscan | 1 |
| check_netstat_l | 1 |
| open_reverse_shell | 1 |
| check_sudo | 1 |

## 3. Comparison to CIC-IDS2018
AIT-LDS v2.0 provides vastly different labels (e.g., `recon_networks_finish`, `check_uname_r`, `crack_wphash`) compared to CIC-IDS2018's high-level categories (e.g., `Brute Force -Web`, `DDoS attacks-LOIC-HTTP`).
Additionally, AIT attack stages are much finer-grained and occur sequentially over a longer period, resulting in many `attackA->attackB` transitions that do not exist in CIC-IDS2018 (which mostly has `benign->attack->benign` transitions).