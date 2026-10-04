# AIT Log Data Set V2.0 — Data Engineering

## Dataset

**AIT Log Data Set V2.0** is a collection of synthetic log data from eight simulated
small-enterprise testbeds built at the Austrian Institute of Technology (AIT). Each
scenario spans 4–6 days and includes mail server, file share, WordPress, VPN, firewall,
DNS, and 9–27 user hosts. Normal user behaviour is simulated as background noise; a
multi-step attack chain is launched once per scenario.

- **Zenodo Record**: <https://zenodo.org/records/5789064>
- **DOI**: 10.5281/zenodo.5789064
- **License**: CC BY-NC-SA 4.0

## Citation

> M. Landauer, F. Skopik, M. Frank, W. Hotwagner, M. Wurzenberger, and A. Rauber.
> "Maintainable Log Datasets for Evaluation of Intrusion Detection Systems".
> IEEE Transactions on Dependable and Secure Computing, vol. 20, no. 4,
> pp. 3466–3482, 2023. DOI: 10.1109/TDSC.2022.3201582. arXiv: 2203.08580.

## Scenarios

| Scenario        | Simulation period               | Attack window              | Scan volume | Unpacked size | Zip size  |
|-----------------|---------------------------------|----------------------------|-------------|---------------|-----------|
| fox             | 2022-01-15 – 2022-01-20 (5 d)  | 2022-01-18 11:59 – 13:15  | High        | 26 GB         | 14.75 GB  |
| harrison        | 2022-02-04 – 2022-02-09 (5 d)  | 2022-02-08 07:07 – 08:38  | High        | 27 GB         | 15.63 GB  |
| russellmitchell | 2022-01-21 – 2022-01-25 (4 d)  | 2022-01-24 03:01 – 04:39  | Low         | 14 GB         |  6.64 GB  |
| santos          | 2022-01-14 – 2022-01-18 (4 d)  | 2022-01-17 11:15 – 11:59  | Low         | 17 GB         |  9.32 GB  |
| shaw            | 2022-01-25 – 2022-01-31 (6 d)  | 2022-01-29 14:37 – 15:21  | Low         | 27 GB         | 16.38 GB  |
| wardbeck        | 2022-01-19 – 2022-01-24 (5 d)  | 2022-01-23 12:10 – 12:56  | Low         | 26 GB         | 15.99 GB  |
| wheeler         | 2022-01-26 – 2022-01-31 (5 d)  | 2022-01-30 07:35 – 17:53  | High        | 30 GB         | 18.25 GB  |
| wilson          | 2022-02-03 – 2022-02-09 (6 d)  | 2022-02-07 10:57 – 11:49  | High        | 39 GB         | 24.73 GB  |

**Notes**:
- shaw: Data exfiltration is not visible in DNS logs.
- wheeler: No password cracking in attack chain.
- Attack parameters and execution orders vary across scenarios.

## Attack Steps

1. nmap scan
2. WPScan
3. dirb
4. Webshell upload (CVE-2020-24186)
5. Webshell commands
6. WordPress DB dump
7. Password cracking (John the Ripper)
8. Privilege escalation (system-user login)
9. Reverse shell / root commands
10. DNS exfiltration (DNSteal)

## Disk Space Requirements

| Item                              | Approximate size |
|-----------------------------------|------------------|
| All 8 zips (compressed)           | ~122 GB          |
| All 8 scenarios unpacked          | ~206 GB          |
| Smallest scenario (russellmitchell) zip | 6.64 GB    |
| Smallest scenario unpacked        | 14 GB            |
| **Recommended free space**        | **≥ 50 GB** (process one at a time, deleting raw after extraction) |

## How to Download

Use `data-engineering/ait/download_ait.py`:

```bash
# Set the raw data directory (default: data-engineering/data/ait/raw)
export AIT_RAW_DIR=data-engineering/data/ait/raw

# Download a single scenario (with resume and md5 verification)
python data-engineering/ait/download_ait.py --scenario russellmitchell

# Download all scenarios
for s in fox harrison russellmitchell santos shaw wardbeck wheeler wilson; do
    python data-engineering/ait/download_ait.py --scenario $s
done
```

Each download is logged to `data-engineering/ait/download_log.json`.

## Data Structure (per scenario zip)

```
<scenario>/
├── gather/                    # All collected logs
│   ├── attacker_0/logs/
│   │   └── attacks.log        # Attack step timeline
│   ├── <host_name>/
│   │   ├── logs/              # Host logs (Apache, audit, DNS, syslog, Suricata, pcap, ...)
│   │   ├── configs/           # Host configuration
│   │   └── facts.json         # Host facts
│   └── ...
├── labels/                    # Ground truth (mirrors gather/ structure)
│   └── <host_name>/logs/      # Per-line JSON label files
├── processing/                # Label generation source code
│   └── config/servers.yml     # Host → IP mapping
├── rules/                     # Labeling rules
├── environment/               # Testbed deployment code
└── dataset.yml                # Simulation start/end time
```

## Label Format

Each label file contains one JSON object per labelled log line:
```json
{"line": 1860, "labels": ["attacker_change_user", "escalate"], "rules": {"attacker_change_user": ["attacker.escalate.audit.su.login"], "escalate": ["attacker.escalate.audit.su.login"]}}
```
Lines without a corresponding label entry are normal/benign.
