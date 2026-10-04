# AIT-LDS v2.0 Feasibility Report: `russellmitchell`

Generated: 2026-10-04T11:35:17.800844+00:00

## (a) Directory Tree (depth 3)

```
├── environment/ (6181 items)
│   ├── datasets/ (98 items)
│   │   ├── scenario/ (36 items)
│   │   ├── scenario1/ (33 items)
│   │   └── scenario2/ (26 items)
│   ├── model/ (15 items)
│   │   ├── templates/ (9 items)
│   │   ├── config.yml (260.0 B)
│   │   ├── context.yml (10.2 KB)
│   │   ├── context.yml.j2 (15.0 KB)
│   │   ├── templates.yml (15.9 KB)
│   │   └── templates.yml.j2 (7.5 KB)
│   └── provisioning/ (6065 items)
│       ├── ansible/ (6004 items)
│       ├── packer/ (23 items)
│       └── terragrunt/ (35 items)
├── gather/ (8564 items)
│   ├── attacker_0/ (47 items)
│   │   ├── configs/ (3 items)
│   │   ├── logs/ (41 items)
│   │   └── facts.json (558.9 KB)
│   ├── cloud_share/ (1024 items)
│   │   ├── configs/ (984 items)
│   │   ├── logs/ (37 items)
│   │   └── facts.json (988.6 KB)
│   ├── davey_mail/ (560 items)
│   │   ├── configs/ (531 items)
│   │   ├── logs/ (26 items)
│   │   └── facts.json (505.8 KB)
│   ├── ext_user_0/ (7 items)
│   │   ├── configs/ (2 items)
│   │   ├── logs/ (2 items)
│   │   └── facts.json (1.0 MB)
│   ├── ext_user_1/ (7 items)
│   │   ├── configs/ (2 items)
│   │   ├── logs/ (2 items)
│   │   └── facts.json (1.0 MB)
│   ├── ext_user_2/ (7 items)
│   │   ├── configs/ (2 items)
│   │   ├── logs/ (2 items)
│   │   └── facts.json (1.0 MB)
│   ├── inet-dns/ (419 items)
│   │   ├── configs/ (404 items)
│   │   ├── logs/ (12 items)
│   │   └── facts.json (476.6 KB)
│   ├── inet-firewall/ (536 items)
│   │   ├── configs/ (504 items)
│   │   ├── logs/ (29 items)
│   │   └── facts.json (946.8 KB)
│   ├── internal_employee_0/ (13 items)
│   │   ├── configs/ (2 items)
│   │   ├── logs/ (8 items)
│   │   └── facts.json (1.1 MB)
│   ├── internal_employee_1/ (7 items)
│   │   ├── configs/ (2 items)
│   │   ├── logs/ (2 items)
│   │   └── facts.json (1.1 MB)
│   ├── internal_employee_2/ (14 items)
│   │   ├── configs/ (2 items)
│   │   ├── logs/ (9 items)
│   │   └── facts.json (1.1 MB)
│   ├── internal_employee_3/ (37 items)
│   │   ├── configs/ (2 items)
│   │   ├── logs/ (32 items)
│   │   └── facts.json (1.1 MB)
│   ├── internal_share/ (508 items)
│   │   ├── configs/ (481 items)
│   │   ├── logs/ (24 items)
│   │   └── facts.json (1004.6 KB)
│   ├── intranet_server/ (815 items)
│   │   ├── configs/ (772 items)
│   │   ├── logs/ (40 items)
│   │   └── facts.json (934.0 KB)
│   ├── mail/ (676 items)
│   │   ├── configs/ (637 items)
│   │   ├── logs/ (36 items)
│   │   └── facts.json (920.9 KB)
│   ├── monitoring/ (1968 items)
│   │   ├── configs/ (1789 items)
│   │   ├── logs/ (176 items)
│   │   └── facts.json (954.3 KB)
│   ├── morris_mail/ (552 items)
│   │   ├── configs/ (523 items)
│   │   ├── logs/ (26 items)
│   │   └── facts.json (505.2 KB)
│   ├── remote_employee_0/ (7 items)
│   │   ├── configs/ (2 items)
│   │   ├── logs/ (2 items)
│   │   └── facts.json (1.1 MB)
│   ├── remote_employee_1/ (7 items)
│   │   ├── configs/ (2 items)
│   │   ├── logs/ (2 items)
│   │   └── facts.json (1.1 MB)
│   ├── remote_employee_2/ (9 items)
│   │   ├── configs/ (2 items)
│   │   ├── logs/ (4 items)
│   │   └── facts.json (1.1 MB)
│   ├── vpn/ (512 items)
│   │   ├── configs/ (485 items)
│   │   ├── logs/ (24 items)
│   │   └── facts.json (882.3 KB)
│   └── webserver/ (810 items)
│       ├── configs/ (756 items)
│       ├── logs/ (51 items)
│       └── facts.json (898.9 KB)
├── labels/ (23 items)
│   ├── inet-firewall/ (2 items)
│   │   └── logs/ (1 items)
│   ├── internal_share/ (3 items)
│   │   └── logs/ (2 items)
│   ├── intranet_server/ (7 items)
│   │   └── logs/ (6 items)
│   ├── monitoring/ (4 items)
│   │   └── logs/ (3 items)
│   └── vpn/ (2 items)
│       └── logs/ (1 items)
├── processing/ (131 items)
│   ├── config/ (10 items)
│   │   ├── attacker/ (3 items)
│   │   ├── attack_queries.yaml (10.5 KB)
│   │   ├── attacker.yaml (498.0 B)
│   │   ├── groups.yaml (1.7 KB)
│   │   ├── logs.yaml (2.7 KB)
│   │   ├── logs.yaml.j2 (2.7 KB)
│   │   └── servers.yaml (14.5 KB)
│   ├── logstash/ (103 items)
│   │   ├── conf.d/ (11 items)
│   │   ├── data/ (73 items)
│   │   ├── log/ (4 items)
│   │   ├── auditd-index-template.json (235.4 KB)
│   │   ├── auditd-ingest.yml (52.6 KB)
│   │   ├── filebeat-index-template.json (817.3 KB)
│   │   ├── jvm.options (2.0 KB)
│   │   ├── log4j2.properties (7.4 KB)
│   │   ├── logstash.yml (11.0 KB)
│   │   ├── metricsbeat-index-template.json (748.7 KB)
│   │   ├── openvpn-index-template.json (5.7 KB)
│   │   ├── pcap-index-template.json (17.0 MB)
│   │   ├── pipelines.yml (363.0 B)
│   │   ├── processed_russellmitchell_scenario-index-template.json (78.5 KB)
│   │   └── processed_russellmitchell_scenario-legacy-index-template.json (72.7 KB)
│   ├── templates/ (14 items)
│   │   ├── attacker/ (3 items)
│   │   ├── rules/ (6 items)
│   │   ├── attacker.yaml.j2 (2.3 KB)
│   │   ├── groups.json.j2 (314.0 B)
│   │   └── servers.json.j2 (2.5 KB)
│   └── process.yaml (9.5 KB)
├── rules/ (6 items)
│   ├── 0_auth.yaml (8.0 KB)
│   ├── apache.yaml (25.5 KB)
│   ├── audit.yaml (9.7 KB)
│   ├── dnsmasq.yaml (37.2 KB)
│   ├── monitoring.yaml (518.0 B)
│   └── openvpn.yaml (4.1 KB)
└── dataset.yaml (97.0 B)
```

## (b) Network Captures

| File | Host | Start (UTC) | End (UTC) | Packets | Total Bytes | File Size |
|------|------|-------------|-----------|---------|-------------|-----------|
| `gather\attacker_0\logs\ait.aecid.attacker.wpdiscuz\traffic.pcap` | attacker_0 | 2022-01-20T13:47:50.743391+00:00 | 2022-01-24T04:38:17.102834+00:00 | 682,840 | 298.7 MB | 309.1 MB |
| `gather\attacker_0\logs\dnsteal\traffic.pcap` | attacker_0 | 2022-01-20T12:48:19.071873+00:00 | 2022-01-24T13:50:03.248663+00:00 | 1,192,295 | 403.4 MB | 421.6 MB |

## (c) Label Schema

**Total label files**: 8

**All unique label values**: attacker, attacker_change_user, attacker_http, attacker_vpn, crack_passwords, dirb, dns_scan, dnsteal, dnsteal-dropped, dnsteal-received, escalate, escalated_command, escalated_sudo_command, escalated_sudo_session, exfiltration-service, foothold, network_scan, service_scan, traceroute, webshell_cmd, webshell_upload, wpscan


### Label files with per-line labels

| File | Label count | Unique labels | Fields |
|------|-------------|---------------|--------|
| `labels\inet-firewall\logs\dnsmasq.log` | 54035 | attacker, dirb, dns_scan, dnsteal, dnsteal-dropped, dnsteal-received, escalate, foothold, network_scan, service_scan, traceroute, webshell_cmd, wpscan | labels, line, rules |
| `labels\internal_share\logs\audit\audit.log` | 2 | attacker, dnsteal, exfiltration-service | labels, line, rules |
| `labels\intranet_server\logs\apache2\intranet.smith.russellmitchell.com-access.log.2` | 7695 | attacker_http, dirb, escalate, foothold, service_scan, webshell_cmd, webshell_upload, wpscan | labels, line, rules |
| `labels\intranet_server\logs\apache2\intranet.smith.russellmitchell.com-error.log.2` | 36 | attacker_http, dirb, foothold, wpscan | labels, line, rules |
| `labels\intranet_server\logs\audit\audit.log` | 9 | attacker_change_user, escalate, escalated_command, escalated_sudo_command | labels, line, rules |
| `labels\intranet_server\logs\auth.log` | 8 | attacker_change_user, escalate, escalated_command, escalated_sudo_command, escalated_sudo_session | labels, line, rules |
| `labels\monitoring\logs\logstash\intranet-server\2022-01-24-system.cpu.log` | 49 | crack_passwords, escalate | labels, line, rules |
| `labels\vpn\logs\openvpn.log` | 28 | attacker_vpn, foothold | labels, line, rules |

### Attacker Timeline (attacks.log)

**Path**: `gather\attacker_0\logs\attacks.log`

**Entries**: 55


```
{"timestamp": "2022-01-24 03:01:00.063877+00:00", "message": "vpn_connect", "raw": "2022-01-24 03:01:00.063877+00:00 vpn_connect"}
{"timestamp": "2022-01-24 03:01:17.723042+00:00", "message": "traceroute_internet", "raw": "2022-01-24 03:01:17.723042+00:00 traceroute_internet"}
{"timestamp": "2022-01-24 03:01:20.969099+00:00", "message": "dns_brute_force", "raw": "2022-01-24 03:01:20.969099+00:00 dns_brute_force"}
{"timestamp": "2022-01-24 03:01:28.456313+00:00", "message": "host_discover_dmz", "raw": "2022-01-24 03:01:28.456313+00:00 host_discover_dmz"}
{"timestamp": "2022-01-24 03:36:38.963403+00:00", "message": "host_discover_local", "raw": "2022-01-24 03:36:38.963403+00:00 host_discover_local"}
{"timestamp": "2022-01-24 03:56:46.825579+00:00", "message": "service_scan", "raw": "2022-01-24 03:56:46.825579+00:00 service_scan"}
{"timestamp": "2022-01-24 03:57:26.435577+00:00", "message": "recon_networks_finish", "raw": "2022-01-24 03:57:26.435577+00:00 recon_networks_finish"}
{"timestamp": "2022-01-24 03:57:26.436714+00:00", "message": "dirb_scan", "raw": "2022-01-24 03:57:26.436714+00:00 dirb_scan"}
{"timestamp": "2022-01-24 03:57:48.934650+00:00", "message": "wpscan", "raw": "2022-01-24 03:57:48.934650+00:00 wpscan"}
{"timestamp": "2022-01-24 03:58:20.020490+00:00", "message": "upload_rce_shell", "raw": "2022-01-24 03:58:20.020490+00:00 upload_rce_shell"}
{"timestamp": "2022-01-24 03:58:23.348462+00:00", "message": "check_whoami", "raw": "2022-01-24 03:58:23.348462+00:00 check_whoami"}
{"timestamp": "2022-01-24 03:58:25.108981+00:00", "message": "check_uname_r", "raw": "2022-01-24 03:58:25.108981+00:00 check_uname_r"}
{"timestamp": "2022-01-24 03:58:27.041422+00:00", "message": "read_profile", "raw": "2022-01-24 03:58:27.041422+00:00 read_profile"}
{"timestamp": "2022-01-24 03:58:28.528493+00:00", "message": "check_who", "raw": "2022-01-24 03:58:28.528493+00:00 check_who"}
{"timestamp": "2022-01-24 03:58:30.292820+00:00", "message": "check_meminfo", "raw": "2022-01-24 03:58:30.292820+00:00 check_meminfo"}
{"timestamp": "2022-01-24 03:58:33.828771+00:00", "message": "check_uname_a", "raw": "2022-01-24 03:58:33.828771+00:00 check_uname_a"}
{"timestamp": "2022-01-24 03:58:36.383526+00:00", "message": "check_user_id", "raw": "2022-01-24 03:58:36.383526+00:00 check_user_id"}
{"timestamp": "2022-01-24 03:58:39.247353+00:00", "message": "check_df", "raw": "2022-01-24 03:58:39.247353+00:00 check_df"}
{"timestamp": "2022-01-24 03:58:40.944504+00:00", "message": "check_netstat_nat", "raw": "2022-01-24 03:58:40.944504+00:00 check_netstat_nat"}
{"timestamp": "2022-01-24 03:58:42.198977+00:00", "message": "check_id", "raw": "2022-01-24 03:58:42.198977+00:00 check_id"}
{"timestamp": "2022-01-24 03:58:43.400876+00:00", "message": "read_resolv", "raw": "2022-01-24 03:58:43.400876+00:00 read_resolv"}
{"timestamp": "2022-01-24 03:58:44.632721+00:00", "message": "check_netstat_t", "raw": "2022-01-24 03:58:44.632721+00:00 check_netstat_t"}
{"timestamp": "2022-01-24 03:58:48.259736+00:00", "message": "list_home", "raw": "2022-01-24 03:58:48.259736+00:00 list_home"}
{"timestamp": "2022-01-24 03:58:49.562813+00:00", "message": "check_last", "raw": "2022-01-24 03:58:49.562813+00:00 check_last"}
{"timestamp": "2022-01-24 03:58:51.585727+00:00", "message": "list_web_dir", "raw": "2022-01-24 03:58:51.585727+00:00 list_web_dir"}
{"timestamp": "2022-01-24 03:58:54.219952+00:00", "message": "check_date", "raw": "2022-01-24 03:58:54.219952+00:00 check_date"}
{"timestamp": "2022-01-24 03:58:55.463292+00:00", "message": "list_www", "raw": "2022-01-24 03:58:55.463292+00:00 list_www"}
{"timestamp": "2022-01-24 03:58:57.641124+00:00", "message": "check_netstat_l", "raw": "2022-01-24 03:58:57.641124+00:00 check_netstat_l"}
{"timestamp": "2022-01-24 03:59:00.791636+00:00", "message": "clear", "raw": "2022-01-24 03:59:00.791636+00:00 clear"}
{"timestamp": "2022-01-24 03:59:02.283150+00:00", "message": "check_wp_config", "raw": "2022-01-24 03:59:02.283150+00:00 check_wp_config"}
{"timestamp": "2022-01-24 03:59:03.740619+00:00", "message": "check_ps_a", "raw": "2022-01-24 03:59:03.740619+00:00 check_ps_a"}
{"timestamp": "2022-01-24 03:59:05.928341+00:00", "message": "read_passwd", "raw": "2022-01-24 03:59:05.928341+00:00 read_passwd"}
{"timestamp": "2022-01-24 03:59:09.511060+00:00", "message": "check_release", "raw": "2022-01-24 03:59:09.511060+00:00 check_release"}
{"timestamp": "2022-01-24 03:59:12.977191+00:00", "message": "check_cpuinfo", "raw": "2022-01-24 03:59:12.977191+00:00 check_cpuinfo"}
{"timestamp": "2022-01-24 03:59:14.207842+00:00", "message": "dump_wp_users", "raw": "2022-01-24 03:59:14.207842+00:00 dump_wp_users"}
{"timestamp": "2022-01-24 03:59:16.923431+00:00", "message": "read_group", "raw": "2022-01-24 03:59:16.923431+00:00 read_group"}
{"timestamp": "2022-01-24 03:59:18.310719+00:00", "message": "check_uptime", "raw": "2022-01-24 03:59:18.310719+00:00 check_uptime"}
{"timestamp": "2022-01-24 03:59:19.790753+00:00", "message": "check_network_config", "raw": "2022-01-24 03:59:19.790753+00:00 check_network_config"}
{"timestamp": "2022-01-24 03:59:22.176229+00:00", "message": "recon_host_finish", "raw": "2022-01-24 03:59:22.176229+00:00 recon_host_finish"}
{"timestamp": "2022-01-24 03:59:22.177425+00:00", "message": "decide_crack_method", "raw": "2022-01-24 03:59:22.177425+00:00 decide_crack_method"}
{"timestamp": "2022-01-24 03:59:22.182453+00:00", "message": "crack_wphash", "raw": "2022-01-24 03:59:22.182453+00:00 crack_wphash"}
{"timestamp": "2022-01-24 04:36:57.885350+00:00", "message": "reverse_shell_listen", "raw": "2022-01-24 04:36:57.885350+00:00 reverse_shell_listen"}
{"timestamp": "2022-01-24 04:36:59.577594+00:00", "message": "open_reverse_shell", "raw": "2022-01-24 04:36:59.577594+00:00 open_reverse_shell"}
{"timestamp": "2022-01-24 04:37:25.836115+00:00", "message": "wait_reverse_shell", "raw": "2022-01-24 04:37:25.836115+00:00 wait_reverse_shell"}
{"timestamp": "2022-01-24 04:37:39.739109+00:00", "message": "open_pty", "raw": "2022-01-24 04:37:39.739109+00:00 open_pty"}
{"timestamp": "2022-01-24 04:37:40.530254+00:00", "message": "login_user", "raw": "2022-01-24 04:37:40.530254+00:00 login_user"}
{"timestamp": "2022-01-24 04:37:53.496765+00:00", "message": "read_fstab", "raw": "2022-01-24 04:37:53.496765+00:00 read_fstab"}
{"timestamp": "2022-01-24 04:37:55.374578+00:00", "message": "check_ps_aux", "raw": "2022-01-24 04:37:55.374578+00:00 check_ps_aux"}
{"timestamp": "2022-01-24 04:37:58.370156+00:00", "message": "check_sudo", "raw": "2022-01-24 04:37:58.370156+00:00 check_sudo"}
{"timestamp": "2022-01-24 04:38:01.850376+00:00", "message": "check_uname_ar", "raw": "2022-01-24 04:38:01.850376+00:00 check_uname_ar"}
... (5 more entries)
```

## (d) Attack Step Timestamps

### From attacks.log

| Step | Earliest | Latest | Entry Count |
|------|----------|--------|-------------|
| vpn_connect | 2022-01-24 03:01:00.063877+00:00 | 2022-01-24 03:01:00.063877+00:00 | 1 |
| traceroute_internet | 2022-01-24 03:01:17.723042+00:00 | 2022-01-24 03:01:17.723042+00:00 | 1 |
| dns_brute_force | 2022-01-24 03:01:20.969099+00:00 | 2022-01-24 03:01:20.969099+00:00 | 1 |
| host_discover_dmz | 2022-01-24 03:01:28.456313+00:00 | 2022-01-24 03:01:28.456313+00:00 | 1 |
| host_discover_local | 2022-01-24 03:36:38.963403+00:00 | 2022-01-24 03:36:38.963403+00:00 | 1 |
| service_scan | 2022-01-24 03:56:46.825579+00:00 | 2022-01-24 03:56:46.825579+00:00 | 1 |
| recon_networks_finish | 2022-01-24 03:57:26.435577+00:00 | 2022-01-24 03:57:26.435577+00:00 | 1 |
| dirb_scan | 2022-01-24 03:57:26.436714+00:00 | 2022-01-24 03:57:26.436714+00:00 | 1 |
| wpscan | 2022-01-24 03:57:48.934650+00:00 | 2022-01-24 03:57:48.934650+00:00 | 1 |
| upload_rce_shell | 2022-01-24 03:58:20.020490+00:00 | 2022-01-24 03:58:20.020490+00:00 | 1 |
| check_whoami | 2022-01-24 03:58:23.348462+00:00 | 2022-01-24 03:58:23.348462+00:00 | 1 |
| check_uname_r | 2022-01-24 03:58:25.108981+00:00 | 2022-01-24 03:58:25.108981+00:00 | 1 |
| read_profile | 2022-01-24 03:58:27.041422+00:00 | 2022-01-24 03:58:27.041422+00:00 | 1 |
| check_who | 2022-01-24 03:58:28.528493+00:00 | 2022-01-24 03:58:28.528493+00:00 | 1 |
| check_meminfo | 2022-01-24 03:58:30.292820+00:00 | 2022-01-24 03:58:30.292820+00:00 | 1 |
| check_uname_a | 2022-01-24 03:58:33.828771+00:00 | 2022-01-24 03:58:33.828771+00:00 | 1 |
| check_user_id | 2022-01-24 03:58:36.383526+00:00 | 2022-01-24 03:58:36.383526+00:00 | 1 |
| check_df | 2022-01-24 03:58:39.247353+00:00 | 2022-01-24 03:58:39.247353+00:00 | 1 |
| check_netstat_nat | 2022-01-24 03:58:40.944504+00:00 | 2022-01-24 03:58:40.944504+00:00 | 1 |
| check_id | 2022-01-24 03:58:42.198977+00:00 | 2022-01-24 03:58:42.198977+00:00 | 1 |
| read_resolv | 2022-01-24 03:58:43.400876+00:00 | 2022-01-24 03:58:43.400876+00:00 | 1 |
| check_netstat_t | 2022-01-24 03:58:44.632721+00:00 | 2022-01-24 03:58:44.632721+00:00 | 1 |
| list_home | 2022-01-24 03:58:48.259736+00:00 | 2022-01-24 03:58:48.259736+00:00 | 1 |
| check_last | 2022-01-24 03:58:49.562813+00:00 | 2022-01-24 03:58:49.562813+00:00 | 1 |
| list_web_dir | 2022-01-24 03:58:51.585727+00:00 | 2022-01-24 03:58:51.585727+00:00 | 1 |
| check_date | 2022-01-24 03:58:54.219952+00:00 | 2022-01-24 03:58:54.219952+00:00 | 1 |
| list_www | 2022-01-24 03:58:55.463292+00:00 | 2022-01-24 03:58:55.463292+00:00 | 1 |
| check_netstat_l | 2022-01-24 03:58:57.641124+00:00 | 2022-01-24 03:58:57.641124+00:00 | 1 |
| clear | 2022-01-24 03:59:00.791636+00:00 | 2022-01-24 03:59:00.791636+00:00 | 1 |
| check_wp_config | 2022-01-24 03:59:02.283150+00:00 | 2022-01-24 03:59:02.283150+00:00 | 1 |
| check_ps_a | 2022-01-24 03:59:03.740619+00:00 | 2022-01-24 03:59:03.740619+00:00 | 1 |
| read_passwd | 2022-01-24 03:59:05.928341+00:00 | 2022-01-24 03:59:05.928341+00:00 | 1 |
| check_release | 2022-01-24 03:59:09.511060+00:00 | 2022-01-24 03:59:09.511060+00:00 | 1 |
| check_cpuinfo | 2022-01-24 03:59:12.977191+00:00 | 2022-01-24 03:59:12.977191+00:00 | 1 |
| dump_wp_users | 2022-01-24 03:59:14.207842+00:00 | 2022-01-24 03:59:14.207842+00:00 | 1 |
| read_group | 2022-01-24 03:59:16.923431+00:00 | 2022-01-24 03:59:16.923431+00:00 | 1 |
| check_uptime | 2022-01-24 03:59:18.310719+00:00 | 2022-01-24 03:59:18.310719+00:00 | 1 |
| check_network_config | 2022-01-24 03:59:19.790753+00:00 | 2022-01-24 03:59:19.790753+00:00 | 1 |
| recon_host_finish | 2022-01-24 03:59:22.176229+00:00 | 2022-01-24 03:59:22.176229+00:00 | 1 |
| decide_crack_method | 2022-01-24 03:59:22.177425+00:00 | 2022-01-24 03:59:22.177425+00:00 | 1 |
| crack_wphash | 2022-01-24 03:59:22.182453+00:00 | 2022-01-24 03:59:22.182453+00:00 | 1 |
| reverse_shell_listen | 2022-01-24 04:36:57.885350+00:00 | 2022-01-24 04:36:57.885350+00:00 | 1 |
| open_reverse_shell | 2022-01-24 04:36:59.577594+00:00 | 2022-01-24 04:36:59.577594+00:00 | 1 |
| wait_reverse_shell | 2022-01-24 04:37:25.836115+00:00 | 2022-01-24 04:37:25.836115+00:00 | 1 |
| open_pty | 2022-01-24 04:37:39.739109+00:00 | 2022-01-24 04:37:39.739109+00:00 | 1 |
| login_user | 2022-01-24 04:37:40.530254+00:00 | 2022-01-24 04:37:40.530254+00:00 | 1 |
| read_fstab | 2022-01-24 04:37:53.496765+00:00 | 2022-01-24 04:37:53.496765+00:00 | 1 |
| check_ps_aux | 2022-01-24 04:37:55.374578+00:00 | 2022-01-24 04:37:55.374578+00:00 | 1 |
| check_sudo | 2022-01-24 04:37:58.370156+00:00 | 2022-01-24 04:37:58.370156+00:00 | 1 |
| check_uname_ar | 2022-01-24 04:38:01.850376+00:00 | 2022-01-24 04:38:01.850376+00:00 | 1 |
| list_shadow | 2022-01-24 04:38:04.866547+00:00 | 2022-01-24 04:38:04.866547+00:00 | 1 |
| read_shadow | 2022-01-24 04:38:06.227784+00:00 | 2022-01-24 04:38:06.227784+00:00 | 1 |
| check_ifconfig | 2022-01-24 04:38:09.938128+00:00 | 2022-01-24 04:38:09.938128+00:00 | 1 |
| vpn_disconnect | 2022-01-24 04:38:11.767134+00:00 | 2022-01-24 04:38:11.767134+00:00 | 1 |
| dnsteal_stop | 2022-01-24 13:50:03.250776+00:00 | 2022-01-24 13:50:03.250776+00:00 | 1 |

**Derivation method**: Timestamps extracted directly from `gather/attacker_0/logs/attacks.log`.

### From host-log label correlation


**labels\inet-firewall\logs\dnsmasq.log**:

| Label | Earliest | Latest | Count |
|-------|----------|--------|-------|
| attacker | Jan 21 00:00:09 | Jan 24 13:52:15 | 53054 |
| dirb | Jan 24 03:57:26 | Jan 24 03:57:26 | 8 |
| dns_scan | Jan 24 03:01:21 | Jan 24 03:01:21 | 414 |
| dnsteal | Jan 21 00:00:09 | Jan 24 13:52:15 | 53054 |
| dnsteal-dropped | Jan 24 13:50:24 | Jan 24 13:52:15 | 48 |
| dnsteal-received | Jan 21 00:00:09 | Jan 24 13:50:03 | 53006 |
| escalate | Jan 24 03:59:22 | Jan 24 03:59:22 | 12 |
| foothold | Jan 24 03:01:17 | Jan 24 03:57:49 | 969 |
| network_scan | Jan 24 03:02:33 | Jan 24 03:37:04 | 92 |
| service_scan | Jan 24 03:56:47 | Jan 24 03:57:17 | 443 |
| traceroute | Jan 24 03:01:17 | Jan 24 03:01:17 | 4 |
| webshell_cmd | Jan 24 03:59:22 | Jan 24 03:59:22 | 12 |
| wpscan | Jan 24 03:57:49 | Jan 24 03:57:49 | 8 |

**labels\internal_share\logs\audit\audit.log**:

| Label | Earliest | Latest | Count |
|-------|----------|--------|-------|
| attacker | 1643032239.298 | 1643032239.298 | 2 |
| dnsteal | 1643032239.298 | 1643032239.298 | 2 |
| exfiltration-service | 1643032239.298 | 1643032239.298 | 2 |

**labels\intranet_server\logs\apache2\intranet.smith.russellmitchell.com-access.log.2**:

| Label | Earliest | Latest | Count |
|-------|----------|--------|-------|
| attacker_http | 24/Jan/2022:03:57:01 +0000 | 24/Jan/2022:03:59:19 +0000 | 7687 |
| dirb | 24/Jan/2022:03:57:26 +0000 | 24/Jan/2022:03:57:39 +0000 | 4462 |
| escalate | 24/Jan/2022:03:59:22 +0000 | 24/Jan/2022:03:59:48 +0000 | 4 |
| foothold | 24/Jan/2022:03:56:53 +0000 | 24/Jan/2022:03:59:19 +0000 | 7691 |
| service_scan | 24/Jan/2022:03:56:53 +0000 | 24/Jan/2022:03:57:01 +0000 | 12 |
| webshell_cmd | 24/Jan/2022:03:58:23 +0000 | 24/Jan/2022:03:59:48 +0000 | 32 |
| webshell_upload | 24/Jan/2022:03:58:20 +0000 | 24/Jan/2022:03:58:20 +0000 | 3 |
| wpscan | 24/Jan/2022:03:57:52 +0000 | 24/Jan/2022:03:58:12 +0000 | 3186 |

**labels\intranet_server\logs\apache2\intranet.smith.russellmitchell.com-error.log.2**:

| Label | Earliest | Latest | Count |
|-------|----------|--------|-------|
| attacker_http | Jan 24 03:57:26 | Jan 24 03:58:09 | 36 |
| dirb | Jan 24 03:57:26 | Jan 24 03:57:39 | 23 |
| foothold | Jan 24 03:57:26 | Jan 24 03:58:09 | 36 |
| wpscan | Jan 24 03:57:53 | Jan 24 03:58:09 | 13 |

**labels\intranet_server\logs\audit\audit.log**:

| Label | Earliest | Latest | Count |
|-------|----------|--------|-------|
| attacker_change_user | 1642999060.603 | 1642999060.627 | 4 |
| escalate | 1642999060.603 | 1642999086.247 | 9 |
| escalated_command | 1642999086.239 | 1642999086.247 | 5 |
| escalated_sudo_command | 1642999086.239 | 1642999086.247 | 5 |

**labels\intranet_server\logs\auth.log**:

| Label | Earliest | Latest | Count |
|-------|----------|--------|-------|
| attacker_change_user | Jan 24 04:37:40 | Jan 24 04:37:40 | 4 |
| escalate | Jan 24 04:37:40 | Jan 24 04:38:06 | 8 |
| escalated_command | Jan 24 04:37:40 | Jan 24 04:38:06 | 5 |
| escalated_sudo_command | Jan 24 04:37:40 | Jan 24 04:38:06 | 5 |
| escalated_sudo_session | Jan 24 04:38:06 | Jan 24 04:38:06 | 3 |

## (e) Step Order and Overlap Matrix

### Step Order (sorted by start time)

| # | Step | Start | End |
|---|------|-------|-----|
| 1 | vpn_connect | 2022-01-24 03:01:00.063877+00:00 | 2022-01-24 03:01:00.063877+00:00 |
| 2 | traceroute_internet | 2022-01-24 03:01:17.723042+00:00 | 2022-01-24 03:01:17.723042+00:00 |
| 3 | dns_brute_force | 2022-01-24 03:01:20.969099+00:00 | 2022-01-24 03:01:20.969099+00:00 |
| 4 | host_discover_dmz | 2022-01-24 03:01:28.456313+00:00 | 2022-01-24 03:01:28.456313+00:00 |
| 5 | host_discover_local | 2022-01-24 03:36:38.963403+00:00 | 2022-01-24 03:36:38.963403+00:00 |
| 6 | service_scan | 2022-01-24 03:56:46.825579+00:00 | 2022-01-24 03:56:46.825579+00:00 |
| 7 | recon_networks_finish | 2022-01-24 03:57:26.435577+00:00 | 2022-01-24 03:57:26.435577+00:00 |
| 8 | dirb_scan | 2022-01-24 03:57:26.436714+00:00 | 2022-01-24 03:57:26.436714+00:00 |
| 9 | wpscan | 2022-01-24 03:57:48.934650+00:00 | 2022-01-24 03:57:48.934650+00:00 |
| 10 | upload_rce_shell | 2022-01-24 03:58:20.020490+00:00 | 2022-01-24 03:58:20.020490+00:00 |
| 11 | check_whoami | 2022-01-24 03:58:23.348462+00:00 | 2022-01-24 03:58:23.348462+00:00 |
| 12 | check_uname_r | 2022-01-24 03:58:25.108981+00:00 | 2022-01-24 03:58:25.108981+00:00 |
| 13 | read_profile | 2022-01-24 03:58:27.041422+00:00 | 2022-01-24 03:58:27.041422+00:00 |
| 14 | check_who | 2022-01-24 03:58:28.528493+00:00 | 2022-01-24 03:58:28.528493+00:00 |
| 15 | check_meminfo | 2022-01-24 03:58:30.292820+00:00 | 2022-01-24 03:58:30.292820+00:00 |
| 16 | check_uname_a | 2022-01-24 03:58:33.828771+00:00 | 2022-01-24 03:58:33.828771+00:00 |
| 17 | check_user_id | 2022-01-24 03:58:36.383526+00:00 | 2022-01-24 03:58:36.383526+00:00 |
| 18 | check_df | 2022-01-24 03:58:39.247353+00:00 | 2022-01-24 03:58:39.247353+00:00 |
| 19 | check_netstat_nat | 2022-01-24 03:58:40.944504+00:00 | 2022-01-24 03:58:40.944504+00:00 |
| 20 | check_id | 2022-01-24 03:58:42.198977+00:00 | 2022-01-24 03:58:42.198977+00:00 |
| 21 | read_resolv | 2022-01-24 03:58:43.400876+00:00 | 2022-01-24 03:58:43.400876+00:00 |
| 22 | check_netstat_t | 2022-01-24 03:58:44.632721+00:00 | 2022-01-24 03:58:44.632721+00:00 |
| 23 | list_home | 2022-01-24 03:58:48.259736+00:00 | 2022-01-24 03:58:48.259736+00:00 |
| 24 | check_last | 2022-01-24 03:58:49.562813+00:00 | 2022-01-24 03:58:49.562813+00:00 |
| 25 | list_web_dir | 2022-01-24 03:58:51.585727+00:00 | 2022-01-24 03:58:51.585727+00:00 |
| 26 | check_date | 2022-01-24 03:58:54.219952+00:00 | 2022-01-24 03:58:54.219952+00:00 |
| 27 | list_www | 2022-01-24 03:58:55.463292+00:00 | 2022-01-24 03:58:55.463292+00:00 |
| 28 | check_netstat_l | 2022-01-24 03:58:57.641124+00:00 | 2022-01-24 03:58:57.641124+00:00 |
| 29 | clear | 2022-01-24 03:59:00.791636+00:00 | 2022-01-24 03:59:00.791636+00:00 |
| 30 | check_wp_config | 2022-01-24 03:59:02.283150+00:00 | 2022-01-24 03:59:02.283150+00:00 |
| 31 | check_ps_a | 2022-01-24 03:59:03.740619+00:00 | 2022-01-24 03:59:03.740619+00:00 |
| 32 | read_passwd | 2022-01-24 03:59:05.928341+00:00 | 2022-01-24 03:59:05.928341+00:00 |
| 33 | check_release | 2022-01-24 03:59:09.511060+00:00 | 2022-01-24 03:59:09.511060+00:00 |
| 34 | check_cpuinfo | 2022-01-24 03:59:12.977191+00:00 | 2022-01-24 03:59:12.977191+00:00 |
| 35 | dump_wp_users | 2022-01-24 03:59:14.207842+00:00 | 2022-01-24 03:59:14.207842+00:00 |
| 36 | read_group | 2022-01-24 03:59:16.923431+00:00 | 2022-01-24 03:59:16.923431+00:00 |
| 37 | check_uptime | 2022-01-24 03:59:18.310719+00:00 | 2022-01-24 03:59:18.310719+00:00 |
| 38 | check_network_config | 2022-01-24 03:59:19.790753+00:00 | 2022-01-24 03:59:19.790753+00:00 |
| 39 | recon_host_finish | 2022-01-24 03:59:22.176229+00:00 | 2022-01-24 03:59:22.176229+00:00 |
| 40 | decide_crack_method | 2022-01-24 03:59:22.177425+00:00 | 2022-01-24 03:59:22.177425+00:00 |
| 41 | crack_wphash | 2022-01-24 03:59:22.182453+00:00 | 2022-01-24 03:59:22.182453+00:00 |
| 42 | reverse_shell_listen | 2022-01-24 04:36:57.885350+00:00 | 2022-01-24 04:36:57.885350+00:00 |
| 43 | open_reverse_shell | 2022-01-24 04:36:59.577594+00:00 | 2022-01-24 04:36:59.577594+00:00 |
| 44 | wait_reverse_shell | 2022-01-24 04:37:25.836115+00:00 | 2022-01-24 04:37:25.836115+00:00 |
| 45 | open_pty | 2022-01-24 04:37:39.739109+00:00 | 2022-01-24 04:37:39.739109+00:00 |
| 46 | login_user | 2022-01-24 04:37:40.530254+00:00 | 2022-01-24 04:37:40.530254+00:00 |
| 47 | read_fstab | 2022-01-24 04:37:53.496765+00:00 | 2022-01-24 04:37:53.496765+00:00 |
| 48 | check_ps_aux | 2022-01-24 04:37:55.374578+00:00 | 2022-01-24 04:37:55.374578+00:00 |
| 49 | check_sudo | 2022-01-24 04:37:58.370156+00:00 | 2022-01-24 04:37:58.370156+00:00 |
| 50 | check_uname_ar | 2022-01-24 04:38:01.850376+00:00 | 2022-01-24 04:38:01.850376+00:00 |
| 51 | list_shadow | 2022-01-24 04:38:04.866547+00:00 | 2022-01-24 04:38:04.866547+00:00 |
| 52 | read_shadow | 2022-01-24 04:38:06.227784+00:00 | 2022-01-24 04:38:06.227784+00:00 |
| 53 | check_ifconfig | 2022-01-24 04:38:09.938128+00:00 | 2022-01-24 04:38:09.938128+00:00 |
| 54 | vpn_disconnect | 2022-01-24 04:38:11.767134+00:00 | 2022-01-24 04:38:11.767134+00:00 |
| 55 | dnsteal_stop | 2022-01-24 13:50:03.250776+00:00 | 2022-01-24 13:50:03.250776+00:00 |

**No overlapping steps detected.**


## (f) Timezone Alignment

**Assessment**: ALIGNED — pcap and attacks.log timestamps both appear to be UTC

**First attack step start (UTC)**: 2022-01-24T03:01:00.063877+00:00

**pcaps covering attack start**: 2

- `gather\attacker_0\logs\ait.aecid.attacker.wpdiscuz\traffic.pcap`: 2022-01-20T13:47:50.743391+00:00 → 2022-01-24T04:38:17.102834+00:00
- `gather\attacker_0\logs\dnsteal\traffic.pcap`: 2022-01-20T12:48:19.071873+00:00 → 2022-01-24T13:50:03.248663+00:00

## (g) Can Each 1-Minute Network Window Be Labelled?

**Answer**: **YES**

**Reason**: attacks.log provides per-step start/end timestamps in UTC. pcap files contain packet timestamps in UTC. Timezone alignment confirmed by checking that the first attack step's start time falls within the time range of at least one pcap file. Each 1-minute window can be assigned an attack-step label by checking if the window's time interval overlaps with any step's [start, end] interval.


**Caveats**:
- Overlapping steps (especially exfiltration) require a precedence rule.
- Timestamps come from attacks.log, not from packet-level ground truth.
- Benign windows are those outside all step intervals.

## Dataset Metadata
