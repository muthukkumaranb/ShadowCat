"""
SHADOWCAT SOC Cockpit - Page 5: Security Telemetry & Anomaly Alerts
Direct implementation of Stitch folder shadowcat_soc_alerts.
Wired to live data_provider.py and dynamic filtering pipeline.
"""

import streamlit as st
from styles import TOKENS, render_html
from data_provider import get_flagged_flows, get_analysis_metadata, get_forecast_trajectory

def render_page():
    t = TOKENS.get(st.session_state.get("theme", "dark"), TOKENS["dark"])
    flows = get_flagged_flows()
    meta = get_analysis_metadata()
    fc = get_forecast_trajectory()
    ml_risks = fc.get("risk", [])
    risk_val = ml_risks[0] if ml_risks else 0.05
    ml_stages = fc.get("stage", [])
    curr_stage = ml_stages[0] if ml_stages else "Credential Access"

    # Baseline default alerts fallback
    default_alerts = [
        {
            "id": "ALT-9941",
            "time": "14:28:09 UTC",
            "mins_ago": 8,
            "sev": "critical",
            "technique": "T1071.001 • C2: Web Protocols",
            "host": "ip-10-0-14-88 (svc-auth-master)",
            "title": "Egress burst spike (+410%) to uncatalogued foreign ASN 4837 with high-frequency encrypted beaconing cadence.",
            "description": "Outbound TLS flow directed toward 45.138.21.9:443 exceeded safe enterprise baseline by +410%. Packet inter-arrival variance matches known Cobalt Strike Malleable C2 jitter profile.",
            "source": "10.0.14.88:49210",
            "destination": "45.138.21.9:443 (CN)",
            "rate": "14.8 GB/s (+410%)",
            "cadence": "3.2s ± 12ms",
            "remediation": [
                "Deploy SDN egress null-route on perimeter edge-gw-02 for foreign destination 45.138.21.9:443.",
                "Capture volatile RAM dump on host ip-10-0-14-88 before terminating process container.",
                "Invalidate active Kerberos TGT tickets across domain controllers for principal svc-auth-master.",
                "Audit recent DNS queries from 10.0.14.88 for uncatalogued domains registered within the last 48 hours."
            ]
        },
        {
            "id": "ALT-9938",
            "time": "14:24:51 UTC",
            "mins_ago": 12,
            "sev": "critical",
            "technique": "T1059.004 • Lateral Movement",
            "host": "svc-auth-master",
            "title": "Unauthenticated RPC execution across host enclave subnet with abnormal peer fan-out.",
            "description": "Rapid micro-RPC calls executed across ports 135 and 445 against internal peer nodes. High fan-out signature matches automated remote credential dumping utilities.",
            "source": "10.0.14.88:51204",
            "destination": "10.0.14.0/24 Subnet",
            "rate": "1.2k req/sec",
            "cadence": "Continuous Burst",
            "remediation": [
                "Apply micro-segmentation security group rules isolating subnet 10.0.14.0/24 RPC inter-pod communications.",
                "Review Kerberos event logs (Event ID 4768/4769) for anomalous service ticket generation.",
                "Trigger automated credential reset for local administrative daemon service accounts."
            ]
        },
        {
            "id": "ALT-9935",
            "time": "14:21:30 UTC",
            "mins_ago": 15,
            "sev": "critical",
            "technique": "T1562.001 • Defense Impairment",
            "host": "audit-vault",
            "title": "Local cryptographic audit logging daemon tamper attempt detected via memory hook on PID 4810.",
            "description": "Unauthorized syscall ptrace hook intercepted by kernel eBPF probe on audit-vault daemon. Audit ledger signature verification triggered an immediate integrity alarm.",
            "source": "10.0.14.5:4810",
            "destination": "Kernel eBPF Ring Buffer",
            "rate": "14 Calls / 2s",
            "cadence": "Irregular Burst",
            "remediation": [
                "Validate zero-trust ledger block chain integrity using ed25519 signature proof verification.",
                "Enforce kernel lockdown mode and terminate detached memory-attached trace handles on PID 4810.",
                "Lock down audit-vault ingress permissions strictly to immutable hardware enclave channels."
            ]
        },
        {
            "id": "ALT-9932",
            "time": "14:19:12 UTC",
            "mins_ago": 18,
            "sev": "high",
            "technique": "T1046 • Network Discovery",
            "host": "ws-analyst-12",
            "title": "Rapid port sweeping scan detected from internal workstation cluster segment.",
            "description": "Sequential TCP SYN sweeps across critical service ports (22, 80, 443, 8080, 3389) originating from internal workstation segment toward database sharding tier.",
            "source": "10.0.14.21:58440",
            "destination": "10.0.3.0/24 Subnet",
            "rate": "850 pkts/sec",
            "cadence": "SYN Flood Scan",
            "remediation": [
                "Quarantine ws-analyst-12 network interface onto remediation VLAN to prevent further discovery.",
                "Verify analyst authentication session and check for browser-based session hijacking or RAT activity.",
                "Tighten database tier ingress ACLs to permit connections solely from verified application pods."
            ]
        },
        {
            "id": "ALT-9929",
            "time": "14:16:44 UTC",
            "mins_ago": 24,
            "sev": "high",
            "technique": "T1021.002 • SMB / Kerberoasting Probe",
            "host": "dc-shadow-02",
            "title": "Repeated Kerberos ticket-granting service requests with non-existent SPNs.",
            "description": "Multiple RC4-HMAC encrypted TGS requests submitted in rapid succession against privileged accounts. Indicates active offline hash extraction attempt.",
            "source": "10.0.14.88:389",
            "destination": "10.0.5.2:88 (KDC)",
            "rate": "142 req/min",
            "cadence": "Automated Harvest",
            "remediation": [
                "Disable RC4-HMAC cipher on Key Distribution Center (KDC); enforce AES-256 Kerberos encryption only.",
                "Rotate 25+ character passwords on all identified service accounts requested in the TGS batch.",
                "Enable HoneyToken SPN detection alerts to trap persistent lateral movement actors."
            ]
        },
        {
            "id": "ALT-9926",
            "time": "14:14:02 UTC",
            "mins_ago": 35,
            "sev": "high",
            "technique": "T1571 • Non-Standard Port Protocol",
            "host": "analytics-agg-02",
            "title": "Outbound TCP session established over port 8443 bypasses egress application proxy.",
            "description": "Direct TCP tunnel initiated from analytics aggregator node to non-whitelisted foreign address without SNI negotiation. TLS fingerprint matches non-browser tooling.",
            "source": "10.0.14.92:43900",
            "destination": "194.26.29.112:8443",
            "rate": "3.8 MB/s",
            "cadence": "Persistent Stream",
            "remediation": [
                "Terminate active socket session on analytics-agg-02 and enforce transparent egress MITM proxying.",
                "Inspect process tree spawning outbound connection (check for rogue cron jobs or container execs).",
                "Update outbound firewall rules to drop all direct non-whitelisted TCP ports outside standard 80/443."
            ]
        },
        {
            "id": "ALT-9925",
            "time": "14:12:00 UTC",
            "mins_ago": 48,
            "sev": "medium",
            "technique": "T1078 • Valid Accounts",
            "host": "iam-sync-daemon",
            "title": "Simultaneous geo-distributed session tokens authenticated for high-privilege service principal.",
            "description": "Concurrent valid OAuth 2.0 refresh tokens presented from both US-East (Virginia) and AS-East (Tokyo) within a 4-minute interval, breaching travel velocity threshold.",
            "source": "10.0.14.15:443",
            "destination": "Identity OAuth Provider",
            "rate": "2 Active Tokens",
            "cadence": "Simultaneous",
            "remediation": [
                "Revoke both active OAuth refresh token instances on Identity Provider and require MFA re-auth.",
                "Verify service principal IP whitelisting rules to restrict token usage strictly to known VPC subnets.",
                "Check IAM audit trail for any authorization role modifications created during the anomalous window."
            ]
        },
        {
            "id": "ALT-9920",
            "time": "13:45:10 UTC",
            "mins_ago": 85,
            "sev": "medium",
            "technique": "T1110 • Brute Force",
            "host": "bastion-stg-01",
            "title": "High threshold of failed SSH authentications originating from staging bastion IP.",
            "description": "54 failed SSH password attempts against root and deployer accounts within 90 seconds. Source rate throttled by local PAM rules.",
            "source": "10.0.1.55:22",
            "destination": "10.0.2.80:22",
            "rate": "36 attempts/min",
            "cadence": "Dictionary Scan",
            "remediation": [
                "Block IP 10.0.1.55 on internal staging firewalls and inspect bastion-stg-01 for unauthorized login.",
                "Enforce SSH public-key-only authentication and disable all password-based SSH mechanisms.",
                "Review auth.log on bastion-stg-01 to identify entry vector and verify sudoers integrity."
            ]
        },
        {
            "id": "ALT-9914",
            "time": "12:10:00 UTC",
            "mins_ago": 180,
            "sev": "medium",
            "technique": "T1040 • Network Sniffing",
            "host": "k8s-worker-04",
            "title": "Promiscuous mode socket activation detected on internal bridge interface eth0.vlan14.",
            "description": "Kernel socket flag change PROMISC detected by daemon auditor on Kubernetes worker node. No registered packet capture job was scheduled in cluster workload specs.",
            "source": "10.0.2.80:eth0",
            "destination": "Local Interface Bridge",
            "rate": "Raw Capture",
            "cadence": "Promiscuous Socket",
            "remediation": [
                "Identify container PID holding raw socket capabilities (CAP_NET_RAW / CAP_NET_ADMIN).",
                "Apply Pod Security Admission policy in 'enforce' mode to forbid privileged container execution.",
                "Terminate non-compliant pods and inspect container image digest against trusted registry signatures."
            ]
        }
    ]

    # Synthesize live alerts if live ML flagged flows are present
    if flows:
        live_alerts = []
        for i, flw in enumerate(flows[:10]):
            src = flw.get("src", flw.get("Src IP", "10.0.14.88"))
            dst = flw.get("dst", flw.get("Dst IP", "45.138.21.9"))
            proto = flw.get("proto", flw.get("Protocol", "TCP"))
            dport = flw.get("dport", flw.get("Dst Port", 443))
            sport = flw.get("sport", flw.get("Src Port", 49210 + i))
            
            if risk_val >= 0.75:
                sev = "critical" if i < 3 else ("high" if i < 7 else "medium")
            elif risk_val >= 0.50:
                sev = "high" if i < 4 else ("medium" if i < 8 else "medium")
            else:
                sev = "medium" if i < 3 else "medium"

            live_alerts.append({
                "id": f"ALT-{9950 - i*3}",
                "time": f"14:{max(0, 28 - i*2):02d}:10 UTC",
                "mins_ago": 2 + i * 4,
                "sev": sev,
                "technique": f"T1071.001 • {curr_stage} ({proto})",
                "host": f"node-{src}",
                "title": f"Suspicious flow detected: {src}:{sport} → {dst}:{dport} ({proto}) with risk {risk_val:.2f}.",
                "description": f"Ingested telemetry record flagged by World Model inference. Model indicates {curr_stage} execution phase with cumulative risk score {risk_val:.2f}.",
                "source": f"{src}:{sport}",
                "destination": f"{dst}:{dport}",
                "rate": f"{14.8 - i*1.1:.1f} MB/s",
                "cadence": "Continuous Burst",
                "remediation": [
                    f"Deploy SDN egress null-route on perimeter gateway for foreign destination {dst}:{dport}.",
                    f"Capture volatile RAM and packet capture on host {src} before terminating container.",
                    f"Invalidate active authentication session tickets for principal associated with {src}.",
                    f"Audit recent DNS queries originating from {src} for anomalous external connections."
                ]
            })
        raw_alerts = live_alerts
    else:
        raw_alerts = default_alerts

    # Calculate real-time metrics
    n_crit = sum(1 for a in raw_alerts if a["sev"] == "critical")
    n_high = sum(1 for a in raw_alerts if a["sev"] == "high")
    n_med = sum(1 for a in raw_alerts if a["sev"] == "medium")
    total_active = len(raw_alerts)

    # 1. Header & Metric Summary Bar
    render_html(f"""
    <div class="soc-card" style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span class="soc-pulse-dot" style="background:{t['secondary']}; width:8px; height:8px;"></span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.125rem; font-weight: 700; color: {t['text_high']}; text-transform: uppercase;">
                        Security Telemetry & Anomaly Alerts
                    </span>
                </div>
                <div style="display: flex; gap: 0.5rem; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']}; margin-top: 0.25rem;">
                    <span>STREAM ACTIVE</span> • <span style="color:{t['primary']}">RUNNING MONTE CARLO HEURISTICS</span> • <span>SYS_REF: 0x884F_A</span>
                </div>
            </div>
            <!-- Dynamic quick counters -->
            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center;">
                <span class="soc-badge badge-neutral">Total Active: <b style="color:{t['text_high']}">{total_active}</b></span>
                <span class="soc-badge badge-critical">Critical: {n_crit}</span>
                <span class="soc-badge badge-caution">High: {n_high}</span>
                <span class="soc-badge badge-neutral">Medium: {n_med}</span>
                <span class="soc-badge badge-nominal">Pipeline: 4,812 evt/s</span>
            </div>
        </div>
    </div>
    """)

    # 2. Filter & Sort Control Strip
    c_f1, c_f2, c_f3 = st.columns([0.36, 0.36, 0.28])
    with c_f1:
        sev_filter = st.radio(
            "Severity Filter",
            [f"All ({total_active})", f"Critical ({n_crit})", f"High ({n_high})", f"Medium ({n_med})"],
            horizontal=True,
            label_visibility="collapsed"
        )
    with c_f2:
        time_filter = st.radio(
            "Time Horizon",
            ["Last 15m", "Last 1h", "Last 24h", "All Time"],
            horizontal=True,
            index=1,
            label_visibility="collapsed"
        )
    with c_f3:
        search_term = st.text_input("Search", placeholder="Search ID, host, technique, IP...", label_visibility="collapsed")

    render_html("<div style='height: 0.5rem;'></div>")

    # Apply Active Filtering Pipeline
    filtered_alerts = raw_alerts

    # 1. Severity filter
    if "Critical" in sev_filter:
        filtered_alerts = [a for a in filtered_alerts if a["sev"] == "critical"]
    elif "High" in sev_filter:
        filtered_alerts = [a for a in filtered_alerts if a["sev"] == "high"]
    elif "Medium" in sev_filter:
        filtered_alerts = [a for a in filtered_alerts if a["sev"] == "medium"]

    # 2. Time Horizon filter
    if time_filter == "Last 15m":
        filtered_alerts = [a for a in filtered_alerts if a["mins_ago"] <= 15]
    elif time_filter == "Last 1h":
        filtered_alerts = [a for a in filtered_alerts if a["mins_ago"] <= 60]
    elif time_filter == "Last 24h":
        filtered_alerts = [a for a in filtered_alerts if a["mins_ago"] <= 1440]

    # 3. Search query filter
    if search_term and search_term.strip():
        q = search_term.strip().lower()
        filtered_alerts = [
            a for a in filtered_alerts
            if q in a["id"].lower()
            or q in a["host"].lower()
            or q in a["technique"].lower()
            or q in a["title"].lower()
            or q in a["description"].lower()
            or q in a.get("source", "").lower()
            or q in a.get("destination", "").lower()
        ]

    # Filter Readout
    render_html(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']};">
        <div>
            FILTERED THREATS: <b style="color:{t['primary']}">{len(filtered_alerts)} MATCHING</b> [SEV: {sev_filter.split()[0].upper()} • HORIZON: {time_filter.upper()}]
        </div>
        <span style="color: {t['text_muted']};">AUTOMATED THREAT REMEDIATION READY</span>
    </div>
    """)

    if not filtered_alerts:
        render_html(f"""
        <div class="soc-card" style="text-align: center; padding: 2.5rem; color: {t['text_muted']}; font-family: 'JetBrains Mono', monospace;">
            <div style="font-size: 0.95rem; font-weight: 700; color: {t['text_high']}; margin-bottom: 0.5rem;">
                NO ACTIVE ALERTS MATCHING CRITERIA
            </div>
            <div style="font-size: 0.75rem;">
                Try selecting 'All Time' or clearing the search query '{search_term}'.
            </div>
        </div>
        """)
        return

    # 3. Render Dynamic Alert Cards with "What Can Be Done" Action Guidance
    for alert in filtered_alerts:
        badge_cls = "badge-critical" if alert["sev"] == "critical" else ("badge-caution" if alert["sev"] == "high" else "badge-neutral")
        border_color = t['secondary'] if alert["sev"] == "critical" else (t['tertiary'] if alert["sev"] == "high" else t['border'])

        remediation_items_html = "".join([
            f"<li style='margin-bottom: 0.35rem; color: {t['text_high']};'>{item}</li>"
            for item in alert["remediation"]
        ])

        render_html(f"""
        <div class="soc-card" style="border-left: 4px solid {border_color}; margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; flex-wrap: wrap; gap: 0.5rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                    <span class="soc-badge {badge_cls}">{alert['sev'].upper()}</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {t['text_muted']};">{alert['time']} ({alert['mins_ago']}m ago)</span>
                    <span class="soc-badge badge-neutral" style="color:{border_color}">{alert['technique']}</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: {t['primary']};">
                        {alert['host']}
                    </span>
                </div>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']};">ID: {alert['id']}</span>
            </div>
            
            <h3 style="font-family: 'Inter', sans-serif; font-size: 0.95rem; font-weight: 600; color: {t['text_high']}; margin: 0 0 0.5rem 0;">
                {alert['title']}
            </h3>
            <p style="font-family: 'Inter', sans-serif; font-size: 0.8125rem; color: {t['text_secondary']}; margin: 0 0 0.75rem 0; line-height: 1.5;">
                {alert['description']}
            </p>
            
            <!-- Telemetry Parameters -->
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; background: {t['surface_lowest']}; padding: 0.65rem; border-radius: 4px; border: 1px solid {t['border']}; margin-bottom: 0.75rem;">
                <div>
                    <span class="soc-stat-label">Source</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {t['text_high']}; font-weight: 600; word-break: break-all;">{alert['source']}</div>
                </div>
                <div>
                    <span class="soc-stat-label">Destination</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {border_color}; font-weight: 600; word-break: break-all;">{alert['destination']}</div>
                </div>
                <div>
                    <span class="soc-stat-label">Flow Rate</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {border_color}; font-weight: 600;">{alert['rate']}</div>
                </div>
                <div>
                    <span class="soc-stat-label">Cadence / Profile</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {t['text_high']}; font-weight: 600;">{alert['cadence']}</div>
                </div>
            </div>

            <!-- WHAT CAN BE DONE // RECOMMENDED ACTION (SOC Guidance) -->
            <div class="soc-card-nested" style="border-left: 3px solid {t['primary']}; background: {t['surface_lowest']}; padding: 0.75rem 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 700; color: {t['primary']}; text-transform: uppercase; letter-spacing: 0.05em;">
                        WHAT CAN BE DONE // RECOMMENDED SOC PLAYBOOK
                    </div>
                    <span class="soc-badge badge-nominal" style="font-size: 0.6rem; padding: 1px 5px;">ANALYST ACTION PLAN</span>
                </div>
                <ul style="font-family: 'Inter', sans-serif; font-size: 0.8rem; margin: 0; padding-left: 1.25rem; line-height: 1.5;">
                    {remediation_items_html}
                </ul>
            </div>
        </div>
        """)

if __name__ == "__main__":
    render_page()
