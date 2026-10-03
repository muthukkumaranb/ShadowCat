"""
SHADOWCAT - Observed State S(t) Component
"""

from styles import COLORS, render_html


def render_current_state(data):
    """Renders the Unified Cyber State card with clean executive telemetry breakdown."""
    state = data["current_state"]
    analysis = data["analysis"]

    render_html(f"""
    <div class="glass-card">
        <div class="card-title">
            <span>Observed Network State &nbsp;<code style="color: #FFFFFF; font-size: 0.95rem;">{state['state_id']}</code></span>
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 24px; align-items: stretch;">
            <div style="flex: 2.2; min-width: 280px; border-right: 1px solid #262626; padding-right: 24px;">
                <div class="metric-label">Observed Behavioral Pattern</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin: 6px 0 10px 0;">
                    {state['dominant_behavior']}
                </div>
                <div style="font-size: 0.8rem; color: #8A8A8A; line-height: 1.5;">
                    Aggregated 10-second traffic window across monitored endpoints.
                </div>
            </div>
            <div style="flex: 1.3; min-width: 180px; border-right: 1px solid #262626; padding-right: 24px;">
                <div class="metric-label">Novelty Score</div>
                <div class="metric-value-huge" style="color: #2FB872; font-size: 2rem;">
                    {state['novelty_score']:.2f} <span style="font-size: 0.85rem; color: #8A8A8A; font-weight: normal;">/ 1.0</span>
                </div>
                <div style="font-size: 0.76rem; color: #8A8A8A;">
                    Status: <b style="color: #2FB872;">Normal Baseline</b>
                </div>
            </div>
            <div style="flex: 2.2; min-width: 240px; display: grid; grid-template-columns: 1fr 1fr; gap: 14px;">
                <div>
                    <div class="metric-label">Active Flows</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; font-family: 'JetBrains Mono', monospace;">
                        {analysis['flows_analyzed']:,}
                    </div>
                </div>
                <div>
                    <div class="metric-label">Packets</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; font-family: 'JetBrains Mono', monospace;">
                        {analysis['packets_analyzed']:,}
                    </div>
                </div>
                <div>
                    <div class="metric-label">Endpoints</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; font-family: 'JetBrains Mono', monospace;">
                        {state['active_endpoints']} <span style="font-size: 0.8rem; color: #8A8A8A; font-weight: normal;">hosts</span>
                    </div>
                </div>
                <div>
                    <div class="metric-label">SYN / ACK Ratio</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #E0982B; font-family: 'JetBrains Mono', monospace;">
                        {state['syn_ack_ratio']:.1f}x <span style="font-size: 0.72rem; color: #E0982B; font-weight: 600;">(Elevated)</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """)
