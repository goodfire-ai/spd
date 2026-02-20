"""MCP tools for generating Distill.pub-style figures.

Each tool returns standalone HTML that can be embedded in the dashboard.
"""

import hashlib
import json
import re
from typing import Any


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))


# =============================================================================
# COLLAPSIBLE TABLE HELPER
# =============================================================================

def make_table_collapsible(
    table_html: str,
    row_count: int,
    threshold: int = 6,
    show_first: int = 4
) -> str:
    """Wrap a table in a collapsible if it has many rows.

    Args:
        table_html: The full table HTML
        row_count: Number of data rows (excluding header)
        threshold: Collapse if row_count > threshold
        show_first: Show this many rows before collapsing

    Returns:
        Either the original table or table wrapped in collapsible
    """
    if row_count <= threshold:
        return table_html

    hidden_count = row_count - show_first

    # Generate unique ID for this collapsible
    import hashlib
    unique_id = hashlib.md5(table_html[:100].encode()).hexdigest()[:8]

    return f'''
    <div class="table-collapsible" id="tc-{unique_id}">
        <style>
            #tc-{unique_id} .table-rows-hidden {{ display: none; }}
            #tc-{unique_id}.expanded .table-rows-hidden {{ display: table-row-group; }}
            #tc-{unique_id}.expanded .expand-btn {{ display: none; }}
            #tc-{unique_id} .collapse-btn {{ display: none; }}
            #tc-{unique_id}.expanded .collapse-btn {{ display: inline-flex; }}
        </style>
        {table_html}
        <div style="text-align: center; margin-top: 12px;">
            <button class="expand-btn" onclick="document.getElementById('tc-{unique_id}').classList.add('expanded')"
                style="background: var(--bg-secondary, #f3f4f6); border: 1px solid var(--border, #e5e7eb);
                       border-radius: 6px; padding: 6px 16px; font-size: 13px; color: var(--text-secondary, #666);
                       cursor: pointer; display: inline-flex; align-items: center; gap: 6px;">
                <span>Show {hidden_count} more rows</span>
                <span style="font-size: 10px;">▼</span>
            </button>
            <button class="collapse-btn" onclick="document.getElementById('tc-{unique_id}').classList.remove('expanded')"
                style="background: var(--bg-secondary, #f3f4f6); border: 1px solid var(--border, #e5e7eb);
                       border-radius: 6px; padding: 6px 16px; font-size: 13px; color: var(--text-secondary, #666);
                       cursor: pointer; align-items: center; gap: 6px;">
                <span>Show less</span>
                <span style="font-size: 10px;">▲</span>
            </button>
        </div>
    </div>
    '''


def split_table_rows(rows_html: str, show_first: int) -> tuple:
    """Split table rows into visible and hidden portions.

    Args:
        rows_html: HTML string containing <tr>...</tr> rows
        show_first: Number of rows to keep visible

    Returns:
        Tuple of (visible_rows_html, hidden_rows_html)
    """
    # Find all rows
    row_pattern = re.compile(r'<tr[^>]*>.*?</tr>', re.DOTALL)
    rows = row_pattern.findall(rows_html)

    if len(rows) <= show_first:
        return rows_html, ""

    visible = '\n'.join(rows[:show_first])
    hidden = '\n'.join(rows[show_first:])

    return visible, hidden


def escape_html_preserve_tags(text: str) -> str:
    """Escape HTML but preserve allowed formatting tags.

    Allows common formatting tags that the agent might use in prose:
    - Inline: <strong>, <em>, <mark>, <br>
    - Block: <p>, <ul>, <ol>, <li>, <blockquote>
    - Headings: <h1> through <h6>
    - Special: <div class="executive-summary"> for summary boxes
    """
    # First escape everything
    escaped = escape_html(text)

    # Then restore allowed tags
    # Inline formatting
    allowed_tags = [
        ('&lt;strong&gt;', '<strong>'),
        ('&lt;/strong&gt;', '</strong>'),
        ('&lt;em&gt;', '<em>'),
        ('&lt;/em&gt;', '</em>'),
        ('&lt;mark&gt;', '<mark>'),
        ('&lt;/mark&gt;', '</mark>'),
        ('&lt;br&gt;', '<br>'),
        ('&lt;br/&gt;', '<br/>'),
        ('&lt;br /&gt;', '<br />'),
        # Block elements
        ('&lt;p&gt;', '<p>'),
        ('&lt;/p&gt;', '</p>'),
        ('&lt;ul&gt;', '<ul>'),
        ('&lt;/ul&gt;', '</ul>'),
        ('&lt;ol&gt;', '<ol>'),
        ('&lt;/ol&gt;', '</ol>'),
        ('&lt;li&gt;', '<li>'),
        ('&lt;/li&gt;', '</li>'),
        ('&lt;blockquote&gt;', '<blockquote>'),
        ('&lt;/blockquote&gt;', '</blockquote>'),
        ('&lt;/div&gt;', '</div>'),
        # Executive summary div (specific class only for security)
        ('&lt;div class=&quot;executive-summary&quot;&gt;', '<div class="executive-summary">'),
    ]

    # Add headings h1-h6
    for i in range(1, 7):
        allowed_tags.append((f'&lt;h{i}&gt;', f'<h{i}>'))
        allowed_tags.append((f'&lt;/h{i}&gt;', f'</h{i}>'))

    for escaped_tag, original_tag in allowed_tags:
        escaped = escaped.replace(escaped_tag, original_tag)

    return escaped


def linkify_neuron_ids(text: str) -> str:
    """Convert neuron IDs like L3/N9778 to clickable links."""
    return re.sub(
        r'(L\d+/N\d+)',
        lambda m: f'<a href="{m.group(1).replace("/", "_")}.html" class="neuron-link">{m.group(1)}</a>',
        text
    )


def clean_token(token: str) -> str:
    """Clean BPE artifacts from token display."""
    # Replace Ġ (BPE leading space marker) with readable format
    if token.startswith("Ġ"):
        return token[1:]  # Just strip the marker
    return token


# =============================================================================
# FIGURE CSS (shared across all figures)
# =============================================================================

FIGURE_CSS = """
/* Figure container base styles */
.figure-container {
    margin: 32px 0;
    background: var(--bg-elevated, #ffffff);
    border-radius: 12px;
    padding: 24px;
    border: 1px solid var(--border, #e0e0e0);
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}

/* Full-width figures that break out of text column */
.figure-container.full-width {
    width: 100vw;
    position: relative;
    left: 50%;
    right: 50%;
    margin-left: -50vw;
    margin-right: -50vw;
    border-radius: 0;
    padding: 32px max(24px, calc((100vw - 1200px) / 2 + 24px));
}

.figure-title {
    font-size: 15px;
    font-weight: 600;
    color: var(--text, #111111);
    margin-bottom: 16px;
}

.figure-caption {
    font-size: 13px;
    color: var(--text-secondary, #555555);
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid var(--border, #e5e5e5);
    line-height: 1.5;
}

/* Activation Grid */
.activation-grid .grid-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
}

.activation-grid .grid-col {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.activation-grid .col-header {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.activation-grid .fires .col-header { color: #16a34a; }
.activation-grid .ignores .col-header { color: #9ca3af; }

.activation-grid .example {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    padding: 12px;
    background: var(--bg-inset, #f5f5f7);
    border-radius: 8px;
    font-size: 14px;
}

.activation-grid .prompt {
    flex: 1;
    line-height: 1.5;
}

.activation-grid .activation {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    flex-shrink: 0;
}

.activation-grid .activation.high { color: #16a34a; font-weight: 600; }
.activation-grid .activation.low { color: #9ca3af; }

.activation-grid mark {
    background: linear-gradient(to top, #fde68a 40%, transparent 40%);
    padding: 0 2px;
}

/* Token Bar Chart */
.token-chart .chart-container {
    position: relative;
    padding: 20px 0;
}

.token-chart .bar-row {
    display: flex;
    align-items: center;
    margin-bottom: 8px;
    height: 28px;
}

.token-chart .token-label {
    width: 80px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    text-align: right;
    padding-right: 12px;
    flex-shrink: 0;
}

.token-chart .bar-area {
    flex: 1;
    display: flex;
    align-items: center;
    position: relative;
}

.token-chart .bar {
    height: 20px;
    border-radius: 4px;
    position: absolute;
}

.token-chart .bar.promote {
    background: linear-gradient(90deg, #86efac, #22c55e);
    left: 50%;
}

.token-chart .bar.suppress {
    background: linear-gradient(90deg, #ef4444, #fca5a5);
    right: 50%;
}

.token-chart .magnitude {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-tertiary, #888888);
    position: absolute;
}

.token-chart .center-line {
    position: absolute;
    left: 50%;
    top: 0;
    bottom: 0;
    width: 1px;
    background: var(--border, #e5e5e5);
}

/* Hypothesis Timeline */
.hypothesis-timeline .timeline-item {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 16px;
    background: var(--bg-inset, #f5f5f7);
    border-radius: 8px;
    margin-bottom: 12px;
}

.hypothesis-timeline .timeline-item.confirmed {
    border-left: 4px solid #22c55e;
}

.hypothesis-timeline .timeline-item.refuted {
    border-left: 4px solid #ef4444;
}

.hypothesis-timeline .timeline-item.inconclusive {
    border-left: 4px solid #f59e0b;
}

.hypothesis-timeline .timeline-item.weakened {
    border-left: 4px solid #f97316;
}

.hypothesis-timeline .timeline-item.registered {
    border-left: 4px solid #6b7280;
}

.hypothesis-timeline .timeline-item.testing {
    border-left: 4px solid #3b82f6;
}

.hypothesis-timeline .status-badge {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    width: 100px;
    flex-shrink: 0;
}

.hypothesis-timeline .confirmed .status-badge { color: #16a34a; }
.hypothesis-timeline .refuted .status-badge { color: #dc2626; }
.hypothesis-timeline .inconclusive .status-badge { color: #f59e0b; }
.hypothesis-timeline .weakened .status-badge { color: #f97316; }
.hypothesis-timeline .registered .status-badge { color: #6b7280; }
.hypothesis-timeline .testing .status-badge { color: #3b82f6; }

.hypothesis-timeline .hypothesis-text {
    flex: 1;
    font-size: 14px;
    color: var(--text-secondary, #555555);
}

.hypothesis-timeline .probability-shift {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}

.hypothesis-timeline .prob-arrow {
    color: var(--text-tertiary, #888888);
}

/* Circuit Diagram - breaks out for more width */
.figure-container.circuit-diagram {
    margin-left: -150px;
    margin-right: -150px;
    padding-left: 150px;
    padding-right: 150px;
    max-width: none;
    width: auto;
}

@media (max-width: 1100px) {
    .figure-container.circuit-diagram {
        margin-left: -24px;
        margin-right: -24px;
        padding-left: 24px;
        padding-right: 24px;
    }
}

.circuit-diagram .circuit-flow {
    display: grid;
    grid-template-columns: 280px 40px auto 40px 280px;
    gap: 16px;
    align-items: center;
    justify-content: center;
}

@media (max-width: 1000px) {
    .circuit-diagram .circuit-flow {
        grid-template-columns: 1fr;
        gap: 20px;
    }
    .circuit-diagram .arrow-column {
        transform: rotate(90deg);
    }
}

.circuit-diagram .node-column {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.circuit-diagram .node-item {
    padding: 14px 16px;
    background: var(--bg-inset, #f5f5f7);
    border-radius: 8px;
}

.circuit-diagram .node-label {
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 4px;
    line-height: 1.4;
}

.circuit-diagram .node-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--text-secondary, #555555);
}

.circuit-diagram .node-weight {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-tertiary, #888888);
}

.circuit-diagram .center-node {
    padding: 24px 32px;
    background: linear-gradient(135deg, #dbeafe, #bfdbfe);
    border-radius: 12px;
    text-align: center;
    min-width: 180px;
}

.circuit-diagram .center-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px;
    font-weight: 600;
    color: #1d4ed8;
}

.circuit-diagram .arrow-column {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    color: var(--text-tertiary, #888888);
    font-size: 24px;
}

/* Selectivity Gallery */
.selectivity-gallery .gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
}

.selectivity-gallery .category-card {
    background: var(--bg-inset, #f5f5f7);
    border-radius: 8px;
    padding: 16px;
}

.selectivity-gallery .category-header {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
}

.selectivity-gallery .fires .category-header { color: #16a34a; }
.selectivity-gallery .ignores .category-header { color: #9ca3af; }

.selectivity-gallery .gallery-example {
    font-size: 13px;
    padding: 8px 0;
    border-bottom: 1px solid var(--border, #e5e5e5);
    display: flex;
    justify-content: space-between;
}

.selectivity-gallery .gallery-example:last-child {
    border-bottom: none;
}

/* Ablation Matrix */
.ablation-matrix table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}

.ablation-matrix th {
    text-align: left;
    padding: 12px;
    border-bottom: 2px solid var(--border, #e5e5e5);
    font-weight: 600;
}

.ablation-matrix td {
    padding: 12px;
    border-bottom: 1px solid var(--border, #e5e5e5);
    vertical-align: top;
}

.ablation-matrix .prompt-cell {
    max-width: 300px;
    font-style: italic;
    color: var(--text-secondary, #555555);
}

.ablation-matrix .token-badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    margin: 2px;
}

.ablation-matrix .token-badge.promote {
    background: rgba(34, 197, 94, 0.1);
    color: #16a34a;
}

.ablation-matrix .token-badge.suppress {
    background: rgba(239, 68, 68, 0.1);
    color: #dc2626;
}

/* Steering Curves - Table format for dose-response */
.steering-curves .dose-response-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}

.steering-curves .dose-response-table th {
    text-align: left;
    padding: 10px 12px;
    border-bottom: 2px solid var(--border, #e5e5e5);
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-secondary, #555);
}

.steering-curves .dose-response-table td {
    padding: 10px 12px;
    border-bottom: 1px solid var(--border, #e5e5e5);
    vertical-align: middle;
}

.steering-curves .steering-value {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 14px;
    white-space: nowrap;
}

.steering-curves .steering-value.positive { color: #16a34a; }
.steering-curves .steering-value.negative { color: #dc2626; }

.steering-curves .effect-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.steering-curves .effect-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    border-radius: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
}

.steering-curves .effect-badge.promote {
    background: rgba(34, 197, 94, 0.1);
    color: #16a34a;
}

.steering-curves .effect-badge.suppress {
    background: rgba(239, 68, 68, 0.1);
    color: #dc2626;
}

.steering-curves .no-data {
    color: var(--text-tertiary, #888);
    font-style: italic;
}

/* Evidence Card */
.evidence-card {
    display: flex;
    gap: 16px;
    padding: 20px;
}

.evidence-card.confirmation {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    border-left: 4px solid #22c55e;
}

.evidence-card.refutation {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    border-left: 4px solid #ef4444;
}

.evidence-card.anomaly {
    background: linear-gradient(135deg, #fef3c7, #fde68a);
    border-left: 4px solid #f59e0b;
}

.evidence-card .evidence-icon {
    font-size: 24px;
    flex-shrink: 0;
}

.evidence-card .evidence-content {
    flex: 1;
}

.evidence-card .evidence-finding {
    font-size: 15px;
    font-weight: 500;
    margin-bottom: 8px;
}

.evidence-card .evidence-data {
    font-size: 13px;
    color: var(--text-secondary, #555555);
}

/* Anomaly Box */
.anomaly-box {
    background: linear-gradient(135deg, #fef3c7, #fde68a);
    border: 2px solid #f59e0b;
}

.anomaly-box .anomaly-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
}

.anomaly-box .anomaly-icon {
    font-size: 24px;
}

.anomaly-box .anomaly-title {
    font-size: 16px;
    font-weight: 600;
    color: #92400e;
}

.anomaly-box .comparison-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
}

.anomaly-box .comparison-item {
    padding: 12px;
    background: rgba(255, 255, 255, 0.5);
    border-radius: 8px;
}

.anomaly-box .comparison-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    color: #92400e;
    margin-bottom: 4px;
}

.anomaly-box .explanations-list {
    margin: 0;
    padding-left: 20px;
    font-size: 14px;
    color: var(--text-secondary, #555555);
}

.anomaly-box .explanations-list li {
    margin-bottom: 6px;
}

/* Output Projections - Compact badge format */
.output-projections .projection-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    margin-bottom: 12px;
}

.output-projections .projection-row-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    min-width: 80px;
}

.output-projections .promotes-row .projection-row-label { color: #16a34a; }
.output-projections .suppresses-row .projection-row-label { color: #dc2626; }

.output-projections .token-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    border-radius: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    white-space: nowrap;
}

.output-projections .promotes-row .token-badge {
    background: rgba(34, 197, 94, 0.1);
    color: #16a34a;
    border: 1px solid rgba(34, 197, 94, 0.2);
}

.output-projections .suppresses-row .token-badge {
    background: rgba(239, 68, 68, 0.1);
    color: #dc2626;
    border: 1px solid rgba(239, 68, 68, 0.2);
}

.output-projections .token-badge .token-name {
    font-weight: 500;
}

.output-projections .token-badge .token-weight {
    opacity: 0.8;
    font-size: 11px;
}

/* Homograph Comparison Grid - breaks out of container for more width */
.figure-container.homograph-comparison {
    margin-left: -100px;
    margin-right: -100px;
    padding-left: 24px;
    padding-right: 24px;
    max-width: none;
    width: calc(100% + 200px);
}

@media (max-width: 900px) {
    .figure-container.homograph-comparison {
        margin-left: 0;
        margin-right: 0;
        width: 100%;
    }
}

.homograph-grid {
    display: flex;
    gap: 24px;
    margin: 24px 0;
    flex-wrap: nowrap;
}

.homograph-pair {
    background: #f8f5f0;
    border-radius: 16px;
    overflow: hidden;
    padding: 20px 24px;
    flex: 1;
    min-width: 0;
}

.homograph-word {
    text-align: center;
    padding: 8px 0 12px;
    font-size: 26px;
    font-weight: 700;
    color: var(--text, #1a1a1a);
    letter-spacing: -0.5px;
}

.homograph-contexts {
    display: flex;
    gap: 0;
}

.homograph-context {
    padding: 12px 16px;
    text-align: center;
    flex: 1;
    min-width: 100px;
}

.homograph-context + .homograph-context {
    border-left: 1px solid rgba(0,0,0,0.08);
}

.context-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
    font-weight: 700;
    white-space: nowrap;
}

.context-label.malware { color: #9a6b4c; }
.context-label.biological { color: #7c5db5; }
.context-label.animal { color: #7c5db5; }
.context-label.mythology { color: #7c5db5; }
.context-label.neutral { color: #6b7280; }

.context-example {
    font-size: 12px;
    color: var(--text-secondary, #555);
    margin-bottom: 10px;
    font-style: italic;
    line-height: 1.35;
    min-height: 2.7em;
}

.context-activation {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 700;
}

.context-activation.high { color: #b8860b; }
.context-activation.low { color: #9ca3af; }

/* Stats Row */
.stats-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin: 20px 0;
}

.stat-card {
    background: #f8f5f0;
    border-radius: 16px;
    padding: 20px 28px;
    text-align: center;
    min-width: 140px;
    flex: 1;
}

.stat-card.highlight {
    background: #f0ebe3;
}

.stat-value {
    font-size: 36px;
    font-weight: 700;
    color: #1a1a1a;
    letter-spacing: -1px;
}

.stat-value.accent {
    color: #b8860b;
}

.stat-label {
    font-size: 12px;
    color: var(--text-secondary, #666);
    margin-top: 6px;
    line-height: 1.3;
}

/* Stacked Density Chart */
.density-chart-container {
    position: relative;
    margin: 20px 0;
}

.density-chart {
    background: var(--bg-elevated, #ffffff);
    border: 1px solid var(--border, #e5e5e5);
    border-radius: 14px;
    padding: 20px;
    overflow: hidden;
}

.density-chart canvas {
    width: 100%;
    height: auto;
    display: block;
}

.category-legend {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    margin: 16px 0;
    padding: 12px 16px;
    background: var(--bg-inset, #f5f5f7);
    border-radius: 8px;
}

.category-item {
    display: flex;
    align-items: center;
    gap: 8px;
}

.category-color {
    width: 16px;
    height: 16px;
    border-radius: 4px;
}

.category-name {
    font-weight: 600;
    font-size: 13px;
}

.density-bars {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.density-bar-row {
    display: flex;
    align-items: center;
    gap: 12px;
    height: 28px;
}

.density-zscore {
    width: 60px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    text-align: right;
    color: var(--text-secondary, #666);
}

.density-bar-container {
    flex: 1;
    height: 20px;
    display: flex;
    border-radius: 4px;
    overflow: hidden;
}

.density-segment {
    height: 100%;
    transition: width 0.3s ease;
}

.density-count {
    width: 50px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-tertiary, #888);
}

/* Patching Comparison */
.patching-comparison {
    max-width: none;
}

.patching-experiments-grid {
    display: flex;
    flex-direction: column;
    gap: 20px;
    margin: 20px 0;
}

.patching-experiment {
    background: var(--bg-inset, #f8f5f0);
    border-radius: 12px;
    padding: 20px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
}

@media (max-width: 768px) {
    .patching-experiment {
        grid-template-columns: 1fr;
    }
}

.patching-prompts {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.prompt-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 14px;
    background: var(--bg-elevated, #ffffff);
    border-radius: 8px;
}

.prompt-row.source {
    border-left: 3px solid #b8860b;
}

.prompt-row.target {
    border-left: 3px solid #9ca3af;
}

.prompt-label {
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    color: var(--text-tertiary, #888);
    min-width: 50px;
}

.prompt-text {
    flex: 1;
    font-size: 13px;
    color: var(--text-secondary, #555);
    font-style: italic;
    line-height: 1.4;
}

.activation-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
}

.activation-badge.high {
    background: rgba(184, 134, 11, 0.15);
    color: #b8860b;
}

.activation-badge.low {
    background: rgba(156, 163, 175, 0.15);
    color: #6b7280;
}

.patching-arrow {
    text-align: center;
    font-size: 12px;
    color: var(--text-tertiary, #888);
    padding: 4px 0;
}

.patching-effects {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.patching-effects .effect-row {
    display: flex;
    align-items: center;
    gap: 12px;
}

.patching-effects .effect-label {
    font-size: 12px;
    color: var(--text-secondary, #666);
    min-width: 90px;
}

.patching-effects .effect-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 16px;
    font-weight: 600;
}

.patching-effects .effect-value.positive { color: #16a34a; }
.patching-effects .effect-value.negative { color: #dc2626; }
.patching-effects .effect-value.neutral { color: #6b7280; }

.patching-effects .effect-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.patching-effects .effect-summary {
    margin-top: 8px;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    text-align: center;
}

.patching-effects .effect-summary.strong {
    background: rgba(22, 163, 74, 0.1);
    color: #16a34a;
}

.patching-effects .effect-summary.moderate {
    background: rgba(234, 179, 8, 0.1);
    color: #ca8a04;
}

.patching-effects .effect-summary.weak {
    background: rgba(156, 163, 175, 0.1);
    color: #6b7280;
}

/* Investigation Flow Visualization */
.investigation-flow .flow-timeline {
    display: flex;
    flex-direction: column;
    gap: 0;
}

.investigation-flow .flow-phase {
    background: var(--bg-inset, #f5f5f7);
    border-radius: 8px;
    padding: 16px 20px;
    margin: 0;
    border-left: 4px solid #6c757d;
}

.investigation-flow .flow-phase.initial-hypotheses {
    border-left-color: #6c757d;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
}

.investigation-flow .flow-phase.exploration {
    border-left-color: #0d6efd;
    background: linear-gradient(135deg, #e7f1ff 0%, #cfe2ff 100%);
}

.investigation-flow .flow-phase.skeptic {
    border-left-color: #dc3545;
    background: linear-gradient(135deg, #fff5f5 0%, #ffe5e5 100%);
}

.investigation-flow .flow-phase.reviewer {
    border-left-color: #ffc107;
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
}

.investigation-flow .flow-phase.conclusion {
    border-left-color: #198754;
    background: linear-gradient(135deg, #d1e7dd 0%, #a3cfbb 100%);
}

.investigation-flow .flow-connector {
    text-align: center;
    color: var(--text-tertiary, #6c757d);
    font-size: 18px;
    padding: 4px 0;
    line-height: 1;
}

.investigation-flow .phase-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}

.investigation-flow .phase-header h4 {
    font-size: 14px;
    font-weight: 600;
    color: var(--text, #111);
    margin: 0;
    flex: 1;
}

.investigation-flow .phase-icon {
    font-size: 16px;
}

.investigation-flow .experiment-count,
.investigation-flow .iteration-count {
    font-size: 11px;
    color: var(--text-tertiary, #888);
    background: rgba(0,0,0,0.05);
    padding: 2px 8px;
    border-radius: 10px;
}

.investigation-flow .hypothesis-bar {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 10px 12px;
    background: white;
    border-radius: 6px;
    margin-bottom: 8px;
    border: 1px solid rgba(0,0,0,0.06);
}

.investigation-flow .hypothesis-bar:last-child {
    margin-bottom: 0;
}

.investigation-flow .hypothesis-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-secondary, #555);
}

.investigation-flow .hypothesis-bar.input-trigger .hypothesis-label {
    color: #0d6efd;
}

.investigation-flow .hypothesis-bar.output-function .hypothesis-label {
    color: #198754;
}

.investigation-flow .hypothesis-description {
    font-size: 13px;
    color: var(--text, #111);
    line-height: 1.4;
}

.investigation-flow .experiment-cards {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.investigation-flow .experiment-card {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 8px 12px;
    background: white;
    border-radius: 6px;
    border: 1px solid rgba(0,0,0,0.08);
    min-width: 120px;
    max-width: 200px;
}

.investigation-flow .experiment-card.positive {
    border-color: rgba(25, 135, 84, 0.3);
}

.investigation-flow .experiment-card.negative {
    border-color: rgba(220, 53, 69, 0.3);
}

.investigation-flow .experiment-card.neutral {
    border-color: rgba(108, 117, 125, 0.3);
}

.investigation-flow .exp-type {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--text-tertiary, #888);
}

.investigation-flow .exp-input {
    font-size: 12px;
    color: var(--text-secondary, #555);
    font-style: italic;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.investigation-flow .exp-result {
    font-size: 12px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

.investigation-flow .experiment-card.positive .exp-result {
    color: #198754;
}

.investigation-flow .experiment-card.negative .exp-result {
    color: #dc3545;
}

.investigation-flow .challenges-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 10px;
}

.investigation-flow .challenge-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 8px 10px;
    background: white;
    border-radius: 6px;
    font-size: 12px;
}

.investigation-flow .challenge-item.refuted {
    border-left: 3px solid #dc3545;
}

.investigation-flow .challenge-item.partial {
    border-left: 3px solid #ffc107;
}

.investigation-flow .challenge-item.supported {
    border-left: 3px solid #198754;
}

.investigation-flow .challenge-icon {
    font-size: 14px;
    flex-shrink: 0;
}

.investigation-flow .challenge-text {
    color: var(--text-secondary, #555);
    line-height: 1.4;
}

.investigation-flow .boundary-summary {
    display: flex;
    gap: 16px;
}

.investigation-flow .boundary-stat {
    font-size: 12px;
    font-weight: 500;
}

.investigation-flow .boundary-stat.passed {
    color: #198754;
}

.investigation-flow .boundary-stat.failed {
    color: #dc3545;
}

.investigation-flow .verdict-badge,
.investigation-flow .confidence-badge {
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 12px;
    text-transform: uppercase;
}

.investigation-flow .verdict-badge.supported {
    background: rgba(25, 135, 84, 0.15);
    color: #198754;
}

.investigation-flow .verdict-badge.modified {
    background: rgba(255, 193, 7, 0.2);
    color: #856404;
}

.investigation-flow .confidence-badge {
    background: rgba(25, 135, 84, 0.15);
    color: #198754;
}

.investigation-flow .review-gaps ul {
    margin: 0;
    padding-left: 20px;
    font-size: 12px;
    color: var(--text-secondary, #555);
}

.investigation-flow .review-gaps li {
    margin-bottom: 4px;
}

.investigation-flow .review-verdict {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid rgba(0,0,0,0.1);
}

.investigation-flow .verdict-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-tertiary, #888);
}

.investigation-flow .verdict-value {
    font-size: 12px;
    font-weight: 600;
    color: var(--text, #111);
}

.investigation-flow .final-hypothesis {
    font-size: 13px;
    color: var(--text, #111);
    line-height: 1.5;
    padding: 10px 12px;
    background: white;
    border-radius: 6px;
    border: 1px solid rgba(0,0,0,0.06);
}

.investigation-flow .key-findings {
    margin-top: 10px;
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.investigation-flow .finding-item {
    font-size: 11px;
    color: var(--text-secondary, #555);
    padding: 4px 8px;
    background: rgba(255,255,255,0.5);
    border-radius: 4px;
    border-left: 2px solid #198754;
}

.investigation-flow .no-data,
.investigation-flow .no-challenges {
    font-size: 12px;
    color: var(--text-tertiary, #888);
    font-style: italic;
    padding: 8px;
}

/* Investigation Flow - Hypothesis Box (full text) */
.investigation-flow .hypothesis-box {
    padding: 12px 16px;
    background: white;
    border-radius: 8px;
    margin-bottom: 10px;
    border: 1px solid rgba(0,0,0,0.06);
}

.investigation-flow .hypothesis-box:last-child {
    margin-bottom: 0;
}

.investigation-flow .hypothesis-box .hypothesis-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #0d6efd;
    margin-bottom: 6px;
}

.investigation-flow .hypothesis-box .hypothesis-text {
    font-size: 13px;
    color: var(--text, #111);
    line-height: 1.5;
}

/* Investigation Flow - Hypothesis Updates Phase */
.investigation-flow .flow-phase.hypothesis-updates {
    border-left-color: #6366f1;
    background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
}

.investigation-flow .update-count {
    font-size: 11px;
    color: var(--text-tertiary, #888);
    background: rgba(0,0,0,0.05);
    padding: 2px 8px;
    border-radius: 10px;
}

.investigation-flow .updates-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.investigation-flow .hypothesis-update {
    padding: 14px 16px;
    background: white;
    border-radius: 8px;
    border-left: 4px solid #6c757d;
}

.investigation-flow .hypothesis-update.confirmed {
    border-left-color: #198754;
}

.investigation-flow .hypothesis-update.refuted {
    border-left-color: #dc3545;
}

.investigation-flow .hypothesis-update.inconclusive {
    border-left-color: #ffc107;
}

.investigation-flow .hypothesis-update-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
    flex-wrap: wrap;
}

.investigation-flow .hypothesis-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary, #555);
    background: rgba(0,0,0,0.05);
    padding: 2px 8px;
    border-radius: 4px;
}

.investigation-flow .hypothesis-status {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
}

.investigation-flow .hypothesis-status.confirmed { color: #198754; }
.investigation-flow .hypothesis-status.refuted { color: #dc3545; }
.investigation-flow .hypothesis-status.inconclusive { color: #856404; }

.investigation-flow .probability-change {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    margin-left: auto;
}

.investigation-flow .probability-change.increased { color: #198754; }
.investigation-flow .probability-change.decreased { color: #dc3545; }
.investigation-flow .probability-change.unchanged { color: #6c757d; }

.investigation-flow .hypothesis-update-text {
    font-size: 13px;
    color: var(--text, #111);
    line-height: 1.5;
}

.investigation-flow .hypothesis-evidence {
    font-size: 12px;
    color: var(--text-secondary, #555);
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid rgba(0,0,0,0.06);
    line-height: 1.4;
}

/* Investigation Flow - Exploration Categories */
.investigation-flow .exploration-categories {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.investigation-flow .exploration-category {
    background: white;
    border-radius: 8px;
    padding: 14px 16px;
    border: 1px solid rgba(0,0,0,0.06);
}

.investigation-flow .exploration-category-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    flex-wrap: wrap;
    gap: 8px;
}

.investigation-flow .category-name {
    font-size: 13px;
    font-weight: 600;
    color: var(--text, #111);
}

.investigation-flow .category-stats {
    font-size: 11px;
    color: var(--text-tertiary, #888);
    font-family: 'JetBrains Mono', monospace;
}

.investigation-flow .exploration-examples {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.investigation-flow .experiment-example {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    padding: 8px 10px;
    background: var(--bg-inset, #f5f5f7);
    border-radius: 6px;
    border-left: 3px solid #6c757d;
}

.investigation-flow .experiment-example.positive {
    border-left-color: #198754;
}

.investigation-flow .experiment-example.negative {
    border-left-color: #dc3545;
}

.investigation-flow .example-prompt {
    font-size: 12px;
    color: var(--text-secondary, #555);
    flex: 1;
    line-height: 1.4;
}

.investigation-flow .example-result {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-tertiary, #888);
    white-space: nowrap;
}

/* Investigation Flow - Skeptic Section Improvements */
.investigation-flow .section-subheader {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-secondary, #555);
    margin-bottom: 10px;
}

.investigation-flow .alternatives-section,
.investigation-flow .boundary-tests-section,
.investigation-flow .revised-hypothesis {
    margin-bottom: 16px;
}

.investigation-flow .alternatives-section:last-child,
.investigation-flow .boundary-tests-section:last-child,
.investigation-flow .revised-hypothesis:last-child {
    margin-bottom: 0;
}

.investigation-flow .alternative-hypothesis {
    padding: 12px 14px;
    background: white;
    border-radius: 8px;
    margin-bottom: 8px;
    border-left: 4px solid #6c757d;
}

.investigation-flow .alternative-hypothesis.refuted {
    border-left-color: #dc3545;
}

.investigation-flow .alternative-hypothesis.partial {
    border-left-color: #ffc107;
}

.investigation-flow .alternative-hypothesis.supported {
    border-left-color: #198754;
}

.investigation-flow .alternative-header {
    margin-bottom: 6px;
}

.investigation-flow .alternative-verdict {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
}

.investigation-flow .alternative-hypothesis.refuted .alternative-verdict { color: #dc3545; }
.investigation-flow .alternative-hypothesis.partial .alternative-verdict { color: #856404; }
.investigation-flow .alternative-hypothesis.supported .alternative-verdict { color: #198754; }

.investigation-flow .alternative-text {
    font-size: 13px;
    color: var(--text, #111);
    line-height: 1.4;
    margin-bottom: 6px;
}

.investigation-flow .alternative-evidence {
    font-size: 12px;
    color: var(--text-secondary, #555);
    line-height: 1.4;
}

/* Boundary Tests */
.investigation-flow .boundary-section {
    margin-bottom: 12px;
}

.investigation-flow .boundary-section-header {
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 8px;
}

.investigation-flow .failed-section .boundary-section-header {
    color: #dc3545;
}

.investigation-flow .passed-section .boundary-section-header {
    color: #198754;
}

.investigation-flow .boundary-test {
    padding: 10px 12px;
    background: white;
    border-radius: 6px;
    margin-bottom: 6px;
    border-left: 3px solid #6c757d;
}

.investigation-flow .boundary-test.failed {
    border-left-color: #dc3545;
}

.investigation-flow .boundary-test.passed {
    border-left-color: #198754;
}

.investigation-flow .boundary-test-header {
    font-size: 13px;
    font-weight: 500;
    color: var(--text, #111);
    margin-bottom: 6px;
}

.investigation-flow .boundary-test-detail {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 12px;
    color: var(--text-secondary, #555);
}

.investigation-flow .test-prompt {
    font-style: italic;
}

.investigation-flow .test-expected,
.investigation-flow .test-actual {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
}

.investigation-flow .boundary-test-notes {
    font-size: 11px;
    color: var(--text-tertiary, #888);
    margin-top: 6px;
    padding-top: 6px;
    border-top: 1px solid rgba(0,0,0,0.06);
}

.investigation-flow .boundary-test-compact {
    font-size: 12px;
    padding: 6px 10px;
    background: rgba(255,255,255,0.7);
    border-radius: 4px;
    margin-bottom: 4px;
}

.investigation-flow .boundary-test-compact.passed {
    color: #198754;
}

.investigation-flow .revised-text {
    font-size: 13px;
    color: var(--text, #111);
    line-height: 1.5;
    padding: 12px 14px;
    background: white;
    border-radius: 8px;
    border: 1px solid rgba(0,0,0,0.06);
}

/* Reviewer Section Improvements */
.investigation-flow .reviews-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 12px;
}

.investigation-flow .review-iteration {
    padding: 12px 14px;
    background: white;
    border-radius: 8px;
    border: 1px solid rgba(0,0,0,0.06);
}

.investigation-flow .review-iteration-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
    flex-wrap: wrap;
}

.investigation-flow .iteration-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary, #555);
}

.investigation-flow .iteration-verdict {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    color: #856404;
}

.investigation-flow .confidence-assessment {
    font-size: 11px;
    color: #dc3545;
    background: rgba(220, 53, 69, 0.1);
    padding: 2px 8px;
    border-radius: 4px;
}

.investigation-flow .review-gaps {
    margin: 0;
    padding-left: 20px;
    font-size: 12px;
    color: var(--text-secondary, #555);
    line-height: 1.5;
}

.investigation-flow .review-gaps li {
    margin-bottom: 4px;
}

.investigation-flow .final-verdict-box {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: white;
    border-radius: 8px;
    border: 1px solid rgba(0,0,0,0.06);
}

.investigation-flow .final-verdict-box .verdict-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary, #555);
}

.investigation-flow .final-verdict-box .verdict-value {
    font-size: 13px;
    font-weight: 600;
    color: var(--text, #111);
}

/* Conclusion Section Improvements */
.investigation-flow .final-hypothesis-box {
    padding: 14px 16px;
    background: white;
    border-radius: 8px;
    border: 1px solid rgba(0,0,0,0.06);
    margin-bottom: 12px;
}

.investigation-flow .final-hypothesis-text {
    font-size: 13px;
    color: var(--text, #111);
    line-height: 1.5;
}

.investigation-flow .key-findings-box {
    padding: 14px 16px;
    background: rgba(255,255,255,0.7);
    border-radius: 8px;
}

.investigation-flow .findings-list {
    margin: 0;
    padding-left: 20px;
    font-size: 12px;
    color: var(--text-secondary, #555);
    line-height: 1.5;
}

.investigation-flow .findings-list li {
    margin-bottom: 6px;
}

.investigation-flow .findings-list li:last-child {
    margin-bottom: 0;
}

/* Claims to Verify phase - the initial hypotheses from prior LLM analysis */
.investigation-flow .flow-phase.claims-to-verify {
    border-left-color: #6f42c1;  /* Purple for "claims" */
    background: linear-gradient(135deg, #f8f5ff 0%, #f8f9fa 100%);
}

.investigation-flow .flow-phase.claims-to-verify.no-claims {
    background: #f8f9fa;
    border-left-color: #6c757d;
}

.investigation-flow .phase-subtitle {
    font-size: 0.75rem;
    color: var(--text-tertiary, #6c757d);
    font-weight: normal;
    margin-left: 8px;
}

.investigation-flow .phase-note {
    font-size: 0.85rem;
    color: var(--text-secondary, #495057);
    padding: 8px 12px;
    background: rgba(111, 66, 193, 0.05);
    border-radius: 6px;
    margin-bottom: 16px;
    line-height: 1.5;
}

.investigation-flow .flow-phase.no-claims .phase-note {
    background: rgba(108, 117, 125, 0.05);
}

.investigation-flow .claim-box {
    background: white;
    border: 1px solid #e0d4f7;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
}

.investigation-flow .claim-box:last-child {
    margin-bottom: 0;
}

.investigation-flow .claim-type {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #6f42c1;
    margin-bottom: 6px;
}

.investigation-flow .claim-label {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary, #212529);
    margin-bottom: 6px;
}

.investigation-flow .claim-description {
    font-size: 0.9rem;
    color: var(--text-secondary, #495057);
    line-height: 1.5;
}

/* Final conclusion - characterization boxes */
.investigation-flow .final-characterization {
    margin-top: 16px;
}

.investigation-flow .conclusion-box {
    background: linear-gradient(135deg, #f0fff4 0%, #f8f9fa 100%);
    border: 1px solid #c3e6cb;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
}

.investigation-flow .conclusion-box:last-child {
    margin-bottom: 0;
}

.investigation-flow .conclusion-type {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #198754;
    margin-bottom: 6px;
}

.investigation-flow .conclusion-text {
    font-size: 0.95rem;
    color: var(--text-primary, #212529);
    line-height: 1.5;
}

.investigation-flow .conclusion-change-note {
    font-size: 0.75rem;
    color: #856404;
    background: #fff3cd;
    padding: 3px 8px;
    border-radius: 4px;
    display: inline-block;
    margin-top: 8px;
}
"""


# =============================================================================
# FIGURE GENERATION FUNCTIONS
# =============================================================================

def _get_example_text(ex: dict[str, Any]) -> str:
    """Extract text from an example dict, trying various field names."""
    # Try common field names for the text content
    for field in ["prompt", "text", "example", "content", "input", "sentence"]:
        if ex.get(field):
            return ex[field]
    # If nothing found, return empty string
    return ""


def generate_activation_grid(
    high_examples: list[dict[str, Any]],
    low_examples: list[dict[str, Any]],
    title: str = "Activation Comparison",
    caption: str = ""
) -> str:
    """Generate side-by-side comparison of high vs low activation examples."""

    high_html = ""
    for ex in high_examples[:5]:
        prompt = escape_html(_get_example_text(ex))
        activation = ex.get("activation", 0)
        highlighted = ex.get("highlighted_token", ex.get("token", ""))
        if highlighted and prompt:
            # Try to highlight the token in the prompt
            prompt = prompt.replace(escape_html(highlighted), f"<mark>{escape_html(highlighted)}</mark>")
        high_html += f'''
        <div class="example">
            <span class="prompt">{prompt if prompt else "<em>No text provided</em>"}</span>
            <span class="activation high">{activation:.2f}</span>
        </div>
        '''

    low_html = ""
    for ex in low_examples[:5]:
        prompt = escape_html(_get_example_text(ex))
        activation = ex.get("activation", 0)
        highlighted = ex.get("highlighted_token", ex.get("token", ""))
        if highlighted and prompt:
            # Try to highlight the token in the prompt
            prompt = prompt.replace(escape_html(highlighted), f"<mark>{escape_html(highlighted)}</mark>")
        low_html += f'''
        <div class="example">
            <span class="prompt">{prompt if prompt else "<em>No text provided</em>"}</span>
            <span class="activation low">{activation:.2f}</span>
        </div>
        '''

    caption_html = f'<p class="figure-caption">{escape_html(caption)}</p>' if caption else ""

    return f'''
    <div class="figure-container activation-grid">
        <div class="figure-title">{escape_html(title)}</div>
        <div class="grid-row">
            <div class="grid-col fires">
                <div class="col-header">Activates</div>
                {high_html}
            </div>
            <div class="grid-col ignores">
                <div class="col-header">Silent</div>
                {low_html}
            </div>
        </div>
        {caption_html}
    </div>
    '''


def generate_token_bar_chart(
    promoted_tokens: list[dict[str, Any]],
    suppressed_tokens: list[dict[str, Any]],
    title: str = "Token Effects",
    caption: str = "",
    max_magnitude: float = None
) -> str:
    """Generate horizontal bar chart showing tokens promoted/suppressed."""

    # Calculate max magnitude for scaling
    all_magnitudes = (
        [abs(t.get("magnitude", 0)) for t in promoted_tokens] +
        [abs(t.get("magnitude", 0)) for t in suppressed_tokens]
    )
    if max_magnitude is None:
        max_magnitude = max(all_magnitudes) if all_magnitudes else 1.0

    bars_html = ""

    # Promoted tokens (green, right side)
    for t in promoted_tokens[:8]:
        token = clean_token(t.get("token", ""))
        magnitude = t.get("magnitude", 0)
        width = min(45, (abs(magnitude) / max_magnitude) * 45)  # Max 45% width
        bars_html += f'''
        <div class="bar-row">
            <span class="token-label">{escape_html(token)}</span>
            <div class="bar-area">
                <div class="center-line"></div>
                <div class="bar promote" style="width: {width}%;"></div>
                <span class="magnitude" style="left: calc(50% + {width}% + 8px);">+{magnitude:.3f}</span>
            </div>
        </div>
        '''

    # Suppressed tokens (red, left side)
    for t in suppressed_tokens[:8]:
        token = clean_token(t.get("token", ""))
        magnitude = abs(t.get("magnitude", 0))
        width = min(45, (magnitude / max_magnitude) * 45)
        bars_html += f'''
        <div class="bar-row">
            <span class="token-label">{escape_html(token)}</span>
            <div class="bar-area">
                <div class="center-line"></div>
                <div class="bar suppress" style="width: {width}%; right: 50%;"></div>
                <span class="magnitude" style="right: calc(50% + {width}% + 8px);">-{magnitude:.3f}</span>
            </div>
        </div>
        '''

    caption_html = f'<p class="figure-caption">{escape_html(caption)}</p>' if caption else ""

    return f'''
    <div class="figure-container token-chart">
        <div class="figure-title">{escape_html(title)}</div>
        <div class="chart-container">
            {bars_html}
        </div>
        {caption_html}
    </div>
    '''


def generate_hypothesis_timeline(
    hypotheses: list[dict[str, Any]],
    title: str = "Hypothesis Evolution",
    caption: str = ""
) -> str:
    """Generate visual showing prior->posterior probability evolution."""

    items_html = ""
    for h in hypotheses:
        status = h.get("status")
        text = h.get("text", h.get("hypothesis", ""))[:150]
        prior = h.get("prior", h.get("prior_probability", 50))
        posterior = h.get("posterior", h.get("posterior_probability", prior))

        # Convert decimals to percentages if needed (0.65 -> 65)
        # Values 0 < x < 1 are treated as fractions; values >= 1 are already percentages
        if isinstance(prior, (int, float)) and 0 < prior < 1:
            prior = int(prior * 100)
        else:
            prior = int(prior) if isinstance(prior, (int, float)) else 50
        if isinstance(posterior, (int, float)) and 0 < posterior < 1:
            posterior = int(posterior * 100)
        else:
            posterior = int(posterior) if isinstance(posterior, (int, float)) else 50

        # Infer status from posterior probability when status is missing/null
        if not status:
            if posterior >= 80:
                status = "supported"
            elif posterior <= 20:
                status = "refuted"
            elif posterior < prior:
                status = "weakened"
            else:
                status = "inconclusive"

        # Override status when it contradicts the probability direction
        # (agent sometimes writes "weakened" even when posterior increased)
        if status and status.lower() == "weakened" and posterior > prior:
            status = "inconclusive" if posterior < 70 else "supported"
        elif status and status.lower() in ("supported", "confirmed") and posterior < prior and posterior < 50:
            status = "weakened"

        # Map status to display class and text
        status_lower = status.lower() if status else "testing"
        if status_lower == "confirmed":
            status_class = "confirmed"
            status_text = "CONFIRMED"
        elif status_lower == "supported":
            status_class = "confirmed"
            status_text = "SUPPORTED"
        elif status_lower == "partially_supported":
            status_class = "inconclusive"
            status_text = "PARTIALLY SUPPORTED"
        elif status_lower == "refuted":
            status_class = "refuted"
            status_text = "REFUTED"
        elif status_lower == "inconclusive":
            status_class = "inconclusive"
            status_text = "INCONCLUSIVE"
        elif status_lower == "weakened":
            status_class = "weakened"
            status_text = "WEAKENED"
        elif status_lower == "revised":
            status_class = "testing"
            status_text = "REVISED"
        elif status_lower == "registered":
            status_class = "registered"
            status_text = "REGISTERED"
        else:
            status_class = "testing"
            status_text = "TESTING"

        items_html += f'''
        <div class="timeline-item {status_class}">
            <span class="status-badge">{status_text}</span>
            <span class="hypothesis-text">{escape_html(text)}</span>
            <div class="probability-shift">
                <span>{prior}%</span>
                <span class="prob-arrow">→</span>
                <span>{posterior}%</span>
            </div>
        </div>
        '''

    caption_html = f'<p class="figure-caption">{escape_html(caption)}</p>' if caption else ""

    return f'''
    <div class="figure-container hypothesis-timeline">
        <div class="figure-title">{escape_html(title)}</div>
        {items_html}
        {caption_html}
    </div>
    '''


def generate_circuit_diagram(
    upstream_neurons: list[dict[str, Any]],
    downstream_neurons: list[dict[str, Any]],
    center_neuron_id: str,
    center_neuron_label: str = "",
    title: str = "Circuit Connectivity",
    caption: str = ""
) -> str:
    """Generate upstream -> Neuron -> downstream flow diagram."""

    upstream_html = ""
    for n in upstream_neurons[:4]:
        nid = n.get("id", n.get("neuron_id", ""))
        label = n.get("label", "Unknown")[:50]
        weight = n.get("weight", 0)
        linked_id = linkify_neuron_ids(nid)
        upstream_html += f'''
        <div class="node-item">
            <div class="node-label">{escape_html(label)}</div>
            <div class="node-id">{linked_id}</div>
            <div class="node-weight">weight: {weight:+.3f}</div>
        </div>
        '''

    downstream_html = ""
    for n in downstream_neurons[:4]:
        nid = n.get("id", n.get("neuron_id", ""))
        label = n.get("label", "Unknown")[:50]
        weight = n.get("weight", 0)
        linked_id = linkify_neuron_ids(nid)
        downstream_html += f'''
        <div class="node-item">
            <div class="node-label">{escape_html(label)}</div>
            <div class="node-id">{linked_id}</div>
            <div class="node-weight">weight: {weight:+.3f}</div>
        </div>
        '''

    caption_html = f'<p class="figure-caption">{escape_html(caption)}</p>' if caption else ""

    # Compute appropriate downstream fallback message based on layer
    if not downstream_html:
        try:
            source_layer = int(center_neuron_id.split("/")[0][1:]) if "/" in center_neuron_id else 0
        except (ValueError, IndexError):
            source_layer = 0
        if source_layer >= 31:
            downstream_fallback = '<div class="node-item"><div class="node-label">Final layer: projects to logits</div></div>'
        else:
            downstream_fallback = f'<div class="node-item"><div class="node-label">No identified downstream neurons (L{source_layer + 1}-31)</div></div>'
    else:
        downstream_fallback = downstream_html

    return f'''
    <div class="figure-container circuit-diagram">
        <div class="figure-title">{escape_html(title)}</div>
        <div class="circuit-flow">
            <div class="node-column upstream">
                {upstream_html if upstream_html else '<div class="node-item"><div class="node-label">Layer 0: connects to embeddings</div></div>'}
            </div>
            <div class="arrow-column">
                <span>→</span>
                <span>→</span>
                <span>→</span>
            </div>
            <div class="center-node">
                <div class="center-id">{center_neuron_id}</div>
                {f'<div class="center-label">{escape_html(center_neuron_label)}</div>' if center_neuron_label else ''}
            </div>
            <div class="arrow-column">
                <span>→</span>
                <span>→</span>
                <span>→</span>
            </div>
            <div class="node-column downstream">
                {downstream_fallback}
            </div>
        </div>
        {caption_html}
    </div>
    '''


def generate_selectivity_gallery(
    categories: list[dict[str, Any]],
    title: str = "Selectivity Patterns",
    caption: str = ""
) -> str:
    """Generate grid of examples organized by category."""

    cards_html = ""
    for cat in categories[:6]:
        # Try to get label, fall back to name, then generate from first example
        label = cat.get("label") or cat.get("name")
        if not label:
            # Generate label from first example text (first few words)
            examples_list = cat.get("examples", [])
            if examples_list:
                first_text = examples_list[0].get("text", "")[:30].strip()
                if first_text:
                    label = f'"{first_text}..."' if len(first_text) >= 28 else f'"{first_text}"'
        label = label or "Examples"  # Final fallback
        cat_type = cat.get("type", "fires")
        examples = cat.get("examples", [])

        examples_html = ""
        for ex in examples[:4]:
            text = ex.get("text", "")[:60]
            activation = ex.get("activation", 0)

            # Highlight the token that triggered activation
            highlighted_token = ex.get("highlighted_token", ex.get("token", ""))
            display_text = escape_html(text)
            if highlighted_token and text:
                # Try to highlight the token in the text
                escaped_token = escape_html(highlighted_token)
                if escaped_token in display_text:
                    display_text = display_text.replace(escaped_token, f"<mark>{escaped_token}</mark>", 1)

            examples_html += f'''
            <div class="gallery-example">
                <span>{display_text}</span>
                <span style="font-family: monospace; font-size: 11px; color: {'#16a34a' if cat_type == 'fires' else '#9ca3af'};">{activation:.2f}</span>
            </div>
            '''

        cards_html += f'''
        <div class="category-card {cat_type}">
            <div class="category-header">{'✓' if cat_type == 'fires' else '✗'} {escape_html(label)}</div>
            {examples_html}
        </div>
        '''

    caption_html = f'<p class="figure-caption">{escape_html(caption)}</p>' if caption else ""

    return f'''
    <div class="figure-container selectivity-gallery">
        <div class="figure-title">{escape_html(title)}</div>
        <div class="gallery-grid">
            {cards_html}
        </div>
        {caption_html}
    </div>
    '''


def generate_ablation_matrix(
    experiments: list[dict[str, Any]],
    title: str = "Ablation Effects",
    caption: str = "",
    total_prompts: int = 0,
    change_rate: float = 0.0
) -> str:
    """Generate table showing token effects across prompts.

    Args:
        experiments: List of ablation experiment results
        title: Figure title
        caption: Optional caption
        total_prompts: Total prompts tested (for batch experiments)
        change_rate: Fraction of prompts where output changed
    """

    # Collect rows for potential collapsing
    row_list = []
    for exp in experiments[:8]:
        prompt = exp.get("prompt", "")[:50]
        promotes = exp.get("promotes", [])
        suppresses = exp.get("suppresses", [])

        promotes_html = "".join([
            f'<span class="token-badge promote">{clean_token(t[0])} +{t[1]:.2f}</span>'
            for t in promotes[:4] if len(t) >= 2
        ]) or '<span style="color: #9ca3af;">—</span>'

        suppresses_html = "".join([
            f'<span class="token-badge suppress">{clean_token(t[0])} {t[1]:.2f}</span>'
            for t in suppresses[:4] if len(t) >= 2
        ]) or '<span style="color: #9ca3af;">—</span>'

        row_list.append(f'''
        <tr>
            <td class="prompt-cell">"{escape_html(prompt)}..."</td>
            <td>{promotes_html}</td>
            <td>{suppresses_html}</td>
        </tr>
        ''')

    caption_html = f'<p class="figure-caption">{escape_html(caption)}</p>' if caption else ""

    # Sample size and change rate indicator
    sample_size_html = ""
    n_shown = min(len(experiments), 8)
    n_total = total_prompts if total_prompts > 0 else len(experiments)

    if n_total > 0:
        batch_label = "BATCH TEST" if n_total >= 10 else "LIMITED SAMPLE"
        batch_color = "#16a34a" if n_total >= 10 else "#f97316"

        change_info = ""
        if change_rate > 0 or total_prompts > 0:
            pct = change_rate * 100 if change_rate else 0
            change_info = f' • <strong>{pct:.1f}%</strong> of outputs changed'

        sample_size_html = f'''
        <div class="sample-size-badge" style="display: inline-flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            <span style="background: {batch_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">{batch_label}</span>
            <span style="color: var(--text-secondary, #666); font-size: 13px;">
                Tested on <strong>{n_total}</strong> prompts (showing {n_shown}){change_info}
            </span>
        </div>
        '''

    # Collapsible for tables with many rows
    row_count = len(row_list)
    collapse_threshold = 4
    show_first = 2

    if row_count > collapse_threshold:
        visible_rows = '\n'.join(row_list[:show_first])
        hidden_rows = '\n'.join(row_list[show_first:])
        rows_html = visible_rows
        hidden_tbody = f'<tbody class="table-rows-hidden">{hidden_rows}</tbody>'
        hidden_count = row_count - show_first
    else:
        rows_html = '\n'.join(row_list)
        hidden_tbody = ""
        hidden_count = 0

    # Build table HTML
    table_html = f'''
        <table>
            <thead>
                <tr>
                    <th>Prompt</th>
                    <th>Promotes (when removed)</th>
                    <th>Suppresses (when removed)</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
            {hidden_tbody}
        </table>
    '''

    # Collapsible wrapper if needed
    if hidden_count > 0:
        import hashlib
        unique_id = hashlib.md5(f"ablation-{title}".encode()).hexdigest()[:8]
        collapsible_wrapper = f'''
        <div class="table-collapsible" id="tc-{unique_id}">
            <style>
                #tc-{unique_id} .table-rows-hidden {{ display: none; }}
                #tc-{unique_id}.expanded .table-rows-hidden {{ display: table-row-group; }}
                #tc-{unique_id}.expanded .expand-btn {{ display: none; }}
                #tc-{unique_id} .collapse-btn {{ display: none; }}
                #tc-{unique_id}.expanded .collapse-btn {{ display: inline-flex; }}
            </style>
            {table_html}
            <div style="text-align: center; margin-top: 12px;">
                <button class="expand-btn" onclick="document.getElementById('tc-{unique_id}').classList.add('expanded')"
                    style="background: var(--bg-secondary, #f3f4f6); border: 1px solid var(--border, #e5e7eb);
                           border-radius: 6px; padding: 6px 16px; font-size: 13px; color: var(--text-secondary, #666);
                           cursor: pointer; display: inline-flex; align-items: center; gap: 6px;">
                    <span>Show {hidden_count} more rows</span>
                    <span style="font-size: 10px;">▼</span>
                </button>
                <button class="collapse-btn" onclick="document.getElementById('tc-{unique_id}').classList.remove('expanded')"
                    style="background: var(--bg-secondary, #f3f4f6); border: 1px solid var(--border, #e5e7eb);
                           border-radius: 6px; padding: 6px 16px; font-size: 13px; color: var(--text-secondary, #666);
                           cursor: pointer; align-items: center; gap: 6px;">
                    <span>Show less</span>
                    <span style="font-size: 10px;">▲</span>
                </button>
            </div>
        </div>
        '''
    else:
        collapsible_wrapper = table_html

    return f'''
    <div class="figure-container ablation-matrix">
        <div class="figure-title">{escape_html(title)}</div>
        {sample_size_html}
        <div class="figure-explanation" style="background: var(--bg-secondary, #f8f9fa); padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 13px; color: var(--text-secondary, #666);">
            <strong>How to read:</strong> We ablate (zero out) this neuron and compare the model's output before/after.
            <span style="color: #16a34a;"><strong>Promotes</strong></span> = tokens that become MORE likely when neuron is removed (neuron was suppressing them).
            <span style="color: #dc2626;"><strong>Suppresses</strong></span> = tokens that become LESS likely (neuron was promoting them).
        </div>
        {collapsible_wrapper}
        {caption_html}
    </div>
    '''


def generate_batch_ablation_summary(
    batch_results: list[dict[str, Any]],
    title: str = "Batch Ablation Results",
    caption: str = ""
) -> str:
    """Generate summary card for batch ablation experiments.

    Shows total prompts tested, change rate, and interpretation.

    Args:
        batch_results: List of batch ablation results from multi_token_ablation_results
        title: Figure title
        caption: Optional caption
    """
    if not batch_results:
        return ""

    # Aggregate results across all batch runs
    total_prompts = 0
    total_changed = 0
    category_stats = {}

    for result in batch_results:
        if result.get("type") == "batch":
            total_prompts += result.get("total_prompts", 0)
            total_changed += result.get("total_changed", 0)
            # Merge category stats
            for cat, stats in result.get("category_stats", {}).items():
                if cat not in category_stats:
                    category_stats[cat] = {"changed": 0, "total": 0}
                category_stats[cat]["changed"] += stats.get("changed", 0)
                category_stats[cat]["total"] += stats.get("total", 0)

    if total_prompts == 0:
        return ""

    change_rate = total_changed / total_prompts if total_prompts > 0 else 0

    # Interpretation based on change rate
    if change_rate == 0:
        interpretation = "Ablating this neuron has <strong>no effect</strong> on model outputs across all tested prompts. This indicates <em>highly redundant circuit pathways</em>—other neurons can compensate for this one's removal."
        status_color = "#9ca3af"
        status_text = "NO EFFECT"
    elif change_rate < 0.1:
        interpretation = f"Ablating this neuron changes outputs in only <strong>{change_rate:.1%}</strong> of cases. The neuron's function appears to be largely <em>redundant</em> with other circuit components."
        status_color = "#f97316"
        status_text = "MINIMAL EFFECT"
    elif change_rate < 0.5:
        interpretation = f"Ablating this neuron changes outputs in <strong>{change_rate:.1%}</strong> of cases, indicating <em>moderate causal influence</em>. The neuron contributes meaningfully but other pathways can sometimes compensate."
        status_color = "#eab308"
        status_text = "MODERATE EFFECT"
    else:
        interpretation = f"Ablating this neuron changes outputs in <strong>{change_rate:.1%}</strong> of cases—a <em>strong causal effect</em>. This neuron is critical for the model's behavior in these contexts."
        status_color = "#16a34a"
        status_text = "STRONG EFFECT"

    # Category breakdown if available
    category_html = ""
    if category_stats:
        sorted_cats = sorted(category_stats.items(), key=lambda x: x[1]["total"], reverse=True)[:5]
        cat_rows = ""
        for cat, stats in sorted_cats:
            cat_rate = stats["changed"] / stats["total"] if stats["total"] > 0 else 0
            cat_rows += f'<tr><td>{escape_html(cat)}</td><td>{stats["changed"]}/{stats["total"]}</td><td>{cat_rate:.0%}</td></tr>'
        if cat_rows:
            category_html = f'''
            <div style="margin-top: 16px;">
                <div style="font-weight: 600; margin-bottom: 8px; font-size: 13px;">By Category:</div>
                <table style="width: 100%; font-size: 13px;">
                    <thead><tr><th style="text-align: left;">Category</th><th>Changed</th><th>Rate</th></tr></thead>
                    <tbody>{cat_rows}</tbody>
                </table>
            </div>
            '''

    caption_html = f'<p class="figure-caption">{escape_html(caption)}</p>' if caption else ""

    return f'''
    <div class="figure-container batch-ablation-summary" style="background: var(--bg-elevated, #ffffff);">
        <div class="figure-title">{escape_html(title)}</div>
        <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 16px;">
            <span style="background: {status_color}; color: white; padding: 4px 12px; border-radius: 4px; font-size: 12px; font-weight: 600;">{status_text}</span>
            <span style="font-size: 13px; color: var(--text-secondary, #666);">
                <strong>{total_changed}</strong> of <strong>{total_prompts}</strong> outputs changed ({change_rate:.1%})
            </span>
        </div>
        <div class="figure-explanation" style="background: var(--bg-secondary, #f8f9fa); padding: 12px 16px; border-radius: 8px; font-size: 13px; color: var(--text-secondary, #666);">
            <strong>What this means:</strong> {interpretation}
        </div>
        {category_html}
        {caption_html}
    </div>
    '''


def generate_batch_steering_summary(
    batch_results: list[dict[str, Any]],
    title: str = "Batch Steering Results",
    caption: str = ""
) -> str:
    """Generate summary card for batch steering experiments.

    Shows total prompts tested, change rate, and interpretation.

    Args:
        batch_results: List of batch steering results from multi_token_steering_results
        title: Figure title
        caption: Optional caption
    """
    if not batch_results:
        return ""

    # Aggregate results across all batch runs
    total_prompts = 0
    total_changed = 0
    steering_value = None
    category_stats = {}

    for result in batch_results:
        if result.get("type") == "batch":
            total_prompts += result.get("total_prompts", 0)
            total_changed += result.get("total_changed", 0)
            if steering_value is None:
                steering_value = result.get("steering_value", 10.0)
            # Merge category stats
            for cat, stats in result.get("category_stats", {}).items():
                if cat not in category_stats:
                    category_stats[cat] = {"changed": 0, "total": 0}
                category_stats[cat]["changed"] += stats.get("changed", 0)
                category_stats[cat]["total"] += stats.get("total", 0)

    if total_prompts == 0:
        return ""

    change_rate = total_changed / total_prompts if total_prompts > 0 else 0
    steering_str = f"+{steering_value}" if steering_value and steering_value > 0 else str(steering_value)

    # Interpretation based on change rate
    if change_rate == 0:
        interpretation = f"Steering this neuron (value={steering_str}) has <strong>no effect</strong> on model outputs across all tested prompts. This indicates the neuron either has weak output projections or the circuit can fully compensate for amplified activation."
        status_color = "#9ca3af"
        status_text = "NO EFFECT"
    elif change_rate < 0.1:
        interpretation = f"Steering this neuron (value={steering_str}) changes outputs in only <strong>{change_rate:.1%}</strong> of cases. The neuron's amplified signal has <em>minimal downstream impact</em>."
        status_color = "#f97316"
        status_text = "MINIMAL EFFECT"
    elif change_rate < 0.5:
        interpretation = f"Steering this neuron (value={steering_str}) changes outputs in <strong>{change_rate:.1%}</strong> of cases, indicating <em>moderate causal influence</em>. Amplifying this neuron meaningfully shifts model behavior."
        status_color = "#eab308"
        status_text = "MODERATE EFFECT"
    else:
        interpretation = f"Steering this neuron (value={steering_str}) changes outputs in <strong>{change_rate:.1%}</strong> of cases—a <em>strong causal effect</em>. This neuron can reliably steer the model's behavior when amplified."
        status_color = "#16a34a"
        status_text = "STRONG EFFECT"

    # Category breakdown if available
    category_html = ""
    if category_stats:
        sorted_cats = sorted(category_stats.items(), key=lambda x: x[1]["total"], reverse=True)[:5]
        cat_rows = ""
        for cat, stats in sorted_cats:
            cat_rate = stats["changed"] / stats["total"] if stats["total"] > 0 else 0
            cat_rows += f'<tr><td>{escape_html(cat)}</td><td>{stats["changed"]}/{stats["total"]}</td><td>{cat_rate:.0%}</td></tr>'
        if cat_rows:
            category_html = f'''
            <div style="margin-top: 16px;">
                <div style="font-weight: 600; margin-bottom: 8px; font-size: 13px;">By Category:</div>
                <table style="width: 100%; font-size: 13px;">
                    <thead><tr><th style="text-align: left;">Category</th><th>Changed</th><th>Rate</th></tr></thead>
                    <tbody>{cat_rows}</tbody>
                </table>
            </div>
            '''

    caption_html = f'<p class="figure-caption">{escape_html(caption)}</p>' if caption else ""

    return f'''
    <div class="figure-container batch-steering-summary" style="background: var(--bg-elevated, #ffffff);">
        <div class="figure-title">{escape_html(title)}</div>
        <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 16px;">
            <span style="background: {status_color}; color: white; padding: 4px 12px; border-radius: 4px; font-size: 12px; font-weight: 600;">{status_text}</span>
            <span style="font-size: 13px; color: var(--text-secondary, #666);">
                <strong>{total_changed}</strong> of <strong>{total_prompts}</strong> outputs changed ({change_rate:.1%}) at steering={steering_str}
            </span>
        </div>
        <div class="figure-explanation" style="background: var(--bg-secondary, #f8f9fa); padding: 12px 16px; border-radius: 8px; font-size: 13px; color: var(--text-secondary, #666);">
            <strong>What this means:</strong> {interpretation}
        </div>
        {category_html}
        {caption_html}
    </div>
    '''


def generate_intelligent_steering_run(
    run_data: dict[str, Any],
    run_number: int = 1,
    title: str = None,
    initial_visible: int = 3,
) -> str:
    """Generate visualization for a single intelligent steering run.

    Shows the analysis summary, key findings, and illustrative examples
    with full-width before/after comparisons in a card layout.

    Args:
        run_data: A single intelligent_steering result from multi_token_steering_results
        run_number: Which run this is (1, 2, 3, etc.)
        title: Optional custom title (defaults to "Intelligent Steering Run #N")
        initial_visible: Number of examples to show initially (default 3)
    """
    if not run_data or run_data.get("type") != "intelligent_steering":
        return ""

    if title is None:
        focus = run_data.get("additional_instructions")
        if focus:
            title = f"Steering Analysis #{run_number}: {focus[:50]}..."
        else:
            title = f"Steering Analysis #{run_number}"

    # Extract data
    n_prompts = run_data.get("n_prompts", 0)
    steering_values = run_data.get("steering_values", [])
    stats = run_data.get("stats_by_steering_value", {})
    summary = run_data.get("analysis_summary", "")
    key_findings = run_data.get("key_findings", [])
    hypothesis_supported = run_data.get("hypothesis_supported")
    effective_range = run_data.get("effective_steering_range", "")
    illustrative_examples = run_data.get("illustrative_examples", [])

    # Hypothesis status badge - handle both bool and string values
    # (Sonnet sometimes returns "true"/"false" as strings instead of JSON booleans)
    hs = str(hypothesis_supported).lower() if hypothesis_supported is not None else ""
    if hs in ("true", "supported", "yes"):
        status_badge = '<span class="steering-status-badge supported">SUPPORTED</span>'
    elif hs in ("false", "refuted", "no"):
        status_badge = '<span class="steering-status-badge refuted">REFUTED</span>'
    elif hs in ("partial", "partially_supported"):
        status_badge = '<span class="steering-status-badge partial">PARTIAL</span>'
    else:
        status_badge = '<span class="steering-status-badge inconclusive">INCONCLUSIVE</span>'

    # Steering value stats
    sv_html = ""
    if stats:
        sv_items = []
        for sv_key, sv_stats in sorted(stats.items(), key=lambda x: float(x[0])):
            rate = sv_stats.get("rate", 0)
            sv_class = "positive" if float(sv_key) > 0 else "negative"
            sv_items.append(f'<span class="sv-stat {sv_class}"><strong>sv={sv_key}</strong>: {rate:.0%}</span>')
        sv_html = f'<div class="steering-sv-stats">Change rates: {"".join(sv_items)}</div>'

    # Key findings
    findings_html = ""
    if key_findings:
        findings_items = "".join(f'<li>{escape_html(str(f))}</li>' for f in key_findings[:5])
        findings_html = f'''
        <div class="steering-findings">
            <div class="findings-header">Key Findings:</div>
            <ul>{findings_items}</ul>
        </div>
        '''

    # Generate unique ID for this run
    unique_id = hashlib.md5(f"intelligent-steering-{run_number}-{n_prompts}".encode()).hexdigest()[:8]

    # Illustrative examples as expandable cards
    examples_html = ""
    if illustrative_examples:
        example_cards = []
        for i, ex in enumerate(illustrative_examples[:10]):
            # Full prompt - no truncation
            prompt = escape_html(str(ex.get("prompt", "")))

            # Handle both old format (baseline, steered) and new format (baseline_completion, steering_results)
            baseline_raw = ex.get("baseline") or ex.get("baseline_completion", "")
            baseline = escape_html(str(baseline_raw))

            # Extract steered output - prefer highest positive steering value that changed
            steered = ""
            sv = "?"
            steering_results = ex.get("steering_results", {})
            if steering_results:
                # Find the best steered result (highest positive sv with change)
                for test_sv in ["15", "10", "5"]:  # Check positive values first
                    if test_sv in steering_results:
                        result = steering_results[test_sv]
                        if result.get("changed", True):  # Default to showing it
                            steered = escape_html(str(result.get("completion", "")))
                            sv = test_sv
                            break
                # Fallback to any changed result
                if not steered:
                    for test_sv, result in steering_results.items():
                        if test_sv != "0" and result.get("changed", True):
                            steered = escape_html(str(result.get("completion", "")))
                            sv = test_sv
                            break
            else:
                steered = escape_html(str(ex.get("steered", "")))
                sv = ex.get("steering_value", "?")

            # Handle both old format (why_illustrative) and new format (rationale)
            why = ex.get("why_illustrative") or ex.get("rationale", "")
            why = escape_html(str(why))

            # Determine if this example shows a refusal, topic shift, etc.
            effect_type = ""
            steered_lower = steered.lower() if steered else ""
            if "sorry" in steered_lower or "cannot" in steered_lower or "i can't" in steered_lower:
                effect_type = '<span class="effect-tag refusal">Refusal</span>'
            elif baseline and steered and baseline[:20] != steered[:20]:
                effect_type = '<span class="effect-tag changed">Changed</span>'

            # Hidden class for cards beyond initial_visible
            hidden_class = "steering-card-hidden" if i >= initial_visible else ""

            # Get prompt format info for proper display
            prompt_format = ex.get("prompt_format", {})
            fmt_type = prompt_format.get("format", "raw")

            # Build prompt display based on format
            if fmt_type == "response":
                # Chat response: User asks, assistant responds
                user_msg = escape_html(prompt_format.get("user_message", prompt))
                prompt_display_baseline = f'<span class="chat-user">User:</span> <span class="user-text">{user_msg}</span><br><span class="chat-assistant">Assistant:</span> '
                prompt_display_steered = prompt_display_baseline
                format_badge = '<span class="format-badge response-badge">chat</span>'
            elif fmt_type == "continuation":
                # Chat continuation: instruction + prefix → completion
                instruction = escape_html(prompt_format.get("instruction", "Continue:"))
                prefix = escape_html(prompt_format.get("prefix", prompt))
                prompt_display_baseline = f'<span class="instruction-text">[{instruction}]</span> <span class="prefix-text">{prefix}</span>'
                prompt_display_steered = prompt_display_baseline
                format_badge = '<span class="format-badge continuation-badge">prefill</span>'
            else:
                # Raw continuation: text → completion
                prompt_display_baseline = f'<span class="prompt-text">{prompt}</span>'
                prompt_display_steered = prompt_display_baseline
                format_badge = '<span class="format-badge raw-badge">raw</span>'

            # New layout: show prompt + completion based on format
            example_cards.append(f'''
            <div class="steering-example-card {hidden_class}" data-card-idx="{i}">
                <div class="card-sidebar">
                    <div class="sv-badge">sv={sv}</div>
                    {format_badge}
                    {effect_type}
                    <div class="why-illustrative">
                        <div class="why-label">Why shown:</div>
                        <div class="why-text">{why}</div>
                    </div>
                </div>
                <div class="card-main">
                    <div class="card-section completion-section baseline-completion">
                        <div class="section-label">Baseline</div>
                        <div class="section-content continuous-text">{prompt_display_baseline}<span class="completion-text baseline-text">{baseline}</span></div>
                    </div>
                    <div class="card-section completion-section steered-completion">
                        <div class="section-label steered-label">Steered (sv={sv})</div>
                        <div class="section-content continuous-text">{prompt_display_steered}<span class="completion-text steered-text">{steered}</span></div>
                    </div>
                </div>
            </div>
            ''')

        total_examples = len(illustrative_examples)
        hidden_count = max(0, total_examples - initial_visible)

        expand_btn = ""
        if hidden_count > 0:
            expand_btn = f'''
            <button class="steering-expand-btn" onclick="toggleSteeringExamples('{unique_id}')" data-expanded="false">
                Show {hidden_count} more examples ▼
            </button>
            '''

        examples_html = f'''
        <div class="steering-examples-container" id="sec-{unique_id}">
            <div class="examples-header">Illustrative Examples ({total_examples})</div>
            <div class="steering-examples-list">
                {"".join(example_cards)}
            </div>
            {expand_btn}
        </div>
        '''

    # Summary
    summary_html = ""
    if summary:
        summary_html = f'''
        <div class="steering-summary">
            {escape_html(summary)}
        </div>
        '''

    return f'''
    <div class="intelligent-steering-run" id="fig-{unique_id}">
        <div class="steering-header">
            <div class="steering-title">{escape_html(title)}</div>
            {status_badge}
        </div>
        <div class="steering-meta">
            Tested <strong>{n_prompts}</strong> prompts across steering values {steering_values}
        </div>
        {sv_html}
        {summary_html}
        {findings_html}
        {examples_html}
    </div>
    <style>
    .intelligent-steering-run {{
        background: var(--bg-elevated, #ffffff);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        border: 1px solid var(--border-color, #e5e7eb);
    }}
    .steering-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
    }}
    .steering-title {{
        font-size: 18px;
        font-weight: 600;
        color: var(--text-primary, #111);
    }}
    .steering-status-badge {{
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }}
    .steering-status-badge.supported {{ background: #16a34a; color: white; }}
    .steering-status-badge.refuted {{ background: #dc2626; color: white; }}
    .steering-status-badge.partial {{ background: #eab308; color: white; }}
    .steering-status-badge.inconclusive {{ background: #9ca3af; color: white; }}
    .steering-meta {{
        font-size: 13px;
        color: var(--text-secondary, #666);
        margin-bottom: 12px;
    }}
    .steering-sv-stats {{
        margin: 12px 0;
        font-size: 13px;
    }}
    .sv-stat {{
        margin-right: 16px;
    }}
    .sv-stat.positive {{ color: #16a34a; }}
    .sv-stat.negative {{ color: #dc2626; }}
    .steering-summary {{
        background: var(--bg-secondary, #f8f9fa);
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 13px;
        color: var(--text-secondary, #666);
        margin-bottom: 16px;
        line-height: 1.6;
    }}
    .steering-findings {{
        margin: 16px 0;
    }}
    .findings-header {{
        font-weight: 600;
        margin-bottom: 8px;
        font-size: 13px;
    }}
    .steering-findings ul {{
        margin: 0;
        padding-left: 20px;
        font-size: 13px;
        color: var(--text-secondary, #666);
    }}
    .steering-findings li {{
        margin-bottom: 6px;
    }}
    .examples-header {{
        font-weight: 600;
        margin-bottom: 16px;
        font-size: 14px;
        color: var(--text-primary, #111);
    }}
    .steering-example-card {{
        display: flex;
        gap: 16px;
        margin-bottom: 20px;
        padding: 16px;
        background: var(--bg-secondary, #f8f9fa);
        border-radius: 8px;
        border-left: 4px solid #8b5cf6;
    }}
    .steering-card-hidden {{
        display: none;
    }}
    .card-sidebar {{
        flex: 0 0 120px;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }}
    .sv-badge {{
        background: #8b5cf6;
        color: white;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 600;
        text-align: center;
    }}
    .effect-tag {{
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 500;
        text-align: center;
    }}
    .effect-tag.refusal {{
        background: #fef2f2;
        color: #dc2626;
    }}
    .effect-tag.changed {{
        background: #ecfdf5;
        color: #059669;
    }}
    .why-illustrative {{
        margin-top: 8px;
    }}
    .why-label {{
        font-size: 10px;
        color: #9ca3af;
        text-transform: uppercase;
        margin-bottom: 4px;
    }}
    .why-text {{
        font-size: 11px;
        color: var(--text-secondary, #666);
        line-height: 1.4;
    }}
    .card-main {{
        flex: 1;
        min-width: 0;
    }}
    .card-section {{
        margin-bottom: 12px;
    }}
    .card-section:last-child {{
        margin-bottom: 0;
    }}
    .section-label {{
        font-size: 11px;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        margin-bottom: 4px;
    }}
    .section-content {{
        font-family: 'SF Mono', 'Consolas', monospace;
        font-size: 13px;
        color: var(--text-primary, #111);
        line-height: 1.5;
        white-space: pre-wrap;
        word-break: break-word;
        background: var(--bg-elevated, #fff);
        padding: 10px 12px;
        border-radius: 6px;
        border: 1px solid var(--border-color, #e5e7eb);
    }}
    /* Format badges */
    .format-badge {{
        display: inline-block;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 9px;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }}
    .response-badge {{ background: #dbeafe; color: #1e40af; }}
    .continuation-badge {{ background: #fef3c7; color: #92400e; }}
    .raw-badge {{ background: #f3e8ff; color: #7c3aed; }}
    /* Continuous text styling for steering examples */
    .continuous-text {{
        background: #fafafa;
    }}
    .continuous-text .prompt-text {{
        color: var(--text-secondary, #555);
    }}
    /* Chat response format */
    .continuous-text .chat-user {{
        color: #6b7280;
        font-weight: 600;
        font-size: 11px;
    }}
    .continuous-text .user-text {{
        color: #374151;
    }}
    .continuous-text .chat-assistant {{
        color: #059669;
        font-weight: 600;
        font-size: 11px;
    }}
    /* Chat continuation format */
    .continuous-text .instruction-text {{
        color: #9ca3af;
        font-size: 11px;
    }}
    .continuous-text .prefix-text {{
        color: var(--text-secondary, #555);
    }}
    .continuous-text .completion-text {{
        padding: 2px 4px;
        border-radius: 3px;
        margin-left: 0;
    }}
    .continuous-text .baseline-text {{
        background: #f3f4f6;
        color: var(--text-primary, #111);
        border-bottom: 2px solid #9ca3af;
    }}
    .continuous-text .steered-text {{
        background: #ecfdf5;
        color: #065f46;
        border-bottom: 2px solid #16a34a;
    }}
    .baseline-completion .section-content {{
        border-left: 3px solid #9ca3af;
    }}
    .steered-completion .section-content {{
        border-left: 3px solid #16a34a;
    }}
    .steered-label {{
        color: #16a34a;
    }}
    .steering-expand-btn {{
        display: block;
        width: 100%;
        padding: 10px;
        margin-top: 12px;
        background: var(--bg-elevated, #fff);
        border: 1px solid var(--border-color, #e5e7eb);
        border-radius: 6px;
        cursor: pointer;
        font-size: 13px;
        color: var(--text-secondary, #666);
        transition: background 0.2s;
    }}
    .steering-expand-btn:hover {{
        background: var(--bg-secondary, #f3f4f6);
    }}
    </style>
    <script>
    function toggleSteeringExamples(runId) {{
        const container = document.getElementById('sec-' + runId);
        const btn = container.querySelector('.steering-expand-btn');
        const hiddenCards = container.querySelectorAll('.steering-card-hidden');
        const isExpanded = btn.dataset.expanded === 'true';

        hiddenCards.forEach(card => {{
            card.style.display = isExpanded ? 'none' : 'flex';
        }});

        const hiddenCount = hiddenCards.length;
        btn.innerHTML = isExpanded ? `Show ${{hiddenCount}} more examples ▼` : `Show less ▲`;
        btn.dataset.expanded = isExpanded ? 'false' : 'true';
    }}
    </script>
    '''


def generate_intelligent_steering_gallery(
    steering_results: list[dict[str, Any]],
    title: str = "Intelligent Steering Analysis",
) -> str:
    """Generate gallery of all intelligent steering runs.

    Groups results by run and generates visualization for each.

    Args:
        steering_results: List from multi_token_steering_results (may contain multiple types)
        title: Overall section title
    """
    # Filter to only intelligent_steering type
    intelligent_runs = [r for r in steering_results if r.get("type") == "intelligent_steering"]

    if not intelligent_runs:
        return ""

    runs_html = ""
    for i, run in enumerate(intelligent_runs):
        runs_html += generate_intelligent_steering_run(run, run_number=i + 1)

    total_prompts = sum(r.get("n_prompts", 0) for r in intelligent_runs)
    total_examples = sum(len(r.get("illustrative_examples", [])) for r in intelligent_runs)

    return f'''
    <div class="intelligent-steering-gallery">
        <h3 style="margin-bottom: 16px;">{escape_html(title)}</h3>
        <div style="font-size: 13px; color: var(--text-secondary, #666); margin-bottom: 20px;">
            {len(intelligent_runs)} steering run{"s" if len(intelligent_runs) != 1 else ""} •
            {total_prompts} total prompts •
            {total_examples} illustrative examples
        </div>
        {runs_html}
    </div>
    '''


def generate_ablation_cards(
    ablation_results: list[dict[str, Any]],
    title: str = "Ablation Effects on Completions",
    initial_visible: int = 3,
) -> str:
    """Generate ablation completion cards matching the steering card format.

    Shows changed completions with "Why shown" sidebar, matching the format
    used by generate_intelligent_steering_run.

    Args:
        ablation_results: List from multi_token_ablation_results (type='batch')
        title: Section title
        initial_visible: Number of examples to show initially (default 3)

    Returns:
        HTML string with ablation cards
    """
    # Extract changed examples from batch ablation results
    changed_examples = []
    total_prompts = 0
    total_changed = 0

    for run in ablation_results:
        if run.get("type") == "batch":
            total_prompts += run.get("total_prompts", 0)
            total_changed += run.get("total_changed", 0)
            for ex in run.get("changed_examples", []):
                if ex.get("baseline_completion") != ex.get("ablated_completion"):
                    changed_examples.append(ex)

    if not changed_examples:
        return ""

    unique_id = hashlib.md5(f"ablation-cards-{len(changed_examples)}".encode()).hexdigest()[:8]

    # Generate cards
    example_cards = []
    for i, ex in enumerate(changed_examples[:15]):  # Limit to 15 examples
        prompt = escape_html(str(ex.get("prompt", "")))
        baseline = escape_html(str(ex.get("baseline_completion", "")))
        ablated = escape_html(str(ex.get("ablated_completion", "")))
        category = ex.get("category", "")
        max_shift = ex.get("max_shift", 0)

        # Generate "Why shown" rationale based on characteristics
        why_parts = []
        if category:
            why_parts.append(f"Category: {category}")
        if max_shift > 1.0:
            why_parts.append(f"High logit shift ({max_shift:.1f})")
        elif max_shift > 0.5:
            why_parts.append(f"Moderate logit shift ({max_shift:.1f})")

        # Check for semantic changes
        if len(baseline) > 0 and len(ablated) > 0:
            if baseline[:20] != ablated[:20]:
                why_parts.append("Completion changed significantly")
            elif baseline != ablated:
                why_parts.append("Subtle completion change")

        why = "; ".join(why_parts) if why_parts else "Changed completion"

        # Effect type badge
        effect_type = '<span class="effect-tag changed">Changed</span>'
        if "refus" in ablated.lower() or "sorry" in ablated.lower() or "cannot" in ablated.lower():
            if "refus" not in baseline.lower() and "sorry" not in baseline.lower():
                effect_type = '<span class="effect-tag refusal">→ Refusal</span>'

        # Category badge
        category_badge = f'<span class="category-badge">{escape_html(category)}</span>' if category else ""

        # Hidden class for cards beyond initial_visible
        hidden_class = "ablation-card-hidden" if i >= initial_visible else ""

        # New layout: show prompt + completion as continuous text
        example_cards.append(f'''
        <div class="ablation-example-card {hidden_class}" data-card-idx="{i}">
            <div class="card-sidebar">
                <div class="ablation-badge">Ablated</div>
                {effect_type}
                {category_badge}
                <div class="why-illustrative">
                    <div class="why-label">Why shown:</div>
                    <div class="why-text">{why}</div>
                </div>
            </div>
            <div class="card-main">
                <div class="card-section completion-section baseline-completion">
                    <div class="section-label">Baseline completion</div>
                    <div class="section-content continuous-text"><span class="prompt-text">{prompt}</span><span class="completion-text baseline-text">{baseline}</span></div>
                </div>
                <div class="card-section completion-section ablated-completion">
                    <div class="section-label ablated-label">Ablated completion</div>
                    <div class="section-content continuous-text"><span class="prompt-text">{prompt}</span><span class="completion-text ablated-text">{ablated}</span></div>
                </div>
            </div>
        </div>
        ''')

    total_examples = len(changed_examples)
    hidden_count = max(0, min(total_examples, 15) - initial_visible)

    expand_btn = ""
    if hidden_count > 0:
        expand_btn = f'''
        <button class="ablation-expand-btn" onclick="toggleAblationExamples('{unique_id}')" data-expanded="false">
            Show {hidden_count} more examples ▼
        </button>
        '''

    change_rate = (total_changed / total_prompts * 100) if total_prompts > 0 else 0

    return f'''
    <div class="ablation-cards-container" id="abl-{unique_id}">
        <h3 style="margin-bottom: 12px;">{escape_html(title)}</h3>
        <div class="ablation-meta">
            <strong>{total_changed}</strong> of <strong>{total_prompts}</strong> prompts changed ({change_rate:.1f}%)
        </div>
        <div class="ablation-examples-list">
            {"".join(example_cards)}
        </div>
        {expand_btn}
    </div>
    <style>
    .ablation-cards-container {{
        margin: 20px 0;
    }}
    .ablation-meta {{
        font-size: 13px;
        color: var(--text-secondary, #666);
        margin-bottom: 16px;
    }}
    .ablation-example-card {{
        display: flex;
        border: 1px solid var(--border-color, #e0e0e0);
        border-radius: 8px;
        margin-bottom: 12px;
        overflow: hidden;
        background: var(--bg-elevated, #ffffff);
    }}
    .ablation-example-card .card-sidebar {{
        width: 140px;
        min-width: 140px;
        padding: 12px;
        background: var(--bg-secondary, #f8f9fa);
        border-right: 1px solid var(--border-color, #e0e0e0);
        display: flex;
        flex-direction: column;
        gap: 8px;
    }}
    .ablation-badge {{
        display: inline-block;
        background: #dc2626;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-align: center;
    }}
    .category-badge {{
        display: inline-block;
        background: #e5e7eb;
        color: #374151;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 10px;
    }}
    .ablation-example-card .card-main {{
        flex: 1;
        padding: 12px;
    }}
    .ablation-example-card .card-section {{
        margin-bottom: 10px;
    }}
    .ablation-example-card .card-section:last-child {{
        margin-bottom: 0;
    }}
    .ablation-example-card .section-label {{
        font-size: 11px;
        font-weight: 600;
        color: var(--text-secondary, #666);
        margin-bottom: 4px;
        text-transform: uppercase;
    }}
    .ablation-example-card .section-content {{
        font-family: var(--font-mono, monospace);
        font-size: 13px;
        color: var(--text-primary, #111);
        white-space: pre-wrap;
        word-break: break-word;
        background: var(--bg-secondary, #f8f9fa);
        padding: 8px;
        border-radius: 4px;
    }}
    /* Continuous text styling for ablation examples */
    .ablation-example-card .continuous-text {{
        background: #fafafa;
    }}
    .ablation-example-card .continuous-text .prompt-text {{
        color: var(--text-secondary, #555);
    }}
    .ablation-example-card .continuous-text .completion-text {{
        padding: 2px 4px;
        border-radius: 3px;
    }}
    .ablation-example-card .continuous-text .baseline-text {{
        background: #f3f4f6;
        color: var(--text-primary, #111);
        border-bottom: 2px solid #9ca3af;
    }}
    .ablation-example-card .continuous-text .ablated-text {{
        background: #fef2f2;
        color: #991b1b;
        border-bottom: 2px solid #dc2626;
    }}
    .ablation-example-card .ablated-label {{
        color: #dc2626;
    }}
    .ablation-example-card .ablated-completion .section-content {{
        border-left: 3px solid #dc2626;
    }}
    .ablation-example-card .baseline-completion .section-content {{
        border-left: 3px solid #9ca3af;
    }}
    .ablation-example-card .why-illustrative {{
        margin-top: auto;
    }}
    .ablation-example-card .why-label {{
        font-size: 10px;
        font-weight: 600;
        color: var(--text-tertiary, #999);
        text-transform: uppercase;
        margin-bottom: 2px;
    }}
    .ablation-example-card .why-text {{
        font-size: 11px;
        color: var(--text-secondary, #666);
        line-height: 1.4;
    }}
    .ablation-example-card .effect-tag {{
        display: inline-block;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 500;
    }}
    .ablation-example-card .effect-tag.changed {{
        background: #fef3c7;
        color: #92400e;
    }}
    .ablation-example-card .effect-tag.refusal {{
        background: #fee2e2;
        color: #991b1b;
    }}
    .ablation-card-hidden {{
        display: none;
    }}
    .ablation-expand-btn {{
        display: block;
        width: 100%;
        padding: 10px;
        margin-top: 8px;
        background: var(--bg-secondary, #f3f4f6);
        border: 1px solid var(--border-color, #e0e0e0);
        border-radius: 6px;
        color: var(--text-secondary, #666);
        font-size: 13px;
        cursor: pointer;
        transition: background 0.2s;
    }}
    .ablation-expand-btn:hover {{
        background: var(--bg-tertiary, #e5e7eb);
    }}
    </style>
    <script>
    function toggleAblationExamples(uniqueId) {{
        const container = document.getElementById('abl-' + uniqueId);
        const hiddenCards = container.querySelectorAll('.ablation-card-hidden');
        const btn = container.querySelector('.ablation-expand-btn');

        const isExpanded = btn.dataset.expanded === 'true';

        hiddenCards.forEach(card => {{
            card.style.display = isExpanded ? 'none' : 'flex';
        }});

        const hiddenCount = hiddenCards.length;
        btn.innerHTML = isExpanded ? `Show ${{hiddenCount}} more examples ▼` : `Show less ▲`;
        btn.dataset.expanded = isExpanded ? 'false' : 'true';
    }}
    </script>
    '''


def generate_downstream_ablation_effects(
    ablation_results: list[dict[str, Any]],
    neuron_labels: dict[str, str] | None = None,
    title: str = "Downstream Ablation Effects",
) -> str:
    """Generate table showing how ablating target neuron affects downstream neurons.

    Args:
        ablation_results: List from multi_token_ablation_results
        neuron_labels: Optional dict mapping neuron_id to label
        title: Table title

    Returns:
        HTML string with downstream effects table
    """
    neuron_labels = neuron_labels or {}

    # Extract dependency_summary from batch ablation results
    dependency_summary = {}
    downstream_checked = []
    batch_total_prompts = 0

    for run in ablation_results:
        if run.get("type") == "batch":
            if run.get("dependency_summary"):
                dependency_summary.update(run["dependency_summary"])
            if run.get("downstream_neurons_checked"):
                downstream_checked = run["downstream_neurons_checked"]
            # Track total prompts from the batch run as fallback for per-neuron counts
            batch_total_prompts = run.get("total_prompts") or batch_total_prompts

    if not dependency_summary:
        return ""

    # Sort by absolute mean change
    sorted_neurons = sorted(
        dependency_summary.items(),
        key=lambda x: abs(x[1].get("mean_change", 0)),
        reverse=True
    )

    rows = []
    for neuron_id, effects in sorted_neurons:
        # Support both key names: mean_change (new) and mean_change_percent (legacy)
        mean_change = effects.get("mean_change", effects.get("mean_change_percent", 0))
        n_measured = effects.get("n_prompts_measured", effects.get("n_prompts", 0)) or batch_total_prompts
        label = neuron_labels.get(neuron_id, effects.get("label", ""))

        # Color based on direction and magnitude
        if mean_change < -20:
            change_class = "strong-decrease"
            change_color = "#dc2626"
        elif mean_change < -5:
            change_class = "decrease"
            change_color = "#f97316"
        elif mean_change > 20:
            change_class = "strong-increase"
            change_color = "#16a34a"
        elif mean_change > 5:
            change_class = "increase"
            change_color = "#22c55e"
        else:
            change_class = "minimal"
            change_color = "#6b7280"

        # Interpretation
        if mean_change < -50:
            interpretation = "Strong suppression"
        elif mean_change < -20:
            interpretation = "Moderate suppression"
        elif mean_change < -5:
            interpretation = "Mild suppression"
        elif mean_change > 50:
            interpretation = "Strong activation"
        elif mean_change > 20:
            interpretation = "Moderate activation"
        elif mean_change > 5:
            interpretation = "Mild activation"
        else:
            interpretation = "Minimal effect"

        label_tooltip = f' title="{escape_html(label)}"' if label else ""
        rows.append(f'''
        <tr class="{change_class}">
            <td class="neuron-id">{escape_html(neuron_id)}</td>
            <td class="neuron-label"{label_tooltip}>{escape_html(label[:50]) if label else "—"}</td>
            <td class="change-value" style="color: {change_color};">{mean_change:+.1f}%</td>
            <td class="interpretation">{interpretation}</td>
            <td class="n-measured">{n_measured}</td>
        </tr>
        ''')

    return f'''
    <div class="figure-container downstream-ablation-effects">
        <div class="figure-title">{escape_html(title)}</div>
        <p class="table-description">
            Shows how ablating the target neuron changes downstream neuron activations.
            Negative values = reduced activation when target is ablated.
        </p>
        <table class="downstream-effects-table">
            <thead>
                <tr>
                    <th>Neuron</th>
                    <th>Label</th>
                    <th>Δ Activation</th>
                    <th>Interpretation</th>
                    <th>N Prompts</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    </div>
    <style>
    .downstream-ablation-effects .table-description {{
        font-size: 12px;
        color: var(--text-secondary, #666);
        margin-bottom: 12px;
    }}
    .downstream-effects-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }}
    .downstream-effects-table th {{
        background: var(--bg-secondary, #f3f4f6);
        padding: 8px 12px;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid var(--border-color, #e0e0e0);
    }}
    .downstream-effects-table td {{
        padding: 8px 12px;
        border-bottom: 1px solid var(--border-color, #e0e0e0);
    }}
    .downstream-effects-table .neuron-id {{
        font-family: var(--font-mono, monospace);
        font-weight: 500;
    }}
    .downstream-effects-table .neuron-label {{
        color: var(--text-secondary, #666);
        max-width: 200px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .downstream-effects-table .change-value {{
        font-weight: 600;
        font-family: var(--font-mono, monospace);
    }}
    .downstream-effects-table .interpretation {{
        font-size: 12px;
    }}
    .downstream-effects-table .n-measured {{
        color: var(--text-tertiary, #999);
        font-size: 12px;
    }}
    .downstream-effects-table tr.strong-decrease {{
        background: rgba(220, 38, 38, 0.05);
    }}
    .downstream-effects-table tr.strong-increase {{
        background: rgba(22, 163, 74, 0.05);
    }}
    </style>
    '''


def generate_downstream_steering_effects(
    steering_results: list[dict[str, Any]],
    neuron_labels: dict[str, str] | None = None,
    title: str = "Downstream Steering Effects",
) -> str:
    """Generate table showing how steering target neuron affects downstream neurons.

    Args:
        steering_results: List from multi_token_steering_results
        neuron_labels: Optional dict mapping neuron_id to label
        title: Table title

    Returns:
        HTML string with downstream steering effects table
    """
    neuron_labels = neuron_labels or {}

    # Extract downstream_effects_summary from intelligent steering results
    downstream_summary = {}
    steering_value_used = None
    run_total_prompts = 0

    for run in steering_results:
        if run.get("type") == "intelligent_steering":
            if run.get("downstream_effects_summary"):
                downstream_summary.update(run["downstream_effects_summary"])
                # Get the steering value used (should be consistent)
                for ds_id, effects in run["downstream_effects_summary"].items():
                    if steering_value_used is None:
                        steering_value_used = effects.get("steering_value_used")
            run_total_prompts = run.get("n_prompts") or run_total_prompts

    if not downstream_summary:
        return ""

    # Sort by absolute mean change
    sorted_neurons = sorted(
        downstream_summary.items(),
        key=lambda x: abs(x[1].get("mean_change_percent", 0)),
        reverse=True
    )

    rows = []
    for neuron_id, effects in sorted_neurons:
        mean_change = effects.get("mean_change_percent", 0)
        n_measured = effects.get("n_prompts_measured", 0) or run_total_prompts
        label = neuron_labels.get(neuron_id, "")

        # Color based on direction and magnitude
        if mean_change < -20:
            change_class = "strong-decrease"
            change_color = "#dc2626"
        elif mean_change < -5:
            change_class = "decrease"
            change_color = "#f97316"
        elif mean_change > 20:
            change_class = "strong-increase"
            change_color = "#16a34a"
        elif mean_change > 5:
            change_class = "increase"
            change_color = "#22c55e"
        else:
            change_class = "minimal"
            change_color = "#6b7280"

        # Interpretation
        if mean_change < -50:
            interpretation = "Strong suppression"
        elif mean_change < -20:
            interpretation = "Moderate suppression"
        elif mean_change < -5:
            interpretation = "Mild suppression"
        elif mean_change > 50:
            interpretation = "Strong excitation"
        elif mean_change > 20:
            interpretation = "Moderate excitation"
        elif mean_change > 5:
            interpretation = "Mild excitation"
        else:
            interpretation = "Minimal effect"

        label_tooltip = f' title="{escape_html(label)}"' if label else ""
        rows.append(f'''
        <tr class="{change_class}">
            <td class="neuron-id">{escape_html(neuron_id)}</td>
            <td class="neuron-label"{label_tooltip}>{escape_html(label[:50]) if label else "—"}</td>
            <td class="change-value" style="color: {change_color};">{mean_change:+.1f}%</td>
            <td class="interpretation">{interpretation}</td>
            <td class="n-measured">{n_measured}</td>
        </tr>
        ''')

    sv_display = f" (sv={steering_value_used})" if steering_value_used else ""

    return f'''
    <div class="figure-container downstream-steering-effects">
        <div class="figure-title">{escape_html(title)}{sv_display}</div>
        <p class="table-description">
            Shows how steering the target neuron changes downstream neuron activations.
            Positive values = increased activation when target is amplified.
        </p>
        <table class="downstream-effects-table">
            <thead>
                <tr>
                    <th>Neuron</th>
                    <th>Label</th>
                    <th>Δ Activation</th>
                    <th>Interpretation</th>
                    <th>N Prompts</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    </div>
    <style>
    .downstream-steering-effects .table-description {{
        font-size: 12px;
        color: var(--text-secondary, #666);
        margin-bottom: 12px;
    }}
    </style>
    '''


def generate_completion_examples(
    examples: list[dict[str, Any]],
    experiment_type: str = "ablation",
    title: str = "Changed Completion Examples",
    caption: str = "",
    steering_value: float = None
) -> str:
    """Generate before/after comparison of completions that changed.

    Args:
        examples: List of dicts with prompt, baseline_completion, ablated/steered_completion
        experiment_type: "ablation" or "steering"
        title: Figure title
        caption: Optional caption
        steering_value: Optional steering multiplier to display (for steering experiments)
    """
    if not examples:
        return ""

    # Determine the key for the modified completion
    modified_key = "ablated_completion" if experiment_type == "ablation" else "steered_completion"
    modified_label = "Ablated" if experiment_type == "ablation" else "Steered"

    # Extract steering value from first example if not provided
    if steering_value is None and experiment_type == "steering" and examples:
        steering_value = examples[0].get("steering_value")

    # Format steering value for display
    steering_display = ""
    if steering_value is not None and experiment_type == "steering":
        steer_str = f"+{steering_value}" if steering_value > 0 else str(steering_value)
        steering_display = f' <span style="background: #8b5cf6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">multiplier: {steer_str}</span>'

    # Collect examples for potential collapsing
    example_list = []
    for i, ex in enumerate(examples[:10]):  # Limit to 10 examples
        prompt = escape_html(ex.get("prompt", "")[:150])  # Show more of prompt
        baseline = escape_html(ex.get("baseline_completion", ""))
        modified = escape_html(ex.get(modified_key, ex.get("steered_completion", ex.get("ablated_completion", ""))))
        category = ex.get("category", "")

        category_badge = f'<span style="background: #e5e7eb; color: #374151; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-left: 8px;">{escape_html(category)}</span>' if category else ""

        # Truncation indicator
        baseline_truncated = "..." if len(ex.get("baseline_completion", "")) > 500 else ""
        modified_truncated = "..." if len(ex.get(modified_key, "")) > 500 else ""

        example_list.append(f'''
        <div class="completion-example" style="margin-bottom: 16px; padding: 12px; background: var(--bg-secondary, #f8f9fa); border-radius: 8px;">
            <div style="font-size: 12px; color: var(--text-secondary, #666); margin-bottom: 8px;">
                <strong>Prompt {i+1}:</strong> {prompt}...{category_badge}
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                <div style="padding: 8px; background: var(--bg-elevated, #fff); border-radius: 4px; border-left: 3px solid #9ca3af;">
                    <div style="font-size: 11px; color: #6b7280; margin-bottom: 4px;">Baseline:</div>
                    <div style="font-family: monospace; font-size: 13px; color: var(--text-primary, #111); white-space: pre-wrap; word-break: break-word;">{baseline}{baseline_truncated}</div>
                </div>
                <div style="padding: 8px; background: var(--bg-elevated, #fff); border-radius: 4px; border-left: 3px solid #f97316;">
                    <div style="font-size: 11px; color: #f97316; margin-bottom: 4px;">{modified_label}:</div>
                    <div style="font-family: monospace; font-size: 13px; color: var(--text-primary, #111); white-space: pre-wrap; word-break: break-word;">{modified}{modified_truncated}</div>
                </div>
            </div>
        </div>
        ''')

    caption_html = f'<p class="figure-caption">{escape_html(caption)}</p>' if caption else ""

    # Explanation with steering value if applicable
    if experiment_type == "steering":
        explanation = f"These examples show how steering (amplifying the neuron's activation) affected the model's generated text.{steering_display}"
    else:
        explanation = "These examples show how ablation (zeroing out the neuron) affected the model's generated text."

    # Collapsible for many examples
    example_count = len(example_list)
    collapse_threshold = 3
    show_first = 2

    if example_count > collapse_threshold:
        visible_examples = '\n'.join(example_list[:show_first])
        hidden_examples = '\n'.join(example_list[show_first:])
        hidden_count = example_count - show_first

        import hashlib
        unique_id = hashlib.md5(f"completion-{title}-{experiment_type}".encode()).hexdigest()[:8]

        examples_html = f'''
        <div class="examples-collapsible" id="ec-{unique_id}">
            <style>
                #ec-{unique_id} .examples-hidden {{ display: none; }}
                #ec-{unique_id}.expanded .examples-hidden {{ display: block; }}
                #ec-{unique_id}.expanded .expand-btn {{ display: none; }}
                #ec-{unique_id} .collapse-btn {{ display: none; }}
                #ec-{unique_id}.expanded .collapse-btn {{ display: inline-flex; }}
            </style>
            {visible_examples}
            <div class="examples-hidden">
                {hidden_examples}
            </div>
            <div style="text-align: center; margin-top: 12px;">
                <button class="expand-btn" onclick="document.getElementById('ec-{unique_id}').classList.add('expanded')"
                    style="background: var(--bg-secondary, #f3f4f6); border: 1px solid var(--border, #e5e7eb);
                           border-radius: 6px; padding: 6px 16px; font-size: 13px; color: var(--text-secondary, #666);
                           cursor: pointer; display: inline-flex; align-items: center; gap: 6px;">
                    <span>Show {hidden_count} more examples</span>
                    <span style="font-size: 10px;">▼</span>
                </button>
                <button class="collapse-btn" onclick="document.getElementById('ec-{unique_id}').classList.remove('expanded')"
                    style="background: var(--bg-secondary, #f3f4f6); border: 1px solid var(--border, #e5e7eb);
                           border-radius: 6px; padding: 6px 16px; font-size: 13px; color: var(--text-secondary, #666);
                           cursor: pointer; align-items: center; gap: 6px;">
                    <span>Show less</span>
                    <span style="font-size: 10px;">▲</span>
                </button>
            </div>
        </div>
        '''
    else:
        examples_html = '\n'.join(example_list)

    return f'''
    <div class="figure-container completion-examples" style="background: var(--bg-elevated, #ffffff);">
        <div class="figure-title">{escape_html(title)}</div>
        <div class="figure-explanation" style="background: var(--bg-secondary, #f8f9fa); padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 13px; color: var(--text-secondary, #666);">
            <strong>What changed:</strong> {explanation}
        </div>
        {examples_html}
        {caption_html}
    </div>
    '''


def _extract_steering_effects(data_point: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract effects from steering data in various formats.

    Handles:
    - {"effects": [{"token": "X", "shift": 5.0}]}
    - {"promotes": [["X", 5.0]], "suppresses": [["Y", -3.0]]}
    - {"promoted_tokens": [["X", 5.0]], "suppressed_tokens": [["Y", -3.0]]}
    """
    effects = []

    # Try the "effects" format first
    if "effects" in data_point:
        for e in data_point["effects"]:
            if isinstance(e, dict):
                effects.append({"token": e.get("token", ""), "shift": e.get("shift", 0)})
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                effects.append({"token": e[0], "shift": e[1]})

    # Try multiple naming conventions for promotes/suppresses
    for key in ["promotes", "suppresses", "promoted_tokens", "suppressed_tokens"]:
        if key in data_point:
            for item in data_point[key]:
                if isinstance(item, dict):
                    effects.append({"token": item.get("token", ""), "shift": item.get("shift", 0)})
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    effects.append({"token": item[0], "shift": item[1]})

    return effects


def generate_steering_curves(
    dose_response_data: list[dict[str, Any]],
    highlight_tokens: list[str],
    title: str = "Steering Response",
    caption: str = ""
) -> str:
    """Generate dose-response table showing effects at different steering magnitudes.

    Groups results by prompt so each prompt's dose-response curve is clearly separated.
    """

    # Organize by prompt - each prompt gets its own section
    prompts_with_curves = []

    for d in dose_response_data:
        prompt = d.get("prompt", "")[:60]
        if not prompt:
            prompt = "[Unknown prompt]"

        curve_points = []

        # Handle nested dose_response_curve format
        if "dose_response_curve" in d:
            for point in d["dose_response_curve"]:
                steering = point.get("steering_value", 0)
                effects = _extract_steering_effects(point)
                curve_points.append((steering, effects))
        else:
            # Handle flat format - single point
            steering = d.get("steering_value", 0)
            effects = _extract_steering_effects(d)
            if effects:
                curve_points.append((steering, effects))

        if curve_points:
            # Sort curve points by steering value
            curve_points.sort(key=lambda x: x[0])
            prompts_with_curves.append((prompt, curve_points))

    if not prompts_with_curves:
        return f'''
        <div class="figure-container steering-curves">
            <div class="figure-title">{escape_html(title)}</div>
            <p style="color: var(--text-tertiary); text-align: center; padding: 40px;">
                Insufficient dose-response data available
            </p>
        </div>
        '''

    # Build tables for each prompt
    tables_html = ""
    for prompt_idx, (prompt, curve_points) in enumerate(prompts_with_curves[:4]):  # Limit to 4 prompts
        rows_html = ""
        for steering, effects in curve_points:
            # Separate positive and negative effects
            promotes = [(e.get("token", ""), e.get("shift", 0)) for e in effects if e.get("shift", 0) > 0]
            suppresses = [(e.get("token", ""), e.get("shift", 0)) for e in effects if e.get("shift", 0) < 0]

            # Sort by magnitude
            promotes.sort(key=lambda x: -x[1])
            suppresses.sort(key=lambda x: x[1])

            # Steering value cell
            steering_class = "positive" if steering > 0 else ("negative" if steering < 0 else "")
            steering_sign = "+" if steering > 0 else ""

            # Build promotes badges (top 3 for compactness)
            promotes_badges = ""
            for tok, shift in promotes[:3]:
                clean_tok = clean_token(tok)
                promotes_badges += f'<span class="effect-badge promote">{escape_html(clean_tok)} +{shift:.2f}</span>'
            if not promotes_badges:
                promotes_badges = '<span class="no-data">—</span>'

            # Build suppresses badges (top 3 for compactness)
            suppresses_badges = ""
            for tok, shift in suppresses[:3]:
                clean_tok = clean_token(tok)
                suppresses_badges += f'<span class="effect-badge suppress">{escape_html(clean_tok)} {shift:.2f}</span>'
            if not suppresses_badges:
                suppresses_badges = '<span class="no-data">—</span>'

            rows_html += f'''
            <tr>
                <td><span class="steering-value {steering_class}">{steering_sign}{steering}</span></td>
                <td><div class="effect-badges">{promotes_badges}</div></td>
                <td><div class="effect-badges">{suppresses_badges}</div></td>
            </tr>
            '''

        # Add prompt header and table
        tables_html += f'''
        <div class="steering-prompt-group" style="margin-bottom: 20px;">
            <div class="prompt-header" style="font-size: 13px; color: var(--text-secondary); margin-bottom: 8px; padding: 8px 12px; background: var(--bg-inset, #f5f5f7); border-radius: 6px; font-style: italic;">
                "{escape_html(prompt)}..."
            </div>
            <table class="dose-response-table" style="margin-bottom: 0;">
                <thead>
                    <tr>
                        <th style="width: 80px;">Steering</th>
                        <th>Promoted</th>
                        <th>Suppressed</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        '''

    caption_html = f'<p class="figure-caption">{escape_html(caption)}</p>' if caption else ""

    return f'''
    <div class="figure-container steering-curves">
        <div class="figure-title">{escape_html(title)}</div>
        {tables_html}
        {caption_html}
    </div>
    '''


def generate_evidence_card(
    finding: str,
    evidence_type: str = "confirmation",
    supporting_data: str = "",
    caption: str = ""
) -> str:
    """Generate key finding highlight card."""

    icon_map = {
        "confirmation": "✓",
        "refutation": "✗",
        "anomaly": "⚠"
    }
    icon = icon_map.get(evidence_type, "•")

    supporting_html = f'<div class="evidence-data">{escape_html(supporting_data)}</div>' if supporting_data else ""

    return f'''
    <div class="figure-container evidence-card {evidence_type}">
        <span class="evidence-icon">{icon}</span>
        <div class="evidence-content">
            <div class="evidence-finding">{linkify_neuron_ids(escape_html(finding))}</div>
            {supporting_html}
        </div>
    </div>
    '''


def generate_anomaly_box(
    anomaly_description: str,
    expected_behavior: str,
    observed_behavior: str,
    possible_explanations: list[str]
) -> str:
    """Generate special callout for surprising findings."""

    explanations_html = "".join([
        f'<li>{escape_html(exp)}</li>'
        for exp in possible_explanations[:5]
    ])

    return f'''
    <div class="figure-container anomaly-box">
        <div class="anomaly-header">
            <span class="anomaly-icon">⚠️</span>
            <span class="anomaly-title">Unexpected Observation</span>
        </div>
        <p style="margin-bottom: 16px; font-size: 15px;">{linkify_neuron_ids(escape_html(anomaly_description))}</p>
        <div class="comparison-row">
            <div class="comparison-item">
                <div class="comparison-label">Expected</div>
                <div>{escape_html(expected_behavior)}</div>
            </div>
            <div class="comparison-item">
                <div class="comparison-label">Observed</div>
                <div>{escape_html(observed_behavior)}</div>
            </div>
        </div>
        <div>
            <strong style="font-size: 13px;">Possible Explanations:</strong>
            <ul class="explanations-list">
                {explanations_html}
            </ul>
        </div>
    </div>
    '''


def _parse_token_weight(token_str: str) -> tuple[str, float]:
    """Parse token string that may contain embedded weight like '.mount (0.036)'."""
    # Try to extract weight from format "token (weight)" or "token (-weight)"
    match = re.match(r'^(.+?)\s*\(([+-]?\d+\.?\d*)\)$', token_str.strip())
    if match:
        return match.group(1).strip(), float(match.group(2))
    return token_str, 0.0


def generate_output_projections(
    promote: list[dict[str, Any]],
    suppress: list[dict[str, Any]],
    title: str = "Output Projections",
    caption: str = ""
) -> str:
    """Generate output projections section showing promoted/suppressed tokens as compact badges."""

    def extract_weight(t: dict[str, Any]) -> tuple[str, float]:
        """Extract token and weight from various data formats."""
        raw_token = t.get("token", "")
        # Try various weight field names (in order of preference)
        weight = (
            t.get("projection_strength") or  # From tool_get_output_projections
            t.get("weight") or
            t.get("magnitude") or
            t.get("frequency") or
            t.get("value") or
            0
        )
        if weight == 0:
            # Parse from token string if weight is embedded like "token (0.036)"
            parsed_token, parsed_weight = _parse_token_weight(raw_token)
            return clean_token(parsed_token), parsed_weight
        return clean_token(raw_token), float(weight)

    promotes_badges = ""
    for t in promote[:12]:
        token, weight = extract_weight(t)
        promotes_badges += f'''<span class="token-badge"><span class="token-name">{escape_html(token)}</span><span class="token-weight">+{abs(weight):.3f}</span></span>'''

    suppresses_badges = ""
    for t in suppress[:12]:
        token, weight = extract_weight(t)
        suppresses_badges += f'''<span class="token-badge"><span class="token-name">{escape_html(token)}</span><span class="token-weight">−{abs(weight):.3f}</span></span>'''

    caption_html = f'<p class="figure-caption">{escape_html(caption)}</p>' if caption else ""

    promotes_row = f'''
        <div class="projection-row promotes-row">
            <span class="projection-row-label">Promotes</span>
            {promotes_badges if promotes_badges else '<span style="color: #9ca3af; font-size: 13px;">No strong promotions</span>'}
        </div>
    ''' if promote else ""

    suppresses_row = f'''
        <div class="projection-row suppresses-row">
            <span class="projection-row-label">Suppresses</span>
            {suppresses_badges if suppresses_badges else '<span style="color: #9ca3af; font-size: 13px;">No strong suppressions</span>'}
        </div>
    ''' if suppress else ""

    return f'''
    <div class="figure-container output-projections">
        <div class="figure-title">{escape_html(title)}</div>
        {promotes_row}
        {suppresses_row}
        {caption_html}
    </div>
    '''


def generate_homograph_comparison(
    pairs: list[dict[str, Any]],
    title: str = "Semantic Disambiguation",
    caption: str = "",
    explanation: str = ""
) -> str:
    """Generate side-by-side comparison for words with multiple meanings.

    Shows how a neuron discriminates between different contexts for the same word.

    Args:
        pairs: List of dicts with structure:
            {
                "word": "virus",
                "contexts": [
                    {"label": "Malware", "example": "infected computers", "activation": 4.75, "category": "malware"},
                    {"label": "Biological", "example": "infected people", "activation": 0.64, "category": "biological"}
                ]
            }
        title: Figure title
        caption: Optional caption
        explanation: Required narrative explaining the figure's significance
    """
    pairs_html = ""
    for pair in pairs[:6]:  # Limit to 6 pairs
        word = pair.get("word", "")
        contexts = pair.get("contexts", [])

        contexts_html = ""
        for ctx in contexts[:2]:  # Two contexts per word
            label = ctx.get("label", "Context")
            example = ctx.get("example", "")
            activation = ctx.get("activation", 0)
            category = ctx.get("category", "").lower()

            # Determine activation class (threshold at 1.0)
            act_class = "high" if activation > 1.0 else "low"

            # Determine label class based on category
            label_class = "context-label"
            if category in ["malware", "tech", "technical", "cyber", "security"]:
                label_class += " malware"
            elif category in ["biological", "medical", "health", "bio"]:
                label_class += " biological"
            elif category in ["animal", "nature", "wildlife"]:
                label_class += " animal"
            elif category in ["mythology", "myth", "history", "historical"]:
                label_class += " mythology"
            else:
                label_class += " neutral"

            contexts_html += f'''
            <div class="homograph-context">
                <div class="{label_class}">{escape_html(label.upper())}</div>
                <div class="context-example">"{escape_html(example)}"</div>
                <div class="context-activation {act_class}">{activation:.2f}</div>
            </div>
            '''

        pairs_html += f'''
        <div class="homograph-pair">
            <div class="homograph-word">{escape_html(word)}</div>
            <div class="homograph-contexts">
                {contexts_html}
            </div>
        </div>
        '''

    caption_html = f'<p class="figure-caption">{escape_html(caption)}</p>' if caption else ""
    explanation_html = f'<p class="figure-explanation" style="font-size: 14px; color: var(--text-secondary); margin-top: 12px; line-height: 1.6;">{escape_html_preserve_tags(explanation)}</p>' if explanation else ""

    return f'''
    <div class="figure-container homograph-comparison">
        <div class="figure-title">{escape_html(title)}</div>
        <div class="homograph-grid">
            {pairs_html}
        </div>
        {caption_html}
        {explanation_html}
    </div>
    '''


# =============================================================================
# INVESTIGATION FLOW VISUALIZATION
# =============================================================================

def extract_key_experiments(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract key experiments from investigation transcript.

    Parses tool calls to find significant test results:
    - test_activation with notable results
    - batch_activation_test with patterns
    - check_output_projections
    - steering/ablation experiments
    """
    key_experiments = []

    # Tool names that indicate experiments
    experiment_tools = {
        "test_activation": "activation",
        "batch_activation_test": "batch_activation",
        "check_output_projections": "projections",
        "run_ablation": "ablation",
        "run_steering": "steering",
        "analyze_projections": "projections",
    }

    for entry in transcript:
        if entry.get("role") != "assistant":
            continue

        content = entry.get("content", [])
        for block in content:
            if not isinstance(block, dict):
                continue

            # Look for tool_use blocks
            if block.get("type") == "tool_use":
                tool_name = block.get("name", "")
                tool_input = block.get("input", {})

                # Check if this is an experiment tool
                for tool_key, exp_type in experiment_tools.items():
                    if tool_key in tool_name:
                        exp = {
                            "type": exp_type,
                            "tool": tool_key,
                            "input": str(tool_input.get("prompt", tool_input.get("prompts", "")))[:80],
                        }
                        key_experiments.append(exp)
                        break

    return key_experiments[:10]  # Limit to 10 most relevant


def extract_confidence_trajectory(
    hypotheses_tested: list[dict[str, Any]]
) -> list[tuple[str, float, float]]:
    """Extract confidence changes from hypotheses.

    Returns list of (hypothesis_id, prior, posterior) tuples.
    """
    trajectory = []

    for h in hypotheses_tested:
        h_id = h.get("hypothesis_id", h.get("id", "H?"))
        prior = h.get("prior_probability", h.get("prior", 50))
        posterior = h.get("posterior_probability", h.get("posterior", 50))
        trajectory.append((h_id, prior, posterior))

    return trajectory


def extract_skeptic_summary(skeptic_report: dict[str, Any]) -> dict[str, Any]:
    """Extract key information from skeptic report.

    Returns dict with:
    - main_challenges: list of challenges posed
    - boundary_tests: number passed/failed
    - verdict: final verdict
    - revised_hypothesis: refined hypothesis if any
    """
    if not skeptic_report:
        return {}

    # Extract challenges from alternative hypotheses
    challenges = []
    for alt in skeptic_report.get("alternative_hypotheses", []):
        challenges.append({
            "alternative": alt.get("alternative", ""),
            "verdict": alt.get("verdict", ""),
            "evidence": alt.get("evidence", "")[:100],
        })

    # Count boundary test results
    boundary_tests = skeptic_report.get("boundary_tests", [])
    passed = sum(1 for t in boundary_tests if t.get("passed", False))
    failed = len(boundary_tests) - passed

    return {
        "challenges": challenges[:3],
        "boundary_tests_passed": passed,
        "boundary_tests_failed": failed,
        "verdict": skeptic_report.get("verdict", ""),
        "revised_hypothesis": skeptic_report.get("revised_hypothesis", ""),
        "key_challenges": skeptic_report.get("key_challenges", []),
        "confidence_adjustment": skeptic_report.get("confidence_adjustment", 0),
    }


def generate_investigation_flow(
    investigation_data: dict[str, Any],
    pi_result: dict[str, Any] | None = None,
    skeptic_report: dict[str, Any] | None = None,
    prior_knowledge: dict[str, Any] | None = None,
    title: str = "Investigation Flow",
    caption: str = ""
) -> str:
    """Generate an Investigation Flow visualization.

    Shows the evolution of hypotheses through experiments, skeptic challenges,
    and reviewer iterations. Creates a visual narrative of how the investigation
    arrived at its conclusions.

    Args:
        investigation_data: Full investigation JSON with transcript, hypotheses, etc.
        pi_result: Optional PI result with review history and verdict
        skeptic_report: Optional skeptic challenge report
        prior_knowledge: Optional prior knowledge dict containing llm_labels with
            input_label, input_description, output_label (function_label),
            output_description (function_description) - these are the "claims to verify"
        title: Figure title
        caption: Optional caption
    """
    # Extract data
    characterization = investigation_data.get("characterization", {})
    hypotheses_tested = investigation_data.get("hypotheses_tested", [])
    evidence = investigation_data.get("evidence", {})
    total_experiments = investigation_data.get("total_experiments", 0)
    final_confidence = investigation_data.get("confidence", 0.5)
    if isinstance(final_confidence, (int, float)):
        final_confidence = float(final_confidence)
    else:
        final_confidence = 0.5

    # =================================================================
    # PHASE 1: CLAIMS TO VERIFY (from prior knowledge / LLM labels)
    # These are the seed hypotheses the agent was asked to test
    # =================================================================
    llm_labels = prior_knowledge.get("llm_labels", {}) if prior_knowledge else {}

    # Extract prior claims (these came from LLM analysis of max-activating examples & projections)
    prior_input_label = llm_labels.get("input_label", "")
    prior_input_desc = llm_labels.get("input_description", "")
    prior_output_label = llm_labels.get("output_label", llm_labels.get("function_label", ""))
    prior_output_desc = llm_labels.get("output_description", llm_labels.get("function_description", ""))

    # Also check for original labels saved in characterization (if available)
    if not prior_input_label:
        prior_input_label = characterization.get("original_input_label", "")
    if not prior_input_desc:
        prior_input_desc = characterization.get("original_input_description", "")
    if not prior_output_label:
        prior_output_label = characterization.get("original_output_label", "")
    if not prior_output_desc:
        prior_output_desc = characterization.get("original_output_description", "")

    has_prior_claims = bool(prior_input_label or prior_output_label)

    if has_prior_claims:
        # Format claim boxes
        input_claim_html = ""
        if prior_input_label or prior_input_desc:
            input_claim_html = f'''
            <div class="claim-box">
                <div class="claim-type">INPUT FUNCTION CLAIM</div>
                {f'<div class="claim-label">{escape_html(prior_input_label)}</div>' if prior_input_label else ''}
                {f'<div class="claim-description">{escape_html(prior_input_desc)}</div>' if prior_input_desc else ''}
            </div>
            '''

        output_claim_html = ""
        if prior_output_label or prior_output_desc:
            output_claim_html = f'''
            <div class="claim-box">
                <div class="claim-type">OUTPUT FUNCTION CLAIM</div>
                {f'<div class="claim-label">{escape_html(prior_output_label)}</div>' if prior_output_label else ''}
                {f'<div class="claim-description">{escape_html(prior_output_desc)}</div>' if prior_output_desc else ''}
            </div>
            '''

        claims_html = f'''
        <div class="flow-phase claims-to-verify">
            <div class="phase-header">
                <span class="phase-icon">🎯</span>
                <h4>Claims to Verify</h4>
                <span class="phase-subtitle">from prior LLM analysis</span>
            </div>
            <div class="phase-note">
                These claims were generated by analyzing max-activating examples and projection weights.
                The agent's task was to test them experimentally.
            </div>
            {input_claim_html}
            {output_claim_html}
        </div>
        '''
    else:
        # No prior claims available - note this
        claims_html = '''
        <div class="flow-phase claims-to-verify no-claims">
            <div class="phase-header">
                <span class="phase-icon">🎯</span>
                <h4>Claims to Verify</h4>
            </div>
            <div class="phase-note">
                No prior LLM labels available. The agent started with an exploratory investigation.
            </div>
        </div>
        '''

    # =================================================================
    # PHASE 2: HYPOTHESIS EVOLUTION (show updates with probabilities)
    # =================================================================
    hypothesis_updates_html = ""
    if hypotheses_tested:
        updates_list = ""
        for h in hypotheses_tested:
            h_id = h.get("hypothesis_id", h.get("id", "H?"))
            hypothesis_text = h.get("hypothesis", "")
            # Handle None values explicitly - use 50 as default
            prior = h.get("prior_probability") or h.get("prior") or 50
            posterior = h.get("posterior_probability") or h.get("posterior") or 50
            # Ensure they're numeric
            prior = int(prior) if prior is not None else 50
            posterior = int(posterior) if posterior is not None else 50
            status = h.get("status", "unknown")
            evidence_summary = h.get("evidence_summary", "")

            # Determine status styling
            if status == "confirmed":
                status_class = "confirmed"
                status_icon = "✓"
            elif status == "refuted":
                status_class = "refuted"
                status_icon = "✗"
            else:
                status_class = "inconclusive"
                status_icon = "?"

            # Calculate probability change
            prob_change = posterior - prior
            prob_direction = "↑" if prob_change > 0 else ("↓" if prob_change < 0 else "→")
            prob_class = "increased" if prob_change > 0 else ("decreased" if prob_change < 0 else "unchanged")

            updates_list += f'''
            <div class="hypothesis-update {status_class}">
                <div class="hypothesis-update-header">
                    <span class="hypothesis-id">{escape_html(h_id)}</span>
                    <span class="hypothesis-status {status_class}">{status_icon} {escape_html(status)}</span>
                    <span class="probability-change {prob_class}">
                        {prior}% {prob_direction} {posterior}%
                    </span>
                </div>
                <div class="hypothesis-update-text">{escape_html(hypothesis_text)}</div>
                {f'<div class="hypothesis-evidence">{escape_html(evidence_summary)}</div>' if evidence_summary else ''}
            </div>
            '''

        hypothesis_updates_html = f'''
        <div class="flow-connector">▼</div>
        <div class="flow-phase hypothesis-updates">
            <div class="phase-header">
                <span class="phase-icon">🔄</span>
                <h4>Hypothesis Testing</h4>
                <span class="update-count">{len(hypotheses_tested)} hypotheses</span>
            </div>
            <div class="updates-list">
                {updates_list}
            </div>
        </div>
        '''

    # =================================================================
    # PHASE 3: EXPLORATION SUMMARY (clearer experiment descriptions)
    # =================================================================
    activating_prompts = evidence.get("activating_prompts", [])
    non_activating = evidence.get("non_activating_prompts", [])
    relp_results = evidence.get("relp_results", investigation_data.get("relp_results", []))

    exploration_items = ""

    # Show activation testing summary
    if activating_prompts or non_activating:
        n_activating = len(activating_prompts)
        n_non_activating = len(non_activating)
        avg_activation = sum(p.get("activation", 0) for p in activating_prompts) / max(n_activating, 1)
        avg_non_activation = sum(p.get("activation", 0) for p in non_activating) / max(n_non_activating, 1)

        # Show a few example prompts with full context
        example_activating = ""
        for p in activating_prompts[:2]:
            prompt = p.get("prompt", "")
            activation = p.get("activation", 0)
            example_activating += f'''
            <div class="experiment-example positive">
                <span class="example-prompt">"{escape_html(prompt)}"</span>
                <span class="example-result">activation: {activation:.2f}</span>
            </div>
            '''

        example_non_activating = ""
        for p in non_activating[:2]:
            prompt = p.get("prompt", "")
            activation = p.get("activation", 0)
            example_non_activating += f'''
            <div class="experiment-example negative">
                <span class="example-prompt">"{escape_html(prompt)}"</span>
                <span class="example-result">activation: {activation:.2f}</span>
            </div>
            '''

        exploration_items += f'''
        <div class="exploration-category">
            <div class="exploration-category-header">
                <span class="category-name">Activation Testing</span>
                <span class="category-stats">{n_activating} activating (avg {avg_activation:.2f}) · {n_non_activating} non-activating (avg {avg_non_activation:.2f})</span>
            </div>
            <div class="exploration-examples">
                {example_activating}
                {example_non_activating}
            </div>
        </div>
        '''

    # Show RelP testing summary
    if relp_results:
        relp_found = sum(1 for r in relp_results if r.get("in_causal_pathway", False))
        relp_total = len(relp_results)

        relp_examples = ""
        for r in relp_results[:2]:
            prompt = r.get("prompt", "")
            found = r.get("in_causal_pathway", False)
            score = r.get("neuron_relp_score")
            target = r.get("target_tokens", [])
            target_str = ", ".join(target) if target else "auto"
            result_class = "positive" if found else "negative"

            relp_examples += f'''
            <div class="experiment-example {result_class}">
                <span class="example-prompt">"{escape_html(prompt)}" → {escape_html(target_str)}</span>
                <span class="example-result">{"found" if found else "not found"}{f" (score: {score:.3f})" if score else ""}</span>
            </div>
            '''

        exploration_items += f'''
        <div class="exploration-category">
            <div class="exploration-category-header">
                <span class="category-name">Causal Pathway (RelP)</span>
                <span class="category-stats">{relp_found}/{relp_total} prompts found neuron in causal pathway</span>
            </div>
            <div class="exploration-examples">
                {relp_examples}
            </div>
        </div>
        '''

    exploration_html = f'''
    <div class="flow-connector">▼</div>
    <div class="flow-phase exploration">
        <div class="phase-header">
            <span class="phase-icon">🔬</span>
            <h4>Exploration Phase</h4>
            <span class="experiment-count">{total_experiments} total experiments</span>
        </div>
        <div class="exploration-categories">
            {exploration_items if exploration_items else '<div class="no-data">No detailed experiment data available</div>'}
        </div>
    </div>
    '''

    # =================================================================
    # PHASE 4: SKEPTIC CHALLENGE (show BOTH passed and failed tests)
    # =================================================================
    skeptic_html = ""
    if skeptic_report:
        alternative_hypotheses = skeptic_report.get("alternative_hypotheses", [])
        boundary_tests = skeptic_report.get("boundary_tests", [])
        verdict = skeptic_report.get("verdict", "UNKNOWN")
        revised_hypothesis = skeptic_report.get("revised_hypothesis", "")

        # Show alternative hypotheses tested
        alternatives_html = ""
        for alt in alternative_hypotheses:
            alt_text = alt.get("alternative", "")
            alt_verdict = alt.get("verdict", "")
            alt_evidence = alt.get("evidence", "")

            if alt_verdict == "refuted":
                verdict_class = "refuted"
                verdict_icon = "✗"
            elif "partial" in alt_verdict:
                verdict_class = "partial"
                verdict_icon = "~"
            else:
                verdict_class = "supported"
                verdict_icon = "✓"

            alternatives_html += f'''
            <div class="alternative-hypothesis {verdict_class}">
                <div class="alternative-header">
                    <span class="alternative-verdict">{verdict_icon} {escape_html(alt_verdict)}</span>
                </div>
                <div class="alternative-text">{escape_html(alt_text)}</div>
                <div class="alternative-evidence">{escape_html(alt_evidence)}</div>
            </div>
            '''

        # Show boundary tests - BOTH passed and failed
        passed_tests = [t for t in boundary_tests if t.get("passed", False)]
        failed_tests = [t for t in boundary_tests if not t.get("passed", False)]

        boundary_html = ""
        if failed_tests:
            failed_items = ""
            for t in failed_tests:
                desc = t.get("description", "")
                prompt = t.get("prompt", "")
                expected = t.get("expected_behavior", "")
                actual = t.get("actual_activation", 0)
                notes = t.get("notes", "")

                failed_items += f'''
                <div class="boundary-test failed">
                    <div class="boundary-test-header">✗ {escape_html(desc)}</div>
                    <div class="boundary-test-detail">
                        <span class="test-prompt">"{escape_html(prompt)}"</span>
                        <span class="test-expected">Expected: {escape_html(expected)}</span>
                        <span class="test-actual">Actual: {actual}</span>
                    </div>
                    {f'<div class="boundary-test-notes">{escape_html(notes)}</div>' if notes else ''}
                </div>
                '''

            boundary_html += f'''
            <div class="boundary-section failed-section">
                <div class="boundary-section-header">Failed Tests ({len(failed_tests)})</div>
                {failed_items}
            </div>
            '''

        if passed_tests:
            # Show just a summary for passed tests unless there are few
            if len(passed_tests) <= 3:
                passed_items = ""
                for t in passed_tests:
                    desc = t.get("description", "")
                    passed_items += f'<div class="boundary-test-compact passed">✓ {escape_html(desc)}</div>'
            else:
                passed_items = f'<div class="boundary-test-compact passed">✓ {len(passed_tests)} boundary tests passed</div>'

            boundary_html += f'''
            <div class="boundary-section passed-section">
                <div class="boundary-section-header">Passed Tests ({len(passed_tests)})</div>
                {passed_items}
            </div>
            '''

        skeptic_html = f'''
        <div class="flow-connector">▼</div>
        <div class="flow-phase skeptic">
            <div class="phase-header">
                <span class="phase-icon">🔍</span>
                <h4>Skeptic Challenge</h4>
                <span class="verdict-badge {"supported" if verdict == "SUPPORTED" else "modified"}">{escape_html(verdict)}</span>
            </div>
            {f'<div class="alternatives-section"><div class="section-subheader">Alternative Hypotheses Tested</div>{alternatives_html}</div>' if alternatives_html else ''}
            {f'<div class="boundary-tests-section"><div class="section-subheader">Boundary Tests</div>{boundary_html}</div>' if boundary_html else ''}
            {f'<div class="revised-hypothesis"><div class="section-subheader">Revised Hypothesis</div><div class="revised-text">{escape_html(revised_hypothesis)}</div></div>' if revised_hypothesis else ''}
        </div>
        '''

    # =================================================================
    # PHASE 5: REVIEWER FEEDBACK (if pi_result available)
    # =================================================================
    reviewer_html = ""
    if pi_result:
        review_history = pi_result.get("review_history", [])
        final_verdict = pi_result.get("final_verdict", "UNKNOWN")
        iterations = pi_result.get("iterations", len(review_history))

        reviews_html = ""
        for i, review in enumerate(review_history):
            iteration = review.get("iteration", i + 1)
            verdict = review.get("verdict", "")
            conf_assessment = review.get("confidence_assessment", "")
            gaps = review.get("gaps", [])

            gaps_html = "".join([f'<li>{escape_html(gap)}</li>' for gap in gaps])

            reviews_html += f'''
            <div class="review-iteration">
                <div class="review-iteration-header">
                    <span class="iteration-number">Iteration {iteration}</span>
                    <span class="iteration-verdict">{escape_html(verdict)}</span>
                    {f'<span class="confidence-assessment">{escape_html(conf_assessment)}</span>' if conf_assessment else ''}
                </div>
                {f'<ul class="review-gaps">{gaps_html}</ul>' if gaps_html else ''}
            </div>
            '''

        reviewer_html = f'''
        <div class="flow-connector">▼</div>
        <div class="flow-phase reviewer">
            <div class="phase-header">
                <span class="phase-icon">📝</span>
                <h4>Reviewer Feedback</h4>
                <span class="iteration-count">{iterations} iterations</span>
            </div>
            <div class="reviews-list">
                {reviews_html}
            </div>
            <div class="final-verdict-box">
                <span class="verdict-label">Final Verdict:</span>
                <span class="verdict-value">{escape_html(final_verdict)}</span>
            </div>
        </div>
        '''

    # =================================================================
    # PHASE 6: FINAL CONCLUSION (the result of the investigation)
    # =================================================================
    final_input = characterization.get("input_function", "")
    final_output = characterization.get("output_function", "")
    final_hypothesis = characterization.get("final_hypothesis", "")
    if not final_hypothesis and skeptic_report:
        final_hypothesis = skeptic_report.get("revised_hypothesis", "")

    key_findings = investigation_data.get("key_findings", [])

    findings_html = ""
    for finding in key_findings:
        findings_html += f'<li>{escape_html(finding)}</li>'

    # Build final characterization section
    final_char_html = ""
    if final_input or final_output:
        input_box = ""
        if final_input:
            # Show if input function changed from prior claim
            changed_input = has_prior_claims and prior_input_label and final_input != prior_input_label
            input_box = f'''
            <div class="conclusion-box">
                <div class="conclusion-type">INPUT FUNCTION</div>
                <div class="conclusion-text">{escape_html(final_input)}</div>
                {'<div class="conclusion-change-note">Refined from prior claim</div>' if changed_input else ''}
            </div>
            '''

        output_box = ""
        if final_output:
            # Show if output function changed from prior claim
            changed_output = has_prior_claims and prior_output_label and final_output != prior_output_label
            output_box = f'''
            <div class="conclusion-box">
                <div class="conclusion-type">OUTPUT FUNCTION</div>
                <div class="conclusion-text">{escape_html(final_output)}</div>
                {'<div class="conclusion-change-note">Refined from prior claim</div>' if changed_output else ''}
            </div>
            '''

        final_char_html = f'''
        <div class="final-characterization">
            <div class="section-subheader">Final Characterization</div>
            {input_box}
            {output_box}
        </div>
        '''

    conclusion_html = f'''
    <div class="flow-connector">▼</div>
    <div class="flow-phase conclusion">
        <div class="phase-header">
            <span class="phase-icon">✅</span>
            <h4>Final Conclusion</h4>
            <span class="confidence-badge">{int(final_confidence * 100)}% confidence</span>
        </div>
        {f'<div class="final-hypothesis-box"><div class="final-hypothesis-text">{escape_html(final_hypothesis)}</div></div>' if final_hypothesis else ''}
        {final_char_html}
        {f'<div class="key-findings-box"><div class="section-subheader">Key Findings</div><ul class="findings-list">{findings_html}</ul></div>' if findings_html else ''}
    </div>
    '''

    caption_html = f'<p class="figure-caption">{escape_html(caption)}</p>' if caption else ""

    return f'''
    <div class="figure-container investigation-flow">
        <div class="figure-title">{escape_html(title)}</div>
        <div class="flow-timeline">
            {claims_html}
            {hypothesis_updates_html}
            {exploration_html}
            {skeptic_html}
            {reviewer_html}
            {conclusion_html}
        </div>
        {caption_html}
    </div>
    '''


# =============================================================================
# AGENT FLOW CSS (for the new agent-focused investigation flow)
# =============================================================================

AGENT_FLOW_CSS = """
/* Agent Flow Visualization - Shows each agent's contribution clearly */
.agent-flow {
    position: relative;
}

/* Agent Section Base */
.agent-section {
    position: relative;
    margin-bottom: 0;
}

/* Agent Header - the colored tab showing who contributed */
.agent-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 20px;
    border-radius: 12px 12px 0 0;
    position: relative;
    color: white;
}

.agent-header.investigator { background: #3b82f6; }
.agent-header.skeptic { background: #ef4444; }
.agent-header.reviewer { background: #f59e0b; }
.agent-header.conclusion { background: #22c55e; }

.agent-icon { font-size: 20px; }

.agent-name {
    font-weight: 600;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.agent-role {
    font-size: 13px;
    opacity: 0.9;
    margin-left: auto;
}

/* Agent Content */
.agent-content {
    background: var(--bg-elevated, #ffffff);
    border: 2px solid var(--border, #e5e5e5);
    border-top: none;
    border-radius: 0 0 12px 12px;
    padding: 24px;
    margin-bottom: 24px;
}

.agent-section.investigator .agent-content {
    border-color: #3b82f6;
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
}

.agent-section.skeptic .agent-content {
    border-color: #ef4444;
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
}

.agent-section.reviewer .agent-content {
    border-color: #f59e0b;
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
}

.agent-section.conclusion .agent-content {
    border-color: #22c55e;
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
}

/* Flow Connector Arrow */
.agent-flow .flow-connector {
    display: flex;
    justify-content: center;
    padding: 8px 0;
}

.agent-flow .flow-arrow {
    width: 2px;
    height: 32px;
    background: var(--border, #e5e5e5);
    position: relative;
}

.agent-flow .flow-arrow::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 8px solid var(--border, #e5e5e5);
}

/* Content Cards within agents */
.agent-content .content-card {
    background: white;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.agent-content .content-card:last-child { margin-bottom: 0; }

.agent-content .card-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-tertiary, #888);
    margin-bottom: 8px;
}

.agent-content .card-text {
    font-size: 14px;
    line-height: 1.6;
    color: var(--text, #111);
}

/* Hypothesis Cards */
.agent-content .hypothesis-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.agent-content .hypothesis-card {
    background: white;
    border-radius: 8px;
    padding: 16px 20px;
    border-left: 4px solid var(--border, #e5e5e5);
}

.agent-content .hypothesis-card.confirmed { border-left-color: #16a34a; }
.agent-content .hypothesis-card.refuted { border-left-color: #dc2626; }
.agent-content .hypothesis-card.inconclusive { border-left-color: #d97706; }

.agent-content .hypothesis-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
    flex-wrap: wrap;
}

.agent-content .hypothesis-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary, #555);
    background: rgba(0,0,0,0.05);
    padding: 2px 8px;
    border-radius: 4px;
}

.agent-content .hypothesis-status {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}

.agent-content .hypothesis-status.confirmed { color: #16a34a; }
.agent-content .hypothesis-status.refuted { color: #dc2626; }
.agent-content .hypothesis-status.inconclusive { color: #d97706; }

.agent-content .probability-shift {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    margin-left: auto;
}

.agent-content .probability-shift.increased { color: #16a34a; }
.agent-content .probability-shift.decreased { color: #dc2626; }

.agent-content .hypothesis-text {
    font-size: 14px;
    color: var(--text, #111);
    line-height: 1.5;
    margin-bottom: 10px;
}

.agent-content .hypothesis-evidence {
    font-size: 13px;
    color: var(--text-secondary, #555);
    padding-top: 10px;
    border-top: 1px solid rgba(0,0,0,0.06);
    line-height: 1.5;
}

/* Experiment Summary Stats */
.agent-content .experiment-summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}

.agent-content .experiment-stat {
    background: white;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}

.agent-content .stat-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--text, #111);
    letter-spacing: -1px;
}

.agent-content .stat-value.positive { color: #16a34a; }
.agent-content .stat-value.negative { color: #dc2626; }
.agent-content .stat-value.accent { color: #3b82f6; }

.agent-content .stat-label {
    font-size: 12px;
    color: var(--text-secondary, #555);
    margin-top: 4px;
}

/* Example List */
.agent-content .example-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.agent-content .example-item {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
    padding: 10px 14px;
    background: white;
    border-radius: 6px;
    border-left: 3px solid var(--border, #e5e5e5);
    font-size: 13px;
}

.agent-content .example-item.positive { border-left-color: #16a34a; }
.agent-content .example-item.negative { border-left-color: #dc2626; }

.agent-content .example-prompt {
    color: var(--text-secondary, #555);
    flex: 1;
    font-style: italic;
}

.agent-content .example-result {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--text-tertiary, #888);
    white-space: nowrap;
}

/* Section Divider */
.agent-content .section-divider {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-secondary, #555);
    margin: 20px 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(0,0,0,0.1);
}

.agent-content .section-divider:first-child { margin-top: 0; }

/* Verdict Banner */
.agent-content .verdict-banner {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: white;
    border-radius: 8px;
    margin-bottom: 16px;
}

.agent-content .verdict-badge {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 16px;
}

.agent-content .verdict-badge.supported {
    background: rgba(34, 197, 94, 0.15);
    color: #16a34a;
}

.agent-content .verdict-badge.modified {
    background: rgba(245, 158, 11, 0.15);
    color: #d97706;
}

.agent-content .verdict-badge.rejected {
    background: rgba(220, 53, 69, 0.15);
    color: #dc2626;
}

.agent-content .verdict-text {
    font-size: 13px;
    color: var(--text-secondary, #555);
}

/* Alternative Hypotheses */
.agent-content .alternative-card {
    background: white;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    border-left: 4px solid var(--border, #e5e5e5);
}

.agent-content .alternative-card.rejected { border-left-color: #16a34a; }
.agent-content .alternative-card.partial { border-left-color: #d97706; }
.agent-content .alternative-card.supported { border-left-color: #dc2626; }

.agent-content .alternative-verdict {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    margin-bottom: 6px;
}

.agent-content .alternative-card.rejected .alternative-verdict { color: #16a34a; }
.agent-content .alternative-card.partial .alternative-verdict { color: #d97706; }
.agent-content .alternative-card.supported .alternative-verdict { color: #dc2626; }

.agent-content .alternative-text {
    font-size: 14px;
    color: var(--text, #111);
    line-height: 1.5;
    margin-bottom: 8px;
}

.agent-content .alternative-evidence {
    font-size: 13px;
    color: var(--text-secondary, #555);
    line-height: 1.4;
}

/* Boundary Tests */
.agent-content .boundary-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}

@media (max-width: 700px) {
    .agent-content .boundary-grid {
        grid-template-columns: 1fr;
    }
}

.agent-content .boundary-column h4 {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
}

.agent-content .boundary-column.passed h4 { color: #16a34a; }
.agent-content .boundary-column.failed h4 { color: #dc2626; }

.agent-content .boundary-test {
    background: white;
    border-radius: 6px;
    padding: 10px 12px;
    margin-bottom: 8px;
    font-size: 13px;
    border-left: 3px solid var(--border, #e5e5e5);
}

.agent-content .boundary-test.passed { border-left-color: #16a34a; }
.agent-content .boundary-test.failed { border-left-color: #dc2626; }

.agent-content .boundary-test-name {
    font-weight: 500;
    margin-bottom: 4px;
}

.agent-content .boundary-test-result {
    font-size: 12px;
    color: var(--text-secondary, #555);
}

/* Review Cards */
.agent-content .review-card {
    background: white;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.agent-content .review-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
    flex-wrap: wrap;
}

.agent-content .iteration-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary, #555);
}

.agent-content .review-verdict {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 12px;
    background: rgba(0,0,0,0.05);
}

.agent-content .review-verdict.accept {
    background: rgba(34, 197, 94, 0.15);
    color: #16a34a;
}

.agent-content .review-verdict.changes {
    background: rgba(245, 158, 11, 0.15);
    color: #d97706;
}

.agent-content .review-verdict.reject {
    background: rgba(220, 53, 69, 0.15);
    color: #dc2626;
}

.agent-content .confidence-flag {
    font-size: 11px;
    color: #dc2626;
    background: rgba(220, 53, 69, 0.1);
    padding: 2px 8px;
    border-radius: 4px;
}

.agent-content .review-gaps {
    margin: 0;
    padding-left: 20px;
    font-size: 13px;
    color: var(--text-secondary, #555);
    line-height: 1.5;
}

.agent-content .review-gaps li { margin-bottom: 6px; }

/* Conclusion Box */
.agent-content .conclusion-box {
    background: white;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

.agent-content .conclusion-type {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #22c55e;
    margin-bottom: 8px;
}

.agent-content .conclusion-text {
    font-size: 15px;
    color: var(--text, #111);
    line-height: 1.6;
}

/* Confidence Badge */
.agent-content .confidence-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    background: rgba(34, 197, 94, 0.1);
    border-radius: 100px;
    font-size: 13px;
    font-weight: 500;
    color: #16a34a;
}

.agent-content .confidence-badge.low {
    background: rgba(220, 53, 69, 0.1);
    color: #dc2626;
}

.agent-content .confidence-badge.medium {
    background: rgba(245, 158, 11, 0.1);
    color: #d97706;
}

/* Key Findings */
.agent-content .findings-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.agent-content .finding-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    font-size: 13px;
    color: var(--text-secondary, #555);
    padding: 8px 12px;
    background: white;
    border-radius: 6px;
    border-left: 3px solid #22c55e;
}

.agent-content .finding-icon {
    color: #22c55e;
    flex-shrink: 0;
}

.agent-content .finding-item.warning {
    border-left-color: #f59e0b;
}

.agent-content .finding-item.warning .finding-icon {
    color: #f59e0b;
}
"""


def generate_agent_flow(
    investigation_data: dict[str, Any],
    pi_result: dict[str, Any] | None = None,
    skeptic_report: dict[str, Any] | None = None,
    prior_knowledge: dict[str, Any] | None = None,
    title: str = "Investigation Flow",
) -> str:
    """Generate an Agent-focused Investigation Flow visualization.

    This version clearly distinguishes each agent's contribution:
    - Investigator Agent: Hypothesis formation & experimental testing
    - Skeptic Agent: Adversarial testing & alternative hypotheses
    - Reviewer Agent: Quality control & confidence calibration
    - Final Conclusion: Validated characterization

    Args:
        investigation_data: Full investigation JSON with transcript, hypotheses, etc.
        pi_result: Optional PI result with review history and verdict
        skeptic_report: Optional skeptic challenge report
        prior_knowledge: Optional prior knowledge dict containing llm_labels
        title: Figure title
    """
    # Extract data
    neuron_id = investigation_data.get("neuron_id", "Unknown")
    characterization = investigation_data.get("characterization", {})
    hypotheses_tested = investigation_data.get("hypotheses_tested", [])
    evidence = investigation_data.get("evidence", {})
    total_experiments = investigation_data.get("total_experiments", 0)
    key_findings = investigation_data.get("key_findings", [])
    final_confidence = investigation_data.get("confidence", 0.5)
    if isinstance(final_confidence, (int, float)):
        final_confidence = float(final_confidence)
    else:
        final_confidence = 0.5

    # Get prior claims
    llm_labels = prior_knowledge.get("llm_labels", {}) if prior_knowledge else {}
    prior_input_desc = llm_labels.get("input_description", "")
    prior_output_desc = llm_labels.get("output_description", llm_labels.get("function_description", ""))

    # =================================================================
    # INVESTIGATOR AGENT SECTION
    # =================================================================

    # Initial hypotheses
    initial_hypotheses_html = ""
    if prior_input_desc or prior_output_desc:
        if prior_input_desc:
            initial_hypotheses_html += f'''
                <div class="content-card">
                    <div class="card-label">Input Trigger Hypothesis</div>
                    <div class="card-text">{escape_html(prior_input_desc)}</div>
                </div>
            '''
        if prior_output_desc:
            initial_hypotheses_html += f'''
                <div class="content-card">
                    <div class="card-label">Output Function Hypothesis</div>
                    <div class="card-text">{escape_html(prior_output_desc)}</div>
                </div>
            '''
    else:
        # Use characterization if no prior
        input_func = characterization.get("input_function", "")
        output_func = characterization.get("output_function", "")
        if input_func:
            initial_hypotheses_html += f'''
                <div class="content-card">
                    <div class="card-label">Input Trigger</div>
                    <div class="card-text">{escape_html(input_func)}</div>
                </div>
            '''
        if output_func:
            initial_hypotheses_html += f'''
                <div class="content-card">
                    <div class="card-label">Output Function</div>
                    <div class="card-text">{escape_html(output_func)}</div>
                </div>
            '''

    # Hypothesis testing results
    hypothesis_cards_html = ""
    confirmed_count = 0
    for h in hypotheses_tested:
        h_id = h.get("hypothesis_id", h.get("id", "H?"))
        hypothesis_text = h.get("hypothesis", "")
        if not hypothesis_text:
            continue

        prior = h.get("prior_probability") or h.get("prior") or 50
        posterior = h.get("posterior_probability") or h.get("posterior") or 50
        prior = int(prior) if prior is not None else 50
        posterior = int(posterior) if posterior is not None else 50
        status = h.get("status", "unknown")
        evidence_summary = h.get("evidence_summary", "")

        if status == "confirmed":
            status_class = "confirmed"
            status_text = "Confirmed"
            confirmed_count += 1
        elif status == "refuted":
            status_class = "refuted"
            status_text = "Refuted"
        else:
            status_class = "inconclusive"
            status_text = status.replace("_", " ").title() if status else "Unknown"

        prob_change = posterior - prior
        prob_direction = "→" if prob_change == 0 else ("↑" if prob_change > 0 else "↓")
        prob_class = "increased" if prob_change > 0 else ("decreased" if prob_change < 0 else "")

        hypothesis_cards_html += f'''
            <div class="hypothesis-card {status_class}">
                <div class="hypothesis-header">
                    <span class="hypothesis-id">{escape_html(h_id)}</span>
                    <span class="hypothesis-status {status_class}">{status_text}</span>
                    <span class="probability-shift {prob_class}">{prior}% {prob_direction} {posterior}%</span>
                </div>
                <div class="hypothesis-text">{escape_html(hypothesis_text)}</div>
                {f'<div class="hypothesis-evidence">{escape_html(evidence_summary)}</div>' if evidence_summary else ''}
            </div>
        '''

    # Experiment summary
    activating_prompts = evidence.get("activating_prompts", [])
    non_activating = evidence.get("non_activating_prompts", [])
    n_activating = len(activating_prompts)
    n_non_activating = len(non_activating)
    avg_activation = sum(p.get("activation", 0) for p in activating_prompts) / max(n_activating, 1)

    experiment_summary_html = f'''
        <div class="experiment-summary">
            <div class="experiment-stat">
                <div class="stat-value accent">{total_experiments}</div>
                <div class="stat-label">Total Experiments</div>
            </div>
            <div class="experiment-stat">
                <div class="stat-value positive">{n_activating}</div>
                <div class="stat-label">Activating Prompts</div>
            </div>
            <div class="experiment-stat">
                <div class="stat-value negative">{n_non_activating}</div>
                <div class="stat-label">Non-Activating</div>
            </div>
            <div class="experiment-stat">
                <div class="stat-value">{avg_activation:.1f}</div>
                <div class="stat-label">Mean Activation</div>
            </div>
        </div>
    '''

    # Example prompts
    example_items_html = ""
    for p in activating_prompts[:2]:
        prompt = p.get("prompt", "")[:80]
        if len(p.get("prompt", "")) > 80:
            prompt += "..."
        activation = p.get("activation", 0)
        example_items_html += f'''
            <div class="example-item positive">
                <span class="example-prompt">"{escape_html(prompt)}"</span>
                <span class="example-result">act: {activation:.2f}</span>
            </div>
        '''
    for p in non_activating[:2]:
        prompt = p.get("prompt", "")[:80]
        if len(p.get("prompt", "")) > 80:
            prompt += "..."
        activation = p.get("activation", 0)
        example_items_html += f'''
            <div class="example-item negative">
                <span class="example-prompt">"{escape_html(prompt)}"</span>
                <span class="example-result">act: {activation:.2f}</span>
            </div>
        '''

    investigator_html = f'''
        <div class="agent-section investigator">
            <div class="agent-header investigator">
                <span class="agent-icon">🔬</span>
                <span class="agent-name">Investigator Agent</span>
                <span class="agent-role">Hypothesis formation & experimental testing</span>
            </div>
            <div class="agent-content">
                <div class="section-divider">Initial Hypotheses</div>
                {initial_hypotheses_html if initial_hypotheses_html else '<div class="content-card"><div class="card-text">No initial hypotheses recorded.</div></div>'}

                {f'<div class="section-divider">Hypothesis Testing</div><div class="hypothesis-list">{hypothesis_cards_html}</div>' if hypothesis_cards_html else ''}

                <div class="section-divider">Exploration Results</div>
                {experiment_summary_html}
                {f'<div class="example-list">{example_items_html}</div>' if example_items_html else ''}
            </div>
        </div>
    '''

    # =================================================================
    # SKEPTIC AGENT SECTION
    # =================================================================
    skeptic_html = ""
    if skeptic_report:
        alternative_hypotheses = skeptic_report.get("alternative_hypotheses", [])
        boundary_tests = skeptic_report.get("boundary_tests", [])
        verdict = skeptic_report.get("verdict", "UNKNOWN")
        revised_hypothesis = skeptic_report.get("revised_hypothesis", "")

        # Verdict banner
        verdict_class = "supported" if verdict == "SUPPORTED" else "modified"
        verdict_text = "Original hypothesis survived skeptical challenge" if verdict == "SUPPORTED" else "Hypothesis refined based on challenges"

        # Alternative hypotheses
        alternatives_html = ""
        for alt in alternative_hypotheses:
            alt_text = alt.get("alternative", "")
            alt_verdict = alt.get("verdict", "")
            alt_evidence = alt.get("evidence", "")

            if alt_verdict == "refuted":
                card_class = "rejected"
                verdict_label = "Rejected"
            elif "partial" in alt_verdict.lower():
                card_class = "partial"
                verdict_label = "Partially Supported"
            else:
                card_class = "supported"
                verdict_label = alt_verdict.replace("_", " ").title()

            alternatives_html += f'''
                <div class="alternative-card {card_class}">
                    <div class="alternative-verdict">{verdict_label}</div>
                    <div class="alternative-text">{escape_html(alt_text)}</div>
                    {f'<div class="alternative-evidence">{escape_html(alt_evidence)}</div>' if alt_evidence else ''}
                </div>
            '''

        # Boundary tests
        passed_tests = [t for t in boundary_tests if t.get("passed", False)]
        failed_tests = [t for t in boundary_tests if not t.get("passed", False)]

        boundary_html = ""
        if passed_tests or failed_tests:
            passed_items = ""
            for t in passed_tests[:3]:
                desc = t.get("description", "")
                result = t.get("notes", "") or f"act: {t.get('actual_activation', 'N/A')}"
                passed_items += f'''
                    <div class="boundary-test passed">
                        <div class="boundary-test-name">{escape_html(desc[:50])}</div>
                        <div class="boundary-test-result">{escape_html(result[:60])}</div>
                    </div>
                '''
            if len(passed_tests) > 3:
                passed_items += f'<div class="boundary-test passed"><div class="boundary-test-name">...and {len(passed_tests) - 3} more</div></div>'

            failed_items = ""
            for t in failed_tests[:3]:
                desc = t.get("description", "")
                result = t.get("notes", "") or f"act: {t.get('actual_activation', 'N/A')}"
                failed_items += f'''
                    <div class="boundary-test failed">
                        <div class="boundary-test-name">{escape_html(desc[:50])}</div>
                        <div class="boundary-test-result">{escape_html(result[:60])}</div>
                    </div>
                '''

            boundary_html = f'''
                <div class="boundary-grid">
                    <div class="boundary-column passed">
                        <h4>Passed ({len(passed_tests)})</h4>
                        {passed_items if passed_items else '<div class="boundary-test">None</div>'}
                    </div>
                    <div class="boundary-column failed">
                        <h4>Failed ({len(failed_tests)})</h4>
                        {failed_items if failed_items else '<div class="boundary-test">None</div>'}
                    </div>
                </div>
            '''

        skeptic_html = f'''
            <div class="flow-connector"><div class="flow-arrow"></div></div>
            <div class="agent-section skeptic">
                <div class="agent-header skeptic">
                    <span class="agent-icon">🔍</span>
                    <span class="agent-name">Skeptic Agent</span>
                    <span class="agent-role">Adversarial testing & alternative hypotheses</span>
                </div>
                <div class="agent-content">
                    <div class="verdict-banner">
                        <span class="verdict-badge {verdict_class}">{escape_html(verdict)}</span>
                        <span class="verdict-text">{verdict_text}</span>
                    </div>

                    {f'<div class="section-divider">Alternative Hypotheses Tested</div>{alternatives_html}' if alternatives_html else ''}
                    {f'<div class="section-divider">Boundary Tests</div>{boundary_html}' if boundary_html else ''}
                    {f'<div class="section-divider">Revised Hypothesis</div><div class="content-card"><div class="card-text">{escape_html(revised_hypothesis)}</div></div>' if revised_hypothesis else ''}
                </div>
            </div>
        '''

    # =================================================================
    # REVIEWER AGENT SECTION
    # =================================================================
    reviewer_html = ""
    if pi_result:
        review_history = pi_result.get("review_history", [])
        final_verdict = pi_result.get("final_verdict", "UNKNOWN")

        reviews_html = ""
        for i, review in enumerate(review_history):
            iteration = review.get("iteration", i + 1)
            verdict = review.get("verdict", "")
            conf_assessment = review.get("confidence_assessment", "")
            gaps = review.get("gaps", [])

            verdict_class = "accept" if "ACCEPT" in verdict.upper() else ("reject" if "REJECT" in verdict.upper() else "changes")
            gaps_html = "".join([f'<li>{escape_html(gap)}</li>' for gap in gaps])

            reviews_html += f'''
                <div class="review-card">
                    <div class="review-header">
                        <span class="iteration-number">Iteration {iteration}</span>
                        <span class="review-verdict {verdict_class}">{escape_html(verdict)}</span>
                        {f'<span class="confidence-flag">{escape_html(conf_assessment)}</span>' if conf_assessment else ''}
                    </div>
                    {f'<ul class="review-gaps">{gaps_html}</ul>' if gaps_html else ''}
                </div>
            '''

        reviewer_html = f'''
            <div class="flow-connector"><div class="flow-arrow"></div></div>
            <div class="agent-section reviewer">
                <div class="agent-header reviewer">
                    <span class="agent-icon">📝</span>
                    <span class="agent-name">Reviewer Agent</span>
                    <span class="agent-role">Quality control & confidence calibration</span>
                </div>
                <div class="agent-content">
                    {reviews_html}
                    <div class="content-card">
                        <div class="card-label">Final Verdict</div>
                        <div class="card-text"><strong>{escape_html(final_verdict.split(" - ")[0] if " - " in final_verdict else final_verdict)}</strong>{" - " + escape_html(final_verdict.split(" - ", 1)[1]) if " - " in final_verdict else ""}</div>
                    </div>
                </div>
            </div>
        '''

    # =================================================================
    # FINAL CONCLUSION SECTION
    # =================================================================
    final_input = characterization.get("input_function", "")
    final_output = characterization.get("output_function", "")

    # Confidence badge class
    conf_class = "low" if final_confidence < 0.4 else ("medium" if final_confidence < 0.7 else "")

    # Key findings
    findings_html = ""
    for finding in key_findings[:6]:
        icon = "⚠" if "false positive" in finding.lower() or "warning" in finding.lower() else "✓"
        item_class = "warning" if "⚠" in icon else ""
        findings_html += f'''
            <div class="finding-item {item_class}">
                <span class="finding-icon">{icon}</span>
                {escape_html(finding)}
            </div>
        '''

    conclusion_html = f'''
        <div class="flow-connector"><div class="flow-arrow"></div></div>
        <div class="agent-section conclusion">
            <div class="agent-header conclusion">
                <span class="agent-icon">✅</span>
                <span class="agent-name">Final Conclusion</span>
                <span class="agent-role">Validated characterization</span>
            </div>
            <div class="agent-content">
                <div style="text-align: center; margin-bottom: 20px;">
                    <span class="confidence-badge {conf_class}">{int(final_confidence * 100)}% Confidence</span>
                </div>

                {f'<div class="conclusion-box"><div class="conclusion-type">Input Function</div><div class="conclusion-text">{escape_html(final_input)}</div></div>' if final_input else ''}
                {f'<div class="conclusion-box"><div class="conclusion-type">Output Function</div><div class="conclusion-text">{escape_html(final_output)}</div></div>' if final_output else ''}

                {f'<div class="section-divider">Key Findings</div><div class="findings-list">{findings_html}</div>' if findings_html else ''}
            </div>
        </div>
    '''

    return f'''
    <div class="figure-container agent-flow">
        <div class="figure-title">{escape_html(title)}: {escape_html(neuron_id)}</div>
        {investigator_html}
        {skeptic_html}
        {reviewer_html}
        {conclusion_html}
    </div>
    '''


def generate_stacked_density_chart(
    bin_data: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    title: str = "Activation Distribution by Category",
    caption: str = "",
    explanation: str = ""
) -> str:
    """Generate a stacked horizontal bar chart showing category distributions by z-score.

    Args:
        bin_data: List of dicts with structure:
            {"zMid": -2.0, "malware": 5, "biological": 45, "neutral": 20}
            Keys should match category names.
        categories: List of dicts with structure:
            {"name": "malware", "color": "#dc2626", "description": "Technical malware context"}
        title: Figure title
        caption: Optional caption
        explanation: Required narrative explaining the figure's significance
    """
    if not bin_data or not categories:
        return f'''
        <div class="figure-container density-chart">
            <div class="figure-title">{escape_html(title)}</div>
            <p style="color: var(--text-tertiary); text-align: center; padding: 40px;">
                Insufficient data for density chart
            </p>
        </div>
        '''

    # Build category name to color mapping
    cat_colors = {c.get("name", ""): c.get("color", "#888") for c in categories}
    cat_names = [c.get("name", "") for c in categories]

    # Build legend
    legend_html = ""
    for cat in categories:
        name = cat.get("name", "")
        color = cat.get("color", "#888")
        desc = cat.get("description", name)
        legend_html += f'''
        <div class="category-item" title="{escape_html(desc)}">
            <span class="category-color" style="background: {color};"></span>
            <span class="category-name">{escape_html(name)}</span>
        </div>
        '''

    # Build bar rows
    bars_html = ""
    for bin_row in bin_data:
        z_mid = bin_row.get("zMid", 0)

        # Calculate total for this bin
        total = sum(bin_row.get(cat, 0) for cat in cat_names)
        if total == 0:
            total = 1  # Avoid division by zero

        # Build segments
        segments_html = ""
        for cat_name in cat_names:
            count = bin_row.get(cat_name, 0)
            pct = (count / total) * 100
            color = cat_colors.get(cat_name, "#888")
            if pct > 0:
                segments_html += f'<div class="density-segment" style="width: {pct:.1f}%; background: {color};" title="{cat_name}: {count}"></div>'

        bars_html += f'''
        <div class="density-bar-row">
            <span class="density-zscore">z={z_mid:+.1f}</span>
            <div class="density-bar-container">
                {segments_html}
            </div>
            <span class="density-count">n={total}</span>
        </div>
        '''

    caption_html = f'<p class="figure-caption">{escape_html(caption)}</p>' if caption else ""
    explanation_html = f'<p class="figure-explanation" style="font-size: 14px; color: var(--text-secondary); margin-top: 12px; line-height: 1.6;">{escape_html_preserve_tags(explanation)}</p>' if explanation else ""

    return f'''
    <div class="figure-container density-chart">
        <div class="figure-title">{escape_html(title)}</div>
        <div class="category-legend">
            {legend_html}
        </div>
        <div class="density-bars">
            {bars_html}
        </div>
        {caption_html}
        {explanation_html}
    </div>
    '''


def generate_patching_comparison(
    experiments: list[dict[str, Any]],
    title: str = "Counterfactual Patching",
    caption: str = "",
    explanation: str = ""
) -> str:
    """Generate visualization for activation patching experiments.

    Shows counterfactual tests where activation from source prompt is patched into target prompt.

    Args:
        experiments: List of dicts with structure:
            {
                "source_prompt": "The malware infected the server",
                "target_prompt": "The patient was infected by a virus",
                "source_activation": 4.75,
                "target_activation": 0.64,
                "promoted_tokens": [("malware", 2.3), ("virus", 1.1)],
                "suppressed_tokens": [("patient", -1.5)],
                "max_shift": 2.3
            }
        title: Figure title
        caption: Optional caption
        explanation: Required narrative explaining the figure's significance
    """
    if not experiments:
        return f'''
        <div class="figure-container patching-comparison">
            <div class="figure-title">{escape_html(title)}</div>
            <p style="color: var(--text-tertiary); text-align: center; padding: 40px;">
                No patching experiments available
            </p>
        </div>
        '''

    rows_html = ""
    for exp in experiments[:6]:  # Limit to 6 experiments
        source = exp.get("source_prompt", "")[:60]
        target = exp.get("target_prompt", "")[:60]
        source_act = exp.get("source_activation", 0)
        target_act = exp.get("target_activation", 0)
        act_delta = source_act - target_act
        max_shift = exp.get("max_shift", 0)

        promoted = exp.get("promoted_tokens", [])
        suppressed = exp.get("suppressed_tokens", [])

        # Build effect badges
        promoted_badges = ""
        for item in promoted[:3]:
            if isinstance(item, (list, tuple)):
                tok, shift = item[0], item[1]
            elif isinstance(item, dict):
                tok, shift = item.get("token", ""), item.get("shift", 0)
            else:
                continue
            clean_tok = clean_token(tok)
            promoted_badges += f'<span class="effect-badge promote">{escape_html(clean_tok)} +{shift:.2f}</span>'

        suppressed_badges = ""
        for item in suppressed[:3]:
            if isinstance(item, (list, tuple)):
                tok, shift = item[0], item[1]
            elif isinstance(item, dict):
                tok, shift = item.get("token", ""), item.get("shift", 0)
            else:
                continue
            clean_tok = clean_token(tok)
            suppressed_badges += f'<span class="effect-badge suppress">{escape_html(clean_tok)} {shift:.2f}</span>'

        # Activation delta coloring
        delta_class = "positive" if act_delta > 0.5 else ("negative" if act_delta < -0.5 else "neutral")
        delta_sign = "+" if act_delta > 0 else ""

        # Effect strength indicator
        effect_class = "strong" if abs(max_shift) > 1.0 else ("moderate" if abs(max_shift) > 0.3 else "weak")

        rows_html += f'''
        <div class="patching-experiment">
            <div class="patching-prompts">
                <div class="prompt-row source">
                    <span class="prompt-label">Source:</span>
                    <span class="prompt-text">"{escape_html(source)}..."</span>
                    <span class="activation-badge high">{source_act:.2f}</span>
                </div>
                <div class="patching-arrow">↓ patch activation</div>
                <div class="prompt-row target">
                    <span class="prompt-label">Target:</span>
                    <span class="prompt-text">"{escape_html(target)}..."</span>
                    <span class="activation-badge low">{target_act:.2f}</span>
                </div>
            </div>
            <div class="patching-effects">
                <div class="effect-row">
                    <span class="effect-label">Δ activation:</span>
                    <span class="effect-value {delta_class}">{delta_sign}{act_delta:.2f}</span>
                </div>
                <div class="effect-row">
                    <span class="effect-label">Promoted:</span>
                    <div class="effect-badges">{promoted_badges if promoted_badges else '<span class="no-data">—</span>'}</div>
                </div>
                <div class="effect-row">
                    <span class="effect-label">Suppressed:</span>
                    <div class="effect-badges">{suppressed_badges if suppressed_badges else '<span class="no-data">—</span>'}</div>
                </div>
                <div class="effect-summary {effect_class}">
                    Max logit shift: {max_shift:.2f}
                </div>
            </div>
        </div>
        '''

    caption_html = f'<p class="figure-caption">{escape_html(caption)}</p>' if caption else ""
    explanation_html = f'<p class="figure-explanation" style="font-size: 14px; color: var(--text-secondary); margin-top: 12px; line-height: 1.6;">{escape_html_preserve_tags(explanation)}</p>' if explanation else ""

    return f'''
    <div class="figure-container patching-comparison">
        <div class="figure-title">{escape_html(title)}</div>
        <div class="patching-experiments-grid">
            {rows_html}
        </div>
        {caption_html}
        {explanation_html}
    </div>
    '''


# =============================================================================
# Category Selectivity Chart
# =============================================================================

def generate_category_selectivity_chart(
    category_selectivity_data: dict[str, Any],
    neuron_id: str,
    title: str = "",
    caption: str = "",
    explanation: str = "",
    chart_id: str = "",
) -> str:
    """Generate a category selectivity stacked area chart with interactive tooltips.

    This visualization shows the conditional probability of each semantic category
    at different activation z-score levels, with individual prompts as hoverable dots.

    Key design choices:
    - Unrelated domains are merged into a single "Other" category (gray) to reduce noise
    - Target categories get vibrant, distinct colors (reds, oranges, purples)
    - Inhibitory/control categories get muted colors to fade into background
    - Interactive legend with hover highlighting
    - Container width matches chart (not full page)

    Args:
        category_selectivity_data: Dict from tool_run_category_selectivity_test containing:
            - global_mean, global_std: baseline statistics
            - categories: {name: {type, prompts: [{prompt, activation, z_score}...], z_mean}}
            - selectivity_summary: human-readable assessment
        neuron_id: Neuron identifier (e.g., "L21/N6856")
        title: Optional custom title
        caption: Optional caption
        explanation: Optional explanation text
        chart_id: Unique ID for the chart (for multiple charts on page)

    Returns:
        HTML string with D3.js visualization (includes <script> tag for D3 from CDN)
    """
    import numpy as np

    if not category_selectivity_data or "categories" not in category_selectivity_data:
        return '<div class="figure-container"><p class="no-data">No category selectivity data available</p></div>'

    categories = category_selectivity_data.get("categories", {})
    global_mean = category_selectivity_data.get("global_mean", 0)
    global_std = category_selectivity_data.get("global_std", 1)
    selectivity_summary = category_selectivity_data.get("selectivity_summary", "")

    # Default title
    if not title:
        title = f"Category Selectivity: {neuron_id}"

    # Unique chart ID to avoid conflicts
    if not chart_id:
        chart_id = f"selectivity_{neuron_id.replace('/', '_').replace(' ', '_')}"

    # Separate categories by type
    target_cats = []
    inhibitory_cats = []
    control_cats = []
    unrelated_cats = []

    for cat_name, cat_data in categories.items():
        cat_type = cat_data.get("type", "unknown")
        if cat_type == "target":
            target_cats.append(cat_name)
        elif cat_type == "inhibitory":
            inhibitory_cats.append(cat_name)
        elif cat_type == "control":
            control_cats.append(cat_name)
        else:
            unrelated_cats.append(cat_name)

    # Sort each group by z_mean
    target_cats.sort(key=lambda c: categories[c].get("z_mean", 0))
    inhibitory_cats.sort(key=lambda c: categories[c].get("z_mean", 0))
    control_cats.sort(key=lambda c: categories[c].get("z_mean", 0))
    unrelated_cats.sort(key=lambda c: categories[c].get("z_mean", 0))

    # Build display categories: merge unrelated into "Other"
    # Order: Other (bottom) -> Control -> Inhibitory -> Target (top, most visible)
    display_cats = []
    original_cat_mapping = {}  # Maps display cat -> list of original cats

    # Add merged "Other" category for unrelated
    if unrelated_cats:
        display_cats.append("_other_unrelated")
        original_cat_mapping["_other_unrelated"] = unrelated_cats

    # Add individual control categories
    for cat in control_cats:
        display_cats.append(cat)
        original_cat_mapping[cat] = [cat]

    # Add individual inhibitory categories
    for cat in inhibitory_cats:
        display_cats.append(cat)
        original_cat_mapping[cat] = [cat]

    # Add individual target categories (these are the important ones!)
    for cat in target_cats:
        display_cats.append(cat)
        original_cat_mapping[cat] = [cat]

    # Extended color palettes - targets get many distinct colors
    target_colors = [
        "#ef4444",  # red-500
        "#f97316",  # orange-500
        "#8b5cf6",  # violet-500
        "#ec4899",  # pink-500
        "#f43f5e",  # rose-500
        "#a855f7",  # purple-500
        "#dc2626",  # red-600
        "#ea580c",  # orange-600
        "#7c3aed",  # violet-600
        "#db2777",  # pink-600
        "#e11d48",  # rose-600
        "#9333ea",  # purple-600
        "#b91c1c",  # red-700
        "#c2410c",  # orange-700
        "#6d28d9",  # violet-700
        "#be185d",  # pink-700
        "#be123c",  # rose-700
        "#7e22ce",  # purple-700
        "#fb923c",  # orange-400
        "#c084fc",  # purple-400
    ]

    inhibitory_colors = [
        "#94a3b8",  # slate-400 (muted)
        "#64748b",  # slate-500
        "#475569",  # slate-600
        "#78716c",  # stone-500
        "#a8a29e",  # stone-400
        "#737373",  # neutral-500
        "#a3a3a3",  # neutral-400
        "#6b7280",  # gray-500
    ]

    control_colors = [
        "#d6d3d1",  # stone-300 (very muted)
        "#e7e5e4",  # stone-200
        "#d4d4d4",  # neutral-300
        "#e5e5e5",  # neutral-200
        "#d1d5db",  # gray-300
        "#e5e7eb",  # gray-200
        "#cbd5e1",  # slate-300
        "#c7d2fe",  # indigo-200
    ]

    other_color = "#e5e7eb"  # gray-200 - very subtle background

    # Assign colors
    category_colors = {}
    target_idx = 0
    inhibitory_idx = 0
    control_idx = 0

    for cat in display_cats:
        if cat == "_other_unrelated":
            category_colors[cat] = other_color
        elif cat in target_cats:
            category_colors[cat] = target_colors[target_idx % len(target_colors)]
            target_idx += 1
        elif cat in inhibitory_cats:
            category_colors[cat] = inhibitory_colors[inhibitory_idx % len(inhibitory_colors)]
            inhibitory_idx += 1
        elif cat in control_cats:
            category_colors[cat] = control_colors[control_idx % len(control_colors)]
            control_idx += 1
        else:
            category_colors[cat] = other_color

    # Create labels
    category_labels = {}
    for cat in display_cats:
        if cat == "_other_unrelated":
            category_labels[cat] = "Other (Unrelated)"
        else:
            label = cat.replace("gen_", "").replace("_", " ").title()
            cat_type = categories[cat].get("type", "unknown")
            if cat_type == "target":
                label = f"Target: {label}"
            elif cat_type == "inhibitory":
                label = f"Inhibitory: {label}"
            category_labels[cat] = label

    # Collect all z-scores to find range (include negative z-scores for bipolar neurons)
    all_z = []
    for cat_data in categories.values():
        for p in cat_data.get("prompts", []):
            all_z.append(p.get("z_score", 0))
            # Include negative z-scores if they're significantly negative
            neg_z = p.get("neg_z_score", 0)
            neg_act = p.get("min_activation", 0)
            if neg_act < -0.5:
                all_z.append(neg_z)

    if not all_z:
        return '<div class="figure-container"><p class="no-data">No activation data available</p></div>'

    # Clip z-axis to avoid extreme outliers dominating the chart.
    # Use IQR-based range: show from Q1 - 3*IQR to Q3 + 3*IQR, with a floor of ±6σ
    sorted_z = sorted(all_z)
    q1_idx = len(sorted_z) // 4
    q3_idx = 3 * len(sorted_z) // 4
    q1 = sorted_z[q1_idx]
    q3 = sorted_z[q3_idx]
    iqr = q3 - q1
    z_min_raw = q1 - 3 * max(iqr, 1.0)
    z_max_raw = q3 + 3 * max(iqr, 1.0)
    # Ensure at least ±6σ range so we always show meaningful spread
    z_min = min(z_min_raw, -6.0) - 0.5
    z_max = max(z_max_raw, 6.0) + 0.5

    # Create points data for scatter overlay (map to display categories)
    points_data = []
    for orig_cat, cat_data in categories.items():
        # Find display category
        display_cat = orig_cat
        if orig_cat in unrelated_cats:
            display_cat = "_other_unrelated"

        for p in cat_data.get("prompts", []):
            prompt_text = p.get("prompt", "")
            token = p.get("token", "")
            position = p.get("position", -1)

            # Skip special chat template tokens for highlighting
            skip_tokens = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|begin_of_text|>", "system"]

            # Build multi-token intensity-highlighted full prompt
            token_activations = p.get("token_activations", [])
            highlighted_full = ""
            highlighted_prompt = prompt_text[:100]

            if token_activations and len(token_activations) > 0:
                # Multi-token highlighting: wrap each token with intensity-based background
                # Find max activation for normalization
                max_tok_act = max(ta["activation"] for ta in token_activations) if token_activations else 1.0
                if max_tok_act <= 0:
                    max_tok_act = 1.0

                # Build list of (start_idx, end_idx, intensity) for each token found in prompt
                highlights = []
                used_ranges = set()  # Track used character ranges to avoid overlaps
                for ta in token_activations:
                    tok = ta.get("token", "")
                    act = ta.get("activation", 0)
                    if not tok or tok in skip_tokens:
                        continue
                    clean = tok.strip()
                    search = clean if clean else tok
                    if not search:
                        continue

                    # Find token in prompt (case-sensitive first, then insensitive)
                    idx = prompt_text.find(search)
                    if idx < 0 and len(clean) > 1:
                        lower_idx = prompt_text.lower().find(clean.lower())
                        if lower_idx >= 0:
                            idx = lower_idx
                            search = prompt_text[idx:idx + len(clean)]

                    if idx >= 0:
                        end_idx = idx + len(search)
                        # Skip if overlapping with an already-used range
                        overlap = any(idx < er and end_idx > sr for sr, er in used_ranges)
                        if not overlap:
                            intensity = act / max_tok_act
                            highlights.append((idx, end_idx, intensity, search))
                            used_ranges.add((idx, end_idx))

                # Build highlighted_full by inserting spans (right-to-left to preserve indices)
                if highlights:
                    result = prompt_text
                    for start, end, intensity, match_text in sorted(highlights, key=lambda x: x[0], reverse=True):
                        alpha = max(0.15, intensity * 0.85)
                        span = f'<span class="tok-highlight" style="background:rgba(34,197,94,{alpha:.2f})">{escape_html(match_text)}</span>'
                        result = result[:start] + span + result[end:]
                    highlighted_full = result

                # Also build truncated version for inline (keep single strongest token marked)
                if token and token not in skip_tokens:
                    clean_token = token.strip()
                    search_token = clean_token if clean_token else token
                    if search_token and search_token in prompt_text:
                        idx = prompt_text.find(search_token)
                        before = prompt_text[:idx]
                        after = prompt_text[idx + len(search_token):]
                        if len(before) > 40:
                            before = "..." + before[-37:]
                        if len(after) > 40:
                            after = after[:37] + "..."
                        highlighted_prompt = f"{before}<mark>{search_token}</mark>{after}"

            elif token and token not in skip_tokens:
                # Backward compat: single-token highlighting when no token_activations
                clean_token = token.strip()
                search_token = clean_token if clean_token else token
                if search_token and search_token in prompt_text:
                    idx = prompt_text.find(search_token)
                    before = prompt_text[:idx]
                    after = prompt_text[idx + len(search_token):]
                    if len(before) > 40:
                        before = "..." + before[-37:]
                    if len(after) > 40:
                        after = after[:37] + "..."
                    highlighted_prompt = f"{before}<mark>{search_token}</mark>{after}"
                elif len(clean_token) > 1:
                    lower_prompt = prompt_text.lower()
                    lower_token = clean_token.lower()
                    if lower_token in lower_prompt:
                        idx = lower_prompt.find(lower_token)
                        match = prompt_text[idx:idx + len(clean_token)]
                        before = prompt_text[:idx]
                        after = prompt_text[idx + len(clean_token):]
                        if len(before) > 40:
                            before = "..." + before[-37:]
                        if len(after) > 40:
                            after = after[:37] + "..."
                        highlighted_prompt = f"{before}<mark>{match}</mark>{after}"

            points_data.append({
                "z": p.get("z_score", 0),
                "category": display_cat,
                "original_category": orig_cat,
                "prompt": highlighted_prompt,
                "full_prompt": prompt_text,
                "highlighted_full": highlighted_full,
                "activation": p.get("activation", 0),
                "token": token if token not in skip_tokens else "(template)",
            })

            # Also add a negative-polarity point if the prompt has strong negative activation
            neg_act = p.get("min_activation", 0)
            neg_z = p.get("neg_z_score", 0)
            if neg_act < -0.5:
                # Clip to axis range so extreme outliers appear at edge, not off-chart
                clipped_neg_z = max(neg_z, z_min + 0.2)

                # Build highlighted prompt for negative firing token (red highlight)
                neg_token = p.get("min_token", "")
                neg_highlighted = f"[NEG] {prompt_text[:100]}"
                neg_highlighted_full = ""
                if neg_token and neg_token.strip() and neg_token not in skip_tokens:
                    clean_neg = neg_token.strip()
                    search_neg = clean_neg if clean_neg else neg_token
                    # Try exact match first, then case-insensitive
                    found_idx = prompt_text.find(search_neg)
                    if found_idx == -1 and len(clean_neg) > 1:
                        found_idx = prompt_text.lower().find(clean_neg.lower())
                        if found_idx >= 0:
                            search_neg = prompt_text[found_idx:found_idx + len(clean_neg)]
                    if found_idx >= 0:
                        before = prompt_text[:found_idx]
                        after = prompt_text[found_idx + len(search_neg):]
                        if len(before) > 40:
                            before = "..." + before[-37:]
                        if len(after) > 40:
                            after = after[:37] + "..."
                        neg_highlighted = f'{before}<span style="background:#ef4444;color:white;padding:1px 4px;border-radius:3px">{search_neg}</span>{after}'
                        # Full prompt with red highlight
                        neg_highlighted_full = (
                            prompt_text[:found_idx]
                            + f'<span class="tok-highlight" style="background:rgba(239,68,68,0.7)">{search_neg}</span>'
                            + prompt_text[found_idx + len(search_neg):]
                        )

                points_data.append({
                    "z": clipped_neg_z,
                    "category": display_cat,
                    "original_category": orig_cat,
                    "prompt": neg_highlighted,
                    "full_prompt": prompt_text,
                    "highlighted_full": neg_highlighted_full,
                    "activation": neg_act,
                    "token": neg_token if neg_token not in skip_tokens else "(template)",
                    "is_negative": True,
                })

    # Create stacked area data using KDE
    n_bins = 50
    bin_edges = np.linspace(z_min, z_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bandwidth = (z_max - z_min) / 25

    def gaussian_kernel(x, xi, h):
        return np.exp(-0.5 * ((x - xi) / h) ** 2) / (h * np.sqrt(2 * np.pi))

    stacked_data = []
    for z_center in bin_centers:
        densities = {}
        total_density = 0

        for display_cat in display_cats:
            # Sum densities from all original categories mapped to this display cat
            orig_cats = original_cat_mapping[display_cat]
            cat_density = 0
            for orig_cat in orig_cats:
                # Include positive z-scores
                cat_z = [p.get("z_score", 0) for p in categories[orig_cat].get("prompts", [])]
                # Also include negative z-scores for prompts with strong negative activation
                for p in categories[orig_cat].get("prompts", []):
                    neg_act = p.get("min_activation", 0)
                    neg_z = p.get("neg_z_score", 0)
                    if neg_act < -0.5:
                        cat_z.append(neg_z)
                cat_density += sum(gaussian_kernel(z_center, zi, bandwidth) for zi in cat_z)
            densities[display_cat] = cat_density
            total_density += cat_density

        if total_density > 0.001:
            row = {"z": float(z_center)}
            for display_cat in display_cats:
                prop = densities[display_cat] / total_density
                row[display_cat] = prop
            stacked_data.append(row)

    # JSON data for JavaScript
    js_stacked = json.dumps(stacked_data)
    js_points = json.dumps(points_data)
    js_colors = json.dumps(category_colors)
    js_categories = json.dumps(display_cats)
    js_labels = json.dumps(category_labels)

    # Counts per display category (include negative points)
    display_counts = {}
    for display_cat in display_cats:
        orig_cats = original_cat_mapping[display_cat]
        count = 0
        for orig_cat in orig_cats:
            prompts = categories[orig_cat].get("prompts", [])
            count += len(prompts)
            # Count negative points too
            count += sum(1 for p in prompts if p.get("min_activation", 0) < -0.5)
        display_counts[display_cat] = count
    js_counts = json.dumps(display_counts)

    # Stats
    total_prompts = sum(len(cat_data.get("prompts", [])) for cat_data in categories.values())
    num_display_categories = len(display_cats)
    num_original_categories = len(categories)

    # Category stats for legend (by display category)
    category_stats = {}
    for display_cat in display_cats:
        orig_cats = original_cat_mapping[display_cat]
        if display_cat == "_other_unrelated":
            # Compute average z_mean for merged categories
            all_z_means = [categories[c].get("z_mean", 0) for c in orig_cats]
            avg_z_mean = sum(all_z_means) / len(all_z_means) if all_z_means else 0
            category_stats[display_cat] = {
                "type": "unrelated",
                "z_mean": avg_z_mean,
                "count": sum(len(categories[c].get("prompts", [])) for c in orig_cats),
                "merged_count": len(orig_cats),
            }
        else:
            category_stats[display_cat] = {
                "type": categories[display_cat].get("type", "unknown"),
                "z_mean": categories[display_cat].get("z_mean", 0),
                "count": len(categories[display_cat].get("prompts", [])),
            }
    js_stats = json.dumps(category_stats)

    caption_html = f'<p class="figure-caption">{escape_html(caption)}</p>' if caption else ""
    explanation_html = f'<p class="figure-explanation">{escape_html_preserve_tags(explanation)}</p>' if explanation else ""

    # Count merged unrelated categories for display
    merged_unrelated_count = len(unrelated_cats)

    return f'''
    <div class="figure-container category-selectivity-chart">
        <div class="figure-title">{escape_html(title)}</div>

        <style>
            .category-selectivity-chart {{
                padding: 24px;
                /* Break out of text column significantly - 50% wider */
                margin-left: -220px;
                margin-right: -220px;
                padding-left: 24px;
                padding-right: 24px;
                background: var(--surface, #fafafa);
                border-radius: 12px;
                border: 1px solid var(--border, #e5e5e5);
            }}
            .category-selectivity-chart .chart-wrapper {{
                display: flex;
                flex-direction: column;
                gap: 20px;
            }}
            .category-selectivity-chart .chart-area {{
                width: 100%;
            }}
            .category-selectivity-chart .selectivity-legend {{
                display: flex;
                flex-direction: row;
                flex-wrap: wrap;
                gap: 24px 40px;
                padding: 16px 20px;
                background: white;
                border-radius: 8px;
                border: 1px solid var(--border);
                font-size: 11px;
            }}
            .category-selectivity-chart .legend-section {{
                display: flex;
                flex-direction: column;
                gap: 6px;
                min-width: 180px;
            }}
            .category-selectivity-chart .legend-title {{
                font-size: 9px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: var(--text-secondary);
                margin-bottom: 4px;
                border-bottom: 1px solid var(--border, #e5e5e5);
                padding-bottom: 4px;
            }}
            .category-selectivity-chart .legend-items {{
                display: flex;
                flex-direction: column;
                gap: 4px;
            }}
            .category-selectivity-chart .legend-item {{
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 12px;
                cursor: pointer;
                padding: 3px 6px;
                border-radius: 4px;
                transition: background 0.15s ease;
            }}
            .category-selectivity-chart .legend-item:hover {{
                background: var(--surface-hover, rgba(0,0,0,0.05));
            }}
            .category-selectivity-chart .legend-item.dimmed {{
                opacity: 0.3;
            }}
            .category-selectivity-chart .legend-item.highlighted {{
                background: var(--surface-hover, rgba(0,0,0,0.08));
            }}
            .category-selectivity-chart .legend-color {{
                width: 10px;
                height: 10px;
                border-radius: 2px;
                flex-shrink: 0;
            }}
            .category-selectivity-chart .legend-label {{
                flex: 1;
                line-height: 1.3;
            }}
            .category-selectivity-chart .legend-z {{
                font-size: 9px;
                color: var(--text-secondary);
                font-family: var(--mono);
            }}
            .category-selectivity-chart .chart-svg {{
                width: 100%;
                height: 350px;
            }}
            .category-selectivity-chart .axis text {{
                font-size: 11px;
                fill: var(--text-secondary);
            }}
            .category-selectivity-chart .axis-label {{
                font-size: 12px;
                fill: var(--text-primary);
            }}
            .category-selectivity-chart .layer {{
                transition: opacity 0.15s ease;
            }}
            .category-selectivity-chart .layer.dimmed {{
                opacity: 0.15 !important;
            }}
            .category-selectivity-chart .layer.highlighted {{
                opacity: 0.95 !important;
            }}
            .category-selectivity-chart .point {{
                stroke: white;
                stroke-width: 1;
                opacity: 0.9;
                cursor: pointer;
                transition: opacity 0.15s ease;
            }}
            .category-selectivity-chart .point:hover {{
                opacity: 1;
                stroke-width: 2;
            }}
            .category-selectivity-chart .point.dimmed {{
                opacity: 0.1;
            }}
            .category-selectivity-chart .point.highlighted {{
                opacity: 1;
                stroke-width: 2;
            }}
            .category-selectivity-chart .selectivity-tooltip {{
                position: fixed;
                background: rgba(0, 0, 0, 0.9);
                color: white;
                padding: 10px 14px;
                border-radius: 8px;
                font-size: 12px;
                max-width: 500px;
                pointer-events: none;
                z-index: 1000;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                line-height: 1.4;
            }}
            .category-selectivity-chart .tooltip-category {{
                font-size: 10px;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                opacity: 0.7;
                margin-bottom: 4px;
            }}
            .category-selectivity-chart .tooltip-prompt {{
                margin-bottom: 6px;
            }}
            .category-selectivity-chart .tooltip-prompt mark {{
                background: #22c55e;
                color: white;
                padding: 1px 4px;
                border-radius: 3px;
                font-weight: 500;
            }}
            .category-selectivity-chart .tooltip-prompt .tok-highlight {{
                color: white;
                padding: 1px 3px;
                border-radius: 2px;
                /* background set inline via style attribute */
            }}
            .category-selectivity-chart .tooltip-stats {{
                display: flex;
                gap: 12px;
                font-family: var(--mono);
                font-size: 11px;
            }}
            .category-selectivity-chart .tooltip-stat-label {{
                opacity: 0.7;
            }}
            .category-selectivity-chart .selectivity-summary {{
                margin-top: 16px;
                padding: 12px 16px;
                background: var(--surface);
                border-radius: 8px;
                border-left: 4px solid var(--accent);
                font-size: 13px;
            }}
            .category-selectivity-chart .stats-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 24px;
                margin-top: 12px;
                font-size: 13px;
                color: var(--text-secondary);
                align-items: baseline;
            }}
            .category-selectivity-chart .stat-item {{
                display: flex;
                gap: 6px;
                align-items: baseline;
            }}
            .category-selectivity-chart .stat-item span {{
                font-size: 13px !important;
                font-weight: normal !important;
                letter-spacing: normal !important;
            }}
            .category-selectivity-chart .stat-value {{
                font-size: 13px !important;
                font-weight: 600 !important;
                color: var(--text-primary);
                letter-spacing: normal !important;
            }}
        </style>

        <div id="{chart_id}_tooltip" class="selectivity-tooltip" style="display: none;"></div>

        <div class="chart-wrapper">
            <div class="chart-area">
                <svg id="{chart_id}_svg" class="chart-svg"></svg>
            </div>
            <div id="{chart_id}_legend" class="selectivity-legend"></div>
        </div>
        <!-- Legend is now below the chart in a horizontal layout -->

        <div class="selectivity-summary">
            <strong>Selectivity Assessment:</strong> {escape_html(selectivity_summary)}
        </div>

        <div class="stats-row">
            <div class="stat-item">
                <span>Total prompts:</span>
                <span class="stat-value">{total_prompts}</span>
            </div>
            <div class="stat-item">
                <span>Categories:</span>
                <span class="stat-value">{num_display_categories}</span>
                <span style="opacity: 0.6;">({merged_unrelated_count} merged as "Other")</span>
            </div>
            <div class="stat-item">
                <span>Max z-score:</span>
                <span class="stat-value">{z_max:.1f}σ</span>
            </div>
        </div>

        {caption_html}
        {explanation_html}

        <script src="https://d3js.org/d3.v7.min.js"></script>
        <script>
        (function() {{
            const chartId = "{chart_id}";
            const stackedData = {js_stacked};
            const pointsData = {js_points};
            const colors = {js_colors};
            const categories = {js_categories};
            const labels = {js_labels};
            const counts = {js_counts};
            const stats = {js_stats};

            // Track SVG elements for highlighting
            let layerElements = {{}};
            let pointElements = [];

            // Build legend grouped by type (target first since most important)
            const legendContainer = d3.select("#" + chartId + "_legend");
            const typeGroups = {{"target": [], "inhibitory": [], "control": [], "unrelated": []}};
            categories.forEach(cat => {{
                const type = stats[cat].type || "unknown";
                if (!typeGroups[type]) typeGroups[type] = [];
                typeGroups[type].push(cat);
            }});

            const typeLabels = {{
                "target": "Target Categories",
                "inhibitory": "Inhibitory",
                "control": "Related (Control)",
                "unrelated": "Background"
            }};

            // Order: target first (most important), then inhibitory, control, unrelated
            const typeOrder = ["target", "inhibitory", "control", "unrelated"];

            typeOrder.forEach(type => {{
                if (!typeGroups[type] || typeGroups[type].length === 0) return;
                const section = legendContainer.append("div").attr("class", "legend-section");
                section.append("div").attr("class", "legend-title").text(typeLabels[type] || type);
                const items = section.append("div").attr("class", "legend-items");
                typeGroups[type].forEach(cat => {{
                    const item = items.append("div")
                        .attr("class", "legend-item")
                        .attr("data-category", cat);

                    item.append("div")
                        .attr("class", "legend-color")
                        .style("background", colors[cat]);

                    // For merged "Other", show count of merged categories
                    let labelText = labels[cat];
                    if (stats[cat].merged_count) {{
                        labelText += ` (${{stats[cat].merged_count}} categories)`;
                    }}

                    item.append("span")
                        .attr("class", "legend-label")
                        .attr("title", labelText)
                        .text(labelText);

                    item.append("span")
                        .attr("class", "legend-z")
                        .text(`z̄=${{stats[cat].z_mean.toFixed(1)}}`);

                    // Add hover interaction
                    item.on("mouseenter", function() {{
                        highlightCategory(cat);
                    }}).on("mouseleave", function() {{
                        clearHighlight();
                    }});
                }});
            }});

            // Highlight function
            function highlightCategory(cat) {{
                // Dim all legend items except this one
                d3.selectAll("#" + chartId + "_legend .legend-item")
                    .classed("dimmed", d => true)
                    .filter(function() {{ return d3.select(this).attr("data-category") === cat; }})
                    .classed("dimmed", false)
                    .classed("highlighted", true);

                // Dim all layers except this one
                Object.keys(layerElements).forEach(key => {{
                    if (key === cat) {{
                        layerElements[key].classed("dimmed", false).classed("highlighted", true);
                    }} else {{
                        layerElements[key].classed("dimmed", true).classed("highlighted", false);
                    }}
                }});

                // Dim all points except this category
                d3.selectAll("#" + chartId + "_svg .point")
                    .classed("dimmed", d => d.category !== cat)
                    .classed("highlighted", d => d.category === cat);
            }}

            function clearHighlight() {{
                d3.selectAll("#" + chartId + "_legend .legend-item")
                    .classed("dimmed", false)
                    .classed("highlighted", false);

                Object.keys(layerElements).forEach(key => {{
                    layerElements[key].classed("dimmed", false).classed("highlighted", false);
                }});

                d3.selectAll("#" + chartId + "_svg .point")
                    .classed("dimmed", false)
                    .classed("highlighted", false);
            }}

            // Chart setup
            const svg = d3.select("#" + chartId + "_svg");
            const container = svg.node().parentElement;
            const width = container.clientWidth || 900;
            const height = 350;
            const margin = {{top: 20, right: 20, bottom: 45, left: 55}};
            const innerWidth = width - margin.left - margin.right;
            const innerHeight = height - margin.top - margin.bottom;

            svg.attr("viewBox", `0 0 ${{width}} ${{height}}`);

            const g = svg.append("g")
                .attr("transform", `translate(${{margin.left}},${{margin.top}})`);

            // Scales
            const x = d3.scaleLinear()
                .domain([d3.min(stackedData, d => d.z) - 0.3, d3.max(stackedData, d => d.z) + 0.3])
                .range([0, innerWidth]);

            const y = d3.scaleLinear()
                .domain([0, 1])
                .range([innerHeight, 0]);

            // Stack generator
            const stack = d3.stack()
                .keys(categories)
                .order(d3.stackOrderNone)
                .offset(d3.stackOffsetNone);

            const layers = stack(stackedData);

            // Area generator
            const area = d3.area()
                .x(d => x(d.data.z))
                .y0(d => y(d[0]))
                .y1(d => y(d[1]))
                .curve(d3.curveMonotoneX);

            // Draw stacked areas and capture references
            g.selectAll(".layer")
                .data(layers)
                .enter()
                .append("path")
                .attr("class", "layer")
                .attr("d", area)
                .attr("fill", d => colors[d.key])
                .attr("opacity", d => {{
                    // Target categories more opaque, others more transparent
                    const type = stats[d.key]?.type || "unknown";
                    return type === "target" ? 0.85 : (type === "inhibitory" ? 0.6 : 0.5);
                }})
                .each(function(d) {{
                    layerElements[d.key] = d3.select(this);
                }})
                .on("mouseenter", function(event, d) {{
                    highlightCategory(d.key);
                }})
                .on("mouseleave", function() {{
                    clearHighlight();
                }});

            // Category index for positioning
            const categoryIndex = {{}};
            categories.forEach((cat, i) => categoryIndex[cat] = i);

            // Tooltip
            const tooltip = d3.select("#" + chartId + "_tooltip");

            // Helper to get band boundaries for a category at a given stacked data point
            // Data stores individual proportions, so we compute cumulative positions
            function getBandBounds(sd, catIdx) {{
                let bottom = 0, top = 0;
                for (let i = 0; i <= catIdx; i++) {{
                    bottom = top;
                    top += sd[categories[i]] || 0;
                }}
                return {{ bottom, top }};
            }}

            // Draw points with interpolated y-positions to match smooth curves
            g.selectAll(".point")
                .data(pointsData)
                .enter()
                .append("circle")
                .attr("class", "point")
                .attr("cx", d => x(d.z))
                .attr("cy", d => {{
                    const catIdx = categoryIndex[d.category];
                    const zVal = d.z;

                    // Find surrounding bins for interpolation
                    let leftIdx = 0, rightIdx = stackedData.length - 1;
                    for (let i = 0; i < stackedData.length - 1; i++) {{
                        if (stackedData[i].z <= zVal && stackedData[i + 1].z > zVal) {{
                            leftIdx = i;
                            rightIdx = i + 1;
                            break;
                        }}
                    }}

                    // Handle edge cases
                    if (zVal <= stackedData[0].z) {{
                        leftIdx = rightIdx = 0;
                    }} else if (zVal >= stackedData[stackedData.length - 1].z) {{
                        leftIdx = rightIdx = stackedData.length - 1;
                    }}

                    const sdL = stackedData[leftIdx];
                    const sdR = stackedData[rightIdx];
                    const boundsL = getBandBounds(sdL, catIdx);
                    const boundsR = getBandBounds(sdR, catIdx);

                    let bottom, top;
                    if (leftIdx === rightIdx) {{
                        bottom = boundsL.bottom;
                        top = boundsL.top;
                    }} else {{
                        // Linear interpolation between bins
                        const t = (zVal - sdL.z) / (sdR.z - sdL.z);
                        bottom = boundsL.bottom + t * (boundsR.bottom - boundsL.bottom);
                        top = boundsL.top + t * (boundsR.top - boundsL.top);
                    }}

                    const bandCenter = (bottom + top) / 2;
                    const bandHeight = Math.max(0.01, top - bottom);  // Ensure minimum height
                    const jitter = (Math.random() - 0.5) * bandHeight * 0.6;
                    return y(Math.max(0, Math.min(1, bandCenter + jitter)));
                }})
                .attr("r", d => d.is_negative ? 5 : 4)
                .attr("fill", d => colors[d.category])
                .attr("fill-opacity", d => d.is_negative ? 0.25 : 1.0)
                .attr("stroke-width", d => d.is_negative ? 2 : 1)
                .attr("stroke-dasharray", d => d.is_negative ? "3,2" : "none")
                .on("mouseover", function(event, d) {{
                    d3.select(this).attr("r", 6).attr("stroke-width", 2);
                    const polarityLabel = d.is_negative ? " (negative firing)" : "";
                    tooltip.style("display", "block")
                        .style("left", (event.clientX + 15) + "px")
                        .style("top", (event.clientY - 10) + "px")
                        .html(`
                            <div class="tooltip-category">${{labels[d.category]}}${{polarityLabel}}</div>
                            <div class="tooltip-prompt">${{d.highlighted_full || d.full_prompt || d.prompt}}</div>
                            <div class="tooltip-stats">
                                <span class="tooltip-stat-label">Act:</span> ${{d.activation.toFixed(3)}}
                                &nbsp;|&nbsp;
                                <span class="tooltip-stat-label">Z:</span> ${{d.z >= 0 ? '+' : ''}}${{d.z.toFixed(2)}}σ
                            </div>
                        `);
                }})
                .on("mousemove", function(event) {{
                    tooltip
                        .style("left", (event.clientX + 15) + "px")
                        .style("top", (event.clientY - 10) + "px");
                }})
                .on("mouseout", function() {{
                    d3.select(this).attr("r", 4).attr("stroke-width", 1);
                    tooltip.style("display", "none");
                }});

            // Axes
            g.append("g")
                .attr("class", "axis")
                .attr("transform", `translate(0,${{innerHeight}})`)
                .call(d3.axisBottom(x).ticks(10));

            g.append("text")
                .attr("class", "axis-label")
                .attr("x", innerWidth / 2)
                .attr("y", innerHeight + 38)
                .attr("text-anchor", "middle")
                .text("Standard Deviations from Mean Activation");

            g.append("g")
                .attr("class", "axis")
                .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")));

            g.append("text")
                .attr("class", "axis-label")
                .attr("transform", "rotate(-90)")
                .attr("x", -innerHeight / 2)
                .attr("y", -42)
                .attr("text-anchor", "middle")
                .text("Cumulative Proportion");

            // Zero line
            g.append("line")
                .attr("x1", x(0))
                .attr("x2", x(0))
                .attr("y1", 0)
                .attr("y2", innerHeight)
                .attr("stroke", "var(--text-secondary)")
                .attr("stroke-width", 1)
                .attr("stroke-dasharray", "4,4");
        }})();
        </script>
    </div>
    '''


# =============================================================================
# DEPENDENCY TABLES (Upstream/Downstream)
# =============================================================================

def generate_upstream_dependency_table(
    dependency_data: dict[str, Any],
    title: str = "Upstream Dependencies",
    caption: str = "",
    neuron_labels: dict[str, str] = None,
    wiring_weights: dict[str, float] = None,
    relp_weights: dict[str, float] = None,  # Deprecated, use wiring_weights
    wiring_stats: dict[str, Any] = None,  # Full wiring stats for regime annotation
) -> str:
    """Generate table showing how ablating upstream neurons affects target neuron.

    Args:
        dependency_data: From upstream_dependency_results, contains:
            - upstream_neurons: list of neuron IDs
            - individual_ablation: dict mapping neuron_id -> {mean_change_percent, dependency_strength}
            - combined_ablation: {mean_change_percent}
            - total_prompts: number of prompts tested (for batch experiments)
        title: Figure title
        caption: Optional caption
        neuron_labels: Optional dict mapping neuron_id to label
        wiring_weights: Optional dict mapping neuron_id to wiring weight for agreement comparison
        relp_weights: Deprecated, use wiring_weights instead

    Returns:
        HTML string for the dependency table
    """
    # Use wiring_weights if provided, fall back to relp_weights for backward compatibility
    weights = wiring_weights if wiring_weights is not None else relp_weights
    if not dependency_data:
        return ""

    individual = dependency_data.get("individual_ablation", {})
    combined = dependency_data.get("combined_ablation", {})
    # Compute total_prompts from test_prompts list if not provided directly
    total_prompts = dependency_data.get("total_prompts", 0)
    if total_prompts == 0 and "test_prompts" in dependency_data:
        total_prompts = len(dependency_data["test_prompts"])
    neuron_labels = neuron_labels or {}
    weights = weights if weights else {}  # Use the merged weights from above

    if not individual:
        return ""

    # Track wiring agreement stats
    n_agree = 0
    n_total_wiring = 0

    # Collect rows for potential collapsing
    row_list = []
    for neuron_id, info in individual.items():
        change = info.get("mean_change_percent", 0)
        strength = info.get("dependency_strength", "unknown")

        # Color based on change direction and magnitude
        if change < -50:
            change_class = "strong-negative"
            change_color = "#dc2626"
        elif change < -10:
            change_class = "moderate-negative"
            change_color = "#f97316"
        elif change > 50:
            change_class = "strong-positive"
            change_color = "#16a34a"
        elif change > 10:
            change_class = "moderate-positive"
            change_color = "#22c55e"
        else:
            change_class = "weak"
            change_color = "#9ca3af"

        # Strength badge
        strength_badge = {
            "strong": '<span class="strength-badge strong">strong</span>',
            "moderate": '<span class="strength-badge moderate">moderate</span>',
            "weak": '<span class="strength-badge weak">weak</span>',
        }.get(strength, '<span class="strength-badge weak">weak</span>')

        neuron_link = linkify_neuron_ids(neuron_id)
        change_sign = "+" if change > 0 else ""

        # Get neuron label
        label = neuron_labels.get(neuron_id, "")
        label_html = f'<div class="neuron-label">{escape_html(label)}</div>' if label else ""
        tooltip = f' title="{escape_html(label)}"' if label else ""

        # Wiring agreement check
        # Logic: Wiring+ (excitatory) means ablating should DECREASE target (negative change)
        #        Wiring- (inhibitory) means ablating should INCREASE target (positive change)
        wiring_cell = ""
        if neuron_id in weights:
            wiring_wt = weights[neuron_id]
            n_total_wiring += 1
            # Agreement: opposite signs (Wiring+ with ablation-, or Wiring- with ablation+)
            # Or both near zero
            if abs(wiring_wt) < 0.01 and abs(change) < 5:
                agrees = True  # Both negligible
            elif wiring_wt > 0 and change < 0:
                agrees = True  # Excitatory upstream, ablation decreases target
            elif wiring_wt < 0 and change > 0:
                agrees = True  # Inhibitory upstream, ablation increases target
            else:
                agrees = False

            if agrees:
                n_agree += 1
                wiring_cell = f'<td class="wiring-cell" style="color: #16a34a; text-align: center;" title="Wiring: {wiring_wt:+.3f}">✓</td>'
            else:
                wiring_cell = f'<td class="wiring-cell" style="color: #dc2626; text-align: center;" title="Wiring: {wiring_wt:+.3f}">✗</td>'
        elif weights:  # Have wiring data but not for this neuron
            wiring_cell = '<td class="wiring-cell" style="color: #9ca3af; text-align: center;">—</td>'

        row_list.append(f'''
        <tr>
            <td class="neuron-cell"{tooltip}>
                {neuron_link}
                {label_html}
            </td>
            <td class="change-cell" style="color: {change_color};">{change_sign}{change:.1f}%</td>
            <td class="strength-cell">{strength_badge}</td>
            {wiring_cell}
        </tr>
        ''')

    # Split rows for collapsible if needed
    row_count = len(row_list)
    collapse_threshold = 6
    show_first = 4

    if row_count > collapse_threshold:
        visible_rows = '\n'.join(row_list[:show_first])
        hidden_rows = '\n'.join(row_list[show_first:])
        rows_html = visible_rows
        hidden_tbody = f'<tbody class="table-rows-hidden">{hidden_rows}</tbody>'
    else:
        rows_html = '\n'.join(row_list)
        hidden_tbody = ""

    # Combined effect row
    combined_html = ""
    if combined and "mean_change_percent" in combined:
        comb_change = combined["mean_change_percent"]
        comb_color = "#dc2626" if comb_change < -50 else "#f97316" if comb_change < 0 else "#16a34a"
        comb_sign = "+" if comb_change > 0 else ""
        wiring_spacer = '<td class="wiring-cell">—</td>' if weights else ""
        combined_html = f'''
        <tr class="combined-row">
            <td class="neuron-cell"><strong>All combined</strong></td>
            <td class="change-cell" style="color: {comb_color};"><strong>{comb_sign}{comb_change:.1f}%</strong></td>
            <td class="strength-cell">—</td>
            {wiring_spacer}
        </tr>
        '''

    caption_html = f'<p class="figure-caption">{escape_html(caption)}</p>' if caption else ""

    # Wiring agreement header and summary
    wiring_header = '<th title="✓ = ablation effect direction matches wiring polarity prediction; ✗ = opposite direction">Wiring ✓/✗</th>' if weights else ""
    wiring_summary = ""
    if n_total_wiring > 0:
        pct = 100 * n_agree / n_total_wiring
        color = "#16a34a" if pct >= 75 else "#f97316" if pct >= 50 else "#dc2626"
        regime_suffix = ""
        if wiring_stats and wiring_stats.get("regime_correction_applied"):
            regime_suffix = ' <span style="font-style: italic;">(regime-corrected)</span>'
        wiring_summary = f'<p class="wiring-summary" style="margin-top: 8px; font-size: 12px; color: {color};">Wiring direction agreement: {n_agree}/{n_total_wiring} ({pct:.0f}%){regime_suffix}</p>'

    # Sample size indicator
    sample_size_html = ""
    if total_prompts > 0:
        batch_label = "BATCH TEST" if total_prompts >= 10 else "LIMITED SAMPLE"
        batch_color = "#16a34a" if total_prompts >= 10 else "#f97316"
        sample_size_html = f'''
        <div class="sample-size-badge" style="display: inline-flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            <span style="background: {batch_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">{batch_label}</span>
            <span style="color: var(--text-secondary, #666); font-size: 13px;">Tested on <strong>{total_prompts}</strong> prompts where neuron was active</span>
        </div>
        '''

    # Build table HTML
    table_html = f'''
        <table>
            <thead>
                <tr>
                    <th>Upstream Neuron</th>
                    <th>Activation Change</th>
                    <th>Dependency</th>
                    {wiring_header}
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
            {hidden_tbody}
            <tbody>
                {combined_html}
            </tbody>
        </table>
    '''

    # Collapsible wrapper if needed
    hidden_count = row_count - show_first if row_count > collapse_threshold else 0
    if hidden_count > 0:
        import hashlib
        unique_id = hashlib.md5(f"upstream-{title}".encode()).hexdigest()[:8]
        collapsible_wrapper = f'''
        <div class="table-collapsible" id="tc-{unique_id}">
            <style>
                #tc-{unique_id} .table-rows-hidden {{ display: none; }}
                #tc-{unique_id}.expanded .table-rows-hidden {{ display: table-row-group; }}
                #tc-{unique_id}.expanded .expand-btn {{ display: none; }}
                #tc-{unique_id} .collapse-btn {{ display: none; }}
                #tc-{unique_id}.expanded .collapse-btn {{ display: inline-flex; }}
            </style>
            {table_html}
            <div style="text-align: center; margin-top: 12px;">
                <button class="expand-btn" onclick="document.getElementById('tc-{unique_id}').classList.add('expanded')"
                    style="background: var(--bg-secondary, #f3f4f6); border: 1px solid var(--border, #e5e7eb);
                           border-radius: 6px; padding: 6px 16px; font-size: 13px; color: var(--text-secondary, #666);
                           cursor: pointer; display: inline-flex; align-items: center; gap: 6px;">
                    <span>Show {hidden_count} more rows</span>
                    <span style="font-size: 10px;">▼</span>
                </button>
                <button class="collapse-btn" onclick="document.getElementById('tc-{unique_id}').classList.remove('expanded')"
                    style="background: var(--bg-secondary, #f3f4f6); border: 1px solid var(--border, #e5e7eb);
                           border-radius: 6px; padding: 6px 16px; font-size: 13px; color: var(--text-secondary, #666);
                           cursor: pointer; align-items: center; gap: 6px;">
                    <span>Show less</span>
                    <span style="font-size: 10px;">▲</span>
                </button>
            </div>
        </div>
        '''
    else:
        collapsible_wrapper = table_html

    return f'''
    <div class="figure-container dependency-table upstream">
        <div class="figure-title">{escape_html(title)}</div>
        <p class="figure-subtitle">Effect on target neuron when each upstream neuron is ablated</p>
        {sample_size_html}
        <div class="figure-explanation" style="background: var(--bg-secondary, #f8f9fa); padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 13px; color: var(--text-secondary, #666);">
            <strong>How to read:</strong> We ablate (zero out) each upstream neuron and measure how the target neuron's activation changes.
            A <span style="color: #dc2626;">negative change</span> means the upstream neuron was <em>excitatory</em> (helped activate the target).
            A <span style="color: #16a34a;">positive change</span> means it was <em>inhibitory</em> (suppressed the target).
        </div>
        {collapsible_wrapper}
        {wiring_summary}
        {caption_html}
    </div>
    <style>
    .dependency-table table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }}
    .dependency-table th {{
        text-align: left;
        padding: 12px 16px;
        background: var(--bg-secondary, #f8f9fa);
        font-weight: 600;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-secondary, #666);
        border-bottom: 2px solid var(--border, #e0e0e0);
    }}
    .dependency-table td {{
        padding: 12px 16px;
        border-bottom: 1px solid var(--border, #e0e0e0);
    }}
    .dependency-table .neuron-cell {{
        font-family: 'SF Mono', Monaco, monospace;
    }}
    .dependency-table .change-cell {{
        font-weight: 600;
        font-family: 'SF Mono', Monaco, monospace;
    }}
    .dependency-table .combined-row {{
        background: var(--bg-secondary, #f8f9fa);
    }}
    .strength-badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 100px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .strength-badge.strong {{
        background: rgba(220, 38, 38, 0.1);
        color: #dc2626;
    }}
    .strength-badge.moderate {{
        background: rgba(249, 115, 22, 0.1);
        color: #f97316;
    }}
    .strength-badge.weak {{
        background: rgba(156, 163, 175, 0.1);
        color: #9ca3af;
    }}
    .figure-subtitle {{
        font-size: 13px;
        color: var(--text-secondary, #666);
        margin: -8px 0 16px 0;
    }}
    </style>
    '''


def generate_downstream_dependency_table(
    dependency_data: dict[str, Any],
    title: str = "Downstream Ablation Effects",
    caption: str = "",
    neuron_labels: dict[str, str] = None,
    wiring_weights: dict[str, float] = None,
    relp_weights: dict[str, float] = None  # Deprecated, use wiring_weights
) -> str:
    """Generate table showing how ablating target neuron affects downstream neurons.

    Args:
        dependency_data: From downstream_dependency_results, contains:
            - downstream_neurons: list of neuron IDs
            - per_prompt_results: list of {prompt, downstream_effects: {neuron_id: {mean_change_percent}}}
            - dependency_summary: optional aggregated stats
            - total_prompts: number of prompts tested
        title: Figure title
        caption: Optional caption
        neuron_labels: Optional dict mapping neuron_id to label
        wiring_weights: Optional dict mapping neuron_id to wiring weight for comparison
        relp_weights: Deprecated, use wiring_weights instead

    Returns:
        HTML string for the dependency table
    """
    # Use wiring_weights if provided, fall back to relp_weights for backward compatibility
    weights = wiring_weights if wiring_weights is not None else relp_weights

    if not dependency_data:
        return ""

    downstream_neurons = dependency_data.get("downstream_neurons", [])
    per_prompt = dependency_data.get("per_prompt_results", [])
    summary = dependency_data.get("dependency_summary", {})
    # Get total_prompts - handle None explicitly (field may be present but None)
    total_prompts = dependency_data.get("total_prompts") or len(per_prompt)
    neuron_labels = neuron_labels or {}
    weights = weights if weights else {}

    if not downstream_neurons:
        return ""

    # Use pre-calculated dependency_summary if available (preferred - correct values)
    # Only fall back to recalculation if summary is missing
    aggregated = {}

    if summary:
        # Use the pre-calculated summary values - these are the authoritative averages
        for neuron_id in downstream_neurons:
            if neuron_id in summary:
                s = summary[neuron_id]
                aggregated[neuron_id] = {
                    "mean_change": s.get("mean_change_percent", 0),
                    "strength": s.get("dependency_strength", "unknown"),
                    "n_prompts": total_prompts,
                }
    elif per_prompt:
        # Fallback: aggregate from per_prompt_results (no capping - preserve raw values)
        for neuron_id in downstream_neurons:
            changes = []
            for pr in per_prompt:
                effects = pr.get("downstream_effects", {})
                if neuron_id in effects:
                    effect_data = effects[neuron_id]
                    change = effect_data.get("mean_change_percent", effect_data.get("change_percent", 0))
                    changes.append(change)

            if changes:
                aggregated[neuron_id] = {
                    "mean_change": sum(changes) / len(changes),
                    "n_prompts": len(changes),
                }

    if not aggregated:
        return ""

    # Sort by absolute mean change
    sorted_neurons = sorted(aggregated.items(), key=lambda x: abs(x[1]["mean_change"]), reverse=True)

    # Track wiring agreement stats
    n_agree = 0
    n_total_wiring = 0

    # Collect rows for potential collapsing
    row_list = []
    for neuron_id, info in sorted_neurons[:10]:  # Top 10
        mean_change = info["mean_change"]

        # Use pre-calculated strength from summary if available, otherwise calculate from magnitude
        if "strength" in info and info["strength"] in ("strong", "moderate", "weak"):
            strength = info["strength"]
            # Color based on direction and strength
            if strength == "strong":
                change_color = "#dc2626" if mean_change < 0 else "#16a34a"
            elif strength == "moderate":
                change_color = "#f97316" if mean_change < 0 else "#22c55e"
            else:
                change_color = "#9ca3af"
        else:
            # Calculate strength from magnitude
            if abs(mean_change) > 30:
                change_color = "#dc2626" if mean_change < 0 else "#16a34a"
                strength = "strong"
            elif abs(mean_change) > 10:
                change_color = "#f97316" if mean_change < 0 else "#22c55e"
                strength = "moderate"
            else:
                change_color = "#9ca3af"
                strength = "weak"

        strength_badge = f'<span class="strength-badge {strength}">{strength}</span>'
        neuron_link = linkify_neuron_ids(neuron_id)
        mean_sign = "+" if mean_change > 0 else ""

        # Get neuron label
        label = neuron_labels.get(neuron_id, "")
        label_html = f'<div class="neuron-label">{escape_html(label)}</div>' if label else ""
        tooltip = f' title="{escape_html(label)}"' if label else ""

        # Wiring agreement check for downstream
        # Logic: Wiring+ means target excites downstream, so ablating target should DECREASE downstream (negative change)
        #        Wiring- means target inhibits downstream, so ablating target should INCREASE downstream (positive change)
        wiring_cell = ""
        if neuron_id in weights:
            wiring_wt = weights[neuron_id]
            n_total_wiring += 1
            # Agreement: Wiring+ with ablation- (excitatory), or Wiring- with ablation+ (inhibitory)
            if abs(wiring_wt) < 0.01 and abs(mean_change) < 5:
                agrees = True  # Both negligible
            elif wiring_wt > 0 and mean_change < 0:
                agrees = True  # Target excited downstream, ablating reduces it
            elif wiring_wt < 0 and mean_change > 0:
                agrees = True  # Target inhibited downstream, ablating increases it
            else:
                agrees = False

            if agrees:
                n_agree += 1
                wiring_cell = f'<td class="wiring-cell" style="color: #16a34a; text-align: center;" title="Wiring: {wiring_wt:+.3f}">✓</td>'
            else:
                wiring_cell = f'<td class="wiring-cell" style="color: #dc2626; text-align: center;" title="Wiring: {wiring_wt:+.3f}">✗</td>'
        elif weights:  # Have wiring data but not for this neuron
            wiring_cell = '<td class="wiring-cell" style="color: #9ca3af; text-align: center;">—</td>'

        row_list.append(f'''
        <tr>
            <td class="neuron-cell"{tooltip}>
                {neuron_link}
                {label_html}
            </td>
            <td class="change-cell" style="color: {change_color};">{mean_sign}{mean_change:.1f}%</td>
            <td class="strength-cell">{strength_badge}</td>
            {wiring_cell}
        </tr>
        ''')

    # Split rows for collapsible if needed
    row_count = len(row_list)
    collapse_threshold = 6
    show_first = 4

    if row_count > collapse_threshold:
        visible_rows = '\n'.join(row_list[:show_first])
        hidden_rows = '\n'.join(row_list[show_first:])
        rows_html = visible_rows
        hidden_tbody = f'<tbody class="table-rows-hidden">{hidden_rows}</tbody>'
    else:
        rows_html = '\n'.join(row_list)
        hidden_tbody = ""

    caption_html = f'<p class="figure-caption">{escape_html(caption)}</p>' if caption else ""

    # Count prompts tested - use total_prompts (set from data) instead of len(per_prompt)
    sample_size_html = ""
    if total_prompts > 0:
        batch_label = "BATCH TEST" if total_prompts >= 5 else "LIMITED SAMPLE"
        batch_color = "#16a34a" if total_prompts >= 5 else "#f97316"
        sample_size_html = f'''
        <div class="sample-size-badge" style="display: inline-flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            <span style="background: {batch_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">{batch_label}</span>
            <span style="color: var(--text-secondary, #666); font-size: 13px;">Tested on <strong>{total_prompts}</strong> prompts, effects averaged</span>
        </div>
        '''

    # Wiring agreement header and summary
    wiring_header = '<th title="✓ = ablation effect direction matches wiring polarity prediction; ✗ = opposite direction">Wiring ✓/✗</th>' if weights else ""
    wiring_summary = ""
    if n_total_wiring > 0:
        pct = 100 * n_agree / n_total_wiring
        color = "#16a34a" if pct >= 75 else "#f97316" if pct >= 50 else "#dc2626"
        wiring_summary = f'<p class="wiring-summary" style="margin-top: 8px; font-size: 12px; color: {color};">Wiring direction agreement: {n_agree}/{n_total_wiring} ({pct:.0f}%)</p>'

    # Build table HTML
    table_html = f'''
        <table>
            <thead>
                <tr>
                    <th>Downstream Neuron</th>
                    <th>Mean Change</th>
                    <th>Effect</th>
                    {wiring_header}
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
            {hidden_tbody}
        </table>
    '''

    # Collapsible wrapper if needed
    hidden_count = row_count - show_first if row_count > collapse_threshold else 0
    if hidden_count > 0:
        import hashlib
        unique_id = hashlib.md5(f"downstream-{title}".encode()).hexdigest()[:8]
        collapsible_wrapper = f'''
        <div class="table-collapsible" id="tc-{unique_id}">
            <style>
                #tc-{unique_id} .table-rows-hidden {{ display: none; }}
                #tc-{unique_id}.expanded .table-rows-hidden {{ display: table-row-group; }}
                #tc-{unique_id}.expanded .expand-btn {{ display: none; }}
                #tc-{unique_id} .collapse-btn {{ display: none; }}
                #tc-{unique_id}.expanded .collapse-btn {{ display: inline-flex; }}
            </style>
            {table_html}
            <div style="text-align: center; margin-top: 12px;">
                <button class="expand-btn" onclick="document.getElementById('tc-{unique_id}').classList.add('expanded')"
                    style="background: var(--bg-secondary, #f3f4f6); border: 1px solid var(--border, #e5e7eb);
                           border-radius: 6px; padding: 6px 16px; font-size: 13px; color: var(--text-secondary, #666);
                           cursor: pointer; align-items: center; gap: 6px;">
                    <span>Show {hidden_count} more rows</span>
                    <span style="font-size: 10px;">▼</span>
                </button>
                <button class="collapse-btn" onclick="document.getElementById('tc-{unique_id}').classList.remove('expanded')"
                    style="background: var(--bg-secondary, #f3f4f6); border: 1px solid var(--border, #e5e7eb);
                           border-radius: 6px; padding: 6px 16px; font-size: 13px; color: var(--text-secondary, #666);
                           cursor: pointer; align-items: center; gap: 6px;">
                    <span>Show less</span>
                    <span style="font-size: 10px;">▲</span>
                </button>
            </div>
        </div>
        '''
    else:
        collapsible_wrapper = table_html

    return f'''
    <div class="figure-container dependency-table downstream">
        <div class="figure-title">{escape_html(title)}</div>
        <p class="figure-subtitle">Effect on downstream neurons when this neuron is ablated</p>
        {sample_size_html}
        <div class="figure-explanation" style="background: var(--bg-secondary, #f8f9fa); padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 13px; color: var(--text-secondary, #666);">
            <strong>How to read:</strong> We ablate (zero out) the target neuron and measure how downstream neurons' activations change.
            A <span style="color: #dc2626;">negative change</span> means the downstream neuron <em>depends on</em> this neuron for activation.
            <span style="color: #16a34a;">Positive changes</span> indicate inhibitory relationships (rare).
        </div>
        {collapsible_wrapper}
        {wiring_summary}
        {caption_html}
    </div>
    <style>
    .dependency-table .neuron-label {{
        font-size: 12px;
        color: var(--text-secondary, #666);
        margin-top: 2px;
    }}
    </style>
    '''


# =============================================================================
# WIRING POLARITY TABLE (Weight-Based Upstream Connectivity)
# =============================================================================

def generate_wiring_polarity_table(
    wiring_data: dict[str, Any],
    title: str = "Upstream Wiring (Weight-Based)",
    caption: str = "",
    initial_visible: int = 3,
) -> str:
    """Generate a dual-column table showing excitatory vs inhibitory upstream neurons.

    This visualizes the weight-based connectivity analysis that predicts which
    upstream neurons would INCREASE (excitatory) or DECREASE (inhibitory) the
    target neuron's activation when they fire.

    Args:
        wiring_data: From analyze_wiring tool, contains:
            - stats: {total_upstream_neurons, excitatory_count, inhibitory_count, ...}
            - label_coverage_pct: Percent of neurons with NeuronDB labels
            - top_excitatory: List of top excitatory connections
            - top_inhibitory: List of top inhibitory connections
        title: Figure title
        caption: Optional caption
        initial_visible: Number of rows to show initially (default 3)

    Returns:
        HTML string for the dual-column table
    """
    if not wiring_data:
        return ""

    stats = wiring_data.get("stats", {})
    top_excitatory = wiring_data.get("top_excitatory", [])[:15]
    top_inhibitory = wiring_data.get("top_inhibitory", [])[:15]
    coverage = wiring_data.get("label_coverage_pct", 0)

    if not top_excitatory and not top_inhibitory:
        return ""

    # Generate a unique ID for this table instance
    import random
    table_id = f"wiring_{random.randint(10000, 99999)}"

    # Find strength "cliff" - largest drop between consecutive neurons
    def find_cliff_index(connections, min_idx=2):
        """Find where the strength drops off significantly."""
        if len(connections) < 3:
            return len(connections)
        strengths = [abs(c.get("effective_strength", c.get("c_combined", 0))) for c in connections]
        max_drop_idx = min_idx
        max_drop_ratio = 0
        for i in range(min_idx, len(strengths) - 1):
            if strengths[i] > 0:
                drop_ratio = (strengths[i] - strengths[i+1]) / strengths[i]
                if drop_ratio > max_drop_ratio:
                    max_drop_ratio = drop_ratio
                    max_drop_idx = i + 1
        # Only use cliff if there's a significant drop (>30%)
        return max_drop_idx if max_drop_ratio > 0.3 else len(connections)

    exc_cliff = find_cliff_index(top_excitatory)
    inh_cliff = find_cliff_index(top_inhibitory)

    # Helper for RelP confirmation cell
    def relp_cell(conn):
        relp = conn.get("relp_confirmed")
        strength = conn.get("relp_strength")
        if relp is True:
            tip = f' title="RelP confirmed (avg weight: {strength:.4f})"' if strength else ' title="RelP confirmed"'
            return f'<td class="relp-col relp-yes"{tip}>✓</td>'
        elif relp is False:
            return '<td class="relp-col relp-no" title="Not found in corpus graphs">✗</td>'
        else:
            return '<td class="relp-col relp-na" title="Not yet checked">—</td>'

    # Generate excitatory rows
    excitatory_rows = ""
    for i, conn in enumerate(top_excitatory):
        neuron_id = conn.get("neuron_id", "")
        full_label = conn.get("label", "")
        label = full_label[:80]
        c_up = conn.get("c_up", 0)
        c_gate = conn.get("c_gate", 0)
        confidence = conn.get("polarity_confidence", 0)

        neuron_link = linkify_neuron_ids(neuron_id)
        conf_pct = int(confidence * 100)

        # Hide rows beyond initial_visible
        hidden_class = "hidden-row" if i >= initial_visible else ""
        # Mark cliff boundary
        cliff_class = "cliff-row" if i == exc_cliff - 1 else ""

        tooltip = f' title="{escape_html(full_label)}"' if full_label else ""
        excitatory_rows += f'''
        <tr class="{hidden_class} {cliff_class}" data-row="{i}">
            <td class="rank">{i+1}</td>
            <td class="neuron-cell"{tooltip}>
                {neuron_link}
                <div class="neuron-label">{escape_html(label)}</div>
            </td>
            <td class="weights">
                <span class="c-up" title="c_up: contribution to up channel">↑{c_up:+.3f}</span>
                <span class="c-gate" title="c_gate: contribution to gate channel">⊗{c_gate:+.3f}</span>
            </td>
            <td class="confidence">{conf_pct}%</td>
            {relp_cell(conn)}
        </tr>
        '''

    # Generate inhibitory rows
    inhibitory_rows = ""
    for i, conn in enumerate(top_inhibitory):
        neuron_id = conn.get("neuron_id", "")
        full_label = conn.get("label", "")
        label = full_label[:80]
        c_up = conn.get("c_up", 0)
        c_gate = conn.get("c_gate", 0)
        confidence = conn.get("polarity_confidence", 0)

        neuron_link = linkify_neuron_ids(neuron_id)
        conf_pct = int(confidence * 100)

        # Hide rows beyond initial_visible
        hidden_class = "hidden-row" if i >= initial_visible else ""
        cliff_class = "cliff-row" if i == inh_cliff - 1 else ""

        tooltip = f' title="{escape_html(full_label)}"' if full_label else ""
        inhibitory_rows += f'''
        <tr class="{hidden_class} {cliff_class}" data-row="{i}">
            <td class="rank">{i+1}</td>
            <td class="neuron-cell"{tooltip}>
                {neuron_link}
                <div class="neuron-label">{escape_html(label)}</div>
            </td>
            <td class="weights">
                <span class="c-up" title="c_up: contribution to up channel">↑{c_up:+.3f}</span>
                <span class="c-gate" title="c_gate: contribution to gate channel">⊗{c_gate:+.3f}</span>
            </td>
            <td class="confidence">{conf_pct}%</td>
            {relp_cell(conn)}
        </tr>
        '''

    # Stats summary - clarify what was actually analyzed
    exc_shown = len(top_excitatory)
    inh_shown = len(top_inhibitory)

    # Default caption explaining the methodology
    default_caption = (
        "Weights are from the SwiGLU gate projection matrix. "
        "<strong>c_up</strong> (↑) shows contribution to the up-projection channel; "
        "<strong>c_gate</strong> (⊗) shows contribution to the gating channel. "
        "Polarity confidence is based on weight sign agreement between channels."
    )
    # Append regime correction note if applicable
    if stats.get("regime_correction_applied"):
        default_caption += (
            " <strong>Polarity labels have been regime-corrected</strong>"
            " (target operates in inverted SwiGLU regime where gate and up channels are both negative)."
        )
    elif stats.get("regime_warning"):
        default_caption += (
            f" <em>Warning: {stats['regime_warning']}</em>"
        )
    caption_text = caption if caption else default_caption

    return f'''
    <div class="wiring-polarity-container" id="{table_id}">
        <h4>{escape_html(title)}</h4>
        <p class="stats-summary">
            Top <strong>{exc_shown}</strong> excitatory and <strong>{inh_shown}</strong> inhibitory connections
            <span class="coverage">({coverage:.0f}% have labels)</span>
        </p>

        <div class="dual-column">
            <div class="column excitatory">
                <h5 class="column-header exc-header">🔺 Excitatory (Increase activation)</h5>
                <table class="wiring-table exc-table" data-total="{exc_shown}">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Neuron</th>
                            <th>Weights</th>
                            <th>Conf</th>
                            <th class="relp-col" title="RelP corpus confirmation">RelP</th>
                        </tr>
                    </thead>
                    <tbody>
                        {excitatory_rows}
                    </tbody>
                </table>
                <button class="expand-btn" onclick="toggleWiringRows('{table_id}', 'exc')" data-expanded="false">
                    Show all {exc_shown} ▼
                </button>
            </div>

            <div class="column inhibitory">
                <h5 class="column-header inh-header">🔻 Inhibitory (Decrease activation)</h5>
                <table class="wiring-table inh-table" data-total="{inh_shown}">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Neuron</th>
                            <th>Weights</th>
                            <th>Conf</th>
                            <th class="relp-col" title="RelP corpus confirmation">RelP</th>
                        </tr>
                    </thead>
                    <tbody>
                        {inhibitory_rows}
                    </tbody>
                </table>
                <button class="expand-btn" onclick="toggleWiringRows('{table_id}', 'inh')" data-expanded="false">
                    Show all {inh_shown} ▼
                </button>
            </div>
        </div>
        <p class="wiring-caption">{caption_text}</p>
    </div>
    <style>
    .wiring-polarity-container {{
        margin: 32px auto;
        margin-left: -120px;
        margin-right: -120px;
        background: var(--bg-elevated, #ffffff);
        border-radius: 12px;
        padding: 24px 40px;
        border: 1px solid var(--border, #e0e0e0);
        width: calc(100% + 240px);
        max-width: 95vw;
        box-sizing: border-box;
    }}
    .wiring-polarity-container h4 {{
        margin-bottom: 10px;
    }}
    .stats-summary {{
        font-size: 14px;
        color: var(--text-secondary, #666);
        margin-bottom: 15px;
    }}
    .exc-badge {{
        background: #dcfce7;
        color: #166534;
        padding: 2px 8px;
        border-radius: 4px;
        margin: 0 4px;
    }}
    .inh-badge {{
        background: #fee2e2;
        color: #991b1b;
        padding: 2px 8px;
        border-radius: 4px;
        margin: 0 4px;
    }}
    .coverage {{
        color: #6b7280;
        font-size: 12px;
    }}
    .dual-column {{
        display: flex;
        gap: 20px;
    }}
    .dual-column .column {{
        flex: 1;
    }}
    .column-header {{
        padding: 8px 12px;
        border-radius: 6px 6px 0 0;
        margin-bottom: 0;
    }}
    .exc-header {{
        background: #dcfce7;
        color: #166534;
    }}
    .inh-header {{
        background: #fee2e2;
        color: #991b1b;
    }}
    .wiring-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }}
    .wiring-table th, .wiring-table td {{
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid var(--border-color, #e5e7eb);
    }}
    .wiring-table .rank {{
        width: 30px;
        color: #9ca3af;
    }}
    .wiring-table .neuron-cell {{
        max-width: 200px;
    }}
    .wiring-table .neuron-label {{
        font-size: 11px;
        color: #6b7280;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 180px;
    }}
    .wiring-table .weights {{
        font-family: monospace;
        font-size: 11px;
    }}
    .c-up {{
        color: #059669;
        margin-right: 4px;
    }}
    .c-gate {{
        color: #7c3aed;
    }}
    .exc-table tbody tr:hover {{
        background: #f0fdf4;
    }}
    .inh-table tbody tr:hover {{
        background: #fef2f2;
    }}
    .wiring-table .confidence {{
        width: 50px;
        color: #6b7280;
    }}
    .wiring-table .relp-col {{
        width: 40px;
        text-align: center;
    }}
    .wiring-table .relp-yes {{
        color: #16a34a;
        font-weight: 600;
    }}
    .wiring-table .relp-no {{
        color: #dc2626;
    }}
    .wiring-table .relp-na {{
        color: #9ca3af;
    }}
    .wiring-table .hidden-row {{
        display: none;
    }}
    .wiring-table .cliff-row td {{
        border-bottom: 2px dashed #9ca3af;
    }}
    .expand-btn {{
        display: block;
        width: 100%;
        padding: 8px;
        margin-top: 4px;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12px;
        color: #6b7280;
        transition: background 0.2s;
    }}
    .expand-btn:hover {{
        background: #f3f4f6;
    }}
    .wiring-caption {{
        font-size: 12px;
        color: #6b7280;
        margin-top: 12px;
        line-height: 1.5;
        border-left: 3px solid #e5e7eb;
        padding-left: 12px;
    }}
    </style>
    <script>
    function toggleWiringRows(tableId, type) {{
        const container = document.getElementById(tableId);
        const table = container.querySelector(type === 'exc' ? '.exc-table' : '.inh-table');
        const btn = table.parentElement.querySelector('.expand-btn');
        const rows = table.querySelectorAll('.hidden-row');
        const isExpanded = btn.dataset.expanded === 'true';

        rows.forEach(row => {{
            row.style.display = isExpanded ? 'none' : 'table-row';
        }});

        const total = table.dataset.total;
        btn.innerHTML = isExpanded ? `Show all ${{total}} ▼` : `Show top 3 ▲`;
        btn.dataset.expanded = isExpanded ? 'false' : 'true';
    }}
    </script>
    '''


# =============================================================================
# DOWNSTREAM WIRING POLARITY TABLE
# =============================================================================

def generate_downstream_wiring_table(
    output_wiring_data: dict[str, Any],
    title: str = "Downstream Wiring (Weight-Based)",
    caption: str = "",
    initial_visible: int = 3,
) -> str:
    """Generate a dual-column table showing downstream neurons this neuron excites vs inhibits.

    This visualizes the weight-based downstream connectivity analysis that predicts which
    downstream neurons this neuron would EXCITE (positive weights) or INHIBIT (negative weights)
    when it fires.

    Args:
        output_wiring_data: From analyze_output_wiring tool, contains:
            - stats: {total_downstream_neurons, excitatory_count, inhibitory_count, ...}
            - label_coverage_pct: Percent of neurons with NeuronDB labels
            - top_excitatory: List of top excitatory connections
            - top_inhibitory: List of top inhibitory connections
        title: Figure title
        caption: Optional caption
        initial_visible: Number of rows to show initially (default 3)

    Returns:
        HTML string for the dual-column table
    """
    if not output_wiring_data:
        return ""

    stats = output_wiring_data.get("stats", {})
    top_excitatory = output_wiring_data.get("top_excitatory", [])[:15]
    top_inhibitory = output_wiring_data.get("top_inhibitory", [])[:15]
    coverage = output_wiring_data.get("label_coverage_pct", 0)

    if not top_excitatory and not top_inhibitory:
        return ""

    # Generate a unique ID for this table instance
    import random
    table_id = f"downstream_wiring_{random.randint(10000, 99999)}"

    # Find strength "cliff" - largest drop between consecutive neurons
    def find_cliff_index(connections, min_idx=2):
        """Find where the strength drops off significantly."""
        if len(connections) < 3:
            return len(connections)
        strengths = [abs(c.get("effective_strength", c.get("weight", 0))) for c in connections]
        max_drop_idx = min_idx
        max_drop_ratio = 0
        for i in range(min_idx, len(strengths) - 1):
            if strengths[i] > 0:
                drop_ratio = (strengths[i] - strengths[i+1]) / strengths[i]
                if drop_ratio > max_drop_ratio:
                    max_drop_ratio = drop_ratio
                    max_drop_idx = i + 1
        # Only use cliff if there's a significant drop (>30%)
        return max_drop_idx if max_drop_ratio > 0.3 else len(connections)

    exc_cliff = find_cliff_index(top_excitatory)
    inh_cliff = find_cliff_index(top_inhibitory)

    # Helper for RelP confirmation cell
    def relp_cell(conn):
        relp = conn.get("relp_confirmed")
        strength = conn.get("relp_strength")
        if relp is True:
            tip = f' title="RelP confirmed (avg weight: {strength:.4f})"' if strength else ' title="RelP confirmed"'
            return f'<td class="relp-col relp-yes"{tip}>✓</td>'
        elif relp is False:
            return '<td class="relp-col relp-no" title="Not found in corpus graphs">✗</td>'
        else:
            return '<td class="relp-col relp-na" title="Not yet checked">—</td>'

    # Generate excitatory rows (neurons this neuron EXCITES)
    excitatory_rows = ""
    for i, conn in enumerate(top_excitatory):
        neuron_id = conn.get("neuron_id", "")
        full_label = conn.get("label", "")
        label = full_label[:80]
        weight = conn.get("effective_strength", conn.get("weight", 0))
        confidence = conn.get("polarity_confidence", 1.0)  # Default high for downstream

        neuron_link = linkify_neuron_ids(neuron_id)
        conf_pct = int(confidence * 100)

        # Hide rows beyond initial_visible
        hidden_class = "hidden-row" if i >= initial_visible else ""
        cliff_class = "cliff-row" if i == exc_cliff - 1 else ""

        tooltip = f' title="{escape_html(full_label)}"' if full_label else ""
        excitatory_rows += f'''
        <tr class="{hidden_class} {cliff_class}" data-row="{i}">
            <td class="rank">{i+1}</td>
            <td class="neuron-cell"{tooltip}>
                {neuron_link}
                <div class="neuron-label">{escape_html(label)}</div>
            </td>
            <td class="weights">
                <span class="weight-val">w={weight:+.3f}</span>
            </td>
            <td class="confidence">{conf_pct}%</td>
            {relp_cell(conn)}
        </tr>
        '''

    # Generate inhibitory rows (neurons this neuron INHIBITS)
    inhibitory_rows = ""
    for i, conn in enumerate(top_inhibitory):
        neuron_id = conn.get("neuron_id", "")
        full_label = conn.get("label", "")
        label = full_label[:80]
        weight = conn.get("effective_strength", conn.get("weight", 0))
        confidence = conn.get("polarity_confidence", 1.0)

        neuron_link = linkify_neuron_ids(neuron_id)
        conf_pct = int(confidence * 100)

        # Hide rows beyond initial_visible
        hidden_class = "hidden-row" if i >= initial_visible else ""
        cliff_class = "cliff-row" if i == inh_cliff - 1 else ""

        tooltip = f' title="{escape_html(full_label)}"' if full_label else ""
        inhibitory_rows += f'''
        <tr class="{hidden_class} {cliff_class}" data-row="{i}">
            <td class="rank">{i+1}</td>
            <td class="neuron-cell"{tooltip}>
                {neuron_link}
                <div class="neuron-label">{escape_html(label)}</div>
            </td>
            <td class="weights">
                <span class="weight-val">w={weight:+.3f}</span>
            </td>
            <td class="confidence">{conf_pct}%</td>
            {relp_cell(conn)}
        </tr>
        '''

    # Stats summary
    exc_shown = len(top_excitatory)
    inh_shown = len(top_inhibitory)

    # Default caption explaining downstream methodology
    default_caption = (
        "Downstream wiring shows which later-layer neurons this neuron influences via weight connections. "
        "<strong>Excitatory</strong> connections (positive weights) increase the downstream neuron's activation; "
        "<strong>inhibitory</strong> connections (negative weights) decrease it."
    )
    caption_text = caption if caption else default_caption

    return f'''
    <div class="wiring-polarity-container downstream-wiring" id="{table_id}">
        <h4>{escape_html(title)}</h4>
        <p class="stats-summary">
            Top <strong>{exc_shown}</strong> excited and <strong>{inh_shown}</strong> inhibited downstream neurons
            <span class="coverage">({coverage:.0f}% have labels)</span>
        </p>

        <div class="dual-column">
            <div class="column excitatory">
                <h5 class="column-header exc-header">🔺 This Neuron Excites</h5>
                <table class="wiring-table exc-table" data-total="{exc_shown}">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Downstream Neuron</th>
                            <th>Weight</th>
                            <th>Conf</th>
                            <th class="relp-col" title="RelP corpus confirmation">RelP</th>
                        </tr>
                    </thead>
                    <tbody>
                        {excitatory_rows}
                    </tbody>
                </table>
                <button class="expand-btn" onclick="toggleWiringRows('{table_id}', 'exc')" data-expanded="false">
                    Show all {exc_shown} ▼
                </button>
            </div>

            <div class="column inhibitory">
                <h5 class="column-header inh-header">🔻 This Neuron Inhibits</h5>
                <table class="wiring-table inh-table" data-total="{inh_shown}">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Downstream Neuron</th>
                            <th>Weight</th>
                            <th>Conf</th>
                            <th class="relp-col" title="RelP corpus confirmation">RelP</th>
                        </tr>
                    </thead>
                    <tbody>
                        {inhibitory_rows}
                    </tbody>
                </table>
                <button class="expand-btn" onclick="toggleWiringRows('{table_id}', 'inh')" data-expanded="false">
                    Show all {inh_shown} ▼
                </button>
            </div>
        </div>
        <p class="wiring-caption">{caption_text}</p>
    </div>
    <style>
    /* Full layout CSS for downstream wiring table (self-contained) */
    .downstream-wiring {{
        margin: 32px auto;
        margin-left: -120px;
        margin-right: -120px;
        background: var(--bg-elevated, #ffffff);
        border-radius: 12px;
        padding: 24px 40px;
        border: 1px solid var(--border, #e0e0e0);
        width: calc(100% + 240px);
        max-width: 95vw;
        box-sizing: border-box;
    }}
    .downstream-wiring h4 {{
        margin-bottom: 10px;
    }}
    .downstream-wiring .stats-summary {{
        font-size: 14px;
        color: var(--text-secondary, #666);
        margin-bottom: 15px;
    }}
    .downstream-wiring .coverage {{
        color: #6b7280;
        font-size: 12px;
    }}
    .downstream-wiring .dual-column {{
        display: flex;
        gap: 20px;
    }}
    .downstream-wiring .dual-column .column {{
        flex: 1;
    }}
    .downstream-wiring .column-header {{
        padding: 8px 12px;
        border-radius: 6px 6px 0 0;
        margin-bottom: 0;
    }}
    .downstream-wiring .exc-header {{
        background: #dcfce7;
        color: #166534;
    }}
    .downstream-wiring .inh-header {{
        background: #fee2e2;
        color: #991b1b;
    }}
    .downstream-wiring .wiring-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }}
    .downstream-wiring .wiring-table th,
    .downstream-wiring .wiring-table td {{
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid var(--border-color, #e5e7eb);
    }}
    .downstream-wiring .wiring-table .rank {{
        width: 30px;
        color: #9ca3af;
    }}
    .downstream-wiring .wiring-table .neuron-cell {{
        max-width: 200px;
    }}
    .downstream-wiring .wiring-table .neuron-label {{
        font-size: 11px;
        color: #6b7280;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 180px;
    }}
    .downstream-wiring .weight-val {{
        font-family: monospace;
        font-size: 12px;
        color: #4b5563;
    }}
    .downstream-wiring .wiring-table .confidence {{
        width: 50px;
        color: #6b7280;
    }}
    .downstream-wiring .wiring-table .relp-col {{
        width: 40px;
        text-align: center;
    }}
    .downstream-wiring .relp-yes {{
        color: #16a34a;
        font-weight: 600;
    }}
    .downstream-wiring .relp-no {{
        color: #dc2626;
    }}
    .downstream-wiring .relp-na {{
        color: #9ca3af;
    }}
    .downstream-wiring .exc-table tbody tr:hover {{
        background: #f0fdf4;
    }}
    .downstream-wiring .inh-table tbody tr:hover {{
        background: #fef2f2;
    }}
    .downstream-wiring .wiring-table .hidden-row {{
        display: none;
    }}
    .downstream-wiring .wiring-table .cliff-row td {{
        border-bottom: 2px dashed #9ca3af;
    }}
    .downstream-wiring .expand-btn {{
        display: block;
        width: 100%;
        padding: 8px;
        margin-top: 4px;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12px;
        color: #6b7280;
        transition: background 0.2s;
    }}
    .downstream-wiring .expand-btn:hover {{
        background: #f3f4f6;
    }}
    .downstream-wiring .wiring-caption {{
        font-size: 12px;
        color: #6b7280;
        margin-top: 12px;
        line-height: 1.5;
        border-left: 3px solid #e5e7eb;
        padding-left: 12px;
    }}
    </style>
    <script>
    if (typeof toggleWiringRows === 'undefined') {{
        function toggleWiringRows(tableId, type) {{
            const container = document.getElementById(tableId);
            const table = container.querySelector(type === 'exc' ? '.exc-table' : '.inh-table');
            const btn = table.parentElement.querySelector('.expand-btn');
            const rows = table.querySelectorAll('.hidden-row');
            const isExpanded = btn.dataset.expanded === 'true';
            rows.forEach(row => {{
                row.style.display = isExpanded ? 'none' : 'table-row';
            }});
            const total = table.dataset.total;
            btn.innerHTML = isExpanded ? `Show all ${{total}} ▼` : `Show top 3 ▲`;
            btn.dataset.expanded = isExpanded ? 'false' : 'true';
        }}
    }}
    </script>
    '''


# =============================================================================
# UPSTREAM STEERING TABLE
# =============================================================================

def generate_upstream_steering_table(
    steering_data: dict[str, Any],
    neuron_labels: dict[str, str] = None,
    title: str = "Upstream Steering Response",
    caption: str = "",
) -> str:
    """Generate table showing how steering upstream neurons affects the target.

    Shows steering slope, R², effect direction, and RelP sign agreement for each
    upstream neuron. This complements the ablation dependency table by revealing
    causal influence strength.

    Args:
        steering_data: From upstream_steering_results[0], contains:
            - upstream_results: dict {neuron_id -> {slope, r_squared, effect_direction}}
            - relp_comparison: dict {neuron_id -> {relp_weight, relp_sign, signs_match}}
            - relp_sign_agreement: str like "8/10 (80%)"
        neuron_labels: Optional dict mapping neuron_id to label
        title: Figure title
        caption: Optional caption

    Returns:
        HTML string for the steering table
    """
    if not steering_data:
        return ""

    upstream_results = steering_data.get("upstream_results", {})
    relp_comparison = steering_data.get("relp_comparison", {})
    sign_agreement = steering_data.get("relp_sign_agreement", "N/A")
    total_prompts = steering_data.get("total_prompts", 0)
    neuron_labels = neuron_labels or {}

    if not upstream_results:
        return ""

    # Build rows sorted by |slope| descending
    rows_html = ""
    sorted_neurons = sorted(upstream_results.items(), key=lambda x: abs(x[1].get("slope", 0)), reverse=True)

    for i, (neuron_id, info) in enumerate(sorted_neurons):
        slope = info.get("slope", 0)
        r2 = info.get("r_squared", 0)
        direction = info.get("effect_direction", "unknown")

        # RelP comparison
        relp = relp_comparison.get(neuron_id, {})
        signs_match = relp.get("signs_match", None)
        relp_weight = relp.get("relp_weight", 0)

        # Color slope by direction and magnitude
        abs_slope = abs(slope)
        if abs_slope > 100:
            intensity = 0.9
        elif abs_slope > 50:
            intensity = 0.7
        elif abs_slope > 10:
            intensity = 0.5
        else:
            intensity = 0.3

        if slope > 0:
            slope_color = f"rgba(34, 197, 94, {intensity})"  # Green for excitatory
            slope_bg = f"rgba(34, 197, 94, {intensity * 0.15})"
        else:
            slope_color = f"rgba(239, 68, 68, {intensity})"  # Red for inhibitory
            slope_bg = f"rgba(239, 68, 68, {intensity * 0.15})"

        # R² quality indicator
        if r2 > 0.5:
            r2_class = "r2-good"
        elif r2 > 0.3:
            r2_class = "r2-ok"
        else:
            r2_class = "r2-weak"

        # Sign match indicator
        if signs_match is True:
            match_html = '<span class="sign-match" title="Steering direction matches wiring prediction">✓</span>'
        elif signs_match is False:
            match_html = '<span class="sign-mismatch" title="Steering direction contradicts wiring prediction">✗</span>'
        else:
            match_html = '<span class="sign-unknown">—</span>'

        label = neuron_labels.get(neuron_id, "")
        neuron_link = linkify_neuron_ids(neuron_id)
        label_html = f'<div class="neuron-label">{escape_html(label[:60])}</div>' if label else ""

        rows_html += f'''
        <tr>
            <td class="neuron-cell">
                {neuron_link}
                {label_html}
            </td>
            <td class="slope-cell" style="background: {slope_bg};">
                <span style="color: {slope_color}; font-weight: 600;">{slope:+.1f}</span>
            </td>
            <td class="r2-cell {r2_class}">{r2:.3f}</td>
            <td class="direction-cell">{direction}</td>
            <td class="match-cell">{match_html}</td>
        </tr>
        '''

    steering_values = steering_data.get("steering_values", [-10, -5, 5, 10])
    n_sv = len(steering_values) if isinstance(steering_values, list) else 4
    sv_text = ", ".join(str(v) for v in steering_values) if isinstance(steering_values, list) else "multiple values"
    default_caption = (
        f"Each upstream neuron was steered at {n_sv} values ({sv_text}) across {total_prompts} prompts. "
        f"<strong>Slope</strong> is the linear fit of target activation change vs steering value — "
        f"larger |slope| means stronger causal influence. "
        f"<strong>R²</strong> measures how linear the relationship is (>0.5 = good fit). "
        f"<strong>Wiring ✓</strong> indicates whether the observed steering direction matches the weight-predicted polarity."
    )
    caption_text = caption if caption else default_caption

    return f'''
    <div class="figure-container upstream-steering-table">
        <div class="figure-title">{escape_html(title)}</div>
        <p class="steering-summary">
            RelP sign agreement: <strong>{sign_agreement}</strong>
        </p>
        <table class="steer-table">
            <thead>
                <tr>
                    <th>Upstream Neuron</th>
                    <th>Slope</th>
                    <th>R²</th>
                    <th>Direction</th>
                    <th title="Does steering direction match wiring weight prediction?">Wiring ✓</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        <p class="wiring-caption">{caption_text}</p>
    </div>
    <style>
    .upstream-steering-table {{
        margin: 24px -60px;
        background: var(--bg-elevated, #ffffff);
        border-radius: 12px;
        padding: 24px 40px;
        border: 1px solid var(--border, #e0e0e0);
        max-width: calc(100% + 120px);
    }}
    .upstream-steering-table .steering-summary {{
        font-size: 14px;
        color: var(--text-secondary, #666);
        margin-bottom: 12px;
    }}
    .steer-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }}
    .steer-table th {{
        text-align: left;
        padding: 8px 10px;
        border-bottom: 2px solid #e5e7eb;
        font-size: 11px;
        text-transform: uppercase;
        color: #6b7280;
        letter-spacing: 0.05em;
    }}
    .steer-table td {{
        padding: 8px 10px;
        border-bottom: 1px solid #f3f4f6;
    }}
    .steer-table tbody tr:hover {{
        background: #f9fafb;
    }}
    .steer-table .neuron-cell {{
        min-width: 120px;
    }}
    .steer-table .neuron-label {{
        font-size: 11px;
        color: #6b7280;
        margin-top: 2px;
    }}
    .steer-table .slope-cell {{
        font-family: monospace;
        text-align: right;
        border-radius: 4px;
    }}
    .steer-table .r2-cell {{
        font-family: monospace;
        text-align: right;
    }}
    .steer-table .r2-good {{ color: #16a34a; }}
    .steer-table .r2-ok {{ color: #d97706; }}
    .steer-table .r2-weak {{ color: #9ca3af; }}
    .steer-table .direction-cell {{
        font-size: 12px;
        color: #4b5563;
    }}
    .steer-table .match-cell {{
        text-align: center;
        font-size: 16px;
    }}
    .steer-table .sign-match {{ color: #16a34a; }}
    .steer-table .sign-mismatch {{ color: #dc2626; }}
    .steer-table .sign-unknown {{ color: #9ca3af; }}
    </style>
    '''


# =============================================================================
# DOWNSTREAM STEERING SLOPE TABLE
# =============================================================================

def generate_downstream_steering_slope_table(
    steering_slopes: dict[str, Any],
    neuron_labels: dict[str, str] = None,
    wiring_weights: dict[str, float] = None,
    title: str = "Downstream Steering Response",
    caption: str = "",
    steering_values_tested: list = None,
    n_prompts: int = 0,
) -> str:
    """Generate table showing how steering the target neuron affects downstream neurons.

    Shows slope, R², and effect direction per downstream neuron — matching the upstream
    steering table format for consistency. Enables direct comparison of upstream causal
    influence (how upstream neurons affect target) with downstream propagation (how target
    affects downstream neurons).

    Args:
        steering_slopes: Dict mapping neuron_id -> {slope, r_squared, effect_direction, dose_response_curve}
        neuron_labels: Optional dict mapping neuron_id to label
        wiring_weights: Optional dict mapping neuron_id to wiring weight for agreement check
        title: Figure title
        caption: Optional caption
        steering_values_tested: List of steering values used
        n_prompts: Number of prompts tested

    Returns:
        HTML string for the downstream steering slope table
    """
    if not steering_slopes:
        return ""

    neuron_labels = neuron_labels or {}
    wiring_weights = wiring_weights or {}

    # Sort by |slope| descending
    sorted_neurons = sorted(
        steering_slopes.items(),
        key=lambda x: abs(x[1].get("slope", 0) or 0),
        reverse=True
    )

    rows_html = ""
    n_agree = 0
    n_wiring_total = 0

    for neuron_id, info in sorted_neurons:
        slope = info.get("slope")
        r2 = info.get("r_squared")
        direction = info.get("effect_direction", "unknown")

        if slope is None:
            # Single-value fallback — show mean_change_percent instead
            change = info.get("mean_change_percent", 0)
            slope_display = f'{change:+.1f}%'
            slope_color = "rgba(107, 114, 128, 0.7)"
            slope_bg = "transparent"
            r2_display = "—"
            r2_class = "r2-weak"
        else:
            abs_slope = abs(slope)
            if abs_slope > 10:
                intensity = 0.9
            elif abs_slope > 5:
                intensity = 0.7
            elif abs_slope > 1:
                intensity = 0.5
            else:
                intensity = 0.3

            if slope > 0:
                slope_color = f"rgba(34, 197, 94, {intensity})"
                slope_bg = f"rgba(34, 197, 94, {intensity * 0.15})"
            else:
                slope_color = f"rgba(239, 68, 68, {intensity})"
                slope_bg = f"rgba(239, 68, 68, {intensity * 0.15})"

            slope_display = f'{slope:+.2f}'
            r2_display = f'{r2:.3f}' if r2 is not None else "—"
            r2_class = "r2-good" if r2 and r2 > 0.5 else "r2-ok" if r2 and r2 > 0.3 else "r2-weak"

        # Wiring agreement check
        wiring_cell = ""
        if neuron_id in wiring_weights:
            wt = wiring_weights[neuron_id]
            n_wiring_total += 1
            # For downstream: positive wiring weight = excitatory (target excites downstream)
            # Steering target positively should increase downstream activation (positive slope)
            if slope is not None:
                agrees = (slope > 0 and wt > 0) or (slope < 0 and wt < 0) or (abs(slope) < 0.1 and abs(wt) < 0.01)
                if agrees:
                    n_agree += 1
                    wiring_cell = f'<td class="match-cell"><span class="sign-match" title="Wiring: {wt:+.3f}">✓</span></td>'
                else:
                    wiring_cell = f'<td class="match-cell"><span class="sign-mismatch" title="Wiring: {wt:+.3f}">✗</span></td>'
            else:
                wiring_cell = '<td class="match-cell"><span class="sign-unknown">—</span></td>'
        elif wiring_weights:
            wiring_cell = '<td class="match-cell"><span class="sign-unknown">—</span></td>'

        label = neuron_labels.get(neuron_id, "")
        neuron_link = linkify_neuron_ids(neuron_id)
        label_html = f'<div class="neuron-label">{escape_html(label[:60])}</div>' if label else ""

        rows_html += f'''
        <tr>
            <td class="neuron-cell">
                {neuron_link}
                {label_html}
            </td>
            <td class="slope-cell" style="background: {slope_bg};">
                <span style="color: {slope_color}; font-weight: 600;">{slope_display}</span>
            </td>
            <td class="r2-cell {r2_class}">{r2_display}</td>
            <td class="direction-cell">{direction}</td>
            {wiring_cell}
        </tr>
        '''

    # Summary
    wiring_agreement = f"{n_agree}/{n_wiring_total} ({n_agree*100//n_wiring_total}%)" if n_wiring_total > 0 else "N/A"
    sv_text = ", ".join(str(v) for v in steering_values_tested) if steering_values_tested else "multiple values"
    n_sv = len(steering_values_tested) if steering_values_tested else 0

    default_caption = (
        f"Each downstream neuron's activation was measured at {n_sv} steering values ({sv_text}) "
        f"across {n_prompts} prompts. "
        f"<strong>Slope</strong> is the linear fit of downstream activation change vs steering value — "
        f"larger |slope| means the target neuron has stronger causal influence on that downstream neuron. "
        f"<strong>R²</strong> measures how linear the relationship is."
    )
    caption_text = caption if caption else default_caption

    # Reuse same CSS classes as upstream steering table
    return f'''
    <div class="figure-container upstream-steering-table downstream-steering-slopes">
        <div class="figure-title">{escape_html(title)}</div>
        <p class="steering-summary">
            Wiring agreement: <strong>{wiring_agreement}</strong>
        </p>
        <table class="steer-table">
            <thead>
                <tr>
                    <th>Downstream Neuron</th>
                    <th>Slope</th>
                    <th>R²</th>
                    <th>Direction</th>
                    {"<th title='Does steering effect direction match wiring weight prediction?'>Wiring ✓</th>" if wiring_weights else ""}
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        <p class="wiring-caption">{caption_text}</p>
    </div>
    '''


# =============================================================================
# BOUNDARY TEST CARDS (Skeptic Results)
# =============================================================================

def generate_boundary_test_cards(
    boundary_tests: list[dict[str, Any]],
    title: str = "Boundary Tests",
    show_only_failures: bool = False,
    caption: str = ""
) -> str:
    """Generate cards showing skeptic boundary test results.

    Args:
        boundary_tests: From skeptic_report.boundary_tests, each contains:
            - description: what was tested
            - prompt: the test prompt
            - expected_behavior: what should have happened
            - actual_activation: what the neuron did
            - passed: bool
            - notes: explanation
        title: Figure title
        show_only_failures: If True, only show failed tests
        caption: Optional caption

    Returns:
        HTML string for boundary test cards
    """
    if not boundary_tests:
        return ""

    tests_to_show = boundary_tests
    if show_only_failures:
        tests_to_show = [t for t in boundary_tests if not t.get("passed", True)]

    if not tests_to_show:
        return ""

    cards_html = ""
    for test in tests_to_show[:8]:  # Max 8 cards
        passed = test.get("passed", False)
        description = test.get("description", "")[:80]
        prompt = test.get("prompt", "")[:60]
        expected = test.get("expected_behavior", "")[:80]
        actual = test.get("actual_activation", 0)
        notes = test.get("notes", "")[:120]

        status_class = "passed" if passed else "failed"
        status_icon = "✓" if passed else "✗"
        status_text = "PASSED" if passed else "FAILED"
        border_color = "#16a34a" if passed else "#dc2626"

        cards_html += f'''
        <div class="boundary-card {status_class}" style="border-left-color: {border_color};">
            <div class="card-header">
                <span class="status-icon">{status_icon}</span>
                <span class="status-text">{status_text}</span>
            </div>
            <div class="card-description">{escape_html(description)}</div>
            <div class="card-prompt">"{escape_html(prompt)}"</div>
            <div class="card-details">
                <div class="detail-row">
                    <span class="detail-label">Expected:</span>
                    <span class="detail-value">{escape_html(expected)}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Actual:</span>
                    <span class="detail-value activation">{actual:.2f}</span>
                </div>
            </div>
            {f'<div class="card-notes">{escape_html(notes)}</div>' if notes else ''}
        </div>
        '''

    # Summary stats
    total = len(boundary_tests)
    passed_count = sum(1 for t in boundary_tests if t.get("passed", False))
    failed_count = total - passed_count

    summary_html = f'''
    <div class="boundary-summary">
        <span class="summary-stat passed">{passed_count} passed</span>
        <span class="summary-stat failed">{failed_count} failed</span>
        <span class="summary-stat total">of {total} tests</span>
    </div>
    '''

    caption_html = f'<p class="figure-caption">{escape_html(caption)}</p>' if caption else ""

    return f'''
    <div class="figure-container boundary-tests">
        <div class="figure-title">{escape_html(title)}</div>
        {summary_html}
        <div class="boundary-cards-grid">
            {cards_html}
        </div>
        {caption_html}
    </div>
    <style>
    .boundary-tests .boundary-summary {{
        display: flex;
        gap: 16px;
        margin-bottom: 20px;
        font-size: 14px;
    }}
    .boundary-tests .summary-stat {{
        padding: 4px 12px;
        border-radius: 100px;
        font-weight: 500;
    }}
    .boundary-tests .summary-stat.passed {{
        background: rgba(22, 163, 74, 0.1);
        color: #16a34a;
    }}
    .boundary-tests .summary-stat.failed {{
        background: rgba(220, 38, 38, 0.1);
        color: #dc2626;
    }}
    .boundary-tests .summary-stat.total {{
        background: var(--bg-secondary, #f8f9fa);
        color: var(--text-secondary, #666);
    }}
    .boundary-cards-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
        gap: 16px;
    }}
    .boundary-card {{
        background: var(--bg-elevated, #fff);
        border: 1px solid var(--border, #e0e0e0);
        border-left-width: 4px;
        border-radius: 8px;
        padding: 16px;
    }}
    .boundary-card .card-header {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }}
    .boundary-card .status-icon {{
        font-size: 16px;
    }}
    .boundary-card.passed .status-icon {{ color: #16a34a; }}
    .boundary-card.failed .status-icon {{ color: #dc2626; }}
    .boundary-card .status-text {{
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .boundary-card.passed .status-text {{ color: #16a34a; }}
    .boundary-card.failed .status-text {{ color: #dc2626; }}
    .boundary-card .card-description {{
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 8px;
    }}
    .boundary-card .card-prompt {{
        font-family: 'SF Mono', Monaco, monospace;
        font-size: 12px;
        color: var(--text-secondary, #666);
        background: var(--bg-secondary, #f8f9fa);
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 12px;
    }}
    .boundary-card .card-details {{
        font-size: 13px;
    }}
    .boundary-card .detail-row {{
        display: flex;
        gap: 8px;
        margin-bottom: 4px;
    }}
    .boundary-card .detail-label {{
        color: var(--text-secondary, #666);
        min-width: 70px;
    }}
    .boundary-card .detail-value.activation {{
        font-family: 'SF Mono', Monaco, monospace;
        font-weight: 600;
    }}
    .boundary-card .card-notes {{
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid var(--border, #e0e0e0);
        font-size: 12px;
        color: var(--text-secondary, #666);
        font-style: italic;
    }}
    </style>
    '''


def generate_alternative_hypothesis_cards(
    alternatives: list[dict[str, Any]],
    title: str = "Alternative Hypotheses Tested",
    caption: str = ""
) -> str:
    """Generate cards showing skeptic alternative hypothesis results.

    Args:
        alternatives: From skeptic_report.alternative_hypotheses, each contains:
            - original_hypothesis: what was claimed
            - alternative: the alternative tested
            - test_description: how it was tested
            - verdict: supported/refuted/partial
            - evidence: what was found

    Returns:
        HTML string for alternative hypothesis cards
    """
    if not alternatives:
        return ""

    cards_html = ""
    for alt in alternatives[:6]:
        original = alt.get("original_hypothesis", "")[:80]
        alternative = alt.get("alternative", "")[:100]
        verdict = alt.get("verdict", "unknown")
        evidence = alt.get("evidence", "")[:200]

        # Verdict styling
        verdict_styles = {
            "alternative_supported": ("#f97316", "ALTERNATIVE SUPPORTED", "⚠"),
            "supported": ("#16a34a", "ORIGINAL SUPPORTED", "✓"),
            "refuted": ("#dc2626", "REFUTED", "✗"),
            "partial": ("#eab308", "PARTIAL", "~"),
        }
        color, text, icon = verdict_styles.get(verdict, ("#9ca3af", verdict.upper(), "?"))

        cards_html += f'''
        <div class="alt-hypo-card" style="border-left-color: {color};">
            <div class="card-header">
                <span class="verdict-icon" style="color: {color};">{icon}</span>
                <span class="verdict-text" style="color: {color};">{text}</span>
            </div>
            <div class="hypothesis-comparison">
                <div class="hypo-box original">
                    <div class="hypo-label">Original claim</div>
                    <div class="hypo-text">{escape_html(original)}</div>
                </div>
                <div class="hypo-arrow">→</div>
                <div class="hypo-box alternative">
                    <div class="hypo-label">Alternative tested</div>
                    <div class="hypo-text">{escape_html(alternative)}</div>
                </div>
            </div>
            <div class="card-evidence">{escape_html(evidence)}</div>
        </div>
        '''

    caption_html = f'<p class="figure-caption">{escape_html(caption)}</p>' if caption else ""

    return f'''
    <div class="figure-container alternative-hypotheses">
        <div class="figure-title">{escape_html(title)}</div>
        <div class="alt-hypo-cards">
            {cards_html}
        </div>
        {caption_html}
    </div>
    <style>
    .alt-hypo-cards {{
        display: flex;
        flex-direction: column;
        gap: 16px;
    }}
    .alt-hypo-card {{
        background: var(--bg-elevated, #fff);
        border: 1px solid var(--border, #e0e0e0);
        border-left-width: 4px;
        border-radius: 8px;
        padding: 16px;
    }}
    .alt-hypo-card .card-header {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
    }}
    .alt-hypo-card .verdict-icon {{
        font-size: 16px;
    }}
    .alt-hypo-card .verdict-text {{
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .hypothesis-comparison {{
        display: flex;
        align-items: stretch;
        gap: 12px;
        margin-bottom: 12px;
    }}
    .hypo-box {{
        flex: 1;
        padding: 12px;
        border-radius: 6px;
        background: var(--bg-secondary, #f8f9fa);
    }}
    .hypo-box.alternative {{
        background: rgba(249, 115, 22, 0.05);
    }}
    .hypo-label {{
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-secondary, #666);
        margin-bottom: 4px;
    }}
    .hypo-text {{
        font-size: 13px;
    }}
    .hypo-arrow {{
        display: flex;
        align-items: center;
        color: var(--text-secondary, #666);
        font-size: 18px;
    }}
    .card-evidence {{
        font-size: 13px;
        color: var(--text-secondary, #666);
        line-height: 1.5;
    }}
    </style>
    '''


# =============================================================================
# STEERING DOWNSTREAM EFFECTS (Circuit Propagation)
# =============================================================================

def generate_steering_downstream_table(
    steering_results: list[dict[str, Any]],
    title: str = "Steering Propagation",
    caption: str = ""
) -> str:
    """Generate table showing how steering this neuron affects downstream neurons.

    Args:
        steering_results: From multi_token_steering_results, each contains:
            - prompt: the test prompt
            - steering_value: the steering magnitude applied
            - downstream_effects: dict of {neuron_id: {baseline_activation, steered_activation, change_percent}}
        title: Figure title
        caption: Optional caption

    Returns:
        HTML string for the steering downstream effects table
    """
    if not steering_results:
        return ""

    # Collect all downstream effects across all steering experiments
    all_effects = []
    for result in steering_results:
        prompt = result.get("prompt", "")[:60]
        steering_value = result.get("steering_value", 0)
        downstream = result.get("downstream_effects", {})

        for neuron_id, effect in downstream.items():
            change_pct = effect.get("change_percent", 0)
            all_effects.append({
                "neuron_id": neuron_id,
                "prompt": prompt,
                "steering_value": steering_value,
                "baseline": effect.get("baseline_activation", 0),
                "steered": effect.get("steered_activation", 0),
                "change_percent": change_pct,
            })

    if not all_effects:
        return ""

    # Sort by absolute change
    all_effects.sort(key=lambda x: abs(x["change_percent"]), reverse=True)

    # Group by steering experiment (prompt + value)
    experiments = {}
    for eff in all_effects:
        key = (eff["prompt"], eff["steering_value"])
        if key not in experiments:
            experiments[key] = []
        experiments[key].append(eff)

    # Build HTML for each experiment
    experiments_html = ""
    for (prompt, steering_value), effects in experiments.items():
        # Take top effects (by absolute change)
        top_effects = sorted(effects, key=lambda x: abs(x["change_percent"]), reverse=True)[:6]

        rows_html = ""
        max_abs_change = max(abs(e["change_percent"]) for e in top_effects) if top_effects else 1

        for eff in top_effects:
            change = eff["change_percent"]
            baseline = eff["baseline"]
            steered = eff["steered"]

            # Color and bar based on magnitude and direction
            if abs(change) > 100:
                color = "#dc2626" if change < 0 else "#16a34a"
                weight = "700"
            elif abs(change) > 30:
                color = "#ea580c" if change < 0 else "#22c55e"
                weight = "600"
            else:
                color = "#6b7280"
                weight = "500"

            # Bar width (log scale for huge effects)
            import math
            bar_pct = min(100, (math.log10(abs(change) + 1) / math.log10(max_abs_change + 1)) * 100) if max_abs_change > 0 else 0
            bar_color = "#ef4444" if change < 0 else "#22c55e"

            # Format values
            baseline_str = f"{baseline:.3f}" if abs(baseline) < 10 else f"{baseline:.1f}"
            steered_str = f"{steered:.3f}" if abs(steered) < 10 else f"{steered:.1f}"

            # Arrow for direction
            arrow = "↓" if change < 0 else "↑"

            rows_html += f'''
            <tr>
                <td class="neuron-cell"><a href="{eff["neuron_id"].replace("/", "_")}.html" class="neuron-link">{eff["neuron_id"]}</a></td>
                <td class="value-cell">{baseline_str}</td>
                <td class="arrow-cell">→</td>
                <td class="value-cell">{steered_str}</td>
                <td class="change-cell" style="color: {color}; font-weight: {weight};">{arrow} {abs(change):.0f}%</td>
                <td class="bar-cell">
                    <div class="change-bar" style="width: {bar_pct}%; background: {bar_color};"></div>
                </td>
            </tr>
            '''

        sign = "+" if steering_value > 0 else ""
        experiments_html += f'''
        <div class="steering-experiment">
            <div class="experiment-header">
                <span class="steering-badge" style="background: {"rgba(34, 197, 94, 0.1); color: #16a34a" if steering_value > 0 else "rgba(239, 68, 68, 0.1); color: #dc2626"};">
                    Steering {sign}{steering_value}
                </span>
                <span class="experiment-prompt">{escape_html(prompt)}...</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Downstream Neuron</th>
                        <th>Baseline</th>
                        <th></th>
                        <th>Steered</th>
                        <th>Change</th>
                        <th></th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        '''

    caption_html = f'<div class="figure-caption">{escape_html(caption)}</div>' if caption else ""

    return f'''
    <div class="figure-container steering-downstream">
        <div class="figure-title">{escape_html(title)}</div>
        <div class="figure-subtitle">How steering this neuron affects connected downstream neurons</div>
        {experiments_html}
        {caption_html}
    </div>
    <style>
    .steering-downstream table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        margin-top: 12px;
    }}
    .steering-downstream th {{
        text-align: left;
        padding: 8px 12px;
        background: var(--bg-secondary, #f8f9fa);
        font-weight: 600;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-secondary, #666);
        border-bottom: 2px solid var(--border, #e0e0e0);
    }}
    .steering-downstream td {{
        padding: 10px 12px;
        border-bottom: 1px solid var(--border, #e0e0e0);
    }}
    .steering-downstream .neuron-cell {{
        font-family: 'SF Mono', Monaco, monospace;
        font-size: 12px;
    }}
    .steering-downstream .value-cell {{
        font-family: 'SF Mono', Monaco, monospace;
        font-size: 12px;
        color: var(--text-secondary, #666);
        text-align: right;
        width: 70px;
    }}
    .steering-downstream .arrow-cell {{
        text-align: center;
        color: var(--text-secondary, #999);
        width: 30px;
    }}
    .steering-downstream .change-cell {{
        font-family: 'SF Mono', Monaco, monospace;
        text-align: right;
        width: 80px;
    }}
    .steering-downstream .bar-cell {{
        width: 120px;
        padding-left: 8px;
    }}
    .steering-downstream .change-bar {{
        height: 8px;
        border-radius: 4px;
        min-width: 2px;
    }}
    .steering-experiment {{
        margin-bottom: 24px;
        padding: 16px;
        background: var(--bg-inset, #fafafa);
        border-radius: 8px;
        border: 1px solid var(--border, #e0e0e0);
    }}
    .steering-experiment:last-child {{
        margin-bottom: 0;
    }}
    .experiment-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
    }}
    .steering-badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 100px;
        font-size: 12px;
        font-weight: 600;
        font-family: 'SF Mono', Monaco, monospace;
    }}
    .experiment-prompt {{
        font-size: 13px;
        color: var(--text-secondary, #666);
        font-style: italic;
    }}
    .figure-subtitle {{
        font-size: 13px;
        color: var(--text-secondary, #666);
        margin: -8px 0 16px 0;
    }}
    </style>
    '''


def generate_custom_visualization(
    title: str,
    html_content: str,
    caption: str = "",
    css: str = "",
    description: str = "",
) -> str:
    """Generate a custom visualization with arbitrary HTML/CSS.

    This tool gives the agent creative freedom to create novel visualizations
    that aren't covered by the pre-built figure types. Use this for:
    - Novel chart types (sankey diagrams, chord diagrams, etc.)
    - Interactive visualizations with custom JS
    - Data-driven layouts not available in other tools
    - Experimental visualization concepts

    Args:
        title: The visualization title
        html_content: The HTML content for the visualization body.
            This should be well-structured HTML. You can use:
            - SVG for custom graphics
            - CSS Grid/Flexbox for layouts
            - Inline styles or classes (prefix custom classes with 'cv-')
            - Basic JavaScript for interactivity (wrap in <script> tags)
        caption: Optional caption explaining the visualization
        css: Optional CSS styles (will be scoped to .custom-viz-{hash})
        description: Internal description of what this visualization shows (for debugging)

    Returns:
        Standalone HTML string for the visualization

    Example:
        generate_custom_visualization(
            title="RelP vs Steering Agreement",
            html_content='''
                <div class="cv-comparison">
                    <div class="cv-neuron cv-agree">L5/N5772: ✓ Both excitatory</div>
                    <div class="cv-neuron cv-disagree">L3/N305: ✗ RelP+ but Steering-</div>
                </div>
            ''',
            css='''
                .cv-comparison { display: flex; flex-direction: column; gap: 8px; }
                .cv-neuron { padding: 12px; border-radius: 8px; }
                .cv-agree { background: #dcfce7; border-left: 4px solid #22c55e; }
                .cv-disagree { background: #fef2f2; border-left: 4px solid #ef4444; }
            ''',
            caption="Comparing RelP edge weights with causal steering effects"
        )
    """
    import hashlib

    # Generate a unique scope ID for CSS
    scope_id = hashlib.md5(f"{title}{html_content[:100]}".encode()).hexdigest()[:8]
    scope_class = f"custom-viz-{scope_id}"

    # Process CSS to scope it
    scoped_css = ""
    if css:
        # Add scope class to all CSS selectors
        lines = css.strip().split('\n')
        for line in lines:
            # Skip empty lines
            if not line.strip():
                scoped_css += "\n"
                continue
            # If line contains a selector (has {), prefix it
            if '{' in line and not line.strip().startswith('@'):
                # Handle multiple selectors separated by comma
                parts = line.split('{')
                selectors = parts[0].split(',')
                scoped_selectors = []
                for sel in selectors:
                    sel = sel.strip()
                    if sel:
                        # Prefix with scope class
                        scoped_selectors.append(f".{scope_class} {sel}")
                scoped_css += ', '.join(scoped_selectors) + ' {' + '{'.join(parts[1:]) + '\n'
            else:
                scoped_css += line + '\n'

    caption_html = f'<div class="figure-caption">{escape_html(caption)}</div>' if caption else ""

    return f'''
    <div class="figure-container {scope_class}">
        <div class="figure-title">{escape_html(title)}</div>
        <div class="custom-viz-content">
            {html_content}
        </div>
        {caption_html}
    </div>
    <style>
    .{scope_class} .custom-viz-content {{
        padding: 16px 0;
    }}
    {scoped_css}
    </style>
    '''
