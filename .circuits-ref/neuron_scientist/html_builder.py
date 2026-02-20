"""HTML builder for neuron dashboard reports.

Provides fixed section generators and page assembly utilities.
Refactored from html_template.py with modular architecture.
"""

import re
from typing import Any

from .figure_tools import (
    FIGURE_CSS,
    escape_html,
    escape_html_preserve_tags,
    generate_output_projections,
    linkify_neuron_ids,
)


def convert_markdown_to_html(text: str) -> str:
    """Convert markdown to proper HTML.

    Handles:
    - Headings (## -> h3, ### -> h4)
    - Paragraphs (double newlines)
    - Bold (**text**)
    - Italics (*text* or _text_)
    - Bullet lists (-, *, •)
    - Numbered lists (1. 2. etc.)

    Preserves existing HTML tags.
    """
    if not text:
        return text

    # First pass: convert inline formatting
    # Bold: **text** -> <strong>text</strong>
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    # Italics: *text* -> <em>text</em> (but not ** which is bold)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', text)
    # Italics: _text_ -> <em>text</em>
    # BUT: Don't match underscores in URLs/filenames (preceded by / or followed by .)
    # This prevents L17_N12426.html from being corrupted
    text = re.sub(r'(?<![/\w])_([^_<>]+)_(?![.\w])', r'<em>\1</em>', text)

    lines = text.split('\n')
    result = []
    in_list = False
    list_type = None
    paragraph_buffer = []

    def flush_paragraph():
        nonlocal paragraph_buffer
        if paragraph_buffer:
            para_text = ' '.join(paragraph_buffer).strip()
            if para_text:
                result.append(f'<p class="prose">{para_text}</p>')
            paragraph_buffer = []

    for line in lines:
        stripped = line.strip()

        # Headings
        h2_match = re.match(r'^##\s+(.+)$', stripped)
        h3_match = re.match(r'^###\s+(.+)$', stripped)

        if h2_match:
            flush_paragraph()
            if in_list:
                result.append(f'</{list_type}>')
                in_list = False
            result.append(f'<h3 class="prose-heading">{h2_match.group(1)}</h3>')
            continue
        if h3_match:
            flush_paragraph()
            if in_list:
                result.append(f'</{list_type}>')
                in_list = False
            result.append(f'<h4 class="prose-subheading">{h3_match.group(1)}</h4>')
            continue

        # Bullet points
        bullet_match = re.match(r'^[•\-\*]\s+(.+)$', stripped)
        number_match = re.match(r'^(\d+)\.\s+(.+)$', stripped)

        if bullet_match:
            flush_paragraph()
            if not in_list or list_type != 'ul':
                if in_list:
                    result.append(f'</{list_type}>')
                result.append('<ul class="prose-list">')
                in_list = True
                list_type = 'ul'
            result.append(f'<li>{bullet_match.group(1)}</li>')
        elif number_match:
            flush_paragraph()
            if not in_list or list_type != 'ol':
                if in_list:
                    result.append(f'</{list_type}>')
                result.append('<ol class="prose-list">')
                in_list = True
                list_type = 'ol'
            result.append(f'<li>{number_match.group(2)}</li>')
        elif not stripped:
            # Empty line - end of paragraph
            if in_list:
                result.append(f'</{list_type}>')
                in_list = False
                list_type = None
            flush_paragraph()
        else:
            # Regular text - add to paragraph buffer
            if in_list:
                result.append(f'</{list_type}>')
                in_list = False
                list_type = None
            paragraph_buffer.append(stripped)

    # Flush any remaining content
    if in_list:
        result.append(f'</{list_type}>')
    flush_paragraph()

    return '\n'.join(result)


# Alias for backward compatibility
convert_bullets_to_html = convert_markdown_to_html


# =============================================================================
# BASE CSS STYLES
# =============================================================================

BASE_CSS = """
:root {
    --bg: #ffffff;
    --bg-elevated: #ffffff;
    --bg-inset: #f5f5f7;
    --border: #e5e5e5;
    --text: #111111;
    --text-secondary: #555555;
    --text-tertiary: #888888;
    --accent: #0066cc;
    --green: #22863a;
    --green-muted: #34d058;
    --red: #d73a49;
    --amber: #e36209;
    --purple: #8957e5;
    /* Variant 5 routing-specific colors */
    --routing-blue: #3b82f6;
    --upstream-gray: #6b7280;
    --downstream-purple: #8b5cf6;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 17px;
    background: var(--bg);
    color: var(--text);
    line-height: 1.65;
    -webkit-font-smoothing: antialiased;
}

.container {
    max-width: 720px;
    margin: 0 auto;
    padding: 16px 24px 60px;
}

.container.wide {
    max-width: 1200px;
}

/* Nav */
.nav {
    margin-bottom: 16px;
}

.nav a {
    font-size: 15px;
    color: var(--accent);
    text-decoration: none;
}

.nav a:hover {
    text-decoration: underline;
}

/* Header */
.header {
    text-align: center;
    margin-bottom: 20px;
}

.neuron-id-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    color: var(--text-tertiary);
    letter-spacing: 0.5px;
    margin-bottom: 16px;
}

.header h1 {
    font-size: 32px;
    font-weight: 600;
    letter-spacing: -0.5px;
    margin-bottom: 12px;
}

.confidence-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-top: 12px;
    padding: 6px 14px;
    background: rgba(52, 199, 89, 0.1);
    border-radius: 100px;
    font-size: 13px;
    font-weight: 500;
    color: #248a3d;
}

.confidence-badge.confidence-adjusted {
    background: rgba(230, 126, 34, 0.1);
    color: #d35400;
    border: 1px solid rgba(230, 126, 34, 0.3);
}

.confidence-badge.confidence-adjusted svg {
    stroke: #e67e22;
}

/* Story section */
.story-section {
    margin-bottom: 0;
}

.story-lead {
    font-size: 17px;
    line-height: 1.6;
    color: var(--text);
    margin-bottom: 8px;
}

.story-lead strong {
    color: var(--accent);
}

.story-body {
    font-size: 15px;
    line-height: 1.6;
    color: var(--text-secondary);
    margin: 0;
}

/* Executive Summary Box */
.executive-summary {
    background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
    border: 1px solid rgba(0, 102, 204, 0.15);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 20px 24px;
    margin: 24px 0 32px;
}

.executive-summary h4 {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--accent);
    margin-bottom: 12px;
}

.executive-summary p {
    font-size: 14px;
    line-height: 1.5;
    color: var(--text);
    margin: 8px 0;
}

.executive-summary p:first-of-type {
    margin-top: 0;
}

.executive-summary p:last-of-type {
    margin-bottom: 0;
}

.executive-summary strong {
    color: var(--text);
    font-weight: 600;
}

/* Full-width circuit wrapper - Distill style */
.circuit-wrapper {
    background: #fcfcfc;
    margin: 20px -9999px 40px;
    padding: 20px 9999px;
    border-top: 1px solid rgba(0, 0, 0, 0.1);
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
}

.circuit-section {
    max-width: 1600px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: 320px 1fr 320px;
    gap: 32px;
}

@media (max-width: 1200px) {
    .circuit-section {
        grid-template-columns: 1fr;
        max-width: 800px;
    }
}

.circuit-col {
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    border: 1px solid rgba(0,0,0,0.06);
}

.circuit-col-header {
    margin-bottom: 16px;
}

.circuit-col-title {
    display: block;
    font-size: 16px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 2px;
}

.circuit-col-sub {
    font-size: 12px;
    color: var(--text-tertiary);
}

.circuit-col.upstream .circuit-col-title {
    color: #6b7280;
}

.circuit-col.center .circuit-col-title {
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px;
}

.circuit-col.downstream .circuit-col-title {
    color: var(--purple);
}

/* Neuron list (upstream/downstream) */
.neuron-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.neuron-item {
}

.neuron-name {
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 4px;
}

.neuron-name.unknown {
    color: var(--text-secondary);
    font-style: italic;
    font-weight: 500;
}

.neuron-meta {
    display: flex;
    flex-direction: column;
    gap: 2px;
    font-size: 11px;
}

.neuron-id {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-secondary);
}

.neuron-conn {
    color: var(--text-tertiary);
}

.neuron-link {
    color: var(--accent);
    text-decoration: none;
    font-family: 'JetBrains Mono', monospace;
}

.neuron-link:hover {
    text-decoration: underline;
}

/* Center selectivity list */
.selectivity-list {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.selectivity-group {
}

.selectivity-header {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}

.selectivity-group.fires .selectivity-header {
    color: #16a34a;
}

.selectivity-group.ignores .selectivity-header {
    color: #9ca3af;
}

.selectivity-examples {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.selectivity-examples .ex {
    font-size: 14px;
    line-height: 1.7;
    color: var(--text);
}

.selectivity-group.ignores .ex {
    color: var(--text-tertiary);
}

.selectivity-examples .act {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-tertiary);
    margin-left: 4px;
}

.selectivity-group.fires .act {
    color: #16a34a;
}

.selectivity-examples mark {
    background: linear-gradient(to top, #fde68a 40%, transparent 40%);
    color: inherit;
}

/* Circuit Hero Section - Variant 5 Enhanced */
.circuit-hero {
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    margin: 12px -9999px;
    padding: 32px 9999px;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
}

.circuit-hero-inner {
    max-width: 1400px;
    margin: 0 auto;
}

.circuit-hero h2 {
    text-align: center;
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 8px;
    color: var(--text);
}

.circuit-hero-subtitle {
    text-align: center;
    font-size: 15px;
    color: var(--text-secondary);
    margin-bottom: 40px;
}

/* Three-Column Circuit Flow */
.circuit-flow {
    display: grid;
    grid-template-columns: 280px 1fr 280px;
    gap: 24px;
    align-items: start;
}

@media (max-width: 1000px) {
    .circuit-flow {
        grid-template-columns: 1fr;
        gap: 32px;
    }
}

.circuit-column {
    background: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border: 1px solid rgba(0,0,0,0.06);
}

.circuit-column-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 2px solid var(--border);
}

.circuit-column-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}

.upstream .circuit-column-icon {
    background: rgba(107, 114, 128, 0.1);
}

.center .circuit-column-icon {
    background: rgba(59, 130, 246, 0.1);
}

.downstream .circuit-column-icon {
    background: rgba(139, 92, 246, 0.1);
}

.circuit-column-title {
    font-size: 14px;
    font-weight: 600;
}

.upstream .circuit-column-title { color: var(--upstream-gray); }
.center .circuit-column-title { color: var(--routing-blue); }
.downstream .circuit-column-title { color: var(--downstream-purple); }

/* Enhanced Neuron Items */
.neuron-item {
    padding: 14px 16px;
    background: var(--bg-inset);
    border-radius: 8px;
    margin-bottom: 12px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.neuron-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.neuron-item:last-child {
    margin-bottom: 0;
}

.neuron-weight {
    color: var(--text-tertiary);
    font-family: 'JetBrains Mono', monospace;
}

.neuron-weight.positive { color: var(--green); }
.neuron-weight.negative { color: var(--red); }

/* Center Neuron Hero Card */
.center-neuron {
    background: linear-gradient(135deg, #dbeafe, #bfdbfe);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    position: relative;
}

.center-neuron-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 24px;
    font-weight: 600;
    color: #1d4ed8;
    margin-bottom: 12px;
}

.center-neuron-function {
    font-size: 16px;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 16px;
}

.der-indicator {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: white;
    border-radius: 100px;
    font-size: 13px;
    color: var(--text-secondary);
}

.der-value {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: var(--routing-blue);
}

/* Routing Explanation Box */
.routing-explanation {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 12px;
    padding: 24px;
    margin: 32px 0;
}

.routing-explanation h3 {
    font-size: 16px;
    font-weight: 600;
    color: #92400e;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.routing-explanation p {
    font-size: 15px;
    line-height: 1.6;
    color: #78350f;
}

.routing-explanation p + p {
    margin-top: 12px;
}

.routing-explanation code {
    font-family: 'JetBrains Mono', monospace;
    background: rgba(0,0,0,0.05);
    padding: 2px 6px;
    border-radius: 4px;
}

/* Downstream Amplification Section */
.amplification-section {
    margin: 48px 0;
}

.amplification-section h2 {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.amplification-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
}

.amplification-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
}

.amplification-card h4 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    color: var(--downstream-purple);
    margin-bottom: 4px;
}

.amplification-card .label {
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 12px;
}

.amplification-bar {
    height: 24px;
    background: var(--bg-inset);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}

.amplification-fill {
    height: 100%;
    background: linear-gradient(90deg, #c4b5fd, #8b5cf6);
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 8px;
    min-width: 60px;
}

.amplification-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    color: white;
}

.amplification-note {
    font-size: 12px;
    color: var(--text-tertiary);
    margin-top: 8px;
}

/* Downstream Puzzle Section */
.puzzle-section {
    background: linear-gradient(135deg, #fdf4ff 0%, #fae8ff 100%);
    border: 1px solid #e879f9;
    border-radius: 12px;
    padding: 24px;
    margin: 32px 0;
}

.puzzle-section h3 {
    font-size: 18px;
    font-weight: 600;
    color: #86198f;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.puzzle-section p {
    font-size: 15px;
    line-height: 1.7;
    color: #701a75;
    margin-bottom: 12px;
}

.puzzle-section ul {
    margin-left: 20px;
    margin-top: 12px;
}

.puzzle-section li {
    font-size: 14px;
    color: #86198f;
    margin-bottom: 8px;
    line-height: 1.5;
}

.puzzle-section code {
    font-family: 'JetBrains Mono', monospace;
    background: rgba(0,0,0,0.05);
    padding: 2px 6px;
    border-radius: 4px;
}

/* Signal Propagation Section */
.signal-section {
    margin: 48px 0;
}

.signal-section h2 {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 8px;
}

.signal-section .section-subtitle {
    font-size: 15px;
    color: var(--text-secondary);
    margin-bottom: 24px;
}

.signal-scenario {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}

.signal-scenario-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 16px;
}

.signal-trigger {
    background: linear-gradient(135deg, #dbeafe, #bfdbfe);
    padding: 12px 16px;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    color: #1d4ed8;
}

.signal-arrow {
    font-size: 20px;
    color: var(--text-tertiary);
}

.signal-activation {
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px;
    font-weight: 600;
    color: var(--green);
}

.signal-effects {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
}

.signal-effect {
    background: var(--bg-inset);
    padding: 12px 16px;
    border-radius: 8px;
}

.signal-effect-target {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--downstream-purple);
    margin-bottom: 4px;
}

.signal-effect-change {
    font-size: 20px;
    font-weight: 600;
}

.signal-effect-change.increase { color: var(--green); }
.signal-effect-change.decrease { color: var(--red); }

/* Function Type Badge */
.function-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: linear-gradient(135deg, #dbeafe, #bfdbfe);
    border-radius: 100px;
    font-size: 14px;
    font-weight: 600;
    color: #1d4ed8;
    margin-top: 8px;
}

.function-badge svg {
    width: 18px;
    height: 18px;
}

/* Key Metrics Row */
.metrics-row {
    display: flex;
    justify-content: center;
    gap: 24px;
    margin: 24px 0;
    flex-wrap: wrap;
}

.metric-card {
    text-align: center;
    padding: 16px 24px;
    background: var(--bg-inset);
    border-radius: 12px;
    min-width: 140px;
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    color: var(--text);
}

.metric-value.routing { color: var(--routing-blue); }
.metric-value.selectivity { color: var(--green); }

.metric-label {
    font-size: 12px;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

/* Section */
.section {
    margin-bottom: 40px;
}

.section-label {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 24px;
}

/* Main Report Sections */
.main-section {
    margin-bottom: 48px;
    padding-top: 32px;
    border-top: 1px solid var(--border);
}

.main-section:first-of-type {
    border-top: none;
    padding-top: 0;
}

.main-section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
}

.main-section-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}

.main-section-icon.input { background: rgba(34, 197, 94, 0.1); }
.main-section-icon.output { background: rgba(99, 102, 241, 0.1); }
.main-section-icon.hypothesis { background: rgba(249, 115, 22, 0.1); }
.main-section-icon.questions { background: rgba(168, 85, 247, 0.1); }

.main-section-title {
    font-size: 20px;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.3px;
}

.main-section-content {
    margin-left: 44px;
}

@media (max-width: 768px) {
    .main-section-content {
        margin-left: 0;
    }
}

/* Collapsible Sections */
.collapsible {
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 16px;
    overflow: hidden;
}

.collapsible-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 18px;
    background: var(--bg-inset);
    cursor: pointer;
    user-select: none;
    transition: background 0.15s ease;
}

.collapsible-header:hover {
    background: #eef0f2;
}

.collapsible-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
}

.collapsible-toggle {
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    background: white;
    border: 1px solid var(--border);
    font-size: 12px;
    color: var(--text-secondary);
    transition: transform 0.2s ease;
}

.collapsible.expanded .collapsible-toggle {
    transform: rotate(180deg);
}

.collapsible-content {
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.3s ease;
}

.collapsible.expanded .collapsible-content {
    max-height: 2000px;
}

.collapsible-body {
    padding: 18px;
}

/* Evidence section (freeform) */
.evidence-section {
    margin-bottom: 40px;
}

.evidence-section h3 {
    font-size: 18px;
    font-weight: 600;
    color: var(--text);
    margin: 32px 0 16px;
}

.evidence-section h3:first-child {
    margin-top: 0;
}

.evidence-section .prose {
    font-size: 16px;
    line-height: 1.7;
    color: var(--text-secondary);
    margin-bottom: 20px;
}

.evidence-section .prose strong {
    color: var(--text);
}

.evidence-section .prose-list,
.prose-list {
    margin: 16px 0;
    padding-left: 24px;
    font-size: 16px;
    line-height: 1.7;
    color: var(--text-secondary);
}

.prose-list li {
    margin-bottom: 12px;
    padding-left: 8px;
}

.prose-list li strong {
    color: var(--text);
}

.evidence-section .prose em {
    font-style: italic;
}

/* Prose headings from markdown */
.prose-heading {
    font-size: 20px;
    font-weight: 600;
    color: var(--text);
    margin: 28px 0 12px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
}

.prose-subheading {
    font-size: 17px;
    font-weight: 600;
    color: var(--text);
    margin: 20px 0 10px;
}

.evidence-section .prose-heading:first-child,
.evidence-section .prose-subheading:first-child {
    margin-top: 0;
}

/* Open questions */
.questions-card {
    background: var(--bg-elevated);
    border-radius: 12px;
    padding: 40px;
    border: 1px solid var(--border);
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}

/* Z-score explanation */
.zscore-explanation {
    background: var(--bg-inset);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 16px 0;
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.5;
    border-left: 3px solid var(--accent);
}

.zscore-explanation strong {
    color: var(--text);
}

.question-item {
    padding: 16px 0;
    border-bottom: 1px solid var(--border);
    font-size: 16px;
    color: var(--text-secondary);
}

.question-item:last-child {
    border-bottom: none;
}
"""


# =============================================================================
# FIXED SECTION GENERATORS
# =============================================================================

def render_header(
    neuron_id: str,
    title: str,
    confidence: float = None,
    total_experiments: int = 0,
    confidence_downgraded: bool = False,
    pre_skeptic_confidence: float = None,
    skeptic_adjustment: float = None,
    hypothesis_count: int = 0,
) -> str:
    """Render the fixed header section.

    Args:
        confidence: If None, no confidence badge is shown (minimal header)
        confidence_downgraded: If True, shows warning that confidence was auto-adjusted
        pre_skeptic_confidence: Original confidence before skeptic adjustment
        skeptic_adjustment: Delta applied by skeptic (-0.1 for WEAKENED, -0.3 for REFUTED)
        hypothesis_count: Number of hypotheses (if >0, label shows "avg hypothesis confidence")
    """
    try:
        layer, neuron = neuron_id.replace("L", "").split("/N")
        display_id = f"L{layer} / N{neuron}"
    except ValueError:
        display_id = neuron_id

    # If confidence is None, render minimal header without metrics
    if confidence is None:
        return f'''
    <header class="header">
        <div class="neuron-id-header">{display_id}</div>
        <h1>{escape_html(title)}</h1>
    </header>
    '''

    # Use "avg hypothesis confidence" label if derived from hypotheses
    confidence_label = "avg hypothesis confidence" if hypothesis_count > 0 else "confidence"

    # Build confidence badge content
    if confidence_downgraded or (pre_skeptic_confidence is not None and pre_skeptic_confidence != confidence):
        # Show both original and adjusted confidence
        original = int((pre_skeptic_confidence or confidence) * 100)
        adjusted = int(confidence * 100)

        # Determine reason for adjustment
        if skeptic_adjustment and skeptic_adjustment != 0:
            reason = f"skeptic {'+' if skeptic_adjustment > 0 else ''}{int(skeptic_adjustment * 100)}%"
        else:
            reason = "evidence-based adjustment"

        confidence_html = f'''
        <div class="confidence-badge confidence-adjusted" title="Average of individual hypothesis posteriors, adjusted based on validation">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#e67e22" stroke-width="3">
                <path d="M12 9v4"/>
                <path d="M12 17h.01"/>
                <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
            </svg>
            <span style="text-decoration: line-through; opacity: 0.6;">{original}%</span>
            → {adjusted}% {confidence_label} ({reason}) &middot; {total_experiments} experiments
        </div>
        '''
    else:
        confidence_html = f'''
        <div class="confidence-badge" title="Average of individual hypothesis posterior probabilities">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
                <path d="M20 6L9 17l-5-5"/>
            </svg>
            {int(confidence * 100)}% {confidence_label} &middot; {total_experiments} experiments
        </div>
        '''

    return f'''
    <header class="header">
        <div class="neuron-id-header">{display_id}</div>
        <h1>{escape_html(title)}</h1>
        {confidence_html}
    </header>
    '''


def render_narrative(
    narrative_lead: str,
    narrative_body: str
) -> str:
    """Render the narrative section.

    Uses escape_html_preserve_tags to allow <strong>, <em>, <mark> formatting.
    """
    return f'''
    <div class="story-section">
        <p class="story-lead">{linkify_neuron_ids(escape_html_preserve_tags(narrative_lead))}</p>
        <p class="story-body">{linkify_neuron_ids(escape_html_preserve_tags(narrative_body))}</p>
    </div>
    '''


def render_executive_summary(executive_summary: str) -> str:
    """Render the executive summary section.

    Placed after the circuit block for quick reference.
    """
    if not executive_summary:
        return ""
    return f'''
    <div class="container">
        {executive_summary}
    </div>
    '''


def render_zscore_explanation(zscore_value: float = None, context: str = "activation") -> str:
    """Render an explanation of what z-score means in this context.

    Args:
        zscore_value: Optional specific z-score to reference
        context: Type of measurement - "activation", "selectivity", or "specificity"
    """
    context_text = {
        "activation": "how strongly this neuron responds to specific inputs compared to baseline",
        "selectivity": "how much more this neuron activates for target stimuli versus random inputs",
        "specificity": "how narrowly tuned this neuron is to particular patterns",
    }.get(context, "the neuron's response relative to baseline")

    value_text = ""
    if zscore_value is not None:
        if zscore_value > 10:
            value_text = f" A z-score of <strong>{zscore_value:.1f}</strong> indicates extremely strong and consistent activation—this response is highly unlikely to occur by chance."
        elif zscore_value > 3:
            value_text = f" A z-score of <strong>{zscore_value:.1f}</strong> indicates significant activation above baseline."
        elif zscore_value > 0:
            value_text = f" A z-score of <strong>{zscore_value:.1f}</strong> indicates mild activation above baseline."
        elif zscore_value < -1:
            value_text = f" A z-score of <strong>{zscore_value:.1f}</strong> indicates this neuron activates <em>less</em> than random baseline neurons on these inputs. This may indicate: (1) a 'dead' neuron that stopped contributing during training, (2) a neuron that only activates in very specific contexts not yet discovered, or (3) a routing neuron that influences downstream computation at near-zero activation."
        elif zscore_value < 0:
            value_text = f" A z-score of <strong>{zscore_value:.1f}</strong> indicates activation slightly below baseline—the neuron does not respond distinctively to the tested patterns."

    return f'''
    <div class="zscore-explanation">
        <strong>Understanding Z-Scores:</strong> The z-score measures {context_text}.
        It compares this neuron's activation to the mean and standard deviation of activations
        across random baseline neurons. Higher values indicate the neuron fires more
        distinctively for the tested pattern.{value_text}
    </div>
    '''


def render_stats_section(stats: list[dict[str, Any]], include_zscore_explanation: bool = False) -> str:
    """Render the investigation metrics section near the top.

    Args:
        stats: List of stat dicts with {value, label, highlight?}
        include_zscore_explanation: If True, adds z-score explanation when z-score stat is present
    """
    if not stats:
        return ""

    cards_html = ""
    zscore_value = None

    for stat in stats[:6]:
        value = stat.get("value", "")
        label = stat.get("label", "")
        highlight = stat.get("highlight", False)
        highlight_class = " highlight" if highlight else ""

        # Check if this is a z-score stat
        if "z-score" in label.lower() or "zscore" in label.lower():
            try:
                # Extract numeric value
                zscore_str = str(value).replace("×", "").replace("x", "").strip()
                zscore_value = float(zscore_str)
            except (ValueError, TypeError):
                pass

        cards_html += f'''
        <div class="stat-card{highlight_class}">
            <div class="stat-value">{escape_html(str(value))}</div>
            <div class="stat-label">{escape_html(label)}</div>
        </div>
        '''

    explanation_html = ""
    if include_zscore_explanation and zscore_value is not None:
        explanation_html = render_zscore_explanation(zscore_value, "selectivity")

    return f'''
    <div class="stats-section" style="margin: 24px 0 32px;">
        <div class="stats-row" style="justify-content: center;">
            {cards_html}
        </div>
        {explanation_html}
    </div>
    '''


def render_circuit_block(
    neuron_id: str,
    upstream_neurons: list[dict[str, Any]],
    downstream_neurons: list[dict[str, Any]],
    selectivity_fires: list[dict[str, Any]],
    selectivity_ignores: list[dict[str, Any]],
    function_description: str = "",
    title: str = ""
) -> str:
    """Render the three-column circuit block (Variant 5 style).

    Args:
        neuron_id: e.g., "L17/N12426"
        upstream_neurons: List of upstream neuron dicts with id, label, weight
        downstream_neurons: List of downstream neuron dicts with id, label, weight
        selectivity_fires: List of selectivity groups for activating examples
        selectivity_ignores: List of selectivity groups for non-activating examples
        function_description: Short description of neuron's function
        title: Optional title (defaults to "Information Flow Through {neuron_id}")
    """

    # Build upstream neurons HTML with enhanced styling
    upstream_html = ""
    for n in upstream_neurons[:4]:
        label = escape_html(str(n.get("label", "Unknown")))
        is_unknown = "unknown" in label.lower() or not label
        name_class = "neuron-name unknown" if is_unknown else "neuron-name"
        nid = n.get("neuron_id", n.get("id", ""))
        try:
            weight = float(n.get("weight", 0))
        except (ValueError, TypeError):
            weight = 0.0
        linked_id = linkify_neuron_ids(nid)
        weight_class = "neuron-weight positive" if weight > 0 else "neuron-weight negative" if weight < 0 else "neuron-weight"
        upstream_html += f'''
        <div class="neuron-item">
            <div class="{name_class}">{label}</div>
            <div class="neuron-meta">
                <a href="{nid.replace('/', '_')}.html" class="neuron-id">{nid}</a>
                <span class="{weight_class}">{weight:+.3f}</span>
            </div>
        </div>
        '''

    if not upstream_html:
        upstream_html = '''
        <div class="neuron-item">
            <div class="neuron-name unknown">Layer 0: connects to embeddings</div>
        </div>
        '''

    # Build downstream neurons HTML with enhanced styling
    downstream_html = ""
    for n in downstream_neurons[:4]:
        label = escape_html(str(n.get("label", "Unknown")))
        is_unknown = "unknown" in label.lower() or not label
        name_class = "neuron-name unknown" if is_unknown else "neuron-name"
        nid = n.get("neuron_id", n.get("id", ""))
        try:
            weight = float(n.get("weight", 0))
        except (ValueError, TypeError):
            weight = 0.0
        linked_id = linkify_neuron_ids(nid)
        weight_class = "neuron-weight positive" if weight > 0 else "neuron-weight negative" if weight < 0 else "neuron-weight"
        downstream_html += f'''
        <div class="neuron-item">
            <div class="{name_class}">{label}</div>
            <div class="neuron-meta">
                <a href="{nid.replace('/', '_')}.html" class="neuron-id">{nid}</a>
                <span class="{weight_class}">{weight:+.3f}</span>
            </div>
        </div>
        '''

    if not downstream_html:
        # Determine if this is actually the final layer (L31 for Llama 3.1-8B)
        try:
            source_layer = int(neuron_id.split("/")[0][1:]) if "/" in neuron_id else 0
        except (ValueError, IndexError):
            source_layer = 0

        if source_layer >= 31:
            # Actually final layer - projects directly to logits
            downstream_html = '''
        <div class="neuron-item">
            <div class="neuron-name unknown">Final layer: projects to logits</div>
        </div>
        '''
        else:
            # Not final layer - just no downstream connections identified
            downstream_html = f'''
        <div class="neuron-item">
            <div class="neuron-name unknown">No identified downstream neurons (layers {source_layer + 1}-31)</div>
        </div>
        '''

    # Use brief version for circuit panel (first sentence only, max 150 chars)
    function_html = ""
    if function_description:
        # Extract first sentence
        first_sentence = function_description.split('.')[0].strip()
        if len(first_sentence) > 150:
            first_sentence = first_sentence[:147] + "..."
        else:
            first_sentence += "."
        function_html = f'<div class="center-neuron-function">{escape_html(first_sentence)}</div>'

    # Build selectivity fires HTML for center panel
    fires_html = ""
    fires_examples = []

    # Tokens to skip (chat template tokens)
    skip_tokens = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|begin_of_text|>", "system"]

    for group in selectivity_fires[:3]:
        # Handle case where group is a string or malformed
        if not isinstance(group, dict):
            continue
        examples = group.get("examples", [])
        for ex in examples[:4]:
            # Handle case where example is not a dict
            if not isinstance(ex, dict):
                continue
            text = ex.get("text", "")
            token = ex.get("token", "")
            activation = ex.get("activation", 0)
            fires_after = ex.get("fires_after", "")

            if text:
                highlighted_text = None

                # Skip template tokens - no highlighting possible
                if token not in skip_tokens:
                    # Strategy 1: Direct token search (handle leading spaces)
                    clean_token = token.strip() if token else ""
                    if clean_token and clean_token in text:
                        idx = text.find(clean_token)
                        before = text[:idx]
                        after = text[idx + len(clean_token):]
                        if len(before) > 35:
                            before = "..." + before[-32:]
                        if len(after) > 25:
                            after = after[:22] + "..."
                        highlighted_text = f'{escape_html(before)}<mark style="background: #bbf7d0; padding: 1px 3px; border-radius: 3px;">{escape_html(clean_token)}</mark>{escape_html(after)}'

                    # Strategy 2: Use fires_after to find the highlight position
                    elif fires_after:
                        clean_fires = fires_after
                        for skip in skip_tokens:
                            clean_fires = clean_fires.replace(skip, "")
                        if clean_fires:
                            for suffix_len in range(min(len(clean_fires), 20), 2, -1):
                                suffix = clean_fires[-suffix_len:]
                                idx = text.find(suffix)
                                if idx >= 0:
                                    end_idx = idx + len(suffix)
                                    highlight_len = min(len(token) + 2, 6) if token else 4
                                    start_highlight = max(idx, end_idx - highlight_len)
                                    before = text[:start_highlight]
                                    highlight = text[start_highlight:end_idx]
                                    after = text[end_idx:]
                                    if len(before) > 35:
                                        before = "..." + before[-32:]
                                    if len(after) > 25:
                                        after = after[:22] + "..."
                                    highlighted_text = f'{escape_html(before)}<mark style="background: #bbf7d0; padding: 1px 3px; border-radius: 3px;">{escape_html(highlight)}</mark>{escape_html(after)}'
                                    break

                # Fallback: just show truncated text (no highlighting)
                if highlighted_text is None:
                    highlighted_text = escape_html(text[:70] + "..." if len(text) > 70 else text)

                fires_examples.append((highlighted_text, activation))

    if fires_examples:
        examples_html = "".join(
            f'<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; padding: 8px 12px; background: white; border-radius: 6px; border-left: 3px solid var(--green);">'
            f'<span style="flex: 1; font-size: 13px; line-height: 1.5;">{text}</span>'
            f'<span style="margin-left: 12px; font-family: \'JetBrains Mono\', monospace; font-size: 12px; color: var(--green); font-weight: 600; white-space: nowrap;">{activation:.2f}</span>'
            f'</div>'
            for text, activation in fires_examples[:4]
        )
        fires_html = f'''
        <div style="margin-top: 24px; padding: 16px; background: var(--bg-inset); border-radius: 8px;">
            <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--green); margin-bottom: 12px;">Fires On</div>
            {examples_html}
        </div>
        '''

    # Silent On section removed - only showing Fires On
    ignores_html = ""

    # Default title
    if not title:
        title = f"Information Flow Through {neuron_id}"

    return f'''
    <div class="circuit-hero">
        <div class="circuit-hero-inner">
            <h2>{escape_html(title)}</h2>
            <p class="circuit-hero-subtitle">Upstream sources feed signals; downstream targets receive processed output</p>

            <div class="circuit-flow">
                <!-- Upstream Column -->
                <div class="circuit-column upstream">
                    <div class="circuit-column-header">
                        <div class="circuit-column-icon">&#8593;</div>
                        <span class="circuit-column-title">Upstream Sources</span>
                    </div>
                    {upstream_html}
                </div>

                <!-- Center Neuron -->
                <div class="circuit-column center">
                    <div class="circuit-column-header">
                        <div class="circuit-column-icon">&#9673;</div>
                        <span class="circuit-column-title">This Neuron</span>
                    </div>

                    <div class="center-neuron">
                        <div class="center-neuron-id">{neuron_id}</div>
                        {function_html}
                    </div>

                    {fires_html}
                    {ignores_html}
                </div>

                <!-- Downstream Column -->
                <div class="circuit-column downstream">
                    <div class="circuit-column-header">
                        <div class="circuit-column-icon">&#8595;</div>
                        <span class="circuit-column-title">Downstream Targets</span>
                    </div>
                    {downstream_html}
                </div>
            </div>
        </div>
    </div>
    '''


def render_output_projections_section(
    promote: list[dict[str, Any]],
    suppress: list[dict[str, Any]]
) -> str:
    """Render the output projections fixed section."""
    return f'''
    <div class="section">
        <div class="section-label">Output Projections</div>
        {generate_output_projections(promote, suppress)}
    </div>
    '''


def render_open_questions(questions: list[str]) -> str:
    """Render open questions section."""
    if not questions:
        return ""

    questions_html = ""
    for q in questions[:5]:
        questions_html += f'<div class="question-item">{linkify_neuron_ids(escape_html(q))}</div>'

    return f'''
    <div class="section">
        <div class="section-label">Open Questions</div>
        <div class="questions-card">{questions_html}</div>
    </div>
    '''


def render_routing_explanation(function_description: str = "") -> str:
    """Render the routing explanation box (deprecated - returns empty).

    Args:
        function_description: Optional description of the neuron's routing function (unused).

    Returns:
        Empty string (DER-based routing explanation has been removed).
    """
    return ""


def render_amplification_section(
    steering_downstream: list[dict[str, Any]],
    steering_magnitude: float = 10.0
) -> str:
    """Render downstream amplification cards showing steering propagation effects.

    Args:
        steering_downstream: List of steering results with downstream_effects.
            Each should have: neuron_id, label, baseline, steered, pct_change
        steering_magnitude: The steering magnitude used (for display).

    Returns:
        HTML string for the amplification section.
    """
    if not steering_downstream:
        return ""

    # Flatten and deduplicate downstream effects
    effects_map = {}  # neuron_id -> {label, baseline, steered, pct_change}
    for result in steering_downstream:
        # Handle case where result is a string or malformed
        if not isinstance(result, dict):
            continue
        downstream_effects = result.get("downstream_effects", [])
        if not isinstance(downstream_effects, list):
            continue
        for effect in downstream_effects:
            # Handle case where effect is a string or malformed
            if not isinstance(effect, dict):
                continue
            nid = effect.get("neuron_id", "")
            if nid and nid not in effects_map:
                pct = effect.get("pct_change", 0)
                # Only show significant amplifications (>100%)
                if abs(pct) > 100:
                    effects_map[nid] = effect

    if not effects_map:
        return ""

    # Sort by absolute pct_change descending
    sorted_effects = sorted(effects_map.values(), key=lambda x: abs(x.get("pct_change", 0)), reverse=True)[:6]

    if not sorted_effects:
        return ""

    # Find max pct for bar scaling
    max_pct = max(abs(e.get("pct_change", 0)) for e in sorted_effects)

    cards_html = ""
    for effect in sorted_effects:
        nid = effect.get("neuron_id", "")
        label = escape_html(effect.get("label", "Downstream neuron"))
        baseline = effect.get("baseline", 0)
        steered = effect.get("steered", 0)
        pct = effect.get("pct_change", 0)

        # Calculate bar width (relative to max)
        bar_width = min(100, abs(pct) / max_pct * 100) if max_pct > 0 else 50

        # Link the neuron ID
        linked_id = linkify_neuron_ids(nid)

        cards_html += f'''
        <div class="amplification-card">
            <h4>{linked_id}</h4>
            <div class="label">{label}</div>
            <div class="amplification-bar">
                <div class="amplification-fill" style="width: {bar_width:.0f}%;">
                    <span class="amplification-value">{pct:+.0f}%</span>
                </div>
            </div>
            <div class="amplification-note">Baseline: {baseline:.2f} &rarr; Steered: {steered:.2f}</div>
        </div>
        '''

    return f'''
    <div class="amplification-section">
        <h2>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M13 17l5-5-5-5"/>
                <path d="M6 17l5-5-5-5"/>
            </svg>
            Downstream Amplification (Steering +{steering_magnitude:.0f})
        </h2>
        <p style="color: var(--text-secondary); margin-bottom: 24px;">
            When we artificially boost this neuron's activation, what happens downstream?
        </p>
        <div class="amplification-grid">
            {cards_html}
        </div>
    </div>
    '''


def render_signal_propagation(
    trigger_text: str,
    activation: float,
    downstream_effects: list[dict[str, Any]],
    direct_token_effect: dict[str, Any] | None = None
) -> str:
    """Render signal propagation example showing trigger -> activation -> effects.

    Args:
        trigger_text: The input text that triggers the neuron (e.g., "COX-2")
        activation: The activation value when triggered
        downstream_effects: List of {neuron_id, label, pct_change}
        direct_token_effect: Optional {token, logit_change} for direct projection effect

    Returns:
        HTML string for the signal propagation section.
    """
    if not trigger_text or not downstream_effects:
        return ""

    effects_html = ""
    for effect in downstream_effects[:3]:
        nid = effect.get("neuron_id", "")
        label = escape_html(effect.get("label", ""))[:30]
        pct = effect.get("pct_change", 0)
        change_class = "increase" if pct > 0 else "decrease"
        linked_id = linkify_neuron_ids(nid)

        effects_html += f'''
        <div class="signal-effect">
            <div class="signal-effect-target">{linked_id}</div>
            <div class="signal-effect-change {change_class}">{pct:+.0f}%</div>
            <div style="font-size: 12px; color: var(--text-tertiary);">{label}</div>
        </div>
        '''

    # Add direct token effect if provided
    if direct_token_effect:
        token = escape_html(direct_token_effect.get("token", ""))
        logit = direct_token_effect.get("logit_change", 0)
        change_class = "increase" if logit > 0 else "decrease"
        effects_html += f'''
        <div class="signal-effect">
            <div class="signal-effect-target">Direct Logits</div>
            <div class="signal-effect-change {change_class}">{logit:+.1f}</div>
            <div style="font-size: 12px; color: var(--text-tertiary);">"{token}" token promoted</div>
        </div>
        '''

    return f'''
    <div class="signal-section">
        <h2>Signal Propagation Example</h2>
        <p class="section-subtitle">What happens when "{escape_html(trigger_text)}" triggers a {activation:.1f} activation?</p>

        <div class="signal-scenario">
            <div class="signal-scenario-header">
                <div class="signal-trigger">"{escape_html(trigger_text)}"</div>
                <div class="signal-arrow">&rarr;</div>
                <div class="signal-activation">{activation:.1f}</div>
            </div>

            <p style="font-size: 14px; color: var(--text-secondary); margin-bottom: 16px;">
                At peak activation, downstream effects cascade:
            </p>

            <div class="signal-effects">
                {effects_html}
            </div>
        </div>
    </div>
    '''


def render_metrics_row(
    selectivity_zscore: float | None = None,
    peak_activation: float | None = None
) -> str:
    """Render key metrics row (selectivity, peak activation).

    Args:
        selectivity_zscore: Category selectivity z-score
        peak_activation: Maximum observed activation value

    Returns:
        HTML string for the metrics row.
    """
    if selectivity_zscore is None and peak_activation is None:
        return ""

    cards_html = ""

    if selectivity_zscore is not None:
        cards_html += f'''
        <div class="metric-card">
            <div class="metric-value selectivity">{selectivity_zscore:.2f}</div>
            <div class="metric-label">Selectivity Gap (z)</div>
        </div>
        '''

    if peak_activation is not None:
        cards_html += f'''
        <div class="metric-card">
            <div class="metric-value">{peak_activation:.1f}</div>
            <div class="metric-label">Peak Activation</div>
        </div>
        '''

    return f'''
    <div class="metrics-row">
        {cards_html}
    </div>
    '''


def render_function_badge(function_type: str) -> str:
    """Render function type badge (routing/projection).

    Args:
        function_type: Type of function (e.g., "routing", "projection", "semantic")

    Returns:
        HTML string for the function badge.
    """
    if not function_type:
        return ""

    # Select appropriate icon based on function type
    if "routing" in function_type.lower():
        icon = '''<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2L2 7l10 5 10-5-10-5z"/>
            <path d="M2 17l10 5 10-5"/>
            <path d="M2 12l10 5 10-5"/>
        </svg>'''
    elif "projection" in function_type.lower():
        icon = '''<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <path d="M8 12l4 4 4-4"/>
            <path d="M12 8v8"/>
        </svg>'''
    else:
        icon = '''<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
        </svg>'''

    return f'''
    <div class="function-badge">
        {icon}
        {escape_html(function_type)} Function
    </div>
    '''


def render_evidence_section(
    figures: list[dict[str, Any]],
    prose_sections: list[dict[str, Any]]
) -> str:
    """Render the freeform evidence section.

    Args:
        figures: List of {html, type} - figures indexed by generation order
        prose_sections: List of {heading, content} where content may contain
                       <FIGURE_X> placeholders (or legacy format without placeholders)

    Placeholder format: The agent can include <FIGURE_0>, <FIGURE_1>, etc. in prose
    to indicate where specific figures should be inserted. Alternatively, figures
    are inserted after each prose section by index (legacy behavior).
    """
    if not figures and not prose_sections:
        return ""

    # Build figure map by index
    figure_map = {i: fig.get("html", "") for i, fig in enumerate(figures)}

    # Track which figures have been used
    used_figures = set()

    content_html = ""

    for i, section in enumerate(prose_sections):
        heading = section.get("heading", "")
        content = section.get("content", "")

        if heading:
            content_html += f'<h3>{escape_html(heading)}</h3>'

        if content:
            # Replace <FIGURE_X> placeholders with actual figure HTML
            # Also handle variants like &lt;FIGURE_X&gt; from escaping
            def replace_figure_placeholder(match):
                fig_idx = int(match.group(1))
                if fig_idx in figure_map:
                    used_figures.add(fig_idx)
                    return f'</p>{figure_map[fig_idx]}<p class="prose">'
                return ""  # Remove placeholder if figure doesn't exist

            # Replace both raw and escaped versions of placeholders
            processed_content = re.sub(
                r'<FIGURE_(\d+)>|&lt;FIGURE_(\d+)&gt;',
                lambda m: replace_figure_placeholder(type('Match', (), {'group': lambda s, i: m.group(1) or m.group(2)})()),
                content
            )

            # Use escape_html_preserve_tags to allow <strong>, <em>, <mark> formatting
            # But preserve our </p>...<p> figure insertions
            if '<FIGURE_' in content or '&lt;FIGURE_' in content:
                # Content had placeholders - process carefully
                content_html += f'<p class="prose">{linkify_neuron_ids(processed_content)}</p>'
                # Clean up empty <p class="prose"></p> tags
                content_html = re.sub(r'<p class="prose">\s*</p>', '', content_html)
            else:
                # No placeholders - use standard escaping
                content_html += f'<p class="prose">{linkify_neuron_ids(escape_html_preserve_tags(content))}</p>'

                # Legacy behavior: insert figure after prose section by index
                if i in figure_map and i not in used_figures:
                    content_html += figure_map[i]
                    used_figures.add(i)

    # Add any remaining figures not used via placeholders or legacy indexing
    for i, fig in enumerate(figures):
        if i not in used_figures:
            content_html += fig.get("html", "")

    return f'''
    <div class="section">
        <div class="section-label">Evidence & Analysis</div>
        <div class="evidence-section">
            {content_html}
        </div>
    </div>
    '''


# =============================================================================
# MAIN SECTION RENDERERS (New 4-Section Structure)
# =============================================================================

def render_main_section(
    section_type: str,
    content_html: str,
    title: str | None = None
) -> str:
    """Render a main section with icon and header.

    Args:
        section_type: One of "input", "output", "hypothesis", "questions"
        content_html: The HTML content for the section body
        title: Optional custom title (defaults based on section_type)
    """
    icons = {
        "input": "📥",
        "output": "📤",
        "hypothesis": "🔬",
        "questions": "❓",
    }
    default_titles = {
        "input": "Input Function",
        "output": "Output Function",
        "hypothesis": "Hypothesis Testing",
        "questions": "Open Questions",
    }

    icon = icons.get(section_type, "📋")
    section_title = title or default_titles.get(section_type, "Section")

    return f'''
    <div class="main-section">
        <div class="main-section-header">
            <div class="main-section-icon {section_type}">{icon}</div>
            <div class="main-section-title">{escape_html(section_title)}</div>
        </div>
        <div class="main-section-content">
            {content_html}
        </div>
    </div>
    '''


def render_collapsible(
    title: str,
    content_html: str,
    expanded: bool = False
) -> str:
    """Render a collapsible section.

    Args:
        title: Header text for the collapsible
        content_html: The HTML content to show when expanded
        expanded: Whether to start expanded (default: collapsed)
    """
    expanded_class = " expanded" if expanded else ""

    return f'''
    <div class="collapsible{expanded_class}">
        <div class="collapsible-header" onclick="this.parentElement.classList.toggle('expanded')">
            <span class="collapsible-title">{escape_html(title)}</span>
            <span class="collapsible-toggle">▼</span>
        </div>
        <div class="collapsible-content">
            <div class="collapsible-body">
                {content_html}
            </div>
        </div>
    </div>
    '''


def render_input_stimuli_section(
    category_selectivity_html: str = "",
    activation_selectivity_html: str = "",
    upstream_dep_html: str = "",
    wiring_polarity_html: str = "",
    upstream_steering_html: str = "",
    prose_html: str = "",
    # Flexible prose slots for Part 1: Behavioral Triggers
    prose_before_selectivity: str = "",
    prose_after_selectivity: str = "",
    prose_after_other_figures: str = "",
    # Flexible prose slots for Part 2: Upstream Circuit Architecture
    prose_before_wiring: str = "",
    prose_after_wiring: str = "",
    prose_before_ablation: str = "",
    prose_after_ablation: str = "",
    prose_before_steering: str = "",
    prose_after_steering: str = "",
    prose_part2: str = "",  # General Part 2 prose (maps to prose_before_wiring)
    # Part 3: Negative Polarity (only when bipolar investigation available)
    prose_part3: str = "",  # Prose describing negative firing triggers
) -> str:
    """Render the Input Function section with two parts.

    PART 1: Behavioral Triggers
    1. prose_before_selectivity (optional intro)
    2. Category Selectivity Chart ← AUTO-GENERATED (title: "Category Selectivity")
    3. prose_after_selectivity (commentary on what activates the neuron)
    4. Other custom figures (activation_selectivity_html)
    5. prose_after_other_figures (commentary on custom figures)

    PART 2: Upstream Circuit Architecture
    6. prose_before_wiring / prose_part2 (intro to circuit architecture)
    7. Upstream Wiring Table ← AUTO-GENERATED (title: "Upstream Wiring (Weight-Based Polarity)")
    8. prose_after_wiring (commentary on wiring predictions)
    9. prose_before_ablation (optional)
    10. Upstream Ablation Table ← AUTO-GENERATED (title: "Upstream Ablation Effects")
    11. prose_after_ablation (commentary on ablation results)
    12. prose_before_steering (optional)
    13. Upstream Steering Table ← AUTO-GENERATED if data exists
    14. prose_after_steering (commentary on steering)

    PART 3: Negative Polarity Triggers (only when bipolar investigation available)
    15. prose_part3 (prose describing what triggers negative firing, from negative investigation)

    Args:
        category_selectivity_html: Category selectivity chart HTML
        activation_selectivity_html: Other activation selectivity figures
        upstream_dep_html: Upstream ablation dependency table
        wiring_polarity_html: Wiring polarity table (weight-based predictions)
        upstream_steering_html: Upstream steering table (if scientist ran steering)
        prose_html: Legacy general prose (backwards compatible, maps to prose_after_selectivity)
        prose_before_selectivity: Prose before category selectivity
        prose_after_selectivity: Prose after category selectivity
        prose_after_other_figures: Prose after other custom figures
        prose_before_wiring: Prose before wiring table
        prose_after_wiring: Prose after wiring table
        prose_before_ablation: Prose before ablation table
        prose_after_ablation: Prose after ablation table
        prose_before_steering: Prose before steering table
        prose_after_steering: Prose after steering table
        prose_part2: General Part 2 prose (maps to prose_before_wiring if not set)
    """

    def add_prose(prose: str) -> str:
        if prose:
            processed = convert_bullets_to_html(prose)
            return f'<div class="evidence-section">{processed}</div>'
        return ""

    content = ""

    # =========================================================================
    # PART 1: Behavioral Triggers
    # =========================================================================
    part1_content = ""

    # 1. Prose before selectivity (optional intro)
    part1_content += add_prose(prose_before_selectivity)

    # 2. Category selectivity chart (AUTO-GENERATED, title: "Category Selectivity")
    if category_selectivity_html:
        part1_content += category_selectivity_html

    # 3. Prose after selectivity (or fallback to legacy prose_html)
    if prose_after_selectivity:
        part1_content += add_prose(prose_after_selectivity)
    elif prose_html:
        part1_content += add_prose(prose_html)
        prose_html = ""  # Clear so it doesn't appear again

    # 4. Other custom figures
    if activation_selectivity_html:
        part1_content += activation_selectivity_html

    # 5. Prose after other figures
    part1_content += add_prose(prose_after_other_figures)

    if part1_content:
        content += f'''
        <div class="input-part behavioral-triggers">
            <h3 class="part-header">Behavioral Triggers</h3>
            {part1_content}
        </div>
        '''

    # =========================================================================
    # PART 2: Upstream Circuit Architecture
    # =========================================================================
    part2_content = ""

    # 6. Prose before wiring (or fallback to prose_part2)
    if prose_before_wiring:
        part2_content += add_prose(prose_before_wiring)
    elif prose_part2:
        part2_content += add_prose(prose_part2)

    # 7. Upstream wiring table (AUTO-GENERATED, title: "Upstream Wiring (Weight-Based Polarity)")
    if wiring_polarity_html:
        part2_content += wiring_polarity_html

    # 8. Prose after wiring
    part2_content += add_prose(prose_after_wiring)

    # 9. Prose before ablation (optional)
    part2_content += add_prose(prose_before_ablation)

    # 10. Upstream ablation table (AUTO-GENERATED, title: "Upstream Ablation Effects")
    if upstream_dep_html:
        part2_content += upstream_dep_html

    # 11. Prose after ablation
    part2_content += add_prose(prose_after_ablation)

    # 12. Prose before steering (optional)
    part2_content += add_prose(prose_before_steering)

    # 13. Upstream steering table (AUTO-GENERATED if data exists)
    if upstream_steering_html:
        part2_content += upstream_steering_html

    # 14. Prose after steering
    part2_content += add_prose(prose_after_steering)

    # Any remaining legacy prose
    part2_content += add_prose(prose_html)

    if part2_content:
        content += f'''
        <div class="input-part upstream-circuit-architecture">
            <h3 class="part-header">Upstream Circuit Architecture</h3>
            {part2_content}
        </div>
        '''

    # =========================================================================
    # PART 3: Negative Polarity Triggers (bipolar neurons only)
    # =========================================================================
    if prose_part3:
        part3_processed = convert_bullets_to_html(prose_part3)
        content += f'''
        <div class="input-part negative-polarity-triggers">
            <h3 class="part-header">Negative Firing Triggers</h3>
            <div class="evidence-section">{part3_processed}</div>
        </div>
        '''

    if not content:
        content = '<p class="prose">No input function data available.</p>'

    return render_main_section("input", content)


def render_output_function_section(
    output_projections_html: str = "",
    ablation_completions_html: str = "",
    steering_completions_html: str = "",
    downstream_wiring_html: str = "",
    downstream_ablation_effects_html: str = "",
    downstream_steering_slopes_html: str = "",
    # Flexible prose slots for Part 1: Direct Token Effects
    prose_before_projections: str = "",
    prose_after_projections: str = "",
    prose_before_ablation: str = "",
    prose_after_ablation: str = "",
    prose_before_steering: str = "",
    prose_after_steering: str = "",
    # Flexible prose slots for Part 2: Downstream Circuit Effects
    prose_before_downstream_wiring: str = "",
    prose_after_downstream_wiring: str = "",
    prose_before_downstream_ablation: str = "",
    prose_after_downstream_ablation: str = "",
    prose_before_downstream_steering: str = "",
    prose_after_downstream_steering: str = "",
    # Simplified prose slots (map to specific slots)
    prose_part1: str = "",  # Maps to prose_after_projections
    prose_part2: str = "",  # Maps to prose_before_downstream_wiring
    # Part 3: Negative Polarity (only when bipolar investigation available)
    prose_part3: str = "",  # Prose describing negative firing output effects
    # Legacy parameters (deprecated, kept for compatibility)
    ablation_table_html: str = "",
    steering_table_html: str = "",
    downstream_dep_html: str = "",
    prose_html: str = "",
) -> str:
    """Render the Output Function section with two parts (plus optional Part 3 for bipolar).

    PART 1: Direct Token Effects
    1. prose_before_projections (optional intro)
    2. Output Projections ← AUTO-GENERATED (title: "Output Projections")
    3. prose_after_projections / prose_part1 (commentary on projections)
    4. prose_before_ablation (optional)
    5. Ablation Completions ← AUTO-GENERATED (title: "Ablation Effects on Completions")
    6. prose_after_ablation (commentary on ablation)
    7. prose_before_steering (optional)
    8. Steering Completions ← AUTO-GENERATED (title: "Intelligent Steering Analysis")
    9. prose_after_steering (commentary on steering)

    PART 2: Downstream Circuit Effects
    10. prose_before_downstream_wiring / prose_part2 (intro to circuit effects)
    11. Downstream Wiring Table ← AUTO-GENERATED (title: "Downstream Wiring (Weight-Based Polarity)")
    12. prose_after_downstream_wiring (commentary on wiring)
    13. prose_before_downstream_ablation (optional)
    14. Downstream Ablation Effects ← AUTO-GENERATED (title: "Downstream Ablation Effects")
    15. prose_after_downstream_ablation (commentary on downstream ablation)
    16. prose_before_downstream_steering (optional)
    17. Downstream Steering Response ← AUTO-GENERATED (slope + R² per downstream neuron)
    18. prose_after_downstream_steering (commentary on downstream steering)

    Args:
        output_projections_html: Compact output projections display
        ablation_completions_html: Ablation changed completions (new card format)
        steering_completions_html: Steering changed completions (intelligent steering gallery)
        downstream_wiring_html: Downstream wiring polarity table (weight-based)
        downstream_ablation_effects_html: How ablating target affects downstream neuron activations
        downstream_steering_slopes_html: Causal slope + R² per downstream neuron (dose-response)
        prose_before_*/prose_after_*: Flexible prose slots before/after each figure
        prose_part1: Simplified slot for Part 1 prose (maps to prose_after_projections)
        prose_part2: Simplified slot for Part 2 prose (maps to prose_before_downstream_wiring)

        # Legacy (deprecated):
        ablation_table_html: Old ablation table format (use ablation_completions_html instead)
        steering_table_html: Old steering table format (use steering_completions_html instead)
        downstream_dep_html: Legacy downstream dependency (use downstream_ablation_effects_html)
        prose_html: Legacy overall prose (use specific prose slots instead)
    """

    def add_prose(prose: str) -> str:
        """Helper to add prose section if non-empty."""
        if not prose:
            return ""
        processed = convert_bullets_to_html(prose)
        # Use evidence-section class (no background) instead of evidence-prose (gray background)
        return f'<div class="evidence-section">{processed}</div>'

    content = ""

    # =========================================================================
    # PART 1: Direct Token Effects
    # =========================================================================
    part1_content = ""

    # 1. Prose before projections (optional intro)
    part1_content += add_prose(prose_before_projections)

    # 2. Output projections (AUTO-GENERATED, title: "Output Projections")
    if output_projections_html:
        part1_content += output_projections_html

    # 3. Prose after projections (or fallback to prose_part1, then legacy prose_html)
    if prose_after_projections:
        part1_content += add_prose(prose_after_projections)
    elif prose_part1:
        part1_content += add_prose(prose_part1)
    elif prose_html:
        part1_content += add_prose(prose_html)
        prose_html = ""  # Clear so it doesn't appear again

    # 4. Prose before ablation (optional)
    part1_content += add_prose(prose_before_ablation)

    # 5. Ablation changed completions (AUTO-GENERATED, title: "Ablation Effects on Completions")
    ablation_html = ablation_completions_html or ablation_table_html
    if ablation_html:
        part1_content += ablation_html

    # 6. Prose after ablation
    part1_content += add_prose(prose_after_ablation)

    # 7. Prose before steering (optional)
    part1_content += add_prose(prose_before_steering)

    # 8. Steering changed completions (AUTO-GENERATED, title: "Intelligent Steering Analysis")
    steering_html = steering_completions_html or steering_table_html
    if steering_html:
        part1_content += steering_html

    # 9. Prose after steering
    part1_content += add_prose(prose_after_steering)

    if part1_content:
        content += f'''
        <div class="output-part direct-token-effects">
            <h3 class="part-header">Direct Token Effects</h3>
            {part1_content}
        </div>
        '''

    # =========================================================================
    # PART 2: Downstream Circuit Effects
    # =========================================================================
    part2_content = ""

    # 10. Prose before downstream wiring (or fallback to prose_part2)
    if prose_before_downstream_wiring:
        part2_content += add_prose(prose_before_downstream_wiring)
    elif prose_part2:
        part2_content += add_prose(prose_part2)

    # 11. Downstream wiring (AUTO-GENERATED, title: "Downstream Wiring (Weight-Based Polarity)")
    if downstream_wiring_html:
        part2_content += downstream_wiring_html

    # 12. Prose after downstream wiring
    part2_content += add_prose(prose_after_downstream_wiring)

    # 13. Prose before downstream ablation (optional)
    part2_content += add_prose(prose_before_downstream_ablation)

    # 14. Downstream ablation effects (AUTO-GENERATED, title: "Downstream Ablation Effects")
    ds_ablation_html = downstream_ablation_effects_html or downstream_dep_html
    if ds_ablation_html:
        part2_content += ds_ablation_html

    # 15. Prose after downstream ablation
    part2_content += add_prose(prose_after_downstream_ablation)

    # 16. Prose before downstream steering (optional)
    part2_content += add_prose(prose_before_downstream_steering)

    # 17. Downstream Steering Response (slope + R² per downstream neuron)
    if downstream_steering_slopes_html:
        part2_content += downstream_steering_slopes_html

    # 18. Prose after downstream steering
    part2_content += add_prose(prose_after_downstream_steering)

    # Any remaining legacy prose
    part2_content += add_prose(prose_html)

    if part2_content:
        content += f'''
        <div class="output-part downstream-circuit-effects">
            <h3 class="part-header">Downstream Circuit Effects</h3>
            {part2_content}
        </div>
        '''

    # =========================================================================
    # PART 3: Negative Polarity Effects (bipolar neurons only)
    # =========================================================================
    if prose_part3:
        part3_processed = convert_bullets_to_html(prose_part3)
        content += f'''
        <div class="output-part negative-polarity-effects">
            <h3 class="part-header">Negative Firing Effects</h3>
            <div class="evidence-section">{part3_processed}</div>
        </div>
        '''

    # Legacy: overall prose at the end if nothing else
    if prose_html and not content:
        processed_prose = convert_bullets_to_html(prose_html)
        content += f'<div class="evidence-section">{processed_prose}</div>'

    if not content:
        content = '<p class="prose">No output function data available.</p>'

    # Add styles for the two-part layout
    content += '''
    <style>
    .output-part {
        margin-bottom: 32px;
    }
    .output-part:last-child {
        margin-bottom: 0;
    }
    .part-header {
        font-size: 16px;
        font-weight: 600;
        color: var(--text-primary, #111);
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid var(--border-color, #e0e0e0);
    }
    .evidence-prose {
        margin: 16px 0;
        padding: 12px 16px;
        background: var(--bg-secondary, #f8f9fa);
        border-radius: 6px;
        font-size: 14px;
        line-height: 1.6;
        color: var(--text-primary, #333);
    }
    </style>
    '''

    return render_main_section("output", content)


def render_hypothesis_testing_section(
    boundary_test_html: str = "",
    alternative_hypo_html: str = "",
    hypothesis_timeline_html: str = "",
    prose_html: str = ""
) -> str:
    """Render the Hypothesis Testing section.

    Args:
        boundary_test_html: Boundary test cards from skeptic
        alternative_hypo_html: Alternative hypothesis test results
        hypothesis_timeline_html: Hypothesis evolution timeline
        prose_html: Summary of hypothesis testing process
    """
    content = ""

    if prose_html:
        processed_prose = convert_bullets_to_html(prose_html)
        content += f'<div class="evidence-section">{processed_prose}</div>'

    if hypothesis_timeline_html:
        content += hypothesis_timeline_html

    if boundary_test_html:
        content += boundary_test_html

    if alternative_hypo_html:
        content += alternative_hypo_html

    if not content:
        content = '<p class="prose">No hypothesis testing data available.</p>'

    return render_main_section("hypothesis", content)


def render_open_questions_section(
    open_questions_html: str = "",
    prose_html: str = ""
) -> str:
    """Render the Open Questions section.

    Args:
        open_questions_html: Open questions for future work
        prose_html: Additional context about unanswered questions
    """
    content = ""

    if prose_html:
        processed_prose = convert_bullets_to_html(prose_html)
        content += f'<div class="evidence-section">{processed_prose}</div>'

    if open_questions_html:
        content += open_questions_html

    if not content:
        content = '<p class="prose">No open questions recorded.</p>'

    return render_main_section("questions", content)


# =============================================================================
# PAGE ASSEMBLY
# =============================================================================

def assemble_page(
    neuron_id: str,
    title: str,
    fixed_sections: dict[str, str],
    freeform_html: str = "",
    additional_css: str = "",
    main_sections: dict[str, str] | None = None
) -> str:
    """Assemble the complete HTML page (Variant 5 style).

    Args:
        neuron_id: e.g., "L4/N10555"
        title: Creative title
        fixed_sections: Dict with keys: header, narrative, circuit, metrics_row, routing_explanation, amplification
        freeform_html: The evidence section HTML (legacy - used if main_sections not provided)
        additional_css: Any extra CSS to include
        main_sections: Dict with keys: input_function, output_function, hypothesis_testing, open_questions
                      If provided, uses new 4-section structure instead of freeform_html
    """
    safe_id = neuron_id.replace("/", "_")

    # Determine which content structure to use
    if main_sections:
        # New 4-section structure: Input Function, Output Function, Hypothesis Testing, Open Questions
        body_content = f'''
        {main_sections.get('input_function', '')}
        {main_sections.get('output_function', '')}
        {main_sections.get('hypothesis_testing', '')}
        {main_sections.get('open_questions', '')}
        '''
    else:
        # Legacy freeform structure (still include projections and questions)
        body_content = f'''
        {fixed_sections.get('projections', '')}
        {freeform_html}
        {fixed_sections.get('questions', '')}
        '''

    # New Variant 5 sections (only show if available)
    metrics_row = fixed_sections.get('metrics_row', '')
    routing_explanation = fixed_sections.get('routing_explanation', '')
    amplification = fixed_sections.get('amplification', '')

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{neuron_id} - {escape_html(title)}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
{BASE_CSS}
{FIGURE_CSS}
{additional_css}
    </style>
</head>
<body>
    <div class="container" style="padding-bottom: 16px;">
        <nav class="nav">
            <a href="index.html">&larr; All Investigations</a>
        </nav>

        {fixed_sections.get('header', '')}
        {fixed_sections.get('narrative', '')}
    </div>

    {fixed_sections.get('circuit', '')}

    <div class="container">
        {body_content}
    </div>
</body>
</html>
'''


def build_fixed_sections(
    neuron_id: str,
    title: str,
    confidence: float,
    total_experiments: int,
    narrative_lead: str,
    narrative_body: str,
    upstream_neurons: list[dict[str, Any]],
    downstream_neurons: list[dict[str, Any]],
    selectivity_fires: list[dict[str, Any]],
    selectivity_ignores: list[dict[str, Any]],
    output_promote: list[dict[str, Any]],
    output_suppress: list[dict[str, Any]],
    open_questions: list[str],
    stats: list[dict[str, Any]] | None = None,
    include_zscore_explanation: bool = True,
    executive_summary: str = "",
    confidence_downgraded: bool = False,
    pre_skeptic_confidence: float = None,
    skeptic_adjustment: float = None,
    hypothesis_count: int = 0,
    # New Variant 5 parameters
    function_description: str = "",
    steering_downstream: list[dict[str, Any]] | None = None,
    selectivity_zscore: float | None = None,
    peak_activation: float | None = None,
) -> dict[str, str]:
    """Build all fixed sections.

    Args:
        include_zscore_explanation: If True, adds z-score explanation when stats include z-score
        executive_summary: HTML string for executive summary box (placed after circuit block)
        confidence_downgraded: If True, shows warning that confidence was adjusted
        pre_skeptic_confidence: Original confidence before skeptic adjustment
        skeptic_adjustment: Delta applied by skeptic
        hypothesis_count: Number of hypotheses (if >0, confidence label shows "avg hypothesis confidence")
        function_description: Short description of neuron's function
        steering_downstream: Steering results with downstream effects for amplification section
        selectivity_zscore: Category selectivity z-score for metrics row
        peak_activation: Maximum observed activation value for metrics row

    Returns dict with keys: header, narrative, executive_summary, stats, circuit, projections, questions,
                           routing_explanation, amplification, metrics_row
    """
    return {
        'header': render_header(
            neuron_id, title, confidence, total_experiments,
            confidence_downgraded=confidence_downgraded,
            pre_skeptic_confidence=pre_skeptic_confidence,
            skeptic_adjustment=skeptic_adjustment,
            hypothesis_count=hypothesis_count,
        ),
        'narrative': render_narrative(narrative_lead, narrative_body),
        'executive_summary': render_executive_summary(executive_summary),
        'stats': render_stats_section(stats or [], include_zscore_explanation=include_zscore_explanation),
        'metrics_row': render_metrics_row(selectivity_zscore, peak_activation),
        'circuit': render_circuit_block(
            neuron_id, upstream_neurons, downstream_neurons,
            selectivity_fires, selectivity_ignores,
            function_description=function_description
        ),
        'routing_explanation': render_routing_explanation(function_description),
        'amplification': render_amplification_section(steering_downstream or []),
        'projections': render_output_projections_section(output_promote, output_suppress),
        'questions': render_open_questions(open_questions),
    }
