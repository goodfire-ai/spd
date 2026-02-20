"""HTML template for neuron dashboard reports.

Distill.pub-inspired design ported from frontend/L4_N10555_v3.html
"""

import re
from typing import Any


def escape_braces(text: str) -> str:
    """Escape curly braces in text for safe use in f-strings.

    Args:
        text: Text that may contain { or }

    Returns:
        Text with { replaced by {{ and } replaced by }}
    """
    return text.replace("{", "{{").replace("}", "}}")


def linkify_neuron_ids(text: str) -> str:
    """Convert neuron IDs like L3/N9778 to clickable links.

    Args:
        text: Text potentially containing neuron IDs

    Returns:
        Text with neuron IDs wrapped in anchor tags
    """
    return re.sub(
        r'(L\d+/N\d+)',
        lambda m: f'<a href="{m.group(1).replace("/", "_")}.html" class="neuron-link">{m.group(1)}</a>',
        text
    )


CSS_STYLES = """
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

/* Full-width circuit wrapper - Distill style */
.circuit-wrapper {
    background: #fcfcfc;
    margin: 20px -9999px 40px;
    padding: 20px 9999px;
    border-top: 1px solid rgba(0, 0, 0, 0.1);
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
}

.circuit-section {
    max-width: 1100px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: 220px 1fr 220px;
    gap: 20px;
}

@media (max-width: 1000px) {
    .circuit-section {
        grid-template-columns: 1fr;
        max-width: 600px;
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

/* Steering */
.steering-card {
    background: var(--bg-elevated);
    border-radius: 20px;
    padding: 40px;
    box-shadow: 0 2px 20px rgba(0,0,0,0.04);
}

.steering-prompt {
    font-size: 15px;
    color: var(--text-tertiary);
    font-style: italic;
    margin-bottom: 32px;
    text-align: center;
}

.steering-row {
    display: flex;
    gap: 32px;
    padding: 28px 0;
    border-top: 1px solid var(--border);
}

.steering-row:first-of-type {
    border-top: none;
    padding-top: 0;
}

.steering-label {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    width: 140px;
    flex-shrink: 0;
}

.steering-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    font-weight: 600;
    flex-shrink: 0;
}

.steering-row.up .steering-icon {
    background: rgba(52, 199, 89, 0.15);
    color: #248a3d;
}

.steering-row.down .steering-icon {
    background: rgba(255, 59, 48, 0.15);
    color: #d70015;
}

.steering-label strong {
    display: block;
    font-size: 15px;
    font-weight: 600;
}

.steering-desc {
    font-size: 13px;
    color: var(--text-tertiary);
}

.steering-result {
    flex: 1;
}

.steering-tokens {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 12px;
}

.steering-tokens .token {
    padding: 10px 16px;
    border-radius: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
}

.steering-tokens .token em {
    font-style: normal;
    margin-left: 6px;
    opacity: 0.7;
    font-size: 12px;
}

.steering-tokens .token.high {
    background: rgba(52, 199, 89, 0.12);
    color: #248a3d;
}

.steering-tokens .token.low {
    background: var(--bg-inset);
    color: var(--text-tertiary);
}

.steering-note {
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.5;
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

/* Finding box */
.finding-box {
    background: linear-gradient(135deg, #fff9e6 0%, #fff4d6 100%);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 40px;
}

.finding-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #b45309;
    margin-bottom: 12px;
}

.finding-box p {
    font-size: 16px;
    line-height: 1.7;
    color: var(--text-secondary);
}

.finding-box strong {
    color: var(--red);
}

/* Hypotheses grid */
.hypotheses-grid {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.hypotheses-grid .hypothesis {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 20px 24px;
    background: var(--bg-elevated);
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.03);
}

.hypotheses-grid .hypothesis.confirmed {
    border-left: 4px solid var(--green);
}

.hypotheses-grid .hypothesis.refuted {
    border-left: 4px solid var(--red);
}

.hypotheses-grid .hypothesis-status {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    width: 80px;
    flex-shrink: 0;
}

.hypotheses-grid .hypothesis.confirmed .hypothesis-status {
    color: #248a3d;
}

.hypotheses-grid .hypothesis.refuted .hypothesis-status {
    color: var(--red);
}

.hypotheses-grid .hypothesis-text {
    flex: 1;
    font-size: 15px;
    color: var(--text-secondary);
}

.hypotheses-grid .hypothesis-delta {
    font-size: 13px;
    color: var(--text-tertiary);
    font-family: 'JetBrains Mono', monospace;
}

/* Activation Examples */
.example-card {
    background: var(--bg-elevated);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.03);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
}

.example-text {
    font-size: 16px;
    line-height: 1.5;
    color: var(--text);
}

.example-card.silent .example-text {
    color: var(--text-tertiary);
}

.example-max {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: var(--text-tertiary);
    flex-shrink: 0;
}

mark {
    background: none;
    padding: 2px 0;
}

mark.hot {
    background: linear-gradient(to top, #fde68a 40%, transparent 40%);
    color: inherit;
}

mark.warm {
    background: linear-gradient(to top, #fef3c7 40%, transparent 40%);
    color: inherit;
}

/* Open questions */
.questions-card {
    background: var(--bg-elevated);
    border-radius: 20px;
    padding: 40px;
    box-shadow: 0 2px 20px rgba(0,0,0,0.04);
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

/* Test */
.test-card {
    background: var(--bg-elevated);
    border-radius: 20px;
    padding: 40px;
    box-shadow: 0 2px 20px rgba(0,0,0,0.04);
}

.test-row {
    display: flex;
    gap: 12px;
}

.test-input {
    flex: 1;
    padding: 16px 20px;
    border: 2px solid var(--border);
    border-radius: 14px;
    font-size: 17px;
    font-family: inherit;
    outline: none;
    transition: border-color 0.2s;
}

.test-input:focus {
    border-color: var(--accent);
}

.test-btn {
    padding: 16px 32px;
    background: var(--accent);
    border: none;
    border-radius: 14px;
    color: white;
    font-size: 17px;
    font-weight: 500;
    cursor: pointer;
    transition: transform 0.1s, opacity 0.15s;
}

.test-btn:hover {
    opacity: 0.9;
}

.test-btn:active {
    transform: scale(0.98);
}

.test-result {
    display: none;
    justify-content: space-between;
    align-items: center;
    margin-top: 20px;
    padding: 20px 24px;
    background: var(--bg-inset);
    border-radius: 14px;
}

.test-result.visible {
    display: flex;
}

.test-result-prompt {
    font-size: 15px;
    color: var(--text-secondary);
}

.test-result-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px;
    font-weight: 600;
}

.test-result-value.high { color: #248a3d; }
.test-result-value.low { color: var(--text-tertiary); }

/* Expandable experiment boxes */
.experiment-box {
    background: var(--bg-elevated);
    border-radius: 12px;
    margin-bottom: 8px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.03);
    border: 1px solid var(--border);
}

.experiment-box summary {
    padding: 16px 20px;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 15px;
    list-style: none;
}

.experiment-box summary::-webkit-details-marker {
    display: none;
}

.experiment-box summary::after {
    content: '\\25B6';
    font-size: 10px;
    color: var(--text-tertiary);
    transition: transform 0.2s;
}

.experiment-box[open] summary::after {
    transform: rotate(90deg);
}

.experiment-box .exp-title {
    color: var(--text);
    font-weight: 500;
}

.experiment-box .exp-result {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: var(--green);
}

.experiment-box .exp-result.negative {
    color: var(--red);
}

.experiment-content {
    padding: 0 20px 20px;
    border-top: 1px solid var(--border);
    margin-top: 0;
}

.experiment-content .promotes,
.experiment-content .suppresses {
    padding: 12px 0;
    font-size: 14px;
    color: var(--text-secondary);
}

.experiment-content .promotes {
    border-bottom: 1px solid var(--border);
}

.experiment-content .token-list {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px;
}

.experiment-content .token-badge {
    padding: 6px 12px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}

.experiment-content .token-badge.promote {
    background: rgba(52, 199, 89, 0.1);
    color: #248a3d;
}

.experiment-content .token-badge.suppress {
    background: rgba(215, 58, 73, 0.1);
    color: var(--red);
}

/* Chat toggle */
.chat-toggle {
    position: fixed;
    bottom: 32px;
    right: 32px;
    width: 60px;
    height: 60px;
    background: linear-gradient(135deg, #d4a574, #c9956c);
    border: none;
    border-radius: 50%;
    color: white;
    font-size: 26px;
    cursor: pointer;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    transition: transform 0.2s;
    z-index: 100;
}

.chat-toggle:hover {
    transform: scale(1.08);
}

.chat-toggle.open { display: none; }

.chat-panel {
    position: fixed;
    top: 0;
    right: 0;
    width: 400px;
    height: 100vh;
    background: white;
    border-left: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    transform: translateX(100%);
    transition: transform 0.3s ease;
    z-index: 200;
}

.chat-panel.open { transform: translateX(0); }

.chat-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.4);
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.3s;
    z-index: 150;
}

.chat-backdrop.open { opacity: 1; visibility: visible; }

.chat-header {
    padding: 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 18px;
    font-weight: 600;
}

.chat-close {
    background: none;
    border: none;
    font-size: 28px;
    cursor: pointer;
    color: var(--text-tertiary);
}

.chat-messages {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
}

.chat-bubble {
    padding: 20px;
    background: var(--bg-inset);
    border-radius: 16px;
    font-size: 15px;
    line-height: 1.6;
}

.chat-input-area {
    padding: 24px;
    border-top: 1px solid var(--border);
    display: flex;
    gap: 12px;
}

.chat-input {
    flex: 1;
    padding: 14px 18px;
    border: 2px solid var(--border);
    border-radius: 12px;
    font-size: 15px;
    outline: none;
}

.chat-send {
    padding: 14px 24px;
    background: var(--accent);
    border: none;
    border-radius: 12px;
    color: white;
    font-size: 15px;
    font-weight: 500;
    cursor: pointer;
}
"""


def generate_html(
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
    steering_results: list[dict[str, Any]],
    activation_examples: list[dict[str, Any]],
    key_finding: str,
    hypotheses: list[dict[str, Any]],
    open_questions: list[str],
    detailed_experiments: dict[str, Any],
    trigger_words: list[str],
    relp_analysis: dict[str, Any] | None = None,
) -> str:
    """Generate complete HTML dashboard.

    Args:
        neuron_id: e.g., "L4/N10555"
        title: LLM-generated title
        confidence: 0-1 confidence score
        total_experiments: Number of experiments run
        narrative_lead: Opening sentence
        narrative_body: Follow-up paragraph
        upstream_neurons: List of upstream neuron dicts
        downstream_neurons: List of downstream neuron dicts
        selectivity_fires: Examples where neuron fires
        selectivity_ignores: Examples where neuron doesn't fire
        steering_results: Steering experiment results
        activation_examples: All activation examples
        key_finding: Most important finding
        hypotheses: Hypothesis status list
        open_questions: Unanswered questions
        detailed_experiments: Full ablation/steering data for expandable boxes
        trigger_words: Words that trigger this neuron (for mock test)

    Returns:
        Complete HTML string
    """
    # Format neuron ID for display
    try:
        layer, neuron = neuron_id.replace("L", "").split("/N")
        display_id = f"L{layer} / N{neuron}"
    except ValueError:
        # Handle malformed neuron_id gracefully
        display_id = neuron_id
        layer = "?"
        neuron = "?"
    safe_id = neuron_id.replace("/", "_")

    # Build upstream neurons HTML
    upstream_html = ""
    for n in upstream_neurons[:4]:
        label = escape_braces(str(n.get("label", "Unknown")))
        is_unknown = "unknown" in label.lower() or not label
        name_class = "neuron-name unknown" if is_unknown else "neuron-name"
        nid = n.get("neuron_id", "")
        try:
            weight = float(n.get("weight", 0))
        except (ValueError, TypeError):
            weight = 0.0
        linked_id = linkify_neuron_ids(nid)
        upstream_html += f"""
        <div class="neuron-item">
            <div class="{name_class}">{label}</div>
            <div class="neuron-meta">
                <span class="neuron-id">{linked_id}</span>
                <span class="neuron-conn">weight {weight:+.3f}</span>
            </div>
        </div>
        """

    # Build downstream neurons HTML
    downstream_html = ""
    for n in downstream_neurons[:4]:
        label = escape_braces(str(n.get("label", "Unknown")))
        is_unknown = "unknown" in label.lower() or not label
        name_class = "neuron-name unknown" if is_unknown else "neuron-name"
        nid = n.get("neuron_id", "")
        try:
            weight = float(n.get("weight", 0))
        except (ValueError, TypeError):
            weight = 0.0
        linked_id = linkify_neuron_ids(nid)
        downstream_html += f"""
        <div class="neuron-item">
            <div class="{name_class}">{label}</div>
            <div class="neuron-meta">
                <span class="neuron-id">{linked_id}</span>
                <span class="neuron-conn">weight {weight:+.3f}</span>
            </div>
        </div>
        """

    # Build selectivity fires HTML
    fires_html = ""
    for group in selectivity_fires[:3]:
        group_label = group.get("label", "Fires on")
        examples = group.get("examples", [])
        examples_html = ""
        for ex in examples[:3]:
            text = ex.get("text", "")
            activation = ex.get("activation", 0)
            examples_html += f'<div class="ex">{text} <span class="act">{activation:.2f}</span></div>'
        fires_html += f"""
        <div class="selectivity-group fires">
            <div class="selectivity-header">&#10003; {group_label}</div>
            <div class="selectivity-examples">{examples_html}</div>
        </div>
        """

    # Build selectivity ignores HTML
    ignores_html = ""
    for group in selectivity_ignores[:3]:
        group_label = group.get("label", "Ignores")
        examples = group.get("examples", [])
        examples_html = ""
        for ex in examples[:2]:
            text = ex.get("text", "")
            activation = ex.get("activation", 0)
            examples_html += f'<div class="ex">{text} <span class="act">{activation:.2f}</span></div>'
        ignores_html += f"""
        <div class="selectivity-group ignores">
            <div class="selectivity-header">&#10007; {group_label}</div>
            <div class="selectivity-examples">{examples_html}</div>
        </div>
        """

    # Build steering HTML
    steering_html = ""
    if steering_results:
        # Find positive and negative steering
        positive_steer = None
        negative_steer = None
        for s in steering_results:
            try:
                steer_val = float(s.get("steering_value", 0))
            except (ValueError, TypeError):
                steer_val = 0
            if steer_val > 0 and not positive_steer:
                positive_steer = s
            elif steer_val < 0 and not negative_steer:
                negative_steer = s

        prompt = positive_steer.get("prompt", "") if positive_steer else "Test prompt"

        if positive_steer:
            promotes = positive_steer.get("promotes", [])[:3]
            tokens_html = "".join([
                f'<span class="token high">{t[0]} <em>+{abs(t[1]):.1f}</em></span>'
                for t in promotes
            ])
            steering_html += f"""
            <div class="steering-row up">
                <div class="steering-label">
                    <span class="steering-icon">&uarr;</span>
                    <div>
                        <strong>Steer +{positive_steer.get('steering_value', 10)}</strong>
                        <span class="steering-desc">force gate open</span>
                    </div>
                </div>
                <div class="steering-result">
                    <div class="steering-tokens">{tokens_html}</div>
                    <p class="steering-note">Promotes these tokens when neuron is amplified</p>
                </div>
            </div>
            """

        if negative_steer:
            promotes = negative_steer.get("promotes", [])[:3]
            tokens_html = "".join([
                f'<span class="token low">{t[0]} <em>+{abs(t[1]):.1f}</em></span>'
                for t in promotes
            ])
            steering_html += f"""
            <div class="steering-row down">
                <div class="steering-label">
                    <span class="steering-icon">&darr;</span>
                    <div>
                        <strong>Ablate</strong>
                        <span class="steering-desc">force gate closed</span>
                    </div>
                </div>
                <div class="steering-result">
                    <div class="steering-tokens">{tokens_html}</div>
                    <p class="steering-note">Model produces generic continuations without this neuron</p>
                </div>
            </div>
            """

        steering_html = f"""
        <p class="steering-prompt">Prompt: "{prompt[:80]}..."</p>
        {steering_html}
        """

    # Build activation examples HTML
    examples_html = ""
    for ex in activation_examples[:7]:
        prompt = ex.get("prompt", "")
        activation = ex.get("activation", 0)
        is_positive = ex.get("is_positive", True)
        highlighted_text = ex.get("highlighted_text", prompt)
        card_class = "example-card" if is_positive else "example-card silent"
        examples_html += f"""
        <div class="{card_class}">
            <div class="example-text">{highlighted_text}</div>
            <span class="example-max">max {activation:.2f}</span>
        </div>
        """

    # Build hypotheses HTML
    hypotheses_html = ""
    for h in hypotheses[:4]:
        status = h.get("status", "testing")
        hypothesis_text = h.get("hypothesis", "")[:100]
        prior = h.get("prior_probability", 50)
        posterior = h.get("posterior_probability", 50)

        if status == "confirmed":
            card_class = "hypothesis confirmed"
            status_text = "Confirmed"
        elif status == "refuted":
            card_class = "hypothesis refuted"
            status_text = "Refuted"
        else:
            card_class = "hypothesis"
            status_text = "Testing"

        hypotheses_html += f"""
        <div class="{card_class}">
            <span class="hypothesis-status">{status_text}</span>
            <span class="hypothesis-text">{hypothesis_text}</span>
            <span class="hypothesis-delta">{prior}% &rarr; {posterior}%</span>
        </div>
        """

    # Build open questions HTML
    questions_html = ""
    for q in open_questions[:4]:
        questions_html += f'<div class="question-item">{linkify_neuron_ids(q)}</div>'

    # Build expandable experiments HTML
    experiments_html = ""
    def safe_float(val, default=0.0):
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    ablations = detailed_experiments.get("ablation", [])
    for abl in ablations[:5]:
        prompt = abl.get("prompt", "")[:50]
        promotes = abl.get("promotes", [])
        suppresses = abl.get("suppresses", [])

        top_effect = promotes[0] if promotes else (suppresses[0] if suppresses else ["?", 0])
        top_effect_val = safe_float(top_effect[1]) if len(top_effect) > 1 else 0.0
        result_class = "exp-result" if top_effect_val > 0 else "exp-result negative"

        promotes_html = "".join([
            f'<span class="token-badge promote">{t[0]} +{safe_float(t[1]):.2f}</span>'
            for t in promotes[:5] if len(t) > 1
        ])
        suppresses_html = "".join([
            f'<span class="token-badge suppress">{t[0]} {safe_float(t[1]):.2f}</span>'
            for t in suppresses[:5] if len(t) > 1
        ])

        experiments_html += f"""
        <details class="experiment-box">
            <summary>
                <span class="exp-title">Ablation: "{escape_braces(prompt)}..."</span>
                <span class="{result_class}">{top_effect[0]} {top_effect_val:+.2f}</span>
            </summary>
            <div class="experiment-content">
                <div class="promotes">
                    <strong>Promotes when removed:</strong>
                    <div class="token-list">{promotes_html}</div>
                </div>
                <div class="suppresses">
                    <strong>Suppresses when removed:</strong>
                    <div class="token-list">{suppresses_html}</div>
                </div>
            </div>
        </details>
        """

    # Add steering experiments as expandable boxes
    steerings = detailed_experiments.get("steering", [])
    for steer in steerings[:5]:
        prompt = steer.get("prompt", "")[:50]
        steering_val = safe_float(steer.get("steering_value", 0))
        promotes = steer.get("promotes", [])
        suppresses = steer.get("suppresses", [])

        top_effect = promotes[0] if promotes else (suppresses[0] if suppresses else ["?", 0])
        top_effect_val = safe_float(top_effect[1]) if len(top_effect) > 1 else 0.0
        result_class = "exp-result" if top_effect_val > 0 else "exp-result negative"
        steer_direction = f"+{steering_val:.0f}" if steering_val > 0 else f"{steering_val:.0f}"

        promotes_html = "".join([
            f'<span class="token-badge promote">{t[0]} +{safe_float(t[1]):.2f}</span>'
            for t in promotes[:5] if len(t) > 1
        ])
        suppresses_html = "".join([
            f'<span class="token-badge suppress">{t[0]} {safe_float(t[1]):.2f}</span>'
            for t in suppresses[:5] if len(t) > 1
        ])

        experiments_html += f"""
        <details class="experiment-box">
            <summary>
                <span class="exp-title">Steering ({steer_direction}): "{escape_braces(prompt)}..."</span>
                <span class="{result_class}">{top_effect[0]} {top_effect_val:+.2f}</span>
            </summary>
            <div class="experiment-content">
                <div class="promotes">
                    <strong>Promotes with steering:</strong>
                    <div class="token-list">{promotes_html}</div>
                </div>
                <div class="suppresses">
                    <strong>Suppresses with steering:</strong>
                    <div class="token-list">{suppresses_html}</div>
                </div>
            </div>
        </details>
        """

    # Add RelP experiments as expandable boxes
    if relp_analysis:
        relp_results = relp_analysis.get("results", [])
        for relp in relp_results[:5]:
            prompt = relp.get("prompt", "")
            target_tokens = relp.get("target_tokens", [])
            relp_score = relp.get("neuron_relp_score", 0)
            neuron_found = relp.get("neuron_found", False)
            in_causal = relp.get("in_causal_pathway", False)
            tau = relp.get("tau", 0.005)
            downstream = relp.get("downstream_edges", [])
            upstream = relp.get("upstream_edges", [])
            graph_stats = relp.get("graph_stats", {})

            # Build upstream neurons HTML
            upstream_html = ""
            for edge in upstream[:5]:
                source_info = edge.get("source_info", {})
                clerp = source_info.get("clerp", "?")
                weight = edge.get("weight", 0)
                # Linkify the neuron ID
                linked_clerp = linkify_neuron_ids(clerp)
                upstream_html += f'<span class="token-badge promote">{linked_clerp} w={weight:.3f}</span>'

            # Build downstream connections HTML
            downstream_html = ""
            for edge in downstream[:5]:
                target_info = edge.get("target_info", {})
                weight = edge.get("weight", 0)
                if target_info.get("type") == "logit":
                    token = target_info.get("token", "?")
                    downstream_html += f'<span class="token-badge promote">{token} →{weight:.3f}</span>'
                else:
                    clerp = target_info.get("clerp", "?")
                    linked_clerp = linkify_neuron_ids(clerp)
                    downstream_html += f'<span class="token-badge">{linked_clerp} w={weight:.3f}</span>'

            result_class = "exp-result" if neuron_found else "exp-result negative"
            targets_str = ", ".join(target_tokens[:3]) if target_tokens else "?"
            prompt_short = prompt[:50]

            # Graph stats
            total_nodes = graph_stats.get("total_nodes", 0)
            elapsed = graph_stats.get("elapsed_seconds", 0)

            experiments_html += f"""
        <details class="experiment-box">
            <summary>
                <span class="exp-title">RelP: "{prompt_short}..." → {targets_str}</span>
                <span class="{result_class}">score {relp_score:.2f}</span>
            </summary>
            <div class="experiment-content">
                <p style="margin-bottom: 12px; color: var(--text-secondary); font-size: 14px;">
                    <strong>Full prompt:</strong> {prompt}
                </p>
                <p style="margin-bottom: 12px; font-size: 13px; color: var(--text-tertiary);">
                    τ={tau} · {total_nodes} nodes · {elapsed:.1f}s ·
                    {"✓ In causal pathway" if in_causal else "✗ Not in causal pathway"}
                </p>
                <div class="promotes" style="margin-bottom: 12px;">
                    <strong>Upstream neurons (feeding this neuron):</strong>
                    <div class="token-list">{upstream_html if upstream_html else '<em>No upstream edges found</em>'}</div>
                </div>
                <div class="suppresses">
                    <strong>Downstream connections (to logits):</strong>
                    <div class="token-list">{downstream_html if downstream_html else '<em>No downstream edges found</em>'}</div>
                </div>
            </div>
        </details>
        """

    # Build trigger words JS array
    triggers_js = ", ".join([f'"{w}"' for w in trigger_words[:15]])

    # Assemble full HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{neuron_id} - {escape_braces(title)}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>{CSS_STYLES}</style>
</head>
<body>
    <div class="container">
        <nav class="nav">
            <a href="index.html">&larr; All Investigations</a>
        </nav>

        <header class="header">
            <div class="neuron-id-header">{display_id}</div>
            <h1>{escape_braces(title)}</h1>
            <div class="confidence-badge">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
                    <path d="M20 6L9 17l-5-5"/>
                </svg>
                {int(confidence * 100)}% confidence &middot; {total_experiments} experiments
            </div>
        </header>

        <div class="story-section">
            <p class="story-lead">{linkify_neuron_ids(escape_braces(narrative_lead))}</p>
            <p class="story-body">{linkify_neuron_ids(escape_braces(narrative_body))}</p>
        </div>
    </div>

    <div class="circuit-wrapper">
        <div class="circuit-section">
            <div class="circuit-col upstream">
                <div class="circuit-col-header">
                    <span class="circuit-col-title">Upstream</span>
                    <span class="circuit-col-sub">what builds this neuron</span>
                </div>
                <div class="neuron-list">{upstream_html}</div>
            </div>

            <div class="circuit-col center">
                <div class="circuit-col-header">
                    <span class="circuit-col-title">{neuron_id}</span>
                    <span class="circuit-col-sub">selectivity</span>
                </div>
                <div class="selectivity-list">
                    {fires_html}
                    {ignores_html}
                </div>
            </div>

            <div class="circuit-col downstream">
                <div class="circuit-col-header">
                    <span class="circuit-col-title">Downstream</span>
                    <span class="circuit-col-sub">what this activates</span>
                </div>
                <div class="neuron-list">{downstream_html}</div>
            </div>
        </div>
    </div>

    <div class="container">
        <div class="section">
            <div class="section-label">What Happens When You Turn It Up or Down</div>
            <div class="steering-card">{steering_html}</div>
        </div>

        <div class="section">
            <div class="section-label">Activation Examples</div>
            {examples_html}
        </div>

        <div class="finding-box">
            <div class="finding-label">Key finding</div>
            <p>{linkify_neuron_ids(escape_braces(key_finding))}</p>
        </div>

        <div class="section">
            <div class="section-label">Hypotheses Tested</div>
            <div class="hypotheses-grid">{hypotheses_html}</div>
        </div>

        <div class="section">
            <div class="section-label">Detailed Experiments</div>
            {experiments_html}
        </div>

        <div class="section">
            <div class="section-label">Open Questions</div>
            <div class="questions-card">{questions_html}</div>
        </div>

        <div class="section">
            <div class="section-label">Try It</div>
            <div class="test-card">
                <div class="test-row">
                    <input type="text" class="test-input" id="test-input" placeholder="Enter any prompt...">
                    <button class="test-btn" onclick="runTest()">Test</button>
                </div>
                <div class="test-result" id="test-result">
                    <span class="test-result-prompt" id="test-prompt-display"></span>
                    <span class="test-result-value" id="test-value"></span>
                </div>
            </div>
        </div>
    </div>

    <button class="chat-toggle" onclick="toggleChat()">&#x1F4AC;</button>
    <div class="chat-backdrop" onclick="toggleChat()"></div>
    <aside class="chat-panel">
        <div class="chat-header">
            Ask Claude
            <button class="chat-close" onclick="toggleChat()">&times;</button>
        </div>
        <div class="chat-messages">
            <div class="chat-bubble">
                I can help explore boundary cases, run new experiments, or explain why certain contexts activate or suppress this neuron.
            </div>
        </div>
        <div class="chat-input-area">
            <input type="text" class="chat-input" placeholder="Ask about this neuron...">
            <button class="chat-send">Send</button>
        </div>
    </aside>

    <script>
        const triggers = [{triggers_js}];
        const suppressors = [];

        function mockActivation(prompt) {{
            const lower = prompt.toLowerCase();
            let score = 0.05 + Math.random() * 0.08;
            for (const w of triggers) if (lower.includes(w.toLowerCase())) score += 0.5 + Math.random() * 0.4;
            for (const w of suppressors) if (lower.includes(w.toLowerCase())) score = Math.max(0.03, score - 0.4);
            return Math.min(3.0, score);
        }}

        function runTest() {{
            const prompt = document.getElementById('test-input').value.trim();
            if (!prompt) return;
            const activation = mockActivation(prompt);
            document.getElementById('test-prompt-display').textContent = prompt.length > 50 ? prompt.slice(0,50) + '...' : prompt;
            const valueEl = document.getElementById('test-value');
            valueEl.textContent = activation.toFixed(2) + (activation > 1.0 ? ' \\u2713' : ' \\u2717');
            valueEl.className = 'test-result-value ' + (activation > 1.0 ? 'high' : 'low');
            document.getElementById('test-result').classList.add('visible');
        }}

        document.getElementById('test-input').addEventListener('keypress', e => {{ if (e.key === 'Enter') runTest(); }});

        function toggleChat() {{
            document.querySelector('.chat-panel').classList.toggle('open');
            document.querySelector('.chat-backdrop').classList.toggle('open');
            document.querySelector('.chat-toggle').classList.toggle('open');
        }}
    </script>
</body>
</html>
"""

    return html
