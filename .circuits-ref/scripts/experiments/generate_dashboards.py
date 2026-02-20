#!/usr/bin/env python3
"""Generate HTML dashboard pages from investigation JSON files."""

import html
import json
from pathlib import Path

# Cache for neuron labels (loaded once)
_neuron_labels_cache = None


def _load_neuron_labels():
    """Load neuron labels from interactive_labels.json and v6_enriched data."""
    global _neuron_labels_cache
    if _neuron_labels_cache is not None:
        return _neuron_labels_cache

    labels = {}

    # Load from interactive_labels.json (higher priority)
    try:
        labels_path = Path("data/interactive_labels.json")
        if labels_path.exists():
            with open(labels_path) as f:
                data = json.load(f)
            for nid, info in data.get("neurons", {}).items():
                label = info.get("function_label", "")
                if label:
                    labels[nid] = label
    except Exception:
        pass

    # Load from v6_enriched (if interactive_labels doesn't have it)
    try:
        v6_path = Path("data/medical_edge_stats_v6_enriched.json")
        if v6_path.exists():
            with open(v6_path) as f:
                data = json.load(f)
            for profile in data.get("profiles", []):
                nid = profile.get("neuron_id", "")
                if nid and nid not in labels:
                    # Use max_act_label if available
                    label = profile.get("max_act_label", "")
                    if label:
                        labels[nid] = label
    except Exception:
        pass

    _neuron_labels_cache = labels
    return labels


def _parse_relp_neuron_id(relp_id: str) -> str:
    """Parse RelP neuron ID like '4_10555_18' to 'L4/N10555' (stripping position)."""
    if not relp_id or not isinstance(relp_id, str):
        return relp_id

    # Handle logit nodes (start with 'L_' or 'LOGIT')
    if relp_id.startswith("L_") or relp_id.startswith("LOGIT"):
        return relp_id

    # Format: layer_neuron_position (e.g., '4_10555_18')
    parts = relp_id.split("_")
    if len(parts) >= 2:
        try:
            layer = int(parts[0])
            neuron = int(parts[1])
            return f"L{layer}/N{neuron}"
        except ValueError:
            pass

    return relp_id


def _get_neuron_label(neuron_id: str) -> str:
    """Get label for a neuron ID, returns empty string if not found."""
    labels = _load_neuron_labels()
    return labels.get(neuron_id, "")


def generate_dashboard_html(dashboard: dict) -> str:
    """Generate a complete HTML dashboard from dashboard JSON."""

    sc = dashboard.get("summary_card", {})
    stats = dashboard.get("stats", {})
    ap = dashboard.get("activation_patterns", {})
    ae = dashboard.get("ablation_effects", {})
    conn = dashboard.get("connectivity", {})
    findings = dashboard.get("findings", {})
    meta = dashboard.get("metadata", {})
    output_proj = dashboard.get("output_projections", {})
    detailed_exp = dashboard.get("detailed_experiments", {})
    relp_analysis = dashboard.get("relp_analysis", {})
    hypothesis_timeline = dashboard.get("hypothesis_timeline", {})

    # Helper to escape HTML
    def esc(s):
        return html.escape(str(s)) if s else ""

    # Generate activation examples HTML
    def activation_rows(examples, is_positive=True, limit=20):
        rows = []
        for ex in examples[:limit]:
            prompt = esc(ex.get("prompt", ""))
            act = ex.get("activation", 0)
            token = esc(ex.get("token", ""))
            color = "var(--accent-green)" if is_positive else "var(--accent-red)"
            rows.append(f'''
                <tr>
                    <td style="max-width: 500px; white-space: normal; word-wrap: break-word;">{prompt}</td>
                    <td style="color: {color}; font-weight: 600; white-space: nowrap;">{act:.3f}</td>
                    <td style="white-space: nowrap;"><code>{token}</code></td>
                </tr>''')
        return "\n".join(rows) if rows else "<tr><td colspan='3'>No examples</td></tr>"

    # Generate ablation effects HTML
    def ablation_items(tokens, direction):
        if not tokens:
            return "<li>None detected</li>"
        return "\n".join(f"<li><code>{esc(t)}</code></li>" for t in tokens[:5])

    # Generate connectivity HTML
    def connectivity_items(nodes):
        if not nodes:
            return "<li>No connections found</li>"
        items = []
        for n in nodes[:10]:  # Show up to 10
            nid = esc(n.get("neuron_id", ""))
            label = esc(n.get("label", ""))
            weight = n.get("weight", 0)
            # Color code by sign
            weight_color = "var(--accent-green)" if weight > 0 else "var(--accent-red)"
            sign = "+" if weight >= 0 else ""
            items.append(f"<li><strong>{nid}</strong>: {label} <span style='color: {weight_color}; font-weight: 600;'>({sign}{weight:.3f})</span></li>")
        return "\n".join(items)

    # Generate findings HTML
    def findings_list(items):
        if not items:
            return "<li>No findings recorded</li>"
        return "\n".join(f"<li>{esc(f)}</li>" for f in items)

    # Generate detailed ablation results
    def ablation_details_html(ablations):
        if not ablations:
            return "<p style='color: var(--text-secondary);'>No detailed ablation data</p>"

        html_parts = []
        for abl in ablations[:5]:
            prompt = esc(abl.get("prompt", ""))
            html_parts.append("<div style='margin-bottom: 1rem; padding: 1rem; background: var(--bg-secondary); border-radius: 8px;'>")
            html_parts.append(f"<div style='font-weight: 600; margin-bottom: 0.5rem;'>Prompt: {prompt}</div>")

            promotes = abl.get("promotes", [])
            if promotes:
                html_parts.append("<div style='color: var(--accent-green); margin-top: 0.5rem;'>Promotes:</div>")
                html_parts.append("<div style='display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.25rem;'>")
                for token, shift in promotes[:10]:
                    html_parts.append(f"<span style='background: var(--bg-card); padding: 0.25rem 0.5rem; border-radius: 4px;'><code>{esc(token)}</code> <span style='color: var(--accent-green);'>+{shift:.2f}</span></span>")
                html_parts.append("</div>")

            suppresses = abl.get("suppresses", [])
            if suppresses:
                html_parts.append("<div style='color: var(--accent-red); margin-top: 0.5rem;'>Suppresses:</div>")
                html_parts.append("<div style='display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.25rem;'>")
                for token, shift in suppresses[:10]:
                    html_parts.append(f"<span style='background: var(--bg-card); padding: 0.25rem 0.5rem; border-radius: 4px;'><code>{esc(token)}</code> <span style='color: var(--accent-red);'>{shift:.2f}</span></span>")
                html_parts.append("</div>")

            html_parts.append("</div>")
        return "\n".join(html_parts)

    # Generate detailed steering results
    def steering_details_html(steerings):
        if not steerings:
            return "<p style='color: var(--text-secondary);'>No detailed steering data</p>"

        html_parts = []
        for steer in steerings[:5]:
            prompt = esc(steer.get("prompt", ""))
            value = steer.get("steering_value", 0)
            html_parts.append("<div style='margin-bottom: 1rem; padding: 1rem; background: var(--bg-secondary); border-radius: 8px;'>")
            html_parts.append(f"<div style='font-weight: 600; margin-bottom: 0.5rem;'>Prompt: {prompt}</div>")
            html_parts.append(f"<div style='color: var(--accent-blue); margin-bottom: 0.5rem;'>Steering: {value:+.1f}</div>")

            promotes = steer.get("promotes", [])
            if promotes:
                html_parts.append("<div style='color: var(--accent-green); margin-top: 0.5rem;'>Promotes:</div>")
                html_parts.append("<div style='display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.25rem;'>")
                for token, shift in promotes[:10]:
                    html_parts.append(f"<span style='background: var(--bg-card); padding: 0.25rem 0.5rem; border-radius: 4px;'><code>{esc(token)}</code> <span style='color: var(--accent-green);'>+{shift:.2f}</span></span>")
                html_parts.append("</div>")

            suppresses = steer.get("suppresses", [])
            if suppresses:
                html_parts.append("<div style='color: var(--accent-red); margin-top: 0.5rem;'>Suppresses:</div>")
                html_parts.append("<div style='display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.25rem;'>")
                for token, shift in suppresses[:10]:
                    html_parts.append(f"<span style='background: var(--bg-card); padding: 0.25rem 0.5rem; border-radius: 4px;'><code>{esc(token)}</code> <span style='color: var(--accent-red);'>{shift:.2f}</span></span>")
                html_parts.append("</div>")

            html_parts.append("</div>")
        return "\n".join(html_parts)

    def relp_results_html(relp):
        if not relp or not relp.get("results"):
            return "<p style='color: var(--text-secondary);'>No RelP attribution experiments recorded</p>"

        results = relp.get("results", [])
        found_count = relp.get("neuron_found_count", 0)
        total_runs = relp.get("total_relp_runs", 0)

        html_parts = []
        html_parts.append(f"<div style='margin-bottom: 1rem; color: var(--text-secondary);'>Neuron found in {found_count}/{total_runs} RelP graphs</div>")

        # Count neurons in causal pathway
        causal_count = sum(1 for r in results if r.get("in_causal_pathway"))
        html_parts.append(f"<div style='margin-bottom: 0.5rem; color: var(--text-secondary);'>In causal pathway (found + has downstream edges): {causal_count}/{total_runs}</div>")

        for r in results[:5]:
            prompt = esc(r.get("prompt", ""))
            found = r.get("neuron_found", False)
            score = r.get("neuron_relp_score")
            tau = r.get("tau", 0)
            target_tokens = r.get("target_tokens")
            in_causal = r.get("in_causal_pathway", False)

            # Status badges
            if in_causal:
                found_badge = "<span style='color: var(--accent-green);'>✓ In causal pathway</span>"
            elif found:
                found_badge = "<span style='color: var(--accent-yellow);'>⚠ Found (no downstream edges)</span>"
            else:
                found_badge = "<span style='color: var(--accent-red);'>✗ Not found</span>"
            score_str = f"RelP score: {score:.4f}" if score is not None else ""

            html_parts.append("<div style='margin-bottom: 1rem; padding: 1rem; background: var(--bg-secondary); border-radius: 8px;'>")
            html_parts.append("<div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>")
            html_parts.append(f"<span style='font-weight: 600;'>{found_badge}</span>")
            html_parts.append(f"<span style='color: var(--text-secondary); font-size: 0.875rem;'>τ={tau}</span>")
            html_parts.append("</div>")

            # Show target tokens if traced
            if target_tokens:
                target_str = ", ".join(esc(str(t)) for t in target_tokens[:3])
                html_parts.append(f"<div style='color: var(--accent-purple); font-size: 0.875rem; margin-bottom: 0.5rem;'>Traced tokens: <code>{target_str}</code></div>")

            html_parts.append(f"<div style='color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 0.5rem;'>{prompt}</div>")
            if score_str:
                html_parts.append(f"<div style='color: var(--accent-blue);'>{score_str}</div>")

            # Upstream edges
            upstream = r.get("upstream_edges", [])
            if upstream:
                html_parts.append("<div style='margin-top: 0.5rem;'><span style='color: var(--text-secondary);'>Upstream edges (inputs to this neuron):</span></div>")
                html_parts.append("<div style='display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.25rem;'>")
                for edge in upstream[:5]:
                    raw_source = edge.get("source", "")
                    weight = edge.get("weight", 0)
                    weight_color = "var(--accent-green)" if weight > 0 else "var(--accent-red)"
                    sign = "+" if weight >= 0 else ""
                    # Parse neuron ID and lookup label
                    parsed_id = _parse_relp_neuron_id(raw_source)
                    label = _get_neuron_label(parsed_id)
                    label_html = f"<span style='color: var(--text-secondary); font-style: italic;'> ({esc(label)})</span>" if label else ""
                    html_parts.append(f"<span style='background: var(--bg-card); padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem;'>{esc(parsed_id)}{label_html} <span style='color: {weight_color};'>{sign}{weight:.3f}</span></span>")
                html_parts.append("</div>")

            # Downstream edges
            downstream = r.get("downstream_edges", [])
            if downstream:
                html_parts.append("<div style='margin-top: 0.5rem;'><span style='color: var(--text-secondary);'>Downstream edges (outputs from this neuron):</span></div>")
                html_parts.append("<div style='display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.25rem;'>")
                for edge in downstream[:5]:
                    raw_target = edge.get("target", "")
                    weight = edge.get("weight", 0)
                    weight_color = "var(--accent-green)" if weight > 0 else "var(--accent-red)"
                    sign = "+" if weight >= 0 else ""
                    # Parse neuron ID and lookup label
                    parsed_id = _parse_relp_neuron_id(raw_target)
                    label = _get_neuron_label(parsed_id)
                    label_html = f"<span style='color: var(--text-secondary); font-style: italic;'> ({esc(label)})</span>" if label else ""
                    html_parts.append(f"<span style='background: var(--bg-card); padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem;'>{esc(parsed_id)}{label_html} <span style='color: {weight_color};'>{sign}{weight:.3f}</span></span>")
                html_parts.append("</div>")

            html_parts.append("</div>")
        return "\n".join(html_parts)

    def hypothesis_timeline_html(timeline):
        """Generate HTML for hypothesis testing timeline."""
        hypotheses = timeline.get("hypotheses", [])
        if not hypotheses:
            return ""

        html_parts = []
        for h in hypotheses:
            h_id = esc(h.get("hypothesis_id", ""))
            hypothesis = esc(h.get("hypothesis", ""))[:100]
            status = h.get("status", "")
            prior = h.get("prior_probability")
            posterior = h.get("posterior_probability")

            # Status badge color
            if status == "confirmed":
                status_color = "var(--accent-green)"
                status_icon = "✓"
            elif status == "refuted":
                status_color = "var(--accent-red)"
                status_icon = "✗"
            else:
                status_color = "var(--accent-yellow)"
                status_icon = "?"

            # Probability shift
            if prior is not None and posterior is not None:
                shift = posterior - prior
                shift_str = f"{'+' if shift >= 0 else ''}{shift}%"
                shift_color = "var(--accent-green)" if shift > 0 else "var(--accent-red)" if shift < 0 else "var(--text-secondary)"
            else:
                shift_str = ""
                shift_color = ""

            html_parts.append(f"""
            <div style="margin-bottom: 0.75rem; padding: 0.75rem; background: var(--bg-secondary); border-radius: 8px; border-left: 3px solid {status_color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                    <span style="font-weight: 600; color: {status_color};">{status_icon} {h_id}: {status.upper() if status else 'PENDING'}</span>
                    <span style="font-size: 0.8rem; color: var(--text-secondary);">
                        {f'{prior}%' if prior is not None else '?'}
                        →
                        {f'{posterior}%' if posterior is not None else '?'}
                        {f'<span style="color: {shift_color};">({shift_str})</span>' if shift_str else ''}
                    </span>
                </div>
                <div style="font-size: 0.85rem; color: var(--text-secondary);">{hypothesis}...</div>
            </div>
            """)

        return "\n".join(html_parts)

    # Pre-generate hypothesis timeline
    hypothesis_html = hypothesis_timeline_html(hypothesis_timeline)
    has_hypotheses = bool(hypothesis_timeline.get("hypotheses"))

    # Function type badge
    ft = sc.get("function_type", "unknown")
    ft_colors = {
        "semantic": ("green", "#10b981"),
        "routing": ("blue", "#3b82f6"),
        "syntactic": ("yellow", "#f59e0b"),
        "formatting": ("purple", "#8b5cf6"),
        "hybrid": ("pink", "#ec4899"),
    }
    ft_class, ft_color = ft_colors.get(ft, ("blue", "#3b82f6"))

    # Confidence bar color
    conf = sc.get("confidence", 0)
    conf_pct = int(conf * 100)
    conf_color = "#10b981" if conf >= 0.7 else "#f59e0b" if conf >= 0.4 else "#ef4444"

    # Pre-generate detailed experiments HTML
    ablation_html = ablation_details_html(detailed_exp.get("ablation", []))
    steering_html = steering_details_html(detailed_exp.get("steering", []))
    relp_html = relp_results_html(relp_analysis)
    has_detailed_exp = bool(detailed_exp.get("ablation") or detailed_exp.get("steering"))
    has_relp = bool(relp_analysis.get("results"))

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neuron Investigation: {esc(dashboard.get("neuron_id", ""))}</title>
    <style>
        :root {{
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #1f2937;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-yellow: #f59e0b;
            --accent-purple: #8b5cf6;
            --border-color: #374151;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem; }}
        .header {{
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border-color);
        }}
        .header h1 {{ font-size: 2rem; }}
        .header .neuron-id {{ color: var(--accent-blue); font-family: monospace; }}
        .header .meta {{ text-align: right; color: var(--text-secondary); font-size: 0.875rem; }}
        .card {{
            background: var(--bg-card); border-radius: 12px; padding: 1.5rem;
            margin-bottom: 1.5rem; border: 1px solid var(--border-color);
        }}
        .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }}
        .card-title {{ font-size: 1.125rem; font-weight: 600; }}
        .badge {{
            padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 500;
        }}
        .badge-green {{ background: rgba(16, 185, 129, 0.2); color: var(--accent-green); }}
        .badge-blue {{ background: rgba(59, 130, 246, 0.2); color: var(--accent-blue); }}
        .badge-yellow {{ background: rgba(245, 158, 11, 0.2); color: var(--accent-yellow); }}
        .badge-red {{ background: rgba(239, 68, 68, 0.2); color: var(--accent-red); }}
        .badge-purple {{ background: rgba(139, 92, 246, 0.2); color: #a78bfa; }}
        .grid {{ display: grid; gap: 1.5rem; }}
        .grid-2 {{ grid-template-columns: repeat(2, 1fr); }}
        .grid-3 {{ grid-template-columns: repeat(3, 1fr); }}
        @media (max-width: 900px) {{ .grid-2, .grid-3 {{ grid-template-columns: 1fr; }} }}
        .summary-box {{
            background: var(--bg-secondary); padding: 1rem; border-radius: 8px;
            font-size: 1.1rem; margin-bottom: 1rem;
        }}
        .function-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
        .function-box {{ background: var(--bg-secondary); padding: 1rem; border-radius: 8px; }}
        .function-label {{ font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 0.5rem; }}
        .function-value {{ font-size: 0.95rem; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1rem; }}
        .stat-box {{ text-align: center; padding: 1rem; background: var(--bg-secondary); border-radius: 8px; }}
        .stat-value {{ font-size: 1.5rem; font-weight: 700; color: var(--accent-blue); }}
        .stat-label {{ font-size: 0.75rem; color: var(--text-secondary); }}
        .confidence-bar {{ height: 8px; background: var(--bg-secondary); border-radius: 4px; overflow: hidden; margin-top: 0.5rem; }}
        .confidence-fill {{ height: 100%; border-radius: 4px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; table-layout: auto; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--border-color); vertical-align: top; }}
        th {{ color: var(--text-secondary); font-weight: 500; }}
        td:first-child {{ min-width: 200px; max-width: 600px; word-wrap: break-word; }}
        ul {{ list-style: none; }}
        ul li {{ padding: 0.5rem 0; border-bottom: 1px solid var(--border-color); line-height: 1.6; }}
        ul li:last-child {{ border-bottom: none; }}
        code {{ background: var(--bg-secondary); padding: 0.2rem 0.4rem; border-radius: 4px; font-size: 0.85em; }}
        a {{ color: var(--accent-blue); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>Neuron Investigation: <span class="neuron-id">{esc(dashboard.get("neuron_id", ""))}</span></h1>
                <div style="color: var(--text-secondary); margin-top: 0.5rem;">
                    Layer {dashboard.get("layer", 0)} / Index {dashboard.get("neuron_idx", 0)}
                </div>
            </div>
            <div class="meta">
                <div>{esc(meta.get("timestamp", "")[:10])}</div>
                <div style="margin-top: 0.5rem;">
                    <span class="badge" style="background: rgba({ft_color.replace('#', '')[:2]}, 0.2); color: {ft_color};">{ft}</span>
                </div>
            </div>
        </div>

        <!-- Summary Card -->
        <div class="card">
            <div class="card-header">
                <span class="card-title">Summary</span>
                <span class="badge badge-blue">{conf_pct}% Confidence</span>
            </div>
            <div class="summary-box">{esc(sc.get("summary", "No summary available"))}</div>
            <div class="function-grid">
                <div class="function-box">
                    <div class="function-label">Input Function (What activates it)</div>
                    <div class="function-value">{esc(sc.get("input_function", "Not determined"))}</div>
                </div>
                <div class="function-box">
                    <div class="function-label">Output Function (What it promotes)</div>
                    <div class="function-value">{esc(sc.get("output_function", "Not determined"))}</div>
                </div>
            </div>
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-value">{stats.get("activating_count", 0)}</div>
                    <div class="stat-label">Activating Prompts</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats.get("non_activating_count", 0)}</div>
                    <div class="stat-label">Non-Activating</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats.get("ablation_count", 0)}</div>
                    <div class="stat-label">Ablation Tests</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{sc.get("total_experiments", 0)}</div>
                    <div class="stat-label">Total Experiments</div>
                </div>
            </div>
            <div style="margin-top: 1rem;">
                <div class="function-label">Confidence</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {conf_pct}%; background: {conf_color};"></div>
                </div>
            </div>
        </div>

        <!-- Original Labels & Prior Analysis -->
        {f'''<div class="card">
            <div class="card-header">
                <span class="card-title">Prior Analysis (Claims Tested)</span>
                <span class="badge badge-yellow">Original Labels</span>
            </div>
            <div class="function-grid">
                <div class="function-box">
                    <div class="function-label">Original Output Label</div>
                    <div class="function-value">{esc(sc.get("original_output_label", "Not provided"))}</div>
                    {f'<div style="margin-top: 0.5rem; font-size: 0.85rem; color: var(--text-secondary);">{esc(sc.get("original_output_description", ""))}</div>' if sc.get("original_output_description") else ""}
                </div>
                <div class="function-box">
                    <div class="function-label">Original Input Label</div>
                    <div class="function-value">{esc(sc.get("original_input_label", "Not provided"))}</div>
                    {f'<div style="margin-top: 0.5rem; font-size: 0.85rem; color: var(--text-secondary);">{esc(sc.get("original_input_description", ""))}</div>' if sc.get("original_input_description") else ""}
                </div>
            </div>
            <div class="stat-grid" style="grid-template-columns: repeat(3, 1fr); margin-top: 1rem;">
                <div class="stat-box">
                    <div class="stat-value">{sc.get("direct_effect_ratio", 0):.1%}</div>
                    <div class="stat-label">Direct Effect Ratio</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{len(output_proj.get("promote", []))}</div>
                    <div class="stat-label">Output Projections +</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{len(output_proj.get("suppress", []))}</div>
                    <div class="stat-label">Output Projections −</div>
                </div>
            </div>
            {f"""<div style="margin-top: 1rem;">
                <div class="function-label">Output Tokens Promoted <span style="font-weight: normal; font-size: 0.8rem; color: var(--text-secondary);">(frequency in top-k when active)</span></div>
                <div style="margin-top: 0.5rem;">
                    <table style="width: 100%; font-size: 0.85rem;">
                        <tr style="color: var(--text-secondary); border-bottom: 1px solid var(--border-color);">
                            <th style="text-align: left; padding: 0.25rem;">Token</th>
                            <th style="text-align: right; padding: 0.25rem;">Frequency</th>
                            <th style="text-align: right; padding: 0.25rem;">Count</th>
                        </tr>
                        {"".join(f'<tr><td style="padding: 0.25rem;"><code>{esc(t.get("token", t) if isinstance(t, dict) else t)}</code></td><td style="text-align: right; padding: 0.25rem; color: var(--accent-green);">{t.get("frequency", 0):.2%}</td><td style="text-align: right; padding: 0.25rem; color: var(--text-secondary);">{t.get("count", 0)}</td></tr>' if isinstance(t, dict) else f'<tr><td style="padding: 0.25rem;"><code>{esc(t)}</code></td><td style="text-align: right; padding: 0.25rem;">-</td><td style="text-align: right; padding: 0.25rem;">-</td></tr>' for t in output_proj.get("promote", [])[:5])}
                    </table>
                </div>
                {f'''<div class="function-label" style="margin-top: 1rem;">Output Tokens Suppressed <span style="font-weight: normal; font-size: 0.8rem; color: var(--text-secondary);">(frequency in top-k when active)</span></div>
                <div style="margin-top: 0.5rem;">
                    <table style="width: 100%; font-size: 0.85rem;">
                        <tr style="color: var(--text-secondary); border-bottom: 1px solid var(--border-color);">
                            <th style="text-align: left; padding: 0.25rem;">Token</th>
                            <th style="text-align: right; padding: 0.25rem;">Frequency</th>
                            <th style="text-align: right; padding: 0.25rem;">Count</th>
                        </tr>
                        {"".join(f'<tr><td style="padding: 0.25rem;"><code>{esc(t.get("token", t) if isinstance(t, dict) else t)}</code></td><td style="text-align: right; padding: 0.25rem; color: var(--accent-red);">{t.get("frequency", 0):.2%}</td><td style="text-align: right; padding: 0.25rem; color: var(--text-secondary);">{t.get("count", 0)}</td></tr>' if isinstance(t, dict) else f'<tr><td style="padding: 0.25rem;"><code>{esc(t)}</code></td><td style="text-align: right; padding: 0.25rem;">-</td><td style="text-align: right; padding: 0.25rem;">-</td></tr>' for t in output_proj.get("suppress", [])[:5])}
                    </table>
                </div>''' if output_proj.get("suppress") else ""}
            </div>""" if output_proj.get("promote") or output_proj.get("suppress") else ""}
        </div>''' if sc.get("original_output_label") or sc.get("original_input_label") or sc.get("direct_effect_ratio", 0) > 0 or output_proj.get("promote") or output_proj.get("suppress") else ""}

        <!-- Activation Patterns -->
        <div class="grid grid-2">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Activating Prompts</span>
                    <span class="badge badge-green">Positive Examples</span>
                </div>
                <table>
                    <thead><tr><th>Prompt</th><th>Activation</th><th>Token</th></tr></thead>
                    <tbody>{activation_rows(ap.get("positive_examples", []), True, limit=20)}</tbody>
                </table>
                <div style="margin-top: 0.5rem; font-size: 0.875rem; color: var(--text-secondary);">
                    Showing {min(len(ap.get("positive_examples", [])), 20)} of {len(ap.get("positive_examples", []))} activating examples
                </div>
            </div>
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Non-Activating Prompts</span>
                    <span class="badge badge-red">Negative Controls</span>
                </div>
                <table>
                    <thead><tr><th>Prompt</th><th>Activation</th><th>Token</th></tr></thead>
                    <tbody>{activation_rows(ap.get("negative_examples", []), False, limit=15)}</tbody>
                </table>
                <div style="margin-top: 0.5rem; font-size: 0.875rem; color: var(--text-secondary);">
                    Showing {min(len(ap.get("negative_examples", [])), 15)} of {len(ap.get("negative_examples", []))} control examples
                </div>
            </div>
        </div>

        <!-- Ablation Effects -->
        <div class="grid grid-2">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Tokens Promoted</span>
                    <span class="badge badge-green">Ablation: ↑ when active</span>
                </div>
                <ul>{ablation_items(ae.get("consistent_promotes", []), "promotes")}</ul>
            </div>
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Tokens Suppressed</span>
                    <span class="badge badge-red">Ablation: ↓ when active</span>
                </div>
                <ul>{ablation_items(ae.get("consistent_suppresses", []), "suppresses")}</ul>
            </div>
        </div>

        <!-- Connectivity -->
        <div class="grid grid-2">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Upstream Neurons</span>
                    <span class="badge badge-blue">Inputs</span>
                </div>
                <ul>{connectivity_items(conn.get("upstream", []))}</ul>
                <div style="margin-top: 0.5rem; font-size: 0.875rem; color: var(--text-secondary);">
                    Showing {len(conn.get("upstream", []))} upstream connections
                </div>
            </div>
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Downstream Neurons</span>
                    <span class="badge badge-purple">Outputs</span>
                </div>
                <ul>{connectivity_items(conn.get("downstream", []))}</ul>
                <div style="margin-top: 0.5rem; font-size: 0.875rem; color: var(--text-secondary);">
                    Showing {len(conn.get("downstream", []))} downstream connections • Green (+) = promotes, Red (−) = suppresses
                </div>
            </div>
        </div>

        <!-- Detailed Experimental Results -->
        {f'''<div class="grid grid-2">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Ablation Details</span>
                    <span class="badge badge-red">Causal Interventions</span>
                </div>
                {ablation_html}
            </div>
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Steering Details</span>
                    <span class="badge badge-blue">Amplification Tests</span>
                </div>
                {steering_html}
            </div>
        </div>''' if has_detailed_exp else ""}

        {f'''<!-- RelP Attribution Analysis -->
        <div class="card">
            <div class="card-header">
                <span class="card-title">RelP Attribution Analysis</span>
                <span class="badge badge-purple">Causal Pathways</span>
            </div>
            {relp_html}
        </div>''' if has_relp else ""}

        {f'''<!-- Hypothesis Testing Timeline -->
        <div class="card">
            <div class="card-header">
                <span class="card-title">Hypothesis Testing Timeline</span>
                <span class="badge badge-purple">Pre-Registered</span>
            </div>
            <div style="padding: 0.5rem;">
                <div style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 1rem;">
                    {hypothesis_timeline.get("pre_registration_count", 0)} hypotheses registered and tested
                </div>
                {hypothesis_html}
            </div>
        </div>''' if has_hypotheses else ""}

        <!-- Key Findings -->
        <div class="grid grid-2">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Key Findings</span>
                    <span class="badge badge-green">Evidence</span>
                </div>
                <ul>{findings_list(findings.get("key_findings", []))}</ul>
            </div>
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Open Questions</span>
                    <span class="badge badge-yellow">Future Work</span>
                </div>
                <ul>{findings_list(findings.get("open_questions", []))}</ul>
            </div>
        </div>

        <p style="text-align: center; color: var(--text-secondary); margin-top: 2rem;">
            <a href="index.html">← Back to all investigations</a>
        </p>
    </div>
</body>
</html>'''


def main():
    """Generate HTML dashboards for all investigations."""
    out_dir = Path("frontend/dashboards")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort by modification time (newest first) to prefer latest version of duplicates
    all_dash_paths = list(Path("outputs/investigations").glob("*_dashboard.json"))
    all_dash_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    dashboards = []
    seen_neurons = set()  # Track neuron_ids to skip duplicates
    skipped = []

    for dash_path in all_dash_paths:
        # Skip test files
        if "TEST" in dash_path.name:
            skipped.append(f"{dash_path.name} (test file)")
            continue

        with open(dash_path) as f:
            dash = json.load(f)

        neuron_id = dash.get("neuron_id", "")

        # Skip if we've already processed this neuron (prefer newer files)
        if neuron_id in seen_neurons:
            skipped.append(f"{dash_path.name} (duplicate of {neuron_id})")
            continue

        seen_neurons.add(neuron_id)
        safe_id = neuron_id.replace("/", "_")

        # Generate HTML
        html_content = generate_dashboard_html(dash)

        # Save
        out_path = out_dir / f"{safe_id}.html"
        out_path.write_text(html_content)
        print(f"Created: {out_path.name}")

        dashboards.append({
            "neuron_id": neuron_id,
            "summary": dash.get("summary_card", {}).get("summary", ""),
            "function_type": dash.get("summary_card", {}).get("function_type", "unknown"),
            "confidence": dash.get("summary_card", {}).get("confidence", 0),
            "file": f"{safe_id}.html"
        })

    # Generate index
    generate_index(dashboards, out_dir)
    print(f"\nGenerated {len(dashboards)} dashboards")
    if skipped:
        print(f"Skipped {len(skipped)} files: {', '.join(skipped[:5])}{'...' if len(skipped) > 5 else ''}")
    print(f"View at: file://{out_dir.absolute()}/index.html")


def generate_index(dashboards, out_dir):
    """Generate index page."""
    items = []
    for d in dashboards:
        ft = d["function_type"] or "unknown"
        conf = int(d["confidence"] * 100)
        ft_colors = {
            "semantic": "#10b981", "routing": "#3b82f6", "syntactic": "#f59e0b",
            "formatting": "#8b5cf6", "hybrid": "#ec4899"
        }
        color = ft_colors.get(ft, "#3b82f6")
        summary_text = d['summary'][:200] + "..." if len(d['summary']) > 200 else d['summary']
        items.append(f'''
        <a href="{d['file']}" class="card">
            <div class="neuron-id">{d['neuron_id']}</div>
            <div class="meta">
                <span class="badge" style="background: {color}22; color: {color};">{ft}</span>
                <span>{conf}% confidence</span>
            </div>
            <div class="summary">{summary_text}</div>
        </a>''')

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Neuron Investigations</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; background: #1a1a2e; color: #e2e8f0; padding: 2rem; }}
        h1 {{ margin-bottom: 0.5rem; }}
        .subtitle {{ color: #94a3b8; margin-bottom: 2rem; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 1rem; }}
        .card {{ background: #1f2937; padding: 1.5rem; border-radius: 8px; border: 1px solid #374151; text-decoration: none; color: inherit; display: block; transition: border-color 0.2s; }}
        .card:hover {{ border-color: #3b82f6; }}
        .neuron-id {{ color: #3b82f6; font-size: 1.25rem; font-weight: 600; font-family: monospace; }}
        .meta {{ color: #94a3b8; font-size: 0.875rem; margin: 0.5rem 0; display: flex; gap: 1rem; align-items: center; }}
        .badge {{ padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; }}
        .summary {{ color: #cbd5e1; font-size: 0.9rem; line-height: 1.5; }}
    </style>
</head>
<body>
    <h1>Neuron Investigations</h1>
    <div class="subtitle">{len(dashboards)} neurons investigated</div>
    <div class="grid">{"".join(items)}</div>
</body>
</html>'''

    (out_dir / "index.html").write_text(html)


if __name__ == "__main__":
    main()
