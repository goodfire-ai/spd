"""Generate L1 attention clustering HTML report from a cluster mapping JSON."""

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

HARVEST_SUMMARY = "/mnt/polished-lake/artifacts/mechanisms/spd/harvest/s-275c8f21/activation_contexts/summary.json"
INTERP_DB = "/mnt/polished-lake/artifacts/mechanisms/spd/autointerp/s-275c8f21/a-20260212_150336/interp.db"
LAYER1_ATTN_MODULES = ["h.1.attn.k_proj", "h.1.attn.q_proj", "h.1.attn.v_proj", "h.1.attn.o_proj"]


def ci_color(ci: float) -> str:
    return f"rgb({int(min(255, ci * 510))}, 40, {int(min(255, (1 - ci) * 510))})"


CONF_COLORS = {"high": "#22c55e", "medium": "#eab308", "low": "#ef4444"}
L1_COLORS = {
    "h.1.attn.k_proj": "#3b82f6",
    "h.1.attn.q_proj": "#8b5cf6",
    "h.1.attn.v_proj": "#ec4899",
    "h.1.attn.o_proj": "#f97316",
}


def confidence_badge(conf: str) -> str:
    color = CONF_COLORS.get(conf, "#888")
    return f'<span style="background:{color};color:white;padding:1px 6px;border-radius:3px;font-size:11px">{conf}</span>'


def module_badge(module: str) -> str:
    color = "#64748b"
    for p, c in L1_COLORS.items():
        if module.startswith(p):
            color = c
            break
    short = module.replace("h.", "L").replace(".attn.", ".").replace(".mlp.", ".mlp.")
    return f'<span style="background:{color};color:white;padding:1px 6px;border-radius:3px;font-size:11px;font-family:monospace">{short}</span>'


def generate_report(cluster_mapping_path: str, output_path: str) -> None:
    cluster_mapping_path = Path(cluster_mapping_path)
    output_path = Path(output_path)

    with open(cluster_mapping_path) as f:
        cluster_data = json.load(f)
    clusters = cluster_data["clusters"]
    iteration = cluster_data["n_iterations"]
    run_idx = cluster_data["run_idx"]
    notes = cluster_data.get("notes", "")

    with open(HARVEST_SUMMARY) as f:
        harvest = json.load(f)

    conn = sqlite3.connect(INTERP_DB)
    interp = {
        r[0]: {"label": r[1], "confidence": r[2]}
        for r in conn.execute("SELECT component_key, label, confidence FROM interpretations").fetchall()
    }
    conn.close()

    layer1_attn_keys = [
        k for k in clusters if any(k.startswith(m + ":") for m in LAYER1_ATTN_MODULES)
    ]

    components = []
    for key in layer1_attn_keys:
        ci = harvest.get(key, {}).get("mean_ci", 0.0)
        module, idx = key.rsplit(":", 1)
        label_info = interp.get(key, {"label": "\u2014", "confidence": "\u2014"})
        components.append({
            "key": key, "module": module, "idx": int(idx), "mean_ci": ci,
            "cluster_id": clusters.get(key),
            "interp_label": label_info["label"], "interp_confidence": label_info["confidence"],
        })
    components.sort(key=lambda c: c["mean_ci"], reverse=True)

    top_n = 50
    top_components = components[:top_n]
    relevant_cluster_ids = {c["cluster_id"] for c in top_components if c["cluster_id"] is not None}

    cluster_to_members: dict[int, list] = defaultdict(list)
    for key, cid in clusters.items():
        if cid in relevant_cluster_ids:
            ci = harvest.get(key, {}).get("mean_ci", 0.0)
            label_info = interp.get(key, {"label": "\u2014", "confidence": "\u2014"})
            module, idx = key.rsplit(":", 1)
            cluster_to_members[cid].append({
                "key": key, "module": module, "idx": int(idx), "mean_ci": ci,
                "interp_label": label_info["label"], "interp_confidence": label_info["confidence"],
                "is_layer1_attn": any(key.startswith(m + ":") for m in LAYER1_ATTN_MODULES),
            })

    for cid in cluster_to_members:
        cluster_to_members[cid].sort(key=lambda m: m["mean_ci"], reverse=True)

    sorted_cluster_ids = sorted(
        relevant_cluster_ids,
        key=lambda cid: max(
            (m["mean_ci"] for m in cluster_to_members[cid] if m["is_layer1_attn"]), default=0
        ),
        reverse=True,
    )

    total_l1_attn = len(layer1_attn_keys)
    n_with_clusters = sum(1 for c in components if c["cluster_id"] is not None)

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Clustering Report: Layer 1 Attention \u2014 iter {iteration}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f8fafc; color: #1e293b; }}
  h1 {{ border-bottom: 3px solid #3b82f6; padding-bottom: 8px; }}
  h2 {{ color: #1e40af; margin-top: 32px; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 16px 0; }}
  .stat {{ background: white; padding: 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .stat-value {{ font-size: 28px; font-weight: 700; color: #1e40af; }}
  .stat-label {{ font-size: 13px; color: #64748b; margin-top: 4px; }}
  .cluster-card {{ background: white; border-radius: 8px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: hidden; }}
  .cluster-header {{ background: #1e293b; color: white; padding: 12px 16px; display: flex; justify-content: space-between; align-items: center; }}
  .cluster-header h3 {{ margin: 0; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ background: #f1f5f9; text-align: left; padding: 8px 12px; font-size: 13px; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
  td {{ padding: 8px 12px; border-top: 1px solid #e2e8f0; font-size: 14px; }}
  tr.highlight {{ background: #eff6ff; }}
  .ci-bar {{ display: inline-block; height: 8px; border-radius: 4px; min-width: 2px; }}
  .top-table {{ margin: 16px 0; }}
  .top-table table {{ background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .meta {{ color: #64748b; font-size: 13px; margin: 8px 0; }}
</style></head><body>
<h1>Clustering Report: Layer 1 Attention Block</h1>
<p class="meta">SPD run: <strong>s-275c8f21</strong> (Pile LlamaSimpleMLP 4L) &nbsp;|&nbsp;
{notes} &nbsp;|&nbsp;
Iteration: <strong>{iteration}</strong></p>
<div class="stats">
  <div class="stat"><div class="stat-value">{total_l1_attn}</div><div class="stat-label">L1 Attn Components</div></div>
  <div class="stat"><div class="stat-value">{n_with_clusters}</div><div class="stat-label">In Multi-Member Clusters</div></div>
  <div class="stat"><div class="stat-value">{len(relevant_cluster_ids)}</div><div class="stat-label">Clusters (from top {top_n})</div></div>
  <div class="stat"><div class="stat-value">{components[0]["mean_ci"]:.4f}</div><div class="stat-label">Max Mean CI (L1 Attn)</div></div>
</div>
<h2>Top {top_n} Layer 1 Attention Components by Mean CI</h2>
<div class="top-table"><table>
<tr><th>#</th><th>Component</th><th>Module</th><th>Mean CI</th><th></th><th>Cluster</th><th>Interpretation</th></tr>""")

    for i, c in enumerate(top_components):
        bar_width = max(2, int(c["mean_ci"] * 200))
        cluster_str = (
            f'#{c["cluster_id"]}' if c["cluster_id"] is not None
            else '<span style="color:#94a3b8">singleton</span>'
        )
        parts.append(
            f'<tr><td>{i+1}</td>'
            f'<td style="font-family:monospace;font-size:13px">{c["key"]}</td>'
            f'<td>{module_badge(c["module"])}</td>'
            f'<td style="font-weight:600;color:{ci_color(c["mean_ci"])}">{c["mean_ci"]:.4f}</td>'
            f'<td><span class="ci-bar" style="width:{bar_width}px;background:{ci_color(c["mean_ci"])}"></span></td>'
            f'<td>{cluster_str}</td>'
            f'<td>{c["interp_label"]} {confidence_badge(c["interp_confidence"])}</td></tr>'
        )

    parts.append(
        f'</table></div>'
        f'<h2>Cluster Details ({len(sorted_cluster_ids)} clusters containing top L1 Attn components)</h2>'
    )

    for cid in sorted_cluster_ids:
        members = cluster_to_members[cid]
        n_l1 = sum(1 for m in members if m["is_layer1_attn"])
        n_other = len(members) - n_l1
        max_ci = max(m["mean_ci"] for m in members)
        module_badges = " ".join(module_badge(mod) for mod in sorted(set(m["module"] for m in members)))
        parts.append(
            f'<div class="cluster-card">'
            f'<div class="cluster-header">'
            f'<h3>Cluster #{cid} &nbsp; ({len(members)} members: {n_l1} L1-attn, {n_other} other)</h3>'
            f'<span style="font-size:13px">max CI: {max_ci:.4f}</span></div>'
            f'<div style="padding:8px 16px;background:#f8fafc;font-size:13px">Modules: {module_badges}</div>'
            f'<table><tr><th>Component</th><th>Module</th><th>Mean CI</th><th></th><th>Interpretation</th></tr>'
        )
        for m in members:
            bar_width = max(2, int(m["mean_ci"] * 200))
            highlight = ' class="highlight"' if m["is_layer1_attn"] else ""
            parts.append(
                f'<tr{highlight}>'
                f'<td style="font-family:monospace;font-size:13px">{m["key"]}</td>'
                f'<td>{module_badge(m["module"])}</td>'
                f'<td style="font-weight:600;color:{ci_color(m["mean_ci"])}">{m["mean_ci"]:.4f}</td>'
                f'<td><span class="ci-bar" style="width:{bar_width}px;background:{ci_color(m["mean_ci"])}"></span></td>'
                f'<td>{m["interp_label"]} {confidence_badge(m["interp_confidence"])}</td></tr>'
            )
        parts.append("</table></div>")

    parts.append("</body></html>")
    output_path.write_text("\n".join(parts))
    print(f"Wrote {output_path}")
    print(f"  {n_with_clusters} clustered / {total_l1_attn} total, {len(relevant_cluster_ids)} clusters, {sum(len(cluster_to_members[c]) for c in sorted_cluster_ids)} members")


if __name__ == "__main__":
    generate_report(sys.argv[1], sys.argv[2])
