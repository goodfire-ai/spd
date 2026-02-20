#!/usr/bin/env python3
"""Consolidate scattered neuron reports into canonical directory structure.

This script:
1. Scans all known locations for neuron report HTMLs and JSONs
2. Moves active files to neuron_reports/{html,json}/
3. Archives everything else to archives/
4. Generates an index.html for browsing reports

Usage:
    # Dry run (default) - show what would happen
    python scripts/consolidate_neuron_reports.py

    # Actually perform consolidation
    python scripts/consolidate_neuron_reports.py --execute

    # Only archive without consolidating (if structure already clean)
    python scripts/consolidate_neuron_reports.py --archive-only --execute
"""

import argparse
import json
import re
import shutil
import tarfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# Canonical output directory
CANONICAL_DIR = Path("neuron_reports")
CANONICAL_HTML = CANONICAL_DIR / "html"  # V2 dashboards with creative titles - NEVER OVERWRITE
CANONICAL_JSON = CANONICAL_DIR / "json"
CANONICAL_INVESTIGATIONS = CANONICAL_DIR / "investigations"  # Simple investigation reports

# Archive directory
ARCHIVE_DIR = Path("archives")

# Known locations to scan for JSON files ONLY
# NOTE: HTML files in neuron_reports/html/ are managed by dashboard_agent_v2 and should
# NEVER be overwritten by this script. The consolidation only handles JSON files.
SCAN_LOCATIONS_JSON = [
    Path("outputs/investigations"),
    Path("outputs/investigations_v2"),
    Path("outputs/investigations_v2_comparison"),
    Path("outputs/investigations_v2_connectivity_test"),
    Path("outputs/investigations_v2_debug"),
    Path("outputs/investigations_v2_final_test"),
    Path("outputs/investigations_v2_fixed"),
    Path("outputs/investigations_v2_registry_test"),
    Path("outputs/test_fix"),
    Path("outputs/test_new_flow"),
    Path("outputs/pi_test"),
]

# Legacy HTML locations to archive (NOT copy from)
LEGACY_HTML_LOCATIONS = [
    Path("frontend/dashboards"),
    Path("frontend/reports"),
    Path("frontend/reports/aspirin-cox"),
    Path("frontend/reports/aspirin-cox-backup-20260117"),
    Path("reports"),
]

# SLURM output directories (to archive)
SLURM_PATTERN = re.compile(r"slurm_\d{8}_\d{6}|slurm_neuronpi_\d{8}_\d{6}")

# Neuron ID pattern
NEURON_PATTERN = re.compile(r"L(\d+)[_/]?N(\d+)")


# =============================================================================
# File Discovery
# =============================================================================

def extract_neuron_id(filename: str) -> tuple[int, int] | None:
    """Extract (layer, neuron_idx) from filename."""
    match = NEURON_PATTERN.search(filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def get_file_type(filename: str) -> str | None:
    """Determine file type from filename pattern."""
    name = filename.lower()
    if "_investigation" in name and name.endswith(".json"):
        return "investigation"
    elif "_dashboard" in name and name.endswith(".json"):
        return "dashboard"
    elif "_pi_result" in name and name.endswith(".json"):
        return "pi_result"
    elif name.endswith(".html"):
        return "html"
    elif name.endswith(".json"):
        return "other_json"
    return None


def scan_for_reports() -> dict:
    """Scan known locations for neuron report JSON files.

    NOTE: This only scans for JSON files. HTML dashboards in neuron_reports/html/
    are managed separately by dashboard_agent_v2 and should NEVER be overwritten.

    Returns:
        dict with structure:
        {
            "L5_N1234": {
                "investigation": [(path, mtime), ...],
                "dashboard": [(path, mtime), ...],
                "pi_result": [(path, mtime), ...],
            },
            ...
        }
    """
    reports = defaultdict(lambda: defaultdict(list))
    seen_paths = set()

    # Scan JSON locations only
    for location in SCAN_LOCATIONS_JSON:
        if not location.exists():
            continue

        # Only scan for JSON files - HTML is managed separately
        for path in location.glob("*.json"):
            if path in seen_paths or not path.is_file():
                continue
            seen_paths.add(path)

            neuron_id = extract_neuron_id(path.name)
            if not neuron_id:
                continue

            file_type = get_file_type(path.name)
            if not file_type or file_type == "html":
                continue

            layer, neuron_idx = neuron_id
            key = f"L{layer}_N{neuron_idx}"
            mtime = path.stat().st_mtime
            reports[key][file_type].append((path, mtime))

    # Also scan SLURM directories for JSON files
    inv_dir = Path("outputs/investigations")
    if inv_dir.exists():
        for subdir in inv_dir.iterdir():
            if subdir.is_dir() and SLURM_PATTERN.match(subdir.name):
                for path in subdir.glob("*.json"):
                    if path in seen_paths:
                        continue
                    seen_paths.add(path)

                    neuron_id = extract_neuron_id(path.name)
                    if not neuron_id:
                        continue

                    file_type = get_file_type(path.name)
                    if not file_type or file_type == "html":
                        continue

                    layer, neuron_idx = neuron_id
                    key = f"L{layer}_N{neuron_idx}"
                    mtime = path.stat().st_mtime
                    reports[key][file_type].append((path, mtime))

    # Also include files already in canonical JSON location
    if CANONICAL_JSON.exists():
        for path in CANONICAL_JSON.glob("*.json"):
            if path in seen_paths or not path.is_file():
                continue
            seen_paths.add(path)

            neuron_id = extract_neuron_id(path.name)
            if not neuron_id:
                continue

            file_type = get_file_type(path.name)
            if not file_type:
                continue

            layer, neuron_idx = neuron_id
            key = f"L{layer}_N{neuron_idx}"
            mtime = path.stat().st_mtime
            reports[key][file_type].append((path, mtime))

    return dict(reports)


def find_stray_files() -> list[Path]:
    """Find neuron-related files in unexpected locations."""
    stray = []

    # Root level HTML files
    for path in Path(".").glob("*.html"):
        if extract_neuron_id(path.name):
            stray.append(path)

    return stray


def find_slurm_dirs() -> list[Path]:
    """Find SLURM output directories to archive."""
    dirs = []
    inv_dir = Path("outputs/investigations")
    if inv_dir.exists():
        for subdir in inv_dir.iterdir():
            if subdir.is_dir() and SLURM_PATTERN.match(subdir.name):
                dirs.append(subdir)
    return dirs


def find_backup_dirs() -> list[Path]:
    """Find backup directories to archive."""
    backups = []
    reports_dir = Path("frontend/reports")
    if reports_dir.exists():
        for subdir in reports_dir.iterdir():
            if subdir.is_dir() and "backup" in subdir.name.lower():
                backups.append(subdir)
    return backups


def find_legacy_html_dirs() -> list[Path]:
    """Find legacy HTML directories that should be archived.

    These are old locations where HTML was generated before we moved to
    the canonical neuron_reports/html/ structure.
    """
    legacy = []
    for location in LEGACY_HTML_LOCATIONS:
        if location.exists() and location.is_dir():
            # Only include if it has HTML files
            html_files = list(location.glob("*.html"))
            if html_files:
                legacy.append(location)
    return legacy


# =============================================================================
# Consolidation
# =============================================================================

def select_canonical_file(files: list[tuple[Path, float]]) -> tuple[Path, list[Path]]:
    """Select the newest file as canonical, return (canonical, others_to_archive)."""
    if not files:
        return None, []

    # Sort by mtime descending (newest first)
    sorted_files = sorted(files, key=lambda x: -x[1])
    canonical = sorted_files[0][0]
    others = [f[0] for f in sorted_files[1:]]

    return canonical, others


def consolidate_reports(reports: dict, execute: bool = False) -> dict:
    """Consolidate JSON reports to canonical locations.

    NOTE: This function ONLY handles JSON files. HTML dashboards in neuron_reports/html/
    are managed by dashboard_agent_v2 and should NEVER be touched by this script.

    Returns summary of actions taken.
    """
    actions = {
        "moved_json": [],
        "archived": [],
        "skipped": [],
    }

    if execute:
        CANONICAL_JSON.mkdir(parents=True, exist_ok=True)

    for neuron_key, file_types in sorted(reports.items()):
        # NOTE: We do NOT process HTML here. HTML dashboards are managed separately
        # by dashboard_agent_v2 and stored in neuron_reports/html/

        # Process investigation JSON
        if "investigation" in file_types:
            canonical, others = select_canonical_file(file_types["investigation"])
            if canonical:
                dest = CANONICAL_JSON / f"{neuron_key}_investigation.json"
                if canonical != dest:
                    actions["moved_json"].append((canonical, dest))
                    if execute:
                        shutil.copy2(canonical, dest)
                for other in others:
                    actions["archived"].append(other)

        # Process dashboard JSON
        if "dashboard" in file_types:
            canonical, others = select_canonical_file(file_types["dashboard"])
            if canonical:
                dest = CANONICAL_JSON / f"{neuron_key}_dashboard.json"
                if canonical != dest:
                    actions["moved_json"].append((canonical, dest))
                    if execute:
                        shutil.copy2(canonical, dest)
                for other in others:
                    actions["archived"].append(other)

        # Process pi_result JSON
        if "pi_result" in file_types:
            canonical, others = select_canonical_file(file_types["pi_result"])
            if canonical:
                dest = CANONICAL_JSON / f"{neuron_key}_pi_result.json"
                if canonical != dest:
                    actions["moved_json"].append((canonical, dest))
                    if execute:
                        shutil.copy2(canonical, dest)
                for other in others:
                    actions["archived"].append(other)

    return actions


# =============================================================================
# Archiving
# =============================================================================

def archive_files(files: list[Path], archive_name: str, execute: bool = False) -> Path | None:
    """Archive files to a tarball."""
    if not files:
        return None

    if execute:
        ARCHIVE_DIR.mkdir(exist_ok=True)
        archive_path = ARCHIVE_DIR / f"{archive_name}.tar.gz"

        with tarfile.open(archive_path, "w:gz") as tar:
            for f in files:
                if f.exists():
                    tar.add(f, arcname=f.name)

        return archive_path

    return ARCHIVE_DIR / f"{archive_name}.tar.gz"


def archive_directory(directory: Path, archive_name: str, execute: bool = False) -> Path | None:
    """Archive an entire directory."""
    if not directory.exists():
        return None

    if execute:
        ARCHIVE_DIR.mkdir(exist_ok=True)
        archive_path = ARCHIVE_DIR / f"{archive_name}.tar.gz"

        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(directory, arcname=directory.name)

        return archive_path

    return ARCHIVE_DIR / f"{archive_name}.tar.gz"


def cleanup_after_archive(files: list[Path], directories: list[Path], execute: bool = False):
    """Remove original files/directories after archiving."""
    if not execute:
        return

    for f in files:
        if f.exists():
            f.unlink()

    for d in directories:
        if d.exists():
            shutil.rmtree(d)


# =============================================================================
# Index Generation
# =============================================================================

def extract_title_from_html(html_path: Path) -> tuple[str, str] | None:
    """Extract title and function type from HTML file.

    Returns (title, func_type) or None if not found.
    HTML titles follow format: "L{layer}/N{neuron} - {Name}"
    """
    try:
        with open(html_path, encoding='utf-8') as f:
            # Read just the first few KB to find the title
            content = f.read(4096)

        # Extract <title> content
        import re
        match = re.search(r'<title>([^<]+)</title>', content)
        if match:
            title_text = match.group(1)
            # Split on " - " to get the name part
            if " - " in title_text:
                name = title_text.split(" - ", 1)[1].strip()
                return name, "unknown"
        return None
    except Exception:
        return None


def generate_index(execute: bool = False) -> Path | None:
    """Generate index.html for browsing neuron reports."""
    if not execute:
        return CANONICAL_DIR / "index.html"

    # First, collect data from JSON files
    json_data = {}  # key -> {title, summary, func_type}
    json_dir = CANONICAL_JSON

    if json_dir.exists():
        for path in json_dir.glob("*_dashboard.json"):
            neuron_id = extract_neuron_id(path.name)
            if not neuron_id:
                continue

            layer, neuron_idx = neuron_id
            key = f"L{layer}_N{neuron_idx}"

            try:
                with open(path) as f:
                    data = json.load(f)

                char = data.get("characterization", {})
                json_data[key] = {
                    "title": char.get("title", ""),
                    "summary": char.get("one_line_summary", ""),
                    "func_type": char.get("function_type", "unknown").lower(),
                }
            except (json.JSONDecodeError, KeyError):
                pass

    # Now scan all HTML files and build neuron list
    neurons = []

    if CANONICAL_HTML.exists():
        for html_path in CANONICAL_HTML.glob("*.html"):
            neuron_id = extract_neuron_id(html_path.name)
            if not neuron_id:
                continue

            layer, neuron_idx = neuron_id
            key = f"L{layer}_N{neuron_idx}"

            # Try to get data from JSON first
            if key in json_data and json_data[key]["title"]:
                title = json_data[key]["title"]
                summary = json_data[key]["summary"]
                func_type = json_data[key]["func_type"]
            else:
                # Fall back to extracting from HTML
                html_info = extract_title_from_html(html_path)
                if html_info:
                    title, func_type = html_info
                    summary = ""
                else:
                    title = key
                    summary = ""
                    func_type = "unknown"

            neurons.append({
                "neuron_id": f"L{layer}/N{neuron_idx}",
                "layer": layer,
                "neuron_idx": neuron_idx,
                "filename": f"{key}.html",
                "title": title,
                "summary": summary,
                "func_type": func_type,
            })

    # Sort by layer, then neuron index
    neurons.sort(key=lambda n: (n["layer"], n["neuron_idx"]))

    # Generate index.json for the HTML to load (kept for API/programmatic access)
    index_json_path = CANONICAL_DIR / "index.json"
    with open(index_json_path, "w") as f:
        json.dump(neurons, f, indent=2)

    # Generate the index.html with embedded data (works offline/locally)
    index_html = generate_index_html(neurons)
    index_path = CANONICAL_DIR / "index.html"
    with open(index_path, "w") as f:
        f.write(index_html)

    return index_path


def generate_index_html(neurons: list) -> str:
    """Generate the index.html content with embedded neuron data for offline use."""
    # Embed neuron data as JSON in the HTML to avoid fetch() CORS issues with local files
    neurons_json = json.dumps(neurons)
    neuron_count = len(neurons)
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neuron Reports Index</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
:root {{
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
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 17px;
    background: var(--bg);
    color: var(--text);
    line-height: 1.65;
    -webkit-font-smoothing: antialiased;
}}

.container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 40px 24px 60px;
}}

.header {{
    text-align: center;
    margin-bottom: 48px;
    padding-bottom: 32px;
    border-bottom: 1px solid var(--border);
}}

.header h1 {{
    font-size: 42px;
    font-weight: 700;
    letter-spacing: -0.8px;
    margin-bottom: 12px;
}}

.header .subtitle {{
    font-size: 18px;
    color: var(--text-secondary);
    margin-bottom: 24px;
}}

.stats {{
    display: flex;
    justify-content: center;
    gap: 32px;
    margin-top: 24px;
}}

.stat {{
    text-align: center;
}}

.stat-value {{
    font-size: 32px;
    font-weight: 700;
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
}}

.stat-label {{
    font-size: 14px;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}}

.controls {{
    display: flex;
    gap: 16px;
    margin-bottom: 32px;
    flex-wrap: wrap;
}}

.control-group {{
    display: flex;
    align-items: center;
    gap: 8px;
}}

.control-group label {{
    font-size: 14px;
    font-weight: 500;
    color: var(--text-secondary);
}}

.control-group select, .control-group input[type="text"] {{
    padding: 8px 12px;
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 14px;
    font-family: inherit;
    background: white;
    color: var(--text);
}}

.control-group input[type="text"] {{
    width: 300px;
}}

.legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 32px;
    padding: 16px;
    background: var(--bg-inset);
    border-radius: 8px;
}}

.legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
}}

.legend-color {{
    width: 12px;
    height: 12px;
    border-radius: 3px;
}}

.layer-section {{
    margin-bottom: 48px;
}}

.layer-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 2px solid var(--border);
}}

.layer-header h2 {{
    font-size: 24px;
    font-weight: 600;
    letter-spacing: -0.3px;
}}

.layer-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    padding: 4px 10px;
    background: var(--bg-inset);
    border-radius: 4px;
    color: var(--text-tertiary);
}}

.neuron-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 16px;
}}

.neuron-card {{
    background: white;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    transition: all 0.2s ease;
    cursor: pointer;
    text-decoration: none;
    color: inherit;
    display: block;
}}

.neuron-card:hover {{
    border-color: var(--accent);
    box-shadow: 0 4px 12px rgba(0, 102, 204, 0.1);
    transform: translateY(-2px);
}}

.neuron-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 8px;
}}

.neuron-id {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 500;
    color: var(--accent);
}}

.function-badge {{
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 4px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}}

.func-semantic {{ background: rgba(137, 87, 229, 0.1); color: var(--purple); }}
.func-routing {{ background: rgba(227, 98, 9, 0.1); color: var(--amber); }}
.func-formatting {{ background: rgba(34, 134, 58, 0.1); color: var(--green); }}
.func-hybrid {{ background: rgba(0, 102, 204, 0.1); color: var(--accent); }}
.func-combination {{ background: rgba(215, 58, 73, 0.1); color: var(--red); }}
.func-lexical, .func-syntactic, .func-structural, .func-unknown {{
    background: rgba(136, 136, 136, 0.1);
    color: var(--text-secondary);
}}

.neuron-title {{
    font-size: 16px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 8px;
    line-height: 1.3;
}}

.neuron-summary {{
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.5;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}}

.empty-state, .loading {{
    text-align: center;
    padding: 60px 20px;
    color: var(--text-tertiary);
}}

@media (max-width: 768px) {{
    .header h1 {{ font-size: 32px; }}
    .stats {{ flex-direction: column; gap: 16px; }}
    .controls {{ flex-direction: column; }}
    .control-group input[type="text"] {{ width: 100%; }}
    .neuron-grid {{ grid-template-columns: 1fr; }}
}}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Neuron Reports</h1>
            <p class="subtitle">NeuronPI Investigation Dashboards</p>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value" id="total-neurons">0</div>
                    <div class="stat-label">Total Neurons</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="layer-count">0</div>
                    <div class="stat-label">Layers</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="function-count">0</div>
                    <div class="stat-label">Function Types</div>
                </div>
            </div>
        </div>

        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: rgba(137, 87, 229, 0.3);"></div>
                <span><strong>Semantic:</strong> Content-driven</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: rgba(227, 98, 9, 0.3);"></div>
                <span><strong>Routing:</strong> Information flow</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: rgba(34, 134, 58, 0.3);"></div>
                <span><strong>Formatting:</strong> Structural</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: rgba(0, 102, 204, 0.3);"></div>
                <span><strong>Hybrid:</strong> Multi-function</span>
            </div>
        </div>

        <div class="controls">
            <div class="control-group">
                <label for="groupBy">Group by:</label>
                <select id="groupBy">
                    <option value="layer">Layer</option>
                    <option value="function">Function Type</option>
                    <option value="none">None (flat list)</option>
                </select>
            </div>
            <div class="control-group">
                <label for="filterFunc">Filter by function:</label>
                <select id="filterFunc">
                    <option value="all">All</option>
                    <option value="semantic">Semantic</option>
                    <option value="routing">Routing</option>
                    <option value="formatting">Formatting</option>
                    <option value="hybrid">Hybrid</option>
                    <option value="lexical">Lexical</option>
                </select>
            </div>
            <div class="control-group">
                <label for="search">Search:</label>
                <input type="text" id="search" placeholder="Search neuron ID or title...">
            </div>
        </div>

        <div id="neuron-container" class="loading">
            <p>Loading neurons...</p>
        </div>
    </div>

    <script>
// Embedded neuron data for offline/local file access (no fetch required)
let allNeurons = {neurons_json};

function loadNeurons() {{
    updateStats();
    renderNeurons();
}}

function updateStats() {{
    document.getElementById('total-neurons').textContent = allNeurons.length;
    const layers = new Set(allNeurons.map(n => n.layer));
    document.getElementById('layer-count').textContent = layers.size;
    const functions = new Set(allNeurons.map(n => n.func_type).filter(f => f !== 'unknown'));
    document.getElementById('function-count').textContent = functions.size;
}}

function renderNeurons() {{
    const groupBy = document.getElementById('groupBy').value;
    const filterFunc = document.getElementById('filterFunc').value;
    const searchTerm = document.getElementById('search').value.toLowerCase();

    let filtered = allNeurons.filter(n => {{
        const matchesFunc = filterFunc === 'all' || n.func_type === filterFunc;
        const matchesSearch = searchTerm === '' ||
            n.neuron_id.toLowerCase().includes(searchTerm) ||
            n.title.toLowerCase().includes(searchTerm);
        return matchesFunc && matchesSearch;
    }});

    if (filtered.length === 0) {{
        document.getElementById('neuron-container').innerHTML =
            '<div class="empty-state"><p>No neurons match the current filters.</p></div>';
        return;
    }}

    let grouped = {{}};
    if (groupBy === 'layer') {{
        filtered.forEach(n => {{
            const key = `Layer ${{n.layer}}`;
            if (!grouped[key]) grouped[key] = [];
            grouped[key].push(n);
        }});
    }} else if (groupBy === 'function') {{
        filtered.forEach(n => {{
            const key = n.func_type.charAt(0).toUpperCase() + n.func_type.slice(1);
            if (!grouped[key]) grouped[key] = [];
            grouped[key].push(n);
        }});
    }} else {{
        grouped['All Neurons'] = filtered;
    }}

    const sortedGroups = Object.keys(grouped).sort((a, b) => {{
        if (groupBy === 'layer') {{
            const layerA = parseInt(a.replace('Layer ', ''));
            const layerB = parseInt(b.replace('Layer ', ''));
            return layerA - layerB;
        }}
        return a.localeCompare(b);
    }});

    const container = document.getElementById('neuron-container');
    container.innerHTML = '';

    sortedGroups.forEach(groupName => {{
        const neurons = grouped[groupName];
        const section = document.createElement('div');
        section.className = 'layer-section';

        if (sortedGroups.length > 1) {{
            const header = document.createElement('div');
            header.className = 'layer-header';
            header.innerHTML = `
                <h2>${{groupName}}</h2>
                <span class="layer-badge">${{neurons.length}} neurons</span>
            `;
            section.appendChild(header);
        }}

        const grid = document.createElement('div');
        grid.className = 'neuron-grid';

        neurons.forEach(neuron => {{
            const card = document.createElement('a');
            card.className = 'neuron-card';
            card.href = `html/${{neuron.filename}}`;
            card.innerHTML = `
                <div class="neuron-header">
                    <span class="neuron-id">${{neuron.neuron_id}}</span>
                    <span class="function-badge func-${{neuron.func_type}}">${{neuron.func_type}}</span>
                </div>
                <div class="neuron-title">${{neuron.title}}</div>
                <div class="neuron-summary">${{neuron.summary}}</div>
            `;
            grid.appendChild(card);
        }});

        section.appendChild(grid);
        container.appendChild(section);
    }});
}}

document.getElementById('groupBy').addEventListener('change', renderNeurons);
document.getElementById('filterFunc').addEventListener('change', renderNeurons);
document.getElementById('search').addEventListener('input', renderNeurons);

loadNeurons();
    </script>
</body>
</html>'''


# =============================================================================
# Main
# =============================================================================

def format_size(size_bytes: int) -> str:
    """Format size in human-readable form."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate scattered neuron reports into canonical structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--execute", "-x",
        action="store_true",
        help="Actually perform consolidation (default: dry run)",
    )
    parser.add_argument(
        "--archive-only",
        action="store_true",
        help="Only archive old files, don't consolidate",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't remove original files after archiving (keep both)",
    )

    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("NEURON REPORT CONSOLIDATION")
    print("=" * 60)

    # Scan for reports
    print("\n1. Scanning for neuron reports...")
    reports = scan_for_reports()
    print(f"   Found {len(reports)} unique neurons with reports")

    # Count files by type
    type_counts = defaultdict(int)
    for neuron_key, file_types in reports.items():
        for file_type, files in file_types.items():
            type_counts[file_type] += len(files)

    print("\n   Files by type:")
    for ft, count in sorted(type_counts.items()):
        print(f"     {ft}: {count}")

    # Find items to archive
    stray_files = find_stray_files()
    slurm_dirs = find_slurm_dirs()
    backup_dirs = find_backup_dirs()
    legacy_html_dirs = find_legacy_html_dirs()

    print(f"\n   Stray root-level files: {len(stray_files)}")
    print(f"   SLURM output directories: {len(slurm_dirs)}")
    print(f"   Backup directories: {len(backup_dirs)}")
    print(f"   Legacy HTML directories: {len(legacy_html_dirs)}")
    if legacy_html_dirs:
        for d in legacy_html_dirs:
            print(f"     - {d}")

    if not args.archive_only:
        # Consolidate JSON files only
        print("\n2. Consolidating JSON files to canonical location...")
        print(f"   Target: {CANONICAL_JSON}/")
        print(f"   NOTE: HTML dashboards in {CANONICAL_HTML}/ are NOT touched")

        actions = consolidate_reports(reports, execute=args.execute)

        print(f"\n   JSON files to copy: {len(actions['moved_json'])}")
        print(f"   Duplicate files to archive: {len(actions['archived'])}")

    # Archive
    print("\n3. Archiving old/duplicate files...")

    archive_actions = []

    # Archive stray files
    if stray_files:
        archive_path = archive_files(stray_files, f"stray_files_{timestamp}", execute=args.execute)
        archive_actions.append(("Stray files", stray_files, archive_path))
        print(f"   Stray files -> {archive_path}")

    # Archive SLURM directories
    if slurm_dirs:
        for slurm_dir in slurm_dirs:
            archive_path = archive_directory(slurm_dir, f"slurm_{slurm_dir.name}", execute=args.execute)
            archive_actions.append(("SLURM dir", [slurm_dir], archive_path))
        print(f"   SLURM directories -> {len(slurm_dirs)} archives")

    # Archive backup directories
    if backup_dirs:
        for backup_dir in backup_dirs:
            archive_path = archive_directory(backup_dir, f"backup_{backup_dir.name}", execute=args.execute)
            archive_actions.append(("Backup dir", [backup_dir], archive_path))
        print(f"   Backup directories -> {len(backup_dirs)} archives")

    # Archive legacy HTML directories (old locations that shouldn't be used anymore)
    if legacy_html_dirs:
        for legacy_dir in legacy_html_dirs:
            safe_name = str(legacy_dir).replace("/", "_").replace(".", "_")
            archive_path = archive_directory(legacy_dir, f"legacy_html_{safe_name}_{timestamp}", execute=args.execute)
            archive_actions.append(("Legacy HTML", [legacy_dir], archive_path))
        print(f"   Legacy HTML directories -> {len(legacy_html_dirs)} archives")

    # Generate index
    if not args.archive_only:
        print("\n4. Generating index.html...")
        index_path = generate_index(execute=args.execute)
        print(f"   Index -> {index_path}")

    # Cleanup (optional)
    if args.execute and not args.no_cleanup:
        print("\n5. Cleaning up original files...")
        # Only clean up stray files and directories that were archived
        for desc, items, _ in archive_actions:
            if desc == "Stray files":
                for f in items:
                    if f.exists():
                        f.unlink()
                        print(f"   Removed: {f}")
            elif desc in ("SLURM dir", "Backup dir", "Legacy HTML"):
                for d in items:
                    if d.exists():
                        shutil.rmtree(d)
                        print(f"   Removed: {d}/")

    # Summary
    print("\n" + "=" * 60)
    if args.execute:
        print("CONSOLIDATION COMPLETE")
        print(f"\nCanonical location: {CANONICAL_DIR}/")
        print(f"Archives: {ARCHIVE_DIR}/")
    else:
        print("DRY RUN COMPLETE - No changes made")
        print("\nRun with --execute to perform consolidation")
    print("=" * 60)


if __name__ == "__main__":
    main()
