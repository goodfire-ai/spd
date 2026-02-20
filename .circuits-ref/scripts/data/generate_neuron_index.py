#!/usr/bin/env python3
"""Generate neuron index JSON from dashboard directory.

Uses JSON files as primary source, falls back to HTML parsing.

Usage:
    python scripts/generate_neuron_index.py frontend/reports/aspirin-cox
    python scripts/generate_neuron_index.py frontend/reports/aspirin-cox -o frontend/reports/neuron_index.json
"""

import argparse
import json
import re
from pathlib import Path


def extract_from_json(json_path: Path) -> dict:
    """Extract neuron metadata from a dashboard JSON file."""
    try:
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)

        neuron_id = data.get('neuron_id', '')
        if not neuron_id:
            return None

        # Extract layer number
        layer_match = re.match(r'L(\d+)', neuron_id)
        layer = int(layer_match.group(1)) if layer_match else 0

        # Get summary card data
        summary_card = data.get('summary_card', {})

        # Try to get title from various sources
        title = ""
        # Check if there's a separate title field
        if 'title' in data:
            title = data['title']
        # Or derive from summary
        elif summary_card.get('summary'):
            # Use first sentence or first 50 chars
            summary = summary_card['summary']
            if '.' in summary[:100]:
                title = summary.split('.')[0][:60]
            else:
                title = summary[:50] + "..."

        # Get function type
        func_type = summary_card.get('function_type', 'unknown')
        if not func_type or func_type == 'unknown':
            # Try characterization
            char = data.get('characterization', {})
            func_type = char.get('function_type', 'unknown')

        # Get summary
        summary = summary_card.get('summary', '')[:200]
        if not summary:
            summary = data.get('characterization', {}).get('final_hypothesis', '')[:200]

        # Get confidence
        confidence = summary_card.get('confidence', 0.5)
        if confidence == 0:
            confidence = data.get('confidence', 0.5)

        # Build HTML filename from JSON filename
        html_name = json_path.stem.replace('_dashboard', '') + '.html'
        # Or from the json filename pattern
        if '-' in json_path.stem:
            html_name = json_path.stem + '.html'

        return {
            'neuron_id': neuron_id,
            'title': title or f"Neuron {neuron_id}",
            'func_type': func_type,
            'layer': layer,
            'summary': summary,
            'confidence': confidence,
            'filename': html_name,
        }
    except Exception as e:
        print(f"Error parsing JSON {json_path}: {e}")
        return None


def extract_from_html(html_path: Path) -> dict:
    """Extract neuron metadata from an HTML dashboard (fallback)."""
    try:
        from bs4 import BeautifulSoup

        with open(html_path, encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        # Get title - usually in format "L14/N13662 - Short Name"
        title_tag = soup.find('title')
        full_title = title_tag.text.strip() if title_tag else ""

        # Parse neuron ID and short name from title
        match = re.match(r'(L\d+/N\d+)\s*[-–—]\s*(.+)', full_title)
        if match:
            neuron_id = match.group(1)
            short_name = match.group(2).strip()
        else:
            # Try to extract from filename
            fname_match = re.match(r'(L\d+)_(N\d+)', html_path.stem)
            if fname_match:
                neuron_id = f"{fname_match.group(1)}/{fname_match.group(2)}"
                short_name = full_title or html_path.stem
            else:
                return None

        # Extract layer number
        layer_match = re.match(r'L(\d+)', neuron_id)
        layer = int(layer_match.group(1)) if layer_match else 0

        return {
            'neuron_id': neuron_id,
            'title': short_name,
            'func_type': 'unknown',
            'layer': layer,
            'summary': '',
            'confidence': 0.5,
            'filename': html_path.name,
        }
    except Exception as e:
        print(f"Error parsing HTML {html_path}: {e}")
        return None


def extract_html_title(html_path: Path) -> str:
    """Extract just the title from an HTML file."""
    try:
        with open(html_path, encoding='utf-8') as f:
            # Read first 2000 chars to find title
            content = f.read(2000)
        match = re.search(r'<title>([^<]+)</title>', content)
        if match:
            full_title = match.group(1).strip()
            # Parse "L14/N13662 - Short Name" format
            title_match = re.match(r'L\d+/N\d+\s*[-–—]\s*(.+)', full_title)
            if title_match:
                return title_match.group(1).strip()
        return ""
    except:
        return ""


def generate_index(dashboard_dir: Path, output_path: Path = None) -> list:
    """Generate neuron index from JSON files, with titles from HTML."""

    neurons = []
    processed_neurons = set()

    # Build a map of neuron_id -> HTML title
    html_titles = {}
    html_files = list(dashboard_dir.glob("L*_N*.html"))
    for html_path in html_files:
        title = extract_html_title(html_path)
        if title:
            fname_match = re.match(r'(L\d+)_(N\d+)', html_path.stem)
            if fname_match:
                neuron_id = f"{fname_match.group(1)}/{fname_match.group(2)}"
                html_titles[neuron_id] = title

    # First try JSON files (more reliable for metadata)
    json_files = list(dashboard_dir.glob("*.json"))
    # Filter out index files
    json_files = [f for f in json_files if 'index' not in f.name.lower()]

    print(f"Found {len(json_files)} JSON files, {len(html_files)} HTML files in {dashboard_dir}")

    for json_path in sorted(json_files):
        info = extract_from_json(json_path)
        if info:
            # Override title with HTML title if available
            if info['neuron_id'] in html_titles:
                info['title'] = html_titles[info['neuron_id']]
            neurons.append(info)
            processed_neurons.add(info['neuron_id'])

    # Then try HTML files for any missing neurons
    html_only = 0
    for html_path in sorted(html_files):
        fname_match = re.match(r'(L\d+)_(N\d+)', html_path.stem)
        if fname_match:
            neuron_id = f"{fname_match.group(1)}/{fname_match.group(2)}"
            if neuron_id not in processed_neurons:
                info = extract_from_html(html_path)
                if info:
                    neurons.append(info)
                    html_only += 1

    print(f"Extracted {len(neurons)} neurons ({len(neurons) - html_only} from JSON, {html_only} from HTML only)")

    # Sort by layer (descending), then neuron number
    neurons.sort(key=lambda x: (-x['layer'], x['neuron_id']))

    # Write output
    if output_path is None:
        output_path = dashboard_dir / "neuron_index.json"

    with open(output_path, 'w') as f:
        json.dump(neurons, f, indent=2)

    print(f"Generated index with {len(neurons)} neurons")
    print(f"Output: {output_path}")

    # Print stats
    func_counts = {}
    for n in neurons:
        ft = n['func_type']
        func_counts[ft] = func_counts.get(ft, 0) + 1

    print("\nFunction type distribution:")
    for ft, count in sorted(func_counts.items(), key=lambda x: -x[1]):
        print(f"  {ft}: {count}")

    return neurons


def main():
    parser = argparse.ArgumentParser(description="Generate neuron index from HTML dashboards")
    parser.add_argument("dashboard_dir", help="Directory containing HTML dashboards")
    parser.add_argument("-o", "--output", help="Output JSON path (default: <dir>/neuron_index.json)")

    args = parser.parse_args()

    dashboard_dir = Path(args.dashboard_dir)
    if not dashboard_dir.exists():
        print(f"Error: Directory not found: {dashboard_dir}")
        return 1

    output_path = Path(args.output) if args.output else None
    generate_index(dashboard_dir, output_path)
    return 0


if __name__ == "__main__":
    exit(main())
