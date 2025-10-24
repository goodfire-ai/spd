"""Integration tests for dashboard data generation and JS loading."""

import http.server
import json
import socketserver
import threading
from pathlib import Path

import pytest

from spd.clustering.dashboard.core.dashboard_config import DashboardConfig
from spd.clustering.dashboard.run import main

# Test output directory
TEST_OUTPUT_DIR = Path("tests/.temp/dashboard-integration")

# WandB clustering run to use for tests - SET THIS to a valid run
_WANDB_RUN = "wandb:goodfire/spd-cluster/runs/c-c3623a67"


def find_free_port():
    """Find a free port to use for the test server."""
    with socketserver.TCPServer(("", 0), None) as s:
        return s.server_address[1]


def start_test_server(directory: Path, port: int) -> threading.Thread:
    """Start a simple HTTP server in a background thread."""

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format, *args):
            # Suppress server logs during tests
            pass

    server = socketserver.TCPServer(("", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Store server reference so we can shut it down
    thread.server = server  # type: ignore
    return thread


@pytest.mark.slow
def test_dashboard_end_to_end():
    """End-to-end test of dashboard generation and browser loading.

    This test:
    1. Generates dashboard data (model loading, inference, ZANJ save)
    2. Verifies output file structure
    3. Tests index.html loads without JS errors
    4. Tests cluster.html loads without JS errors for multiple clusters
    """
    from playwright.sync_api import sync_playwright

    # ============================================================================
    # SECTION 1: Generate dashboard data
    # ============================================================================
    output_dir = TEST_OUTPUT_DIR / "end_to_end"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DashboardConfig(
        wandb_run=_WANDB_RUN,
        output_dir=output_dir,
        iteration=-10,  # Late iteration with fewer, larger clusters
        n_samples=2,  # Minimal samples for speed
        n_batches=2,
        batch_size=2,
        context_length=4,
        write_html=True,
    )

    main(config)

    # Get the generated directory
    output_subdirs = list(output_dir.iterdir())
    assert len(output_subdirs) == 1, f"Expected 1 output dir, got {len(output_subdirs)}"
    run_output_dir = output_subdirs[0]

    # ============================================================================
    # SECTION 2: Verify output file structure
    # ============================================================================
    assert (run_output_dir / "dashboard.zanj").exists(), "ZANJ file missing"
    assert (run_output_dir / "data").exists(), "Data directory missing"
    assert (run_output_dir / "data" / "__zanj__.json").exists(), "__zanj__.json missing"
    assert (run_output_dir / "index.html").exists(), "index.html missing"
    assert (run_output_dir / "cluster.html").exists(), "cluster.html missing"

    # Load cluster IDs for testing
    zanj_json_path = run_output_dir / "data" / "__zanj__.json"
    with open(zanj_json_path) as f:
        zanj_data = json.load(f)

    cluster_ids = list(zanj_data.get("clusters", {}).keys())
    assert len(cluster_ids) > 0, "No clusters found in generated data"

    # ============================================================================
    # SECTION 3: Start HTTP server for browser tests
    # ============================================================================
    port = find_free_port()
    server_thread = start_test_server(run_output_dir, port)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()

            # ========================================================================
            # SECTION 4: Test index.html loads without errors
            # ========================================================================
            page = context.new_page()

            # Collect console errors
            console_errors = []
            page_errors = []
            page.on("console", lambda msg: console_errors.append(msg) if msg.type == "error" else None)
            page.on("pageerror", lambda exc: page_errors.append(str(exc)))

            # Load index page
            url = f"http://localhost:{port}/index.html"
            response = page.goto(url, wait_until="networkidle")

            assert response is not None
            assert response.ok, f"Failed to load index.html: {response.status}"

            # Wait for loading to complete
            page.wait_for_selector("#loading", state="hidden", timeout=30000)

            # Check for errors
            if page_errors:
                pytest.fail(f"Page errors on index.html: {page_errors}")

            if console_errors:
                error_text = "\n".join(msg.text for msg in console_errors)
                pytest.fail(f"Console errors on index.html:\n{error_text}")

            # Verify table loaded
            assert page.query_selector("#clusterTableContainer") is not None, \
                "Cluster table container not found"

            page.close()

            # ========================================================================
            # SECTION 5: Test cluster.html loads for multiple clusters
            # ========================================================================
            test_cluster_ids = cluster_ids[:min(3, len(cluster_ids))]

            for cluster_id in test_cluster_ids:
                page = context.new_page()

                # Collect errors for this cluster
                console_errors = []
                page_errors = []
                page.on("console", lambda msg, errors=console_errors: errors.append(msg) if msg.type == "error" else None)
                page.on("pageerror", lambda exc, errors=page_errors: errors.append(str(exc)))

                # Load cluster page
                url = f"http://localhost:{port}/cluster.html?id={cluster_id}"
                response = page.goto(url, wait_until="networkidle")

                assert response is not None
                assert response.ok, f"Failed to load cluster {cluster_id}: {response.status}"

                # Wait for loading
                page.wait_for_selector("#loading", state="hidden", timeout=30000)

                # Check for errors
                if page_errors:
                    pytest.fail(f"Page errors for cluster {cluster_id}: {page_errors}")

                if console_errors:
                    error_text = "\n".join(msg.text for msg in console_errors)
                    pytest.fail(f"Console errors for cluster {cluster_id}:\n{error_text}")

                # Verify key elements
                assert page.query_selector("#clusterTitle") is not None, \
                    f"Cluster title not found for {cluster_id}"
                assert page.query_selector("#componentsTable") is not None, \
                    f"Components table not found for {cluster_id}"
                assert page.query_selector("#samplesTable") is not None, \
                    f"Samples table not found for {cluster_id}"

                page.close()

            browser.close()

    finally:
        # Shutdown server
        if hasattr(server_thread, 'server'):
            server_thread.server.shutdown()  # type: ignore
