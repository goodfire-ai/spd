"""HTTP server for querying the local attribution database.

Usage:
    python -m spd.attributions.serve --db_path ./local_attr.db --port 8765

API Endpoints:
    GET /api/meta              - Database metadata (wandb_path, n_blocks)
    GET /api/activation_contexts - Activation contexts for all components
    GET /api/prompts           - List all prompts (id, tokens preview)
    GET /api/prompt/<id>       - Full prompt data (tokens, pairs)
    GET /api/search?components=a,b,c&mode=all|any - Find matching prompts
    GET /api/components        - List all unique component keys

    Static files served from current directory (for local_attributions.html)
"""

import argparse
import json
import re
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from spd.attributions.db import LocalAttrDB


def make_handler(db: LocalAttrDB) -> type[BaseHTTPRequestHandler]:
    """Create a handler class with db bound via closure."""

    class LocalAttrHandler(BaseHTTPRequestHandler):
        """HTTP request handler for local attribution queries."""

        def log_message(self, format: str, *args: object) -> None:
            """Suppress default logging (too verbose)."""
            pass

        def send_json(self, data: dict[str, Any] | list[Any], status: int = 200) -> None:
            """Send a JSON response."""
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")  # CORS for dev
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))

        def send_error_json(self, status: int, message: str) -> None:
            """Send a JSON error response."""
            self.send_json({"error": message}, status)

        def serve_static(self, path: str) -> None:
            """Serve a static file."""
            # Normalize path and prevent directory traversal
            clean_path = path.lstrip("/")
            if clean_path == "":
                clean_path = "local_attributions.html"

            file_path = THIS_FILE.parent / clean_path

            # unneeded - running locally only
            # # Security: ensure path is within static_dir
            # try:
            #     file_path = file_path.resolve()
            #     static_dir.resolve()
            #     if not str(file_path).startswith(str(static_dir.resolve())):
            #         self.send_error(403, "Forbidden")
            #         return
            # except Exception:
            #     self.send_error(400, "Bad path")
            #     return

            if not file_path.exists() or not file_path.is_file():
                self.send_error(404, "Not found")
                return

            # Determine content type
            suffix = file_path.suffix.lower()
            content_types = {
                ".html": "text/html",
                ".css": "text/css",
                ".js": "application/javascript",
                ".json": "application/json",
                ".png": "image/png",
                ".svg": "image/svg+xml",
            }
            content_type = content_types.get(suffix, "application/octet-stream")

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            with open(file_path, "rb") as f:
                self.wfile.write(f.read())

        def do_GET(self) -> None:
            """Handle GET requests."""
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            query = urllib.parse.parse_qs(parsed.query)

            # API routes
            if path == "/api/meta":
                self.handle_meta()
            elif path == "/api/activation_contexts":
                self.handle_activation_contexts()
            elif path == "/api/prompts":
                self.handle_prompts()
            elif path.startswith("/api/prompt/"):
                match = re.match(r"/api/prompt/(\d+)", path)
                if match:
                    self.handle_prompt(int(match.group(1)))
                else:
                    self.send_error_json(400, "Invalid prompt ID")
            elif path == "/api/search":
                components = query.get("components", [""])[0].split(",")
                components = [c.strip() for c in components if c.strip()]
                mode = query.get("mode", ["all"])[0]
                self.handle_search(components, mode)
            elif path == "/api/components":
                self.handle_components()
            else:
                # Serve static files
                self.serve_static(path)

        def handle_meta(self) -> None:
            """Return database metadata."""
            wandb_info = db.get_meta("wandb_path")
            n_blocks_info = db.get_meta("n_blocks")
            self.send_json(
                {
                    "wandb_path": wandb_info.get("path") if isinstance(wandb_info, dict) else None,
                    "n_blocks": n_blocks_info.get("n_blocks")
                    if isinstance(n_blocks_info, dict)
                    else None,
                    "prompt_count": db.get_prompt_count(),
                }
            )

        def handle_activation_contexts(self) -> None:
            """Return activation contexts."""
            contexts = db.get_activation_contexts()
            if contexts is None:
                self.send_error_json(404, "No activation contexts found")
                return
            self.send_json(contexts)

        def handle_prompts(self) -> None:
            """Return list of all prompts (summaries)."""
            summaries = db.get_all_prompt_summaries()
            self.send_json(
                [
                    {
                        "id": s.id,
                        "tokens": s.tokens,
                        "preview": "".join(s.tokens[:10]) + ("..." if len(s.tokens) > 10 else ""),
                    }
                    for s in summaries
                ]
            )

        def handle_prompt(self, prompt_id: int) -> None:
            """Return full data for a single prompt."""
            prompt = db.get_prompt(prompt_id)
            if prompt is None:
                self.send_error_json(404, f"Prompt {prompt_id} not found")
                return

            # Parse the pairs JSON
            pairs = json.loads(prompt.pairs_json)

            self.send_json(
                {
                    "id": prompt.id,
                    "tokens": prompt.tokens,
                    "pairs": pairs,
                }
            )

        def handle_search(self, components: list[str], mode: str) -> None:
            """Search for prompts with specified components."""
            if not components:
                self.send_error_json(400, "No components specified")
                return

            require_all = mode != "any"
            prompt_ids = db.find_prompts_with_components(components, require_all=require_all)

            # Get summaries for matching prompts
            all_summaries = {s.id: s for s in db.get_all_prompt_summaries()}
            results = []
            for pid in prompt_ids:
                if pid in all_summaries:
                    s = all_summaries[pid]
                    results.append(
                        {
                            "id": s.id,
                            "tokens": s.tokens,
                            "preview": "".join(s.tokens[:10])
                            + ("..." if len(s.tokens) > 10 else ""),
                        }
                    )

            self.send_json(
                {
                    "query": {"components": components, "mode": mode},
                    "count": len(results),
                    "results": results,
                }
            )

        def handle_components(self) -> None:
            """Return all unique component keys."""
            components = db.get_unique_components()
            self.send_json({"components": components, "count": len(components)})

    return LocalAttrHandler


def run_server(db_path: Path, port: int) -> None:
    """Run the HTTP server."""
    db = LocalAttrDB(db_path)

    # Verify database has data
    try:
        count = db.get_prompt_count()
        print(f"Database loaded: {count} prompts")
    except Exception as e:
        print(f"Error loading database: {e}")
        return

    handler_class = make_handler(db)
    server = HTTPServer(("localhost", port), handler_class)
    print(f"Server running at http://localhost:{port}/")
    print(f"  Original:  http://localhost:{port}/local_attributions.html")
    print(f"  Alpine.js: http://localhost:{port}/local_attributions_alpine.html")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        db.close()
        server.server_close()


THIS_FILE = Path(__file__)


def main(
    db_path: str,
    port: int = 8765,
) -> None:
    """Serve local attribution database.

    Args:
        db_path: Path to SQLite database
        port: Port to serve on (default 8765)
    """
    db_path_ = Path(db_path)
    assert db_path_.exists(), f"Database not found: {db_path_}"
    run_server(db_path_, port)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
