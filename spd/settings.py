import os
from pathlib import Path

REPO_ROOT = (
    Path(os.environ["GITHUB_WORKSPACE"]) if os.environ.get("CI") else Path(__file__).parent.parent
)

# Base directory for SPD cache. Defaults to a subdirectory in the home directory.
default_cache_dir = str(Path.home() / "spd_cache")
SPD_CACHE_DIR = Path(os.environ.get("SPD_CACHE_DIR", default_cache_dir))
