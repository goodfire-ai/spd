#!/usr/bin/env python3
"""
Wrapper script for causal importance sweep plotting.

This script can be run from the project root to analyze how causal importance
functions respond as input magnitude increases.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from spd.scripts.plot_causal_importance_sweep_main import main

if __name__ == "__main__":
    exit(main())
