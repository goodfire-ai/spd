#!/usr/bin/env python3
"""
Wrapper script for component matrix visualization.

This script can be run from the project root to analyze SPD component matrices.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from spd.scripts.visualize_component_matrices import main

if __name__ == "__main__":
    exit(main())
