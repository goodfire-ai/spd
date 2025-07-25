#!/bin/bash
# quick-plan.sh - Simplified wrapper for common planwithgemini operations

set -e

# Default to auto mode if not specified
MODE="${1:-auto}"
TASK="$2"

# If only one argument, assume it's the task and use auto mode
if [ -z "$TASK" ]; then
    TASK="$MODE"
    MODE="auto"
fi

# Check if task is provided
if [ -z "$TASK" ]; then
    echo "Usage: $0 [mode] \"task description\""
    echo "Modes: auto (default), full, quick, security"
    echo ""
    echo "Examples:"
    echo "  $0 \"implement caching\"                    # Uses auto mode"
    echo "  $0 full \"refactor authentication system\"  # Full analysis"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the main planwithgemini script
"$SCRIPT_DIR/planwithgemini.sh" "$MODE" "$TASK"