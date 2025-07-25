#!/bin/bash
# Quick integration script - adds Claude commands to existing project

set -e

echo "🚀 Quick Claude Commands Integration"
echo "===================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEMPLATE_ROOT="$(dirname "$SCRIPT_DIR")"

# Create .claude/commands directory if it doesn't exist
mkdir -p .claude/commands

# Copy all commands
cp -r "$TEMPLATE_ROOT/.claude/commands/"* .claude/commands/

echo "✅ Claude commands installed!"
echo ""
echo "Available commands:"
echo "  /multi-mind - Multi-agent analysis"
echo "  /search-all - Search conversation history"
echo "  /page - Save session state"
echo "  /analyze-function - Deep code analysis"
echo "  /crud-claude-commands - Create custom commands"
echo ""
echo "Run 'claude' to start using these commands!"