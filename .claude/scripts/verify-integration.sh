#!/bin/bash
# Verify Agent-Guides Integration
# This script checks that all agent-guides features are properly installed

set -e

echo "🔍 Verifying Agent-Guides Integration"
echo "====================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✅ Found: $1${NC}"
        return 0
    else
        echo -e "${RED}❌ Missing: $1${NC}"
        return 1
    fi
}

# Function to check if directory exists
check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✅ Found directory: $1${NC}"
        return 0
    else
        echo -e "${RED}❌ Missing directory: $1${NC}"
        return 1
    fi
}

# Function to check Python script
check_python_script() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✅ Found: $1${NC}"
        # Test if script runs without errors
        if echo '{}' | python3 "$1" >/dev/null 2>&1; then
            echo -e "${GREEN}   ✓ Script executes successfully${NC}"
        else
            echo -e "${YELLOW}   ⚠ Script exists but may have issues${NC}"
        fi
        return 0
    else
        echo -e "${RED}❌ Missing: $1${NC}"
        return 1
    fi
}

# Track overall success
ALL_GOOD=true

echo ""
echo "1. Checking Claude Commands"
echo "---------------------------"
COMMANDS=(
    ".claude/commands/multi-mind.md"
    ".claude/commands/search-prompts.md"
    ".claude/commands/page.md"
    ".claude/commands/analyze-function.md"
    ".claude/commands/crud-claude-commands.md"
    ".claude/commands/spec-driven.md"
    ".claude/commands/plan-with-context.md"
)

for cmd in "${COMMANDS[@]}"; do
    check_file "$cmd" || ALL_GOOD=false
done

echo ""
echo "2. Checking Hook Scripts"
echo "------------------------"
# List all python scripts in .claude/scripts/
if [ -d ".claude/scripts" ]; then
    for script in .claude/scripts/*.py; do
        if [ -f "$script" ]; then
            check_python_script "$script" || ALL_GOOD=false
        fi
    done
else
    echo -e "${RED}❌ Missing directory: .claude/scripts${NC}"
    ALL_GOOD=false
fi

echo ""
echo "3. Checking Configuration Files"
echo "-------------------------------"
check_file ".claude/settings.json" || ALL_GOOD=false
check_file "CLAUDE.md" || ALL_GOOD=false

echo ""
echo "4. Checking Documentation"
echo "-------------------------"
check_file "docs/claude-hooks-configuration.md" || ALL_GOOD=false
check_file "docs/agent-guides-integration.md" || ALL_GOOD=false
check_file ".claude/scripts/README.md" || ALL_GOOD=false

echo ""
echo "5. Checking Directory Structure"
echo "-------------------------------"
DIRS=(
    ".claude/commands"
    ".claude/scripts"
    "docs/templates"
    "issues"
    "scripts"
    "utils"
    "tests"
    "examples"
)

for dir in "${DIRS[@]}"; do
    check_dir "$dir" || ALL_GOOD=false
done

echo ""
echo "6. Checking Templates"
echo "---------------------"
TEMPLATES=(
    "docs/templates/issue-template.md"
    "docs/templates/plan-template.md"
    "docs/templates/specification-template.md"
    "docs/templates/task-template.md"
)

for template in "${TEMPLATES[@]}"; do
    if [ -f "$template" ]; then
        echo -e "${GREEN}✅ Found: $template${NC}"
    else
        echo -e "${YELLOW}⚠️  Optional: $template (not critical)${NC}"
    fi
done

echo ""
echo "7. Checking Hook Configuration"
echo "------------------------------"
if [ -f ".claude/settings.json" ]; then
    # Check if hooks are configured
    if grep -q "PostToolUse" .claude/settings.json && \
       grep -q "PreToolUse" .claude/settings.json && \
       grep -q "Stop" .claude/settings.json && \
       grep -q "Notification" .claude/settings.json; then
        echo -e "${GREEN}✅ All hook types are configured${NC}"
    else
        echo -e "${YELLOW}⚠️  Some hook types may not be configured${NC}"
    fi
fi

echo ""
echo "8. Checking Script Permissions"
echo "------------------------------"
for script in .claude/scripts/*.py; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            echo -e "${GREEN}✅ Executable: $(basename $script)${NC}"
        else
            echo -e "${YELLOW}⚠️  Not executable: $(basename $script)${NC}"
            echo "   Fix with: chmod +x $script"
        fi
    fi
done

echo ""
echo "====================================="
if [ "$ALL_GOOD" = true ]; then
    echo -e "${GREEN}✨ All critical components are properly installed!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Start Claude Code: claude"
    echo "2. Try /help to see all commands"
    echo "3. Test a command: /multi-mind \"analyze this project\""
    echo "4. Read docs/agent-guides-integration.md for full guide"
else
    echo -e "${RED}⚠️  Some components are missing or need attention${NC}"
    echo ""
    echo "To fix:"
    echo "1. Re-run the setup script: ./scripts/setup.sh"
    echo "2. Check error messages above"
    echo "3. Manually install missing components if needed"
fi

echo ""
echo "💡 Tips:"
echo "- Restart Claude Code after making changes to hooks"
echo "- Use /hooks to reload hook configuration"
echo "- Check ~/.claude/research-activity.log for logged activities"
echo "====================================="