#!/bin/bash
# Setup script for AI Safety Research Template
# This script helps integrate the template into your existing project

set -e

echo "🚀 AI Safety Research Template Setup"
echo "===================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEMPLATE_ROOT="$(dirname "$SCRIPT_DIR")"

# Function to create directory if it doesn't exist
create_dir_if_not_exists() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        echo "✅ Created directory: $1"
    else
        echo "ℹ️  Directory already exists: $1"
    fi
}

# Function to copy file with backup
copy_with_backup() {
    local src="$1"
    local dest="$2"
    
    if [ -f "$dest" ]; then
        backup_file="${dest}.backup.$(date +%Y%m%d_%H%M%S)"
        cp "$dest" "$backup_file"
        echo "📋 Backed up existing file to: $backup_file"
    fi
    
    cp "$src" "$dest"
    echo "✅ Copied: $(basename "$src") to $dest"
}

# Check if we're in a git repository
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo "✅ Git repository detected"
    PROJECT_ROOT=$(git rev-parse --show-toplevel)
else
    echo "⚠️  No git repository detected. Using current directory as project root."
    PROJECT_ROOT=$(pwd)
fi

echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Ask user what they want to set up
echo "What would you like to set up?"
echo "1) Full template (recommended for new projects)"
echo "2) Claude commands only"
echo "3) Scripts and utilities only"
echo "4) Custom selection"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        # Full setup
        echo ""
        echo "Setting up full template..."
        
        # Create directories
        create_dir_if_not_exists "$PROJECT_ROOT/.claude/commands"
        create_dir_if_not_exists "$PROJECT_ROOT/.claude/scripts"
        create_dir_if_not_exists "$PROJECT_ROOT/docs/templates"
        create_dir_if_not_exists "$PROJECT_ROOT/issues"
        create_dir_if_not_exists "$PROJECT_ROOT/scripts"
        create_dir_if_not_exists "$PROJECT_ROOT/utils"
        create_dir_if_not_exists "$PROJECT_ROOT/tests"
        create_dir_if_not_exists "$PROJECT_ROOT/examples"
        
        # Copy Claude commands
        echo ""
        echo "Installing Claude commands..."
        cp -r "$TEMPLATE_ROOT/.claude/commands/"* "$PROJECT_ROOT/.claude/commands/" 2>/dev/null || true
        
        # Copy scripts
        echo "Installing scripts..."
        cp -r "$TEMPLATE_ROOT/.claude/scripts/"* "$PROJECT_ROOT/.claude/scripts/" 2>/dev/null || true
        
        # Copy CLAUDE.md with backup
        if [ -f "$TEMPLATE_ROOT/CLAUDE.md" ]; then
            copy_with_backup "$TEMPLATE_ROOT/CLAUDE.md" "$PROJECT_ROOT/CLAUDE.md"
        fi
        
        # Copy .gitignore entries
        if [ -f "$PROJECT_ROOT/.gitignore" ]; then
            echo ""
            echo "Updating .gitignore..."
            # Add our entries if they don't exist
            grep -q "CLAUDE.local.md" "$PROJECT_ROOT/.gitignore" || echo -e "\n# Claude Code\nCLAUDE.local.md\n.claude/settings.local.json" >> "$PROJECT_ROOT/.gitignore"
        else
            cp "$TEMPLATE_ROOT/.gitignore" "$PROJECT_ROOT/.gitignore"
        fi
        
        # Copy documentation
        echo "Installing documentation..."
        cp -r "$TEMPLATE_ROOT/docs/"* "$PROJECT_ROOT/docs/" 2>/dev/null || true
        ;;
        
    2)
        # Claude commands only
        echo ""
        echo "Installing Claude commands only..."
        create_dir_if_not_exists "$PROJECT_ROOT/.claude/commands"
        cp -r "$TEMPLATE_ROOT/.claude/commands/"* "$PROJECT_ROOT/.claude/commands/"
        ;;
        
    3)
        # Scripts and utilities only
        echo ""
        echo "Installing scripts and utilities..."
        create_dir_if_not_exists "$PROJECT_ROOT/scripts"
        create_dir_if_not_exists "$PROJECT_ROOT/utils"
        create_dir_if_not_exists "$PROJECT_ROOT/.claude/scripts"
        
        cp -r "$TEMPLATE_ROOT/scripts/"* "$PROJECT_ROOT/scripts/" 2>/dev/null || true
        cp -r "$TEMPLATE_ROOT/.claude/scripts/"* "$PROJECT_ROOT/.claude/scripts/" 2>/dev/null || true
        ;;
        
    4)
        # Custom selection
        echo ""
        echo "Custom setup - select what to install:"
        
        read -p "Install Claude commands? (y/n): " install_commands
        read -p "Install scripts? (y/n): " install_scripts
        read -p "Install CLAUDE.md? (y/n): " install_claude_md
        read -p "Install documentation? (y/n): " install_docs
        read -p "Create directory structure? (y/n): " create_dirs
        
        if [ "$create_dirs" = "y" ]; then
            create_dir_if_not_exists "$PROJECT_ROOT/.claude/commands"
            create_dir_if_not_exists "$PROJECT_ROOT/.claude/scripts"
            create_dir_if_not_exists "$PROJECT_ROOT/docs/templates"
            create_dir_if_not_exists "$PROJECT_ROOT/issues"
            create_dir_if_not_exists "$PROJECT_ROOT/scripts"
            create_dir_if_not_exists "$PROJECT_ROOT/utils"
        fi
        
        if [ "$install_commands" = "y" ]; then
            create_dir_if_not_exists "$PROJECT_ROOT/.claude/commands"
            cp -r "$TEMPLATE_ROOT/.claude/commands/"* "$PROJECT_ROOT/.claude/commands/"
        fi
        
        if [ "$install_scripts" = "y" ]; then
            create_dir_if_not_exists "$PROJECT_ROOT/.claude/scripts"
            cp -r "$TEMPLATE_ROOT/.claude/scripts/"* "$PROJECT_ROOT/.claude/scripts/"
        fi
        
        if [ "$install_claude_md" = "y" ] && [ -f "$TEMPLATE_ROOT/CLAUDE.md" ]; then
            copy_with_backup "$TEMPLATE_ROOT/CLAUDE.md" "$PROJECT_ROOT/CLAUDE.md"
        fi
        
        if [ "$install_docs" = "y" ]; then
            create_dir_if_not_exists "$PROJECT_ROOT/docs"
            cp -r "$TEMPLATE_ROOT/docs/"* "$PROJECT_ROOT/docs/" 2>/dev/null || true
        fi
        ;;
        
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "✨ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Review the installed files and customize them for your project"
echo "2. Check the README.md for detailed usage instructions"
echo "3. Run 'claude' in your project directory to start using the template"
echo ""
echo "Available Claude commands:"
echo "  /multi-mind - Multi-agent analysis"
echo "  /search-all - Search conversation history"
echo "  /page - Save session state"
echo "  /analyze-function - Deep code analysis"
echo "  /crud-claude-commands - Create custom commands"
echo ""
echo "Happy researching! 🎯"