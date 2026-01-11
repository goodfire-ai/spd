#!/bin/bash
set -e  # Exit on error

echo "🚀 Starting RunPod setup script..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Navigate to project directory
cd /workspace/spd || { echo "Error: /workspace/spd directory not found"; exit 1; }

# 1. Install uv
print_step "Installing uv..."
if command -v uv &> /dev/null; then
    print_success "uv is already installed"
    uv --version
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH - it installs to $HOME/.local/bin
    export PATH="$HOME/.local/bin:$PATH"
    # Also source the env file if it exists (for future shell sessions)
    if [ -f "$HOME/.local/bin/env" ]; then
        source "$HOME/.local/bin/env"
    fi
    # Add to .bashrc so uv is available in all shells
    if ! grep -q '$HOME/.local/bin' "$HOME/.bashrc" 2>/dev/null; then
        echo '' >> "$HOME/.bashrc"
        echo '# Add uv to PATH' >> "$HOME/.bashrc"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    fi
    print_success "uv installed successfully"
fi

# Ensure uv is in PATH before proceeding
export PATH="$HOME/.local/bin:$PATH"

# 2. Install Python 3.13 via uv
print_step "Installing Python 3.13 via uv..."
uv python install 3.13
print_success "Python 3.13 installed"

# 3. Install nvm (Node Version Manager)
print_step "Installing nvm..."
if [ -s "$HOME/.nvm/nvm.sh" ]; then
    print_success "nvm is already installed"
    source "$HOME/.nvm/nvm.sh"
else
    export NVM_DIR="$HOME/.nvm"
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    [ -s "$HOME/.bashrc" ] && echo 'export NVM_DIR="$HOME/.nvm"' >> "$HOME/.bashrc"
    [ -s "$HOME/.bashrc" ] && echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"' >> "$HOME/.bashrc"
    print_success "nvm installed successfully"
fi

# 4. Install Node.js (using latest LTS version)
print_step "Installing Node.js via nvm..."
# Ensure nvm is loaded
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm install --lts
nvm use --lts
nvm alias default node
print_success "Node.js $(node --version) installed"
print_success "npm $(npm --version) installed"

# 5. Run make install-dev (installs Python dependencies and dev tools)
print_step "Installing Python dependencies (dev mode)..."
# Run copy-templates and uv sync first
make copy-templates
uv sync
# Then install pre-commit hooks using uv run (since pre-commit is in the venv)
uv run pre-commit install
print_success "Python dependencies installed"

# 6. Run make install-app (installs frontend dependencies)
print_step "Installing frontend dependencies..."
make install-app
print_success "Frontend dependencies installed"

# 7. Verify installations
print_step "Verifying installations..."
echo ""
echo "Python version: $(uv run python --version)"
echo "Node version: $(node --version)"
echo "npm version: $(npm --version)"
echo "uv version: $(uv --version)"
echo ""

print_success "Setup complete! 🎉"
echo ""
echo "⚠️  IMPORTANT: To use npm/node in this shell session, run:"
echo "  source ~/.bashrc"
echo ""
echo "Or manually load nvm:"
echo "  export NVM_DIR=\"\$HOME/.nvm\""
echo "  [ -s \"\$NVM_DIR/nvm.sh\" ] && \. \"\$NVM_DIR/nvm.sh\""
echo ""
echo "You can now:"
echo "  - Run 'make app' to start the application"
echo "  - Run 'make test' to run tests"
echo "  - Run 'make check' to run code checks"
echo ""
echo "Note: New terminals will automatically load uv and nvm from .bashrc"

