#!/bin/bash
set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
DIM='\033[2m'
BOLD='\033[1m'
UNDERLINE='\033[4m'
RESET='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGFILE="$SCRIPT_DIR/run.log"
STARTUP_TIMEOUT=30  # seconds to wait for services to start

# Track PIDs for cleanup
BACKEND_PID=""
FRONTEND_PID=""

# Function to kill process and all its descendants
kill_process_tree() {
    local pid=$1
    local signal=${2:-TERM}

    if [ -z "$pid" ]; then
        return
    fi

    # Get all descendant PIDs (children, grandchildren, etc.)
    local descendants=$(pgrep -P "$pid" 2>/dev/null || true)

    # Recursively kill descendants first
    for child in $descendants; do
        kill_process_tree "$child" "$signal"
    done

    # Then kill the process itself
    kill -"$signal" "$pid" 2>/dev/null || true
}

# Cleanup function
cleanup() {
    echo -e "\n👋 Shutting down..."

    # Kill process trees gracefully with TERM first
    if [ -n "$BACKEND_PID" ]; then
        kill_process_tree "$BACKEND_PID" TERM
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill_process_tree "$FRONTEND_PID" TERM
    fi

    # Give processes a moment to cleanup gracefully
    sleep 0.5

    # Force kill if still running
    if [ -n "$BACKEND_PID" ]; then
        kill_process_tree "$BACKEND_PID" KILL
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill_process_tree "$FRONTEND_PID" KILL
    fi

    exit 0
}

trap cleanup EXIT INT TERM

# Function to find an available port
find_available_port() {
    local start_port=${1:-8000}
    local port=$start_port

    while lsof -i :$port >/dev/null 2>&1; do
        port=$((port + 1))
        if [ $port -gt $((start_port + 100)) ]; then
            echo -e "${RED}✗${RESET} Could not find available port after checking 100 ports from $start_port" >&2
            exit 1
        fi
    done

    echo $port
}

# Function to wait for service with timeout
wait_for_service() {
    local url=$1
    local timeout=$2
    local service_name=$3
    local elapsed=0

    while ! curl -s "$url" >/dev/null 2>&1; do
        sleep 0.5
        elapsed=$(echo "$elapsed + 0.5" | bc)

        if (( $(echo "$elapsed >= $timeout" | bc -l) )); then
            echo -e "\r  ${RED}✗${RESET} $service_name failed to start within ${timeout}s        " >&2
            echo -e "${DIM}Check $LOGFILE for details${RESET}" >&2
            exit 1
        fi
    done
}

# Initialize logfile
rm -f "$LOGFILE"
touch "$LOGFILE"

# Find available ports
echo -e "${DIM}Finding available ports...${RESET}"
BACKEND_PORT=$(find_available_port 8000)
FRONTEND_PORT=$(find_available_port 5173)

echo -e "${DIM}  Backend port: $BACKEND_PORT${RESET}"
echo -e "${DIM}  Frontend port: $FRONTEND_PORT${RESET}"
echo ""

# Fancy header
echo -e "${BOLD}🚀 Starting development servers${RESET}"
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

# Start backend in its own process group
echo -en "${DIM}  ▸ Starting backend...${RESET}"
(
    cd "$SCRIPT_DIR/../.."
    uv run python -u spd/app/backend/controller.py --port $BACKEND_PORT 2>&1 | \
    sed -u 's/^/[BACKEND] /' >> "$LOGFILE"
) &

BACKEND_PID=$!

# Wait for backend to be healthy
wait_for_service "http://localhost:$BACKEND_PORT" "$STARTUP_TIMEOUT" "Backend"
echo -e "\r  ${GREEN}✓${RESET} Backend started (port $BACKEND_PORT)        "

# Install frontend dependencies
echo -en "${DIM}  ▸ Installing frontend dependencies...${RESET}"
(
    cd "$SCRIPT_DIR/frontend"
    if npm ci --silent 2>/dev/null; then
        echo -e "\r  ${GREEN}✓${RESET} Frontend dependencies installed        "
    else
        echo -e "\r  ${YELLOW}⚠${RESET} Frontend dependency installation had issues (trying anyway)"
    fi
)

# Start frontend in its own process group
echo -en "${DIM}  ▸ Starting frontend...${RESET}"
(
    cd "$SCRIPT_DIR/frontend"
    VITE_API_URL=http://localhost:$BACKEND_PORT npm run dev -- --port $FRONTEND_PORT 2>&1 | \
    sed -u 's/^/[FRONTEND] /' >> "$LOGFILE"
) &

FRONTEND_PID=$!

# Wait for frontend to be healthy
wait_for_service "http://localhost:$FRONTEND_PORT" "$STARTUP_TIMEOUT" "Frontend"
echo -e "\r  ${GREEN}✓${RESET} Frontend started (port $FRONTEND_PORT)        "

echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

# Success message
echo ""
echo -e "${GREEN}${BOLD}✓ Ready!${RESET}"
echo ""
echo -e "${DIM}Backend   http://localhost:$BACKEND_PORT/${RESET}"
echo -e "${BOLD}Frontend  ${GREEN}${BOLD}${UNDERLINE}http://localhost:$FRONTEND_PORT/${RESET}"
echo -e "${BOLD}Logfile   ${UNDERLINE}$LOGFILE${RESET}"
echo ""
echo -e "${DIM}  Press Ctrl+C to stop${RESET}"

# Wait indefinitely - cleanup trap will handle shutdown
wait
