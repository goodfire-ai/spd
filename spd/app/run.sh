#!/bin/bash
# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
DIM='\033[2m'
BOLD='\033[1m'
UNDERLINE='\033[4m'
RESET='\033[0m'

FRONTEND_PORT=5173
BACKEND_PORT=8000

# Before starting servers
if lsof -i :$FRONTEND_PORT >/dev/null 2>&1; then
    echo -e "  ${YELLOW}⚠${RESET} Port $FRONTEND_PORT already in use"
    exit 1
fi

if lsof -i :$BACKEND_PORT >/dev/null 2>&1; then
    echo -e "  ${YELLOW}⚠${RESET} Port $BACKEND_PORT already in use"
    exit 1
fi

# Fancy header
echo -e "${BOLD}🚀 Starting development servers${RESET}"
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

# Start backend
echo -en "${DIM}  ▸ Starting backend...${RESET}"
uv run python spd/app/backend/controller.py 2>&1 | \
    grep -e "ERROR" -e "CRITICAL" --line-buffered &
BACKEND_PID=$!
echo -e "\r  ${GREEN}✓${RESET} Backend started        "

# Frontend deps
cd spd/app/frontend
echo -en "${DIM}  ▸ Installing frontend dependencies...${RESET}"
if npm ci --silent 2>/dev/null; then
    echo -e "\r  ${GREEN}✓${RESET} Frontend dependencies installed        "
else
    echo -e "\r  ${YELLOW}⚠${RESET} Frontend dependency installation had issues"
fi

# Start frontend
echo -en "${DIM}  ▸ Starting frontend...${RESET}"
npm run dev 2>&1 | grep -e "ERROR" -e "Failed" --line-buffered &
FRONTEND_PID=$!

# Wait for frontend to be ready
until curl -s http://localhost:$FRONTEND_PORT >/dev/null 2>&1; do
    sleep 0.5
done

echo -e "\r  ${GREEN}✓${RESET} Frontend started        "
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

# Success message
echo ""
echo -e "${GREEN}${BOLD}✓ Ready!${RESET}"
echo ""
echo -e "${DIM}Backend   http://localhost:$BACKEND_PORT/${RESET}"
echo -e "${BOLD}Frontend  ${GREEN}${BOLD}${UNDERLINE}http://localhost:$FRONTEND_PORT/${RESET}"
echo 
echo -e "${DIM}  Press Ctrl+C to stop${RESET}"

# Cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo -e '\n${DIM}👋 Shutting down...${RESET}'" EXIT
wait