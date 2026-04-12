#!/usr/bin/env bash
# Run all tests for the drone swarm project (backend + frontend).
# Usage: ./scripts/test.sh
#
# Exit codes:
#   0 — all tests passed
#   1 — one or more test suites failed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

backend_ok=true
frontend_ok=true

# --- Colors (if terminal supports them) ---
if [ -t 1 ]; then
  GREEN='\033[0;32m'
  RED='\033[0;31m'
  BOLD='\033[1m'
  RESET='\033[0m'
else
  GREEN=''
  RED=''
  BOLD=''
  RESET=''
fi

echo ""
echo -e "${BOLD}========================================${RESET}"
echo -e "${BOLD}  Backend Tests (pytest)${RESET}"
echo -e "${BOLD}========================================${RESET}"
echo ""

cd "$BACKEND_DIR"
if uv run pytest -x -q 2>&1; then
  echo ""
  echo -e "${GREEN}Backend tests PASSED${RESET}"
else
  echo ""
  echo -e "${RED}Backend tests FAILED${RESET}"
  backend_ok=false
fi

echo ""
echo -e "${BOLD}========================================${RESET}"
echo -e "${BOLD}  Frontend Tests (vitest)${RESET}"
echo -e "${BOLD}========================================${RESET}"
echo ""

cd "$FRONTEND_DIR"
if npx vitest run 2>&1; then
  echo ""
  echo -e "${GREEN}Frontend tests PASSED${RESET}"
else
  echo ""
  echo -e "${RED}Frontend tests FAILED${RESET}"
  frontend_ok=false
fi

# --- Summary ---
echo ""
echo -e "${BOLD}========================================${RESET}"
echo -e "${BOLD}  Summary${RESET}"
echo -e "${BOLD}========================================${RESET}"
echo ""

if $backend_ok && $frontend_ok; then
  echo -e "${GREEN}All tests passed.${RESET}"
  exit 0
else
  $backend_ok  || echo -e "${RED}  FAIL: Backend${RESET}"
  $frontend_ok || echo -e "${RED}  FAIL: Frontend${RESET}"
  echo ""
  exit 1
fi
