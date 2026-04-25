.PHONY: install install-hooks run backend frontend test lint typecheck clean reset help

# Default: show help
help:
	@echo "Drone Swarm Coordinator — Commands"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make install       Install all dependencies (Python + Node)"
	@echo "  make install-hooks Install git pre-commit hooks (lint + typecheck)"
	@echo "  make run           Start backend + frontend (open http://localhost:5173)"
	@echo "  make backend    Start backend server only"
	@echo "  make frontend   Start frontend dev server only"
	@echo "  make test       Run all tests"
	@echo "  make lint       Lint all code"
	@echo "  make typecheck  Type-check all code"
	@echo "  make clean      Remove build artifacts"
	@echo ""
	@echo "In-browser controls:"
	@echo "  Mouse drag=orbit  Scroll=zoom  Click=select drone  Right-click=send drone"
	@echo "  Space=pause  1/2/3=speed  N=new sim  T=chat"

# Install all dependencies
install:
	@echo "Installing backend dependencies..."
	cd backend && uv venv && uv pip install -e ".[dev]"
	@echo ""
	@echo "Installing frontend dependencies..."
	@eval "$$(fnm env 2>/dev/null)" && cd frontend && npm install
	@echo ""
	@echo "Done! Run 'make install-hooks' to enable pre-commit checks, or 'make run' to start."

# Install git pre-commit hooks — requires pre-commit (pip install pre-commit or
# brew install pre-commit). Runs ruff + tsc + vitest + standard hygiene on every
# git commit.
install-hooks:
	@command -v pre-commit >/dev/null 2>&1 || { \
		echo "pre-commit not found. Install with: pip install pre-commit"; \
		exit 1; \
	}
	pre-commit install
	@echo "Hooks installed. They run on every 'git commit'."
	@echo "To run them manually across the whole tree: pre-commit run --all-files"

# Run both backend and frontend
run:
	@echo "Starting Drone Swarm Coordinator..."
	@echo "Backend: http://localhost:8765  Frontend: http://localhost:5173"
	@echo "Press Ctrl+C to stop."
	@echo ""
	@trap 'kill 0' EXIT; \
	cd backend && .venv/bin/python -m src.server.main & \
	sleep 2 && \
	eval "$$(fnm env 2>/dev/null)" && cd frontend && npx vite --host & \
	wait

# Backend only
backend:
	cd backend && .venv/bin/python -m src.server.main

# Frontend only
frontend:
	@eval "$$(fnm env 2>/dev/null)" && cd frontend && npx vite --host

# Run headless simulation (no browser needed)
headless:
	cd backend && .venv/bin/python -m src.simulation.run --headless --ticks 1000

# Run all tests
test:
	cd backend && .venv/bin/pytest -x -q

# Lint all code
lint:
	cd backend && .venv/bin/ruff check src/ tests/
	@eval "$$(fnm env 2>/dev/null)" && cd frontend && npx tsc --noEmit

# Type-check all code
typecheck:
	@eval "$$(fnm env 2>/dev/null)" && cd frontend && npx tsc --noEmit
	cd backend && .venv/bin/ruff check src/

# Format all code
format:
	cd backend && .venv/bin/ruff format src/ tests/
	@eval "$$(fnm env 2>/dev/null)" && cd frontend && npx prettier --write "src/**/*.ts"

# Clean build artifacts
clean:
	rm -rf frontend/dist frontend/node_modules/.vite
	find backend -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find backend -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
