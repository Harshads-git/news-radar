# ==============================================================================
# Makefile — News Radar Development Commands
# ==============================================================================
# Usage: make <target>
# Run `make help` to see all available targets.
#
# Windows note: Use Git Bash or WSL if native make is not installed.
# Install make on Windows via: winget install GnuWin32.Make
# ==============================================================================

.PHONY: help install lint format check test test-unit test-integration \
        run run-dry setup clean coverage

# Default target — show help
.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Colors for terminal output
# ---------------------------------------------------------------------------
CYAN  := \033[0;36m
GREEN := \033[0;32m
RESET := \033[0m

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help: ## Show this help message
	@echo ""
	@echo "$(CYAN)📡 News Radar — Development Commands$(RESET)"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)  %-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ---------------------------------------------------------------------------
# Setup & Installation
# ---------------------------------------------------------------------------

install: ## Install all dependencies (including dev extras)
	uv sync --extra dev
	@echo "$(GREEN)✓ Dependencies installed$(RESET)"

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------

lint: ## Run ruff linter (check only, no fixes)
	uv run ruff check src/ tests/
	@echo "$(GREEN)✓ Lint passed$(RESET)"

lint-fix: ## Run ruff linter and auto-fix issues
	uv run ruff check src/ tests/ --fix
	@echo "$(GREEN)✓ Lint fixes applied$(RESET)"

format: ## Format code with ruff (replaces Black)
	uv run ruff format src/ tests/
	@echo "$(GREEN)✓ Code formatted$(RESET)"

format-check: ## Check formatting without modifying files
	uv run ruff format src/ tests/ --check

check: lint format-check ## Run all quality checks (lint + format) — no modifications
	@echo "$(GREEN)✓ All quality checks passed$(RESET)"

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test: ## Run full test suite with coverage report
	uv run pytest tests/ -q

test-unit: ## Run only unit tests (fast, no I/O)
	uv run pytest tests/ -m "unit" -q

test-integration: ## Run only integration tests (slower, may need API keys)
	uv run pytest tests/ -m "integration" -q

test-verbose: ## Run tests with full verbose output
	uv run pytest tests/ -v --tb=long

coverage: ## Generate HTML coverage report and open it
	uv run pytest tests/ --cov=src --cov-report=html:htmlcov -q
	@echo "$(GREEN)✓ Coverage report at htmlcov/index.html$(RESET)"

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

run: ## Run the full news radar pipeline
	uv run python -m src.main --run

run-dry: ## Run pipeline in dry-run mode (no saves, no notifications)
	uv run python -m src.main --dry-run

setup: ## Run the interactive setup wizard
	uv run python -m src.main --setup

check-setup: ## Validate environment, API keys, and config
	uv run python scripts/check_setup.py

# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

clean: ## Remove generated files (cache, coverage, __pycache__)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned build artifacts$(RESET)"

clean-data: ## Remove generated briefings and AI cache (CAREFUL: data loss!)
	@echo "WARNING: This will delete all generated briefings and cache."
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	rm -rf data/briefings/*.json data/cache/*.json data/run_log.json
	@echo "$(GREEN)✓ Runtime data cleaned$(RESET)"
