# Contributing to News Radar

Thank you for your interest in contributing! This document explains how to get
the project running locally, the code standards we follow, and the workflow for
submitting changes.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Setup](#local-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Running Tests](#running-tests)
- [Commit Message Convention](#commit-message-convention)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Adding a New Source Type](#adding-a-new-source-type)

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | [python.org](https://python.org) |
| `uv` | Latest | `pip install uv` or [astral.sh/uv](https://github.com/astral-sh/uv) |
| Git | 2.40+ | [git-scm.com](https://git-scm.com) |
| `make` | Any | `winget install GnuWin32.Make` (Windows) / built-in on macOS/Linux |

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/Harshads-git/news-radar.git
cd news-radar

# 2. Install all dependencies (including dev tools)
make install
# or directly:
uv sync --extra dev

# 3. Copy the environment template and fill in your keys
cp .env.example .env
# Edit .env with your preferred editor — at minimum set AI_MODEL + one API key

# 4. Verify your setup
make check-setup
# This validates env vars, imports, and a test AI call

# 5. Run the interactive setup wizard (optional but recommended)
make setup
```

---

## Project Structure

```
news-radar/
├── src/
│   ├── config.py           ← All env var settings (Pydantic BaseSettings)
│   ├── models.py           ← Core data models (NewsItem, Briefing, etc.)
│   ├── orchestrator.py     ← Pipeline controller — wires everything together
│   ├── search.py           ← DuckDuckGo web context fetcher
│   ├── exceptions.py       ← Custom exception hierarchy
│   ├── logger.py           ← Rich-powered centralized logger
│   ├── scrapers/           ← One file per source type
│   ├── ai/                 ← AI provider adapters + scorer + summarizer
│   ├── services/           ← Output: email, webhook, site generator
│   └── storage/            ← Persistence: briefing store + cache
├── tests/
│   ├── conftest.py         ← Shared fixtures (reuse these in your tests!)
│   └── test_*/             ← Mirrors src/ structure
├── data/
│   ├── sources.json        ← What to watch (edit this, don't touch code)
│   ├── briefings/          ← Generated daily outputs (gitignored)
│   └── cache/              ← AI response cache (gitignored)
├── docs/                   ← GitHub Pages site content
├── scripts/                ← Operational shell scripts
├── Makefile                ← All dev commands
└── ruff.toml               ← Linter + formatter config
```

---

## Development Workflow

```bash
# Create a feature branch
git checkout -b feat/your-feature-name

# Make your changes, then run quality checks
make check         # lint + format check (read-only)
make lint-fix      # auto-fix lint issues
make format        # auto-format code

# Run tests
make test          # full suite with coverage
make test-unit     # fast unit tests only

# Commit using conventional commits (see below)
git add .
git commit -m "feat: add Twitter/X scraper"

# Push and open a PR
git push origin feat/your-feature-name
```

---

## Code Standards

### Linting & Formatting

We use **[ruff](https://docs.astral.sh/ruff/)** for both linting and formatting
(replaces flake8 + isort + Black). Configuration is in [`ruff.toml`](ruff.toml).

```bash
make check        # check without modifying
make lint-fix     # auto-fix lint issues
make format       # auto-format
```

Key rules enforced:
- **E/W** — PEP 8 style
- **F** — Pyflakes (unused imports, undefined names)
- **I** — Import ordering (isort)
- **B** — Bugbear (likely bugs)
- **UP** — Pyupgrade (modern Python syntax)

### Type Annotations

All functions in `src/` must have full type annotations. The `ANN` rules are
active for the `src/` directory. Tests have relaxed annotation rules.

```python
# Good
def fetch(self, source: SourceConfig) -> list[NewsItem]:
    ...

# Bad — missing return type
def fetch(self, source: SourceConfig):
    ...
```

### Pydantic Models

- Use `Field(description="...")` for every field — it auto-generates docs.
- Add `@field_validator` for input cleaning (strip whitespace, validate enums).
- Prefer `model_validator(mode="after")` for cross-field checks.
- Never mutate a model after creation — Pydantic models should be treated as
  immutable value objects.

### Error Handling

- Raise from the custom exception hierarchy in `src/exceptions.py`.
- Never bare-except (`except Exception: pass`).
- Scrapers must catch network errors and raise `FetchError`.
- AI calls must catch API errors and raise `AIError`.

---

## Running Tests

```bash
make test              # full suite + coverage report
make test-unit         # mark: @pytest.mark.unit (fast, no I/O)
make test-integration  # mark: @pytest.mark.integration (may hit real APIs)
make test-verbose      # full verbose output for debugging
make coverage          # HTML coverage report → htmlcov/index.html
```

### Writing Tests

1. **Use fixtures from `conftest.py`** — don't recreate objects by hand.
2. **Mock all network calls** — use `pytest-mock` or `httpx` mock transport.
3. **Group tests in classes** — `class TestSomething:` keeps related tests together.
4. **Name tests descriptively** — `test_rss_scraper_returns_news_items_on_success`.
5. **Assert one thing per test** — single, clear assertion per test method.
6. **Mark tests** — `@pytest.mark.unit`, `@pytest.mark.integration`, or `@pytest.mark.slow`.

```python
# Good test structure
class TestRssScraper:
    @pytest.mark.unit
    async def test_fetch_returns_list_of_news_items(self, rss_source, mock_httpx):
        scraper = RssScraper()
        items = await scraper.fetch(rss_source)
        assert isinstance(items, list)
        assert all(isinstance(i, NewsItem) for i in items)
```

**Minimum coverage requirement: 80%** (enforced by `--cov-fail-under=80`).

---

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short description>

[optional body]
[optional footer]
```

| Type | When to use |
|------|-------------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `chore` | Tooling, deps, config (no production code change) |
| `docs` | Documentation only |
| `test` | Adding or modifying tests |
| `refactor` | Code restructuring without behaviour change |
| `perf` | Performance improvement |
| `ci` | GitHub Actions / CI configuration |
| `release` | Version bump or release tag |

**Examples:**
```bash
git commit -m "feat: implement Reddit scraper with rate limiting"
git commit -m "fix: handle empty RSS feed gracefully"
git commit -m "test: add unit tests for deduplication engine"
git commit -m "chore: upgrade pydantic to 2.10.0"
```

---

## Submitting a Pull Request

1. Make sure `make check` passes with zero errors.
2. Make sure `make test` passes with ≥80% coverage.
3. Update `CHANGELOG.md` with a summary of your change.
4. Open a PR with a descriptive title following the commit convention.
5. Fill in the PR template — describe the problem, solution, and how to test.

A maintainer will review your PR within 48 hours. We may request changes before
merging.

---

## Adding a New Source Type

1. Create `src/scrapers/<type>.py` implementing `BaseScraper`.
2. Register it in `src/scrapers/__init__.py` (the `ScraperRegistry`).
3. Add the new `type` string to `SourceConfig.validate_type` in `src/models.py`.
4. Add an example entry with `"enabled": false` to `data/sources.json`.
5. Write tests in `tests/test_scrapers/test_<type>.py` using mocked HTTP.
6. Add a row to the Features table in `README.md`.

---

## Questions?

Open a [GitHub Discussion](https://github.com/Harshads-git/news-radar/discussions)
or file an [Issue](https://github.com/Harshads-git/news-radar/issues).
