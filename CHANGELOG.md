# Changelog

All notable changes to News Radar. Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-07-20

30-day build complete. First stable release.

### Summary
Built from scratch over 30 days (1 hour/day) as a personal project.
Full async Python pipeline: fetch → dedup → score → enrich → summarize → store → render → deliver.

---

## Build Log

### Day 1 — Scaffold
- `pyproject.toml`, `uv` workspace, `src/` layout, `README.md`, `.env.example`
- Pydantic `Settings` loading from `.env`
- `NewsItem`, `ScoredItem`, `SummarizedItem`, `Briefing` data models

### Day 2 — RSS Scraper
- `RssScraper` using `feedparser`; concurrent fetch with `asyncio.gather`
- `SourceConfig` / `SourcesConfig` with `sources.json`
- Structured JSONL event logger (`pipeline/logger.py`)

### Day 3 — Hacker News Scraper
- `HackerNewsScraper` using the HN Firebase REST API
- Top stories + new stories; configurable `limit`
- `ScraperFactory` to dispatch by source type

### Day 4 — Reddit Scraper
- `RedditScraper` using Reddit JSON API (no OAuth needed)
- Hot / new / top / rising sort support
- Per-source fetch health tracking

### Day 5 — Deduplication
- URL normalisation (strip UTM params, trailing slash, www)
- Jaccard coefficient title similarity (threshold 0.65)
- `Deduplicator` — O(n²) for small batches, acceptable for ≤1000 items

### Day 6 — AI Scorer (v1)
- `AIProviderFactory` — OpenAI, Google Gemini, Anthropic via single interface
- `NewsScorer` — async semaphore-limited batch scoring
- Rubric-based prompt: score 0–10 with one-sentence reason + topic labels

### Day 7 — AI Summarizer
- `NewsSummarizer` — 3-paragraph summary + key takeaways + AI headline
- DuckDuckGo web context enrichment per story
- Executive summary for the full briefing

### Day 8 — Briefing Builder + Store
- `BriefingBuilder` assembles ranked `Briefing` from scored + summarized items
- `BriefingStore` persists to `data/briefings/YYYY-MM-DD.json`
- `load_latest()` / `load_by_date()` helpers

### Day 9 — HTML Renderer
- `render_html()` — self-contained dark-mode HTML page
- Score colour coding (green → yellow → red)
- Collapsible story cards via CSS `:target` trick

### Day 10 — Markdown Renderer + GitHub Pages
- `render_markdown()` — GitHub-flavoured Markdown briefing
- `GitHubPagesWriter` — writes `docs/index.html`, `docs/YYYY-MM-DD.html`, `docs/archive.html`
- Atomic file writes (tmp → rename)

### Day 11 — Email Delivery
- `EmailDelivery` with SMTP / STARTTLS
- MIME multipart/alternative (plain text + HTML)
- Subject line: `📡 News Radar — YYYY-MM-DD · N stories · Top: <headline>`

### Day 12 — Discord Delivery
- `DiscordWebhookDelivery` — rich embed cards with score colours
- Top-5 stories as embed fields; footer with full pipeline stats
- Retry on 429 rate limit

### Day 13 — Slack Delivery
- `SlackWebhookDelivery` — Block Kit layout
- Per-story blocks with score badge, source label, CTA button
- Configurable story count (`SLACK_MAX_STORIES`)

### Day 14 — Custom Webhook
- `CustomWebhookDelivery` — POST full briefing JSON
- Configurable headers and timeout
- Delivery dispatcher (`DeliveryDispatcher`) — routes to all configured channels

### Day 15 — Orchestrator + CLI
- `Orchestrator` — wires all 8 stages, handles partial failures
- `main.py` CLI: `--run`, `--dry-run`, `--check`, `--status`, `--version`
- GitHub Actions `daily.yml` cron (07:00 UTC)

### Day 16 — GitHub Actions CI
- `ci.yml` — lint (ruff) + test matrix (Python 3.11, 3.12)
- Coverage enforcement (65% minimum)
- Coverage summary posted to GitHub Actions job summary

### Day 17 — Setup Wizard
- Interactive `--setup` command (questionary-based)
- Validates API keys, writes `.env`, previews first run
- `--sources-list` to inspect source health

### Day 18 — Source Health Tracking
- `SourceHealth` — per-source error counters, consecutive failures
- Auto-disable sources after N consecutive fetch errors
- `--source-stats` CLI with Rich tables

### Day 19 — Score Cache
- `ScoreCache` — JSON-backed cache, 24h TTL per story URL
- Cache-first in `NewsScorer` (hit → skip AI call)
- `--cache-stats` CLI: hit rate, size, oldest/newest entries

### Day 20 — Topic Clustering
- AI assigns topic labels per story (`ai_topics` field on `ScoredItem`)
- `BriefingBuilder` aggregates `top_topics` from all story labels
- Topic pills in HTML renderer

### Day 21 — Semantic Deduplication
- Embedding-based similarity using text chunking + cosine distance
- Configurable threshold (default 0.85)
- Falls back to Jaccard if embedding provider unavailable

### Day 22 — Rubric Scoring Refinement
- Multi-dimension rubric: relevance, novelty, quality, impact
- Weighted score aggregation
- Prompt versioning (stored with scored item)

### Day 23 — Cost Ledger
- `CostLedger` — JSONL append-log of every AI call (tokens, model, cost estimate)
- `--cost-report` CLI: daily spend table + weekly summary
- Token counting via `tiktoken` (or character estimate fallback)

### Day 24 — Async Concurrency Tuning
- Semaphore limits configurable per stage (score / summarize / enrich)
- `asyncio.TaskGroup` for structured concurrency (Python 3.11+)
- Per-stage timing logged to event log

### Day 25 — GitHub Scraper
- `GitHubScraper` — trending repositories via GitHub API
- Stars, forks, language, description → `NewsItem`
- Added to default `sources.json`

### Day 26 — Retry + Backoff
- `with_ai_retry()` decorator — exponential backoff + jitter
- `_extract_retry_after()` — parses `Retry-After` / error message for hints
- Per-provider backoff tuning

### Day 27 — Retry Budget + Circuit Breaker
- `RetryBudget` — three-state FSM (CLOSED → OPEN → HALF_OPEN → CLOSED)
- 20-event rolling window for error-rate calculation
- 40% error rate → `throttle_down` (concurrency − 1); <20% → `throttle_up`
- 5 consecutive failures → circuit OPEN; 60s probe delay → HALF_OPEN
- All events flushed to `data/retry_budget.jsonl`
- `--retry-stats` CLI: circuit event history + event-type summary

### Day 28 — GitHub Pages: Topic Index + Search
- `_write_search_index()` — cumulative `docs/search.json` (90-day window)
- `_write_topic_pages()` — per-topic `docs/topic-{slug}.html` with story history
- `_write_topics_index()` — `docs/topics.html` responsive grid
- `_write_index()` updated — sticky search bar + topic nav pills injected
- Vanilla JS real-time story filter (no external libraries)
- Archive page updated with "Browse Topics" link

### Day 29 — Email Template Redesign
- `render_email_html()` — dedicated email-safe renderer
  - XHTML doctype + MSO conditional comments for Outlook 2007–2019
  - All styles inlined (no `<style>` blocks, no CSS variables)
  - Table-based 600px layout for maximum email client compatibility
  - Dark palette: score circles (green/amber/orange/red), topic pills, key-points bullets
  - First-paragraph summary + "Read article →" CTA per story
- Wired into `EmailDelivery._build_message()`
- `--preview-email` CLI: renders latest briefing, saves `data/email_preview.html`, opens in browser

### Day 30 — v1.0.0 Polish
- Version bumped to `1.0.0` (`pyproject.toml` + `src/__version__.py`)
- `--version` flag reads from `__version__.py` as fallback
- `README.md` full rewrite: badges, full CLI reference, architecture table, GitHub Pages output table
- `CHANGELOG.md` — this file
- CI workflow updated: coverage threshold raised to 70%
- Final push: all test files committed (Days 27–29)

---

## Statistics

| Metric | Count |
|--------|-------|
| Days of work | 30 |
| Lines of source code | ~8,500 |
| Test files | 20+ |
| Unit tests | 550+ |
| CLI commands | 16 |
| Pipeline stages | 8 |
| AI providers supported | 3 (OpenAI, Gemini, Claude) |
| Delivery channels | 4 (Email, Discord, Slack, Webhook) |
| GitHub Pages files per run | 8+ |
| Estimated cost per day | $0.01–$0.03 |

---

[1.0.0]: https://github.com/Harshads-git/news-radar/releases/tag/v1.0.0
