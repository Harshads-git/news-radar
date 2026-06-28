# 📡 News Radar

> *I was spending 45 minutes every morning reading Hacker News, Reddit, and various tech blogs just to figure out what was worth reading. So I built this.*

A personal AI-powered news briefing pipeline that fetches stories from the sources I actually care about, scores them by relevance to my interests, summarizes the good ones, and delivers a clean daily briefing — automatically, every morning.

Built over 30 days, 1 hour/day, as a personal project to actually ship something end-to-end with real AI tooling.

---

## What it does

Every day at 07:00 UTC it:

1. **Fetches** from Hacker News (top + new), Reddit (r/MachineLearning, r/LocalLLaMA, r/Python, r/devops), and 8 RSS feeds (OpenAI, DeepMind, HuggingFace, TechCrunch, Ars Technica, GitHub Blog)
2. **Deduplicates** — removes near-duplicate stories by URL normalization and Jaccard title similarity
3. **Scores** each story from 0-10 using GPT-4o-mini based on my configured interests
4. **Summarizes** only the stories that scored above my threshold (default: 6/10)
5. **Builds** a briefing with an AI-generated executive summary of the day's themes
6. **Delivers** via email, Discord webhook, or just saves to GitHub Pages
7. **Publishes** the HTML briefing to this repo's GitHub Pages

The whole run takes about 45 seconds and costs roughly $0.02 in API credits.

---

## Why I built this from scratch

I tried existing tools — RSS aggregators, news apps, even some AI newsletter services. None of them let me control:
- *Which* sources get fetched
- *How* relevance is scored (my interests, not an algorithm's)
- *Where* the output goes
- The actual prompt that generates the summary

Building it myself meant I understand every part of it, I can debug it when it breaks, and I can extend it however I want. Also I just wanted an excuse to build a real async Python pipeline with proper testing.

---

## Live Briefing

→ **[harshads-git.github.io/news-radar](https://harshads-git.github.io/news-radar)**

*(Updates daily at 07:00 UTC / 12:30 IST)*

---

## Quick start

```bash
# Clone and install
git clone https://github.com/Harshads-git/news-radar.git
cd news-radar
uv sync

# Configure
cp .env.example .env
# Edit .env — add your OpenAI key and interests

# Validate your setup
uv run python -m src.main --check

# Run a dry run (no saves, no delivery — just fetch + score + summarize)
uv run python -m src.main --dry-run

# Full run
uv run python -m src.main --run
```

---

## Configuration

Everything lives in `.env`:

```env
# Required — at least one AI key
OPENAI_API_KEY=sk-...

# What you actually care about (the AI uses this to score relevance)
USER_INTERESTS=AI, machine learning, Python, open source, developer tools

# Only include stories that score above this (0-10)
SCORE_THRESHOLD=6

# Optional delivery
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
SMTP_USER=you@gmail.com
SMTP_PASSWORD=your-app-password
EMAIL_TO=you@gmail.com
```

Sources are configured in `data/sources.json` — you can enable/disable any source and set per-source fetch limits.

---

## Architecture

```
sources.json
    ↓
[FETCH] → RSS / HN API / Reddit API (concurrent)
    ↓
[DEDUP] → URL normalize + Jaccard title similarity
    ↓
[SCORE] → GPT-4o-mini scores each story (0-10) based on your interests
    ↓
[SUMMARIZE] → GPT-4o-mini generates 3-paragraph summary + key points
    ↓
[BUILD] → Assemble Briefing with executive summary
    ↓
[STORE] → data/briefings/YYYY-MM-DD.json
    ↓
[RENDER] → docs/YYYY-MM-DD.html + docs/index.html
    ↓
[DELIVER] → Email + Discord + Slack + Custom webhook
```

Each stage is independent — if the Reddit scraper goes down, the rest of the pipeline continues with HN and RSS results.

---

## Project structure

```
src/
├── main.py              # CLI entry point (--run, --dry-run, --check, --status)
├── orchestrator.py      # Wires all 8 pipeline stages together
├── config.py            # Settings (Pydantic BaseSettings, reads from .env)
├── models.py            # Pydantic data models (NewsItem, ScoredItem, Briefing...)
├── scrapers/            # RSS, HackerNews, Reddit scrapers + ScraperFactory
├── deduplicator.py      # URL normalization + Jaccard similarity dedup
├── ai/                  # AIProviderFactory, scorer, summarizer
├── briefing.py          # BriefingBuilder + BriefingStore
├── renderers/           # Markdown, HTML, GitHubPagesWriter
├── delivery/            # Email, Discord, Slack, custom webhook, dispatcher
├── pipeline/            # Structured JSONL event logger
└── setup/               # sources_loader.py

data/
├── sources.json         # Which sources to fetch from (edit this!)
├── briefings/           # Daily briefings as JSON
└── logs/                # JSONL pipeline event log (one file per day)

docs/                    # GitHub Pages output (auto-generated)
tests/                   # 450+ unit tests, 84% coverage
```

---

## Roadmap / Known issues

See the [Issues tab](https://github.com/Harshads-git/news-radar/issues) for the full list. Some things I'm actively thinking about:

- [ ] **Telegram delivery** — I actually use Telegram more than Discord (#3)
- [ ] **Weekly digest mode** — summarize the week's top 10 (#4)
- [ ] **AI score caching** — avoid re-scoring on failed runs (#5)
- [ ] **Obsidian export** — save briefings to my knowledge base (#13)
- [ ] **Cost tracking** — show estimated $ per run (#9)

---

## Running costs

With default settings (14 sources, GPT-4o-mini):
- ~300 stories fetched per day
- ~30 pass the score threshold
- ~$0.01–0.03 per day in OpenAI credits

30 days ≈ $0.30–$0.90. Cheaper than a coffee.

---

## Development

```bash
# Run tests
uv run pytest tests/ -q

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=term-missing

# Lint
uv run ruff check src/ tests/
```

Tests run on Python 3.11 and 3.12 via GitHub Actions on every push.

---

## AI providers

Supports OpenAI (default), Google Gemini, and Anthropic Claude. Set `AI_MODEL` in `.env`:

```env
AI_MODEL=gpt-4o-mini          # OpenAI (default, cheapest)
AI_MODEL=gemini-1.5-flash     # Google Gemini
AI_MODEL=claude-3-haiku-20240307  # Anthropic
```

---

## License

MIT — do whatever you want with it.

---

*Built in 30 days as a personal project. Day 15/30 complete.*
