# GitHub Secrets Setup

This file documents all secrets required for the automated workflows.

## Setting Secrets

Go to: **GitHub repo → Settings → Secrets and variables → Actions → New repository secret**

## Required Secrets (at least one AI key)

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (for GPT-4o-mini) | `sk-proj-...` |
| `GEMINI_API_KEY` | Google Gemini API key | `AIza...` |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | `sk-ant-...` |

## Optional — Email Delivery

| Secret Name | Description | Default |
|-------------|-------------|---------|
| `SMTP_USER` | Your Gmail / SMTP address | — |
| `SMTP_PASSWORD` | App Password (not your login password!) | — |
| `EMAIL_TO` | Recipient email address | — |
| `SMTP_HOST` | SMTP server hostname | `smtp.gmail.com` |
| `SMTP_PORT` | SMTP server port | `587` |

> **Gmail setup:** Go to myaccount.google.com → Security → 2-Step Verification → App passwords → Create one for "News Radar".

## Optional — Discord Delivery

| Secret Name | Description |
|-------------|-------------|
| `DISCORD_WEBHOOK_URL` | Discord webhook URL (Server Settings → Integrations → Webhooks) |

## Optional — Slack Delivery

| Secret Name | Description |
|-------------|-------------|
| `SLACK_WEBHOOK_URL` | Slack incoming webhook URL (api.slack.com/apps → Incoming Webhooks) |

## Optional — Custom Webhook

| Secret Name | Description |
|-------------|-------------|
| `CUSTOM_WEBHOOK_URL` | Any HTTP endpoint to receive the full briefing as JSON |

## GitHub Pages Setup

1. Go to **repo → Settings → Pages**
2. Set **Source** to `GitHub Actions`
3. The daily workflow will deploy `docs/` automatically each run

## Verify Setup

After adding secrets, run the workflow manually:

```
GitHub → Actions → Daily Briefing → Run workflow → dry_run: true
```

This validates your config without calling the AI APIs.
