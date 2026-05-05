# 🧠 Personal Knowledge OS with RAG Memory

A RAG-powered second brain that ingests your Notion pages, Obsidian vault, PDFs, and Gmail threads into a local vector database — and lets you query your entire knowledge base conversationally via Telegram, powered by Claude.

**Stack:** OpenClaw · ChromaDB · Notion API · Claude API · ngrok · Lightning AI Studio

---

## What It Does

- 📄 Ingests Notion pages, PDFs, Obsidian `.md` files, Gmail threads
- 🔢 Embeds everything into ChromaDB (local vector database)
- 💬 Query your entire knowledge base via Telegram conversationally
- 🧠 Claude synthesises answers from retrieved chunks
- 📅 Weekly "forgotten ideas" digest surfaced automatically
- 🔗 Cross-document link suggestions via similarity threshold

---

## Prerequisites

Before starting, create accounts and collect these keys:

| Service | Purpose | Free? | Link |
|---|---|---|---|
| Lightning AI Studio | Cloud dev environment | Free tier | [lightning.ai](https://lightning.ai) |
| Anthropic Claude | LLM for answer synthesis | Pay-per-use | [console.anthropic.com](https://console.anthropic.com) |
| Telegram @BotFather | Chat interface | Free | Telegram app |
| ngrok | Public tunnel for webhook | Free | [ngrok.com](https://ngrok.com) |
| Notion | Knowledge source | Free | [notion.so/my-integrations](https://notion.so/profile/integrations) |
| OpenAI (optional) | Embeddings (or use Ollama) | Pay-per-use | [platform.openai.com](https://platform.openai.com) |

---

## Setup Guide

### Step 1 — Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/personal-knowledge-os.git
cd personal-knowledge-os
```

### Step 2 — Copy and fill in your environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in all your API keys (see `.env.example` for details).

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install OpenClaw

```bash
npm install -g openclaw@latest
openclaw --version
```

### Step 5 — Install ngrok and start tunnel

```bash
pip install pyngrok
ngrok authtoken YOUR_NGROK_TOKEN
ngrok http 18789
```

Copy the forwarding URL (e.g. `https://xxxx.ngrok-free.app`) — you'll need it next.

### Step 6 — Run OpenClaw onboarding

```bash
openclaw onboard --mode local
```

During onboarding select:
- **Model:** `anthropic/claude-sonnet-4-5` (or haiku for cheaper)
- **Port:** `18789`
- **Bind:** `LAN (0.0.0.0)`
- **Channel:** `Telegram`
- **Search:** `DuckDuckGo`
- **Skills:** `nano-pdf`, `obsidian`, `summarize`
- **Hooks:** `session-memory`

### Step 7 — Add your Telegram bot token

```bash
openclaw config set channels.telegram.botToken "YOUR_BOT_TOKEN"
openclaw config set channels.telegram.webhookUrl "https://YOUR_NGROK_URL"
```

### Step 8 — Start the gateway

```bash
openclaw gateway
```

### Step 9 — Approve yourself on Telegram

Message your bot on Telegram. It will show a pairing code. In a new terminal run:

```bash
openclaw pairing approve telegram YOUR_PAIRING_CODE
```

### Step 10 — Ingest your knowledge base

```bash
# Ingest PDFs (put your PDFs in my_knowledge/pdfs/ first)
python src/ingest/pdf_ingest.py

# Ingest Notion pages
python src/ingest/notion_ingest.py

# Ingest Obsidian vault (put .md files in my_knowledge/obsidian/ first)
python src/ingest/obsidian_ingest.py
```

### Step 11 — Start everything together

```bash
bash start.sh
```

---

## Project Structure

```
personal-knowledge-os/
│
├── README.md                        # This file
├── .env.example                     # Template for API keys
├── .gitignore                       # Keeps secrets and data out of git
├── requirements.txt                 # Python dependencies
├── start.sh                         # One command to start everything
│
├── config/
│   └── agents/
│       └── AGENTS.md                # Agent personality and instructions
│
├── my_knowledge/                    # YOUR personal files go here (gitignored)
│   ├── pdfs/                        # Drop your PDF files here
│   ├── obsidian/                    # Paste your Obsidian vault here
│   └── README.md                    # Instructions for what goes here
│
├── src/
│   ├── ingest/
│   │   ├── pdf_ingest.py            # PDF parser and chunker
│   │   ├── notion_ingest.py         # Notion page fetcher
│   │   ├── obsidian_ingest.py       # Obsidian .md file walker
│   │   └── gmail_ingest.py          # Gmail thread fetcher
│   │
│   ├── embed/
│   │   └── embedder.py              # Embedding pipeline (OpenAI or Ollama)
│   │
│   ├── rag/
│   │   ├── chromadb_store.py        # ChromaDB store and retrieve
│   │   └── rag_query.py             # Top-k retrieval + Claude synthesis
│   │
│   └── skills/
│       ├── rag-search/SKILL.md      # OpenClaw skill: query knowledge base
│       ├── weekly-digest/SKILL.md   # OpenClaw skill: forgotten ideas digest
│       └── ingest-trigger/SKILL.md  # OpenClaw skill: re-ingest on demand
│
├── chromadb/                        # Vector store (gitignored)
└── logs/                            # Runtime logs (gitignored)
```

---

## Usage — Telegram Commands

Once running, message your bot on Telegram:

```
Ask anything from your knowledge base:
"What did I write about RAG last month?"
"Summarise my notes on transformers"
"What are my open tasks from Notion?"
"Find ideas I haven't revisited in 2 weeks"
```

Built-in OpenClaw commands:
```
/help     — show all commands
/memory   — show what the agent remembers about you
/reset    — start a new session
```

---

## Troubleshooting

**Bot not responding?**
```bash
# Check gateway is running
openclaw gateway

# Check ngrok is live
curl https://YOUR_NGROK_URL/health

# Check logs
tail -f /tmp/openclaw/openclaw-$(date +%Y-%m-%d).log
```

**ChromaDB empty after ingest?**
```bash
python -c "import chromadb; c = chromadb.PersistentClient('./chromadb'); print(c.list_collections())"
```

**Auth error for Anthropic?**
```
# In OpenClaw TUI type:
/auth anthropic
# Paste your API key
```

---

## Contributing

1. Fork the repo
2. Add your own ingestion source in `src/ingest/`
3. Add a corresponding skill in `src/skills/`
4. Submit a PR!

---

## Resume Positioning

This project demonstrates:
- **RAG architecture** — chunking, embedding, vector retrieval, synthesis
- **Agent orchestration** — OpenClaw skills, hooks, memory
- **Multi-source ingestion** — Notion, PDF, Obsidian, Gmail
- **Production deployment** — Lightning AI Studio, ngrok tunneling

**Tags:** `RAG` `ChromaDB` `OpenClaw` `Claude` `Telegram` `LangChain` `Vector Database` `Knowledge Management`
