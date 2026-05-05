# Personal Knowledge OS with RAG Memory

A second-brain system that ingests your Notion pages, Obsidian vault, PDFs, and Gmail threads into a local vector database and lets you query your entire knowledge base conversationally via Telegram. Built with OpenClaw, ChromaDB, Ollama, and Claude.

---

## Overview

Knowledge workers accumulate thousands of notes, articles, and documents that are never revisited. This system makes your entire personal knowledge base queryable and proactively surfaces relevant context when you need it.

**What it does:**
- Ingests Notion pages, PDFs, Obsidian markdown files, and Gmail threads
- Chunks and embeds documents into a local ChromaDB vector database
- Answers natural language queries via Telegram using RAG and Claude
- Surfaces forgotten ideas with a weekly digest
- Suggests cross-document links via similarity threshold

**Stack:** OpenClaw · ChromaDB · Ollama · Claude API · Notion API · ngrok · Lightning AI Studio

---

## Architecture

```
Telegram
    |
    v
ngrok (public HTTPS tunnel)
    |
    v
OpenClaw Gateway (Node.js, port 18789)
    |
    v
RAG Pipeline (Python)
    |-- ChromaDB (vector store, local)
    |-- Ollama (embeddings, local)
    |-- Claude API (answer synthesis)
    |
    v
Response back to Telegram
```

---

## Prerequisites

Create accounts and collect the following credentials before starting:

| Service | Purpose | Cost | Link |
|---|---|---|---|
| Lightning AI Studio | Cloud development environment | Free tier | https://lightning.ai |
| Anthropic Claude | Answer synthesis | Pay per use | https://console.anthropic.com |
| Telegram BotFather | Chat interface | Free | Telegram app |
| ngrok | Public HTTPS tunnel for webhook | Free | https://ngrok.com |
| Notion | Knowledge source ingestion | Free | https://notion.so/profile/integrations |

---

## Setup Guide

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/personal-knowledge-os.git
cd personal-knowledge-os
```

### Step 2 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your credentials. See the Environment Variables section below for details on each key.

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install Ollama for local embeddings

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull nomic-embed-text
```

Ollama starts automatically as a system service. Verify it is running:

```bash
curl http://localhost:11434/api/tags
```

### Step 5 — Install OpenClaw

```bash
npm install -g openclaw@latest
openclaw --version
```

If the command is not found after installation, add npm global bin to your PATH:

```bash
echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Step 6 — Start the ngrok tunnel

```bash
ngrok http 18789
```

Copy the forwarding URL shown (e.g. `https://xxxx.ngrok-free.app`). Add it to your `.env` as `NGROK_URL`.

### Step 7 — Run OpenClaw onboarding

```bash
openclaw onboard --mode local
```

During the interactive setup, select the following options:

| Prompt | Selection |
|---|---|
| Model | anthropic/claude-haiku-4-5 |
| Gateway port | 18789 |
| Gateway bind | LAN (0.0.0.0) |
| Channel | Telegram (Bot API) |
| Web search | DuckDuckGo |
| Skills | nano-pdf, summarize |
| Hooks | session-memory |

### Step 8 — Add Telegram bot token

Create a bot via @BotFather on Telegram (`/newbot`), then set the token:

```bash
openclaw config set channels.telegram.botToken "YOUR_BOT_TOKEN"
openclaw config set channels.telegram.webhookUrl "YOUR_NGROK_URL"
```

### Step 9 — Start the gateway

```bash
openclaw gateway
```

### Step 10 — Approve yourself on Telegram

Message your bot on Telegram. It will send a pairing code. In a new terminal, approve it:

```bash
openclaw pairing approve telegram YOUR_PAIRING_CODE
```

### Step 11 — Add your knowledge files

Place your documents in the `my_knowledge/` directory:

```
my_knowledge/
    pdfs/         <- drop PDF files here
    obsidian/     <- copy or symlink your Obsidian vault here
```

### Step 12 — Run ingestion

```bash
# Ingest PDFs
python src/ingest/pdf_ingest.py

# Ingest Notion pages
python src/ingest/notion_ingest.py

# Ingest Obsidian vault
python src/ingest/obsidian_ingest.py
```

### Step 13 — Install RAG skills into OpenClaw

```bash
mkdir -p ~/.openclaw/skills/rag-search
mkdir -p ~/.openclaw/skills/weekly-digest

cp src/skills/rag-search/SKILL.md ~/.openclaw/skills/rag-search/
cp src/skills/weekly-digest/SKILL.md ~/.openclaw/skills/weekly-digest/
```

### Step 14 — Start everything

```bash
bash start.sh
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key from console.anthropic.com |
| `TELEGRAM_BOT_TOKEN` | Yes | Bot token from @BotFather |
| `NGROK_AUTHTOKEN` | Yes | Auth token from dashboard.ngrok.com |
| `NGROK_URL` | Yes | Your ngrok forwarding URL |
| `NOTION_API_KEY` | Optional | Integration token from notion.so/profile/integrations |
| `OLLAMA_BASE_URL` | Yes | Set to http://localhost:11434 |
| `OPENAI_API_KEY` | Optional | Alternative to Ollama for embeddings |
| `VOYAGE_API_KEY` | Optional | Alternative to Ollama for embeddings |
| `CHROMADB_PATH` | Yes | Local path for vector store (default: ./chromadb) |
| `CHROMADB_COLLECTION` | Yes | Collection name (default: knowledge_os) |
| `OPENCLAW_PORT` | Yes | Gateway port (default: 18789) |
| `OPENCLAW_MODEL` | Yes | Claude model (default: anthropic/claude-haiku-4-5) |

---

## Embedding Options

The system supports three embedding providers. Configure one in `.env`:

**Option A — Ollama (recommended, free, local)**
```
OLLAMA_BASE_URL=http://localhost:11434
```
Run `ollama pull nomic-embed-text` once to download the model.

**Option B — Voyage AI (free tier, 200M tokens)**
```
VOYAGE_API_KEY=your-key-from-dash.voyageai.com
```

**Option C — OpenAI**
```
OPENAI_API_KEY=your-key-from-platform.openai.com
```

---

## Project Structure

```
personal-knowledge-os/
|
|-- README.md
|-- .env.example
|-- .gitignore
|-- requirements.txt
|-- start.sh
|
|-- config/
|   |-- agents/
|       |-- AGENTS.md          Agent personality and instructions
|
|-- my_knowledge/              Personal files — gitignored
|   |-- pdfs/                  Drop PDF files here
|   |-- obsidian/              Copy Obsidian vault here
|   |-- README.md
|
|-- src/
|   |-- ingest/
|   |   |-- pdf_ingest.py      PDF parser and chunker
|   |   |-- notion_ingest.py   Notion page fetcher
|   |   |-- obsidian_ingest.py Obsidian markdown walker
|   |   |-- gmail_ingest.py    Gmail thread fetcher
|   |
|   |-- embed/
|   |   |-- embedder.py        Embedding pipeline (Ollama / Voyage / OpenAI)
|   |
|   |-- rag/
|   |   |-- chromadb_store.py  Vector store operations
|   |   |-- rag_query.py       Retrieval and Claude synthesis
|   |
|   |-- skills/
|       |-- rag-search/        OpenClaw skill for knowledge queries
|       |-- weekly-digest/     OpenClaw skill for forgotten ideas
|       |-- ingest-trigger/    OpenClaw skill for on-demand ingestion
|
|-- chromadb/                  Vector store — gitignored
|-- logs/                      Runtime logs — gitignored
```

---

## Usage

Once the gateway is running, message your Telegram bot directly:

```
What did I write about attention mechanisms?
Summarise my notes on BERT
What are my open tasks from Notion?
Weekly digest
Find ideas I have not revisited recently
```

**CLI usage:**

```bash
# Query the knowledge base
python src/rag/rag_query.py "your question here"

# Filter by source
python src/rag/rag_query.py "question" --source pdf
python src/rag/rag_query.py "question" --source notion

# Run weekly digest
python src/rag/rag_query.py --digest

# Check knowledge base stats
python src/rag/rag_query.py --stats
```

---

## Testing

Run these tests in order to verify every component is working correctly.

### Test 1 — ChromaDB stats

```bash
python src/rag/rag_query.py --stats
```

Expected output:
```
Total chunks: 855
  pdf: 848
  notion: 7
```

### Test 2 — RAG query from PDFs

```bash
python src/rag/rag_query.py "What is BERT?"
```

Expected: Detailed answer with PDF source attribution at the bottom.

### Test 3 — RAG query from Notion

```bash
python src/rag/rag_query.py "What are my tasks?" --source notion
```

Expected: Answer pulled only from Notion pages.

### Test 4 — Weekly digest

```bash
python src/rag/rag_query.py --digest
```

Expected: 5 ideas from your knowledge base with source attribution.

### Test 5 — Embedder

```bash
python src/embed/embedder.py
```

Expected output:
```
Embedder: Ollama (nomic-embed-text @ http://localhost:11434)
Test successful! Dimension: 768
```

### Test 6 — Telegram bot

Message your bot on Telegram:
```
What is supervised learning?
```

Expected: A synthesised answer from your PDFs delivered in Telegram within a few seconds.

### Test 7 — ngrok tunnel

```bash
curl https://YOUR_NGROK_URL
```

Expected: Any response other than connection refused confirms the tunnel is live.

---

## Troubleshooting

**Bot not responding on Telegram**
```bash
# Check gateway is running
openclaw gateway

# Check ngrok is live
curl https://YOUR_NGROK_URL/health

# Check logs
tail -f /tmp/openclaw/openclaw-$(date +%Y-%m-%d).log
```

**Ingestion errors — dimension mismatch in ChromaDB**

This happens when switching embedding providers. Delete the collection and re-ingest:
```bash
python -c "
import chromadb
client = chromadb.PersistentClient('./chromadb')
client.delete_collection('knowledge_os')
print('Collection deleted')
"
python src/ingest/pdf_ingest.py
```

**Ollama 500 error during ingestion**

The embedder automatically retries with shorter text. If it persists, restart Ollama:
```bash
sudo systemctl restart ollama
```

**OpenClaw command not found**
```bash
echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Anthropic auth error in OpenClaw TUI**
```
/auth anthropic
```
Paste your API key when prompted.

---

## Deployment Notes

This project is designed for Lightning AI Studio but works on any Ubuntu server with Node 22+ and Python 3.12+.

For persistent 24/7 operation on Lightning Studio, enable background execution in the Studio settings. The free tier includes unlimited background execution.

For a permanent webhook URL instead of ngrok, use Cloudflare Tunnel:
```bash
cloudflared tunnel --url http://localhost:18789
```

---

## Contributing

1. Fork the repository
2. Add a new ingestion source in `src/ingest/`
3. Add a corresponding OpenClaw skill in `src/skills/`
4. Update `requirements.txt` if new dependencies are added
5. Submit a pull request with a clear description of the change

---

## Resume Positioning

This project demonstrates end-to-end RAG architecture including multi-source document ingestion, local embedding generation, vector similarity search, and LLM-based answer synthesis with a conversational interface — covering the full ML engineering stack from data pipeline to production deployment.

Tags: RAG, ChromaDB, Vector Database, OpenClaw, Claude, Telegram, Ollama, Notion API, Knowledge Management, Lightning AI