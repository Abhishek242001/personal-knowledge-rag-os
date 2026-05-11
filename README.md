# Personal Knowledge OS — RAG-Powered Second Brain

A production-grade personal knowledge management system that ingests your documents, code, images, and notes into a local vector database and lets you query your entire knowledge base conversationally via Telegram. Supports multi-format ingestion, project-aware retrieval, and direct file delivery over Telegram.

Built with **OpenClaw · ChromaDB · Ollama · Claude API · ngrok · Lightning AI Studio**

---

**Developed by:**
[Abhishek Kumar Shukla](https://www.linkedin.com/in/researcher-abhishek-kumar-shukla/)
Senior AI Developer @ [Uniconverge Technologies Pvt. Ltd.](https://www.uniconvergetech.in/)

---

## What It Does

- Ingests **any file type** — PDF, Word, Excel, PowerPoint, code, images, markdown, CSV
- Organises documents into **project folders** for structured retrieval
- Answers natural language questions via **Telegram** using RAG and Claude
- **Sends files directly to your Telegram** — ask for a document and receive it instantly
- Surfaces forgotten ideas with a **weekly digest**
- Supports **Notion, Obsidian, Gmail** as additional knowledge sources

---

## Architecture

```
You (Telegram)
      |
      v
Telegram Servers
      |
      v
ngrok (public HTTPS tunnel)
      |
      v
OpenClaw Gateway (Node.js · port 18789)
      |
      ├── /rag_search  →  RAG Pipeline (Python)
      |                       |-- ChromaDB (vector store)
      |                       |-- Ollama (local embeddings)
      |                       |-- Claude API (answer synthesis)
      |
      └── /file-send   →  File Sender (Python)
                              |-- ChromaDB (metadata search)
                              |-- Telegram Bot API (file upload)
```

---

## Supported File Types

| Extension | Type | Extraction Method |
|---|---|---|
| `.pdf` | PDF | PyMuPDF |
| `.docx` `.doc` | Word | python-docx |
| `.xlsx` `.xls` `.csv` | Excel / Spreadsheet | pandas + openpyxl |
| `.pptx` `.ppt` | PowerPoint | python-pptx |
| `.jpg` `.jpeg` `.png` `.webp` `.bmp` | Image | Claude Vision API |
| `.py` `.js` `.ts` `.java` `.go` `.rs` etc. | Code | Plain text |
| `.txt` `.md` `.rst` | Text / Markdown | Plain text |

---

## Prerequisites

Collect the following credentials before starting:

| Service | Purpose | Cost | Link |
|---|---|---|---|
| Lightning AI Studio | Cloud development environment | Free | [lightning.ai](https://lightning.ai) |
| Anthropic Claude | Answer synthesis + image description | Pay per use | [console.anthropic.com](https://console.anthropic.com) |
| Telegram BotFather | Chat interface | Free | Telegram app → @BotFather |
| ngrok | Public HTTPS tunnel | Free | [ngrok.com](https://ngrok.com) |
| Notion | Optional knowledge source | Free | [notion.so/profile/integrations](https://notion.so/profile/integrations) |

---

## Setup Guide

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Abhishek242001/personal-knowledge-rag-os.git
cd personal-knowledge-rag-os
```

### Step 2 — Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in your credentials:

```env
ANTHROPIC_API_KEY=sk-ant-...
TELEGRAM_BOT_TOKEN=your-bot-token-from-botfather
NGROK_AUTHTOKEN=your-ngrok-auth-token
NGROK_URL=https://your-ngrok-url.ngrok-free.app
OLLAMA_BASE_URL=http://localhost:11434
CHROMADB_PATH=./chromadb
CHROMADB_COLLECTION=knowledge_os
OPENCLAW_PORT=18789
OPENCLAW_MODEL=anthropic/claude-haiku-4-5
```

### Step 3 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install and Start Ollama (Embeddings)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve &
sleep 5

# Pull the embedding model (274 MB, one-time download)
ollama pull nomic-embed-text

# Verify it is running
curl http://localhost:11434/api/tags
```

> **Note for Lightning AI users:** Ollama does not persist across session restarts. Run `ollama serve & && ollama pull nomic-embed-text` at the start of each session, or switch to Voyage AI (see Embedding Options below).

### Step 5 — Install OpenClaw Gateway

```bash
npm install -g openclaw@latest
```

If `openclaw` command is not found after installation:

```bash
# Find the installation path
find / -name "openclaw.mjs" 2>/dev/null | head -3

# Run directly with node
node /path/to/openclaw.mjs gateway

# Or add alias permanently
echo 'alias openclaw="node /path/to/openclaw.mjs"' >> ~/.zshrc
source ~/.zshrc
```

### Step 6 — Run OpenClaw Onboarding (First Time Only)

```bash
openclaw onboard --mode local
```

During the interactive setup:

| Prompt | Selection |
|---|---|
| Model | `anthropic/claude-haiku-4-5` |
| Gateway port | `18789` |
| Gateway bind | `LAN (0.0.0.0)` |
| Channel | `Telegram (Bot API)` |
| Web search | `DuckDuckGo` |
| Hooks | `session-memory` |

Add your Telegram bot token when prompted, or set it manually:

```bash
openclaw config set channels.telegram.botToken "YOUR_BOT_TOKEN"
```

### Step 7 — Install RAG Skills into OpenClaw

```bash
mkdir -p ~/.openclaw/skills/rag-search
mkdir -p ~/.openclaw/skills/file-send
mkdir -p ~/.openclaw/skills/universal-ingest

cp src/skills/rag-search/SKILL.md ~/.openclaw/skills/rag-search/
cp src/skills/file-send/SKILL.md ~/.openclaw/skills/file-send/
cp src/skills/universal-ingest/SKILL.md ~/.openclaw/skills/universal-ingest/
```

### Step 8 — Add Your Knowledge Files

Create project folders under `my_knowledge/`:

```
my_knowledge/
    Person-Reid-Model/          ← project folder
        documents/
            Project Proposal.docx
            Project Document.docx
        modules/
            main.py
            config.py
        requirements.txt
    Finance-Reports/            ← another project folder
        Q1-2026.xlsx
        Budget.pdf
    README.md
```

Any file type placed here will be automatically detected and ingested.

### Step 9 — Run Ingestion

```bash
# Ingest all files from all project folders
python src/ingest/universal_ingest.py

# Ingest a specific project only
python src/ingest/universal_ingest.py --project "Person-Reid-Model"

# Ingest a single file
python src/ingest/universal_ingest.py --path my_knowledge/report.pdf

# Re-ingest a project (clear old chunks first)
python src/ingest/universal_ingest.py --clean --project "Person-Reid-Model"
```

Expected output:
```
🧠 Personal Knowledge OS — Universal Ingestion
==================================================
✅ Embedder: Ollama (nomic-embed-text @ http://localhost:11434)
✅ ChromaDB connected at: ./chromadb
✅ Collection: knowledge_os (0 chunks)
📁 Scanning: my_knowledge
🔍 Found 17 supported file(s)
  [.DOCX] Person-Reid-Model/Project Proposal.docx
     ✅ 3 chunks stored
  [.PY] Person-Reid-Model/main.py
     ✅ 7 chunks stored
  ...
✅ Ingestion complete — 70 total chunks
```

### Step 10 — Start ngrok Tunnel

```bash
mkdir -p logs
ngrok http 18789 > logs/ngrok.log 2>&1 &
sleep 3

# Get the public URL
curl -s http://localhost:4040/api/tunnels | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(d['tunnels'][0]['public_url'])"
```

### Step 11 — Set Telegram Webhook

```bash
BOT_TOKEN=$(grep TELEGRAM_BOT_TOKEN .env | cut -d'=' -f2)
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(d['tunnels'][0]['public_url'])")

curl "https://api.telegram.org/bot${BOT_TOKEN}/setWebhook?url=${NGROK_URL}"
```

Expected: `{"ok":true,"result":true,"description":"Webhook was set"}`

### Step 12 — Start OpenClaw Gateway

```bash
export $(grep -v '^#' .env | xargs)
openclaw gateway
```

Expected:
```
[gateway] ready
[telegram] starting provider (@your_bot_name)
```

---

## One-Command Startup (After Initial Setup)

Save this script for future sessions:

```bash
cat > ~/start_bot.sh << 'EOF'
#!/bin/bash
echo "📦 Installing Python dependencies..."
pip install -q python-docx python-pptx pandas openpyxl xlrd pymupdf anthropic chromadb python-dotenv requests notion-client

echo "🚀 Installing/starting Ollama..."
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
sleep 5
ollama pull nomic-embed-text

echo "🦞 Installing OpenClaw..."
npm install -g openclaw@latest
export PATH="/system/conda/node/nvm/versions/node/v22.14.0/lib/node_modules/openclaw/bin:$PATH"

echo "📁 Setting up project..."
cd ~/personal-knowledge-rag-os
export $(grep -v '^#' .env | xargs)

echo "🔑 Updating API key in OpenClaw..."
python3 -c "
import json, os
auth_file = '/teamspace/studios/this_studio/.openclaw/agents/main/agent/auth-profiles.json'
with open(auth_file, 'r') as f:
    auth = json.load(f)
auth['profiles']['anthropic:default']['key'] = os.getenv('ANTHROPIC_API_KEY')
with open(auth_file, 'w') as f:
    json.dump(auth, f, indent=2)
print('API key updated')
"

echo "🌐 Starting ngrok..."
mkdir -p logs
ngrok http 18789 > logs/ngrok.log 2>&1 &
sleep 3

echo "🔗 Setting Telegram webhook..."
BOT_TOKEN=$(grep TELEGRAM_BOT_TOKEN .env | cut -d'=' -f2)
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(d['tunnels'][0]['public_url'])")
curl -s "https://api.telegram.org/bot${BOT_TOKEN}/setWebhook?url=${NGROK_URL}" > /dev/null
echo "Webhook set to ${NGROK_URL}"

echo "✅ Starting OpenClaw gateway..."
node /system/conda/node/nvm/versions/node/v22.14.0/lib/node_modules/openclaw/openclaw.mjs gateway
EOF
chmod +x ~/start_bot.sh
```

Next time just run:
```bash
bash ~/start_bot.sh
```

---

## Usage

### Telegram Commands

| Command | Description |
|---|---|
| `/rag_search What does my Reid project do?` | Search knowledge base |
| `/rag_search Explain main.py` | Query code files |
| `/rag_search Summarise the project proposal` | Summarise documents |
| `/file-send Project Proposal` | Receive file on Telegram |
| `/file-send requirements.txt` | Receive any file |
| `/universal-ingest` | Trigger re-ingestion |

### CLI Usage

```bash
# Query the knowledge base
python src/rag/rag_query.py "What does the Person-Reid-Model project do?"

# Filter by project
python src/rag/rag_query.py "explain the config" --project "Person-Reid-Model"

# Filter by file type
python src/rag/rag_query.py "what libraries are used" --source universal

# Send a file to Telegram
python src/rag/rag_query.py "Send me the Project Proposal" --chat-id YOUR_CHAT_ID

# List files in a project
python src/rag/rag_query.py "List files in Reid project" --chat-id YOUR_CHAT_ID

# Weekly digest
python src/rag/rag_query.py --digest

# Knowledge base stats
python src/rag/rag_query.py --stats
```

---

## Embedding Options

Configure one provider in `.env`:

**Option A — Ollama (local, free, recommended for development)**
```env
OLLAMA_BASE_URL=http://localhost:11434
```
Run `ollama pull nomic-embed-text` once to download the model (274 MB).

**Option B — Voyage AI (cloud, free 200M tokens, recommended for Lightning AI)**
```env
VOYAGE_API_KEY=your-key-from-dash.voyageai.com
```
No local installation needed. Survives session restarts.

**Option C — OpenAI**
```env
OPENAI_API_KEY=your-key-from-platform.openai.com
```

---

## Project Structure

```
personal-knowledge-rag-os/
│
├── README.md
├── .env.example
├── .gitignore
├── requirements.txt
├── start.sh
│
├── config/
│   └── agents/
│       └── AGENTS.md              Agent persona and instructions
│
├── my_knowledge/                  Your documents (gitignored)
│   ├── Person-Reid-Model/         Project folder example
│   │   ├── documents/
│   │   └── modules/
│   └── README.md
│
├── src/
│   ├── ingest/
│   │   ├── universal_ingest.py    Universal multi-format ingestion ← NEW
│   │   ├── pdf_ingest.py          PDF-only ingestion
│   │   ├── notion_ingest.py       Notion pages
│   │   ├── obsidian_ingest.py     Obsidian vault
│   │   └── gmail_ingest.py        Gmail threads
│   │
│   ├── embed/
│   │   └── embedder.py            Embedding pipeline (Ollama / Voyage / OpenAI)
│   │
│   ├── rag/
│   │   ├── chromadb_store.py      Vector store operations
│   │   └── rag_query.py           Retrieval + Claude synthesis + file routing
│   │
│   ├── telegram/
│   │   ├── __init__.py
│   │   └── file_sender.py         File search and Telegram delivery ← NEW
│   │
│   └── skills/
│       ├── rag-search/            OpenClaw skill — knowledge queries
│       ├── file-send/             OpenClaw skill — file delivery ← NEW
│       ├── universal-ingest/      OpenClaw skill — ingestion trigger ← NEW
│       └── weekly-digest/         OpenClaw skill — weekly digest
│
├── chromadb/                      Vector store data (gitignored)
└── logs/                          Runtime logs (gitignored)
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key from console.anthropic.com |
| `TELEGRAM_BOT_TOKEN` | Yes | Bot token from @BotFather |
| `NGROK_AUTHTOKEN` | Yes | Auth token from dashboard.ngrok.com |
| `NGROK_URL` | Optional | Fixed ngrok URL (if on paid plan) |
| `OLLAMA_BASE_URL` | Yes* | `http://localhost:11434` |
| `VOYAGE_API_KEY` | Yes* | Alternative to Ollama |
| `OPENAI_API_KEY` | Yes* | Alternative to Ollama |
| `NOTION_API_KEY` | Optional | For Notion ingestion |
| `CHROMADB_PATH` | Yes | Default: `./chromadb` |
| `CHROMADB_COLLECTION` | Yes | Default: `knowledge_os` |
| `OPENCLAW_PORT` | Yes | Default: `18789` |
| `OPENCLAW_MODEL` | Yes | Default: `anthropic/claude-haiku-4-5` |

*One of `OLLAMA_BASE_URL`, `VOYAGE_API_KEY`, or `OPENAI_API_KEY` is required.

---

## Troubleshooting

**Bot not responding on Telegram**
```bash
# Check OpenClaw gateway is running
openclaw gateway

# Check ngrok tunnel is live
curl http://localhost:4040/api/tunnels

# View live logs
tail -f /tmp/openclaw/openclaw-$(date +%Y-%m-%d).log
```

**Ollama connection error**
```bash
# Start Ollama
ollama serve &
sleep 5

# Re-pull model if missing
ollama pull nomic-embed-text

# Verify
curl http://localhost:11434/api/tags
```

**OpenClaw command not found**
```bash
# Find and run directly
find / -name "openclaw.mjs" 2>/dev/null | head -3
node /path/to/openclaw.mjs gateway
```

**Anthropic API key invalid in OpenClaw**
```bash
export $(grep -v '^#' .env | xargs)
python3 -c "
import json, os
auth_file = '/teamspace/studios/this_studio/.openclaw/agents/main/agent/auth-profiles.json'
with open(auth_file) as f: auth = json.load(f)
auth['profiles']['anthropic:default']['key'] = os.getenv('ANTHROPIC_API_KEY')
with open(auth_file, 'w') as f: json.dump(auth, f, indent=2)
print('API key updated')
"
```

**Dimension mismatch in ChromaDB (switching embedding providers)**
```bash
python3 -c "
import chromadb
client = chromadb.PersistentClient('./chromadb')
client.delete_collection('knowledge_os')
print('Collection deleted — re-run ingestion')
"
python src/ingest/universal_ingest.py
```

**Files not found during ingestion**
```bash
# Check what rglob sees in my_knowledge
python3 -c "
from pathlib import Path
for f in Path('my_knowledge').rglob('*'):
    if f.is_file(): print(f)
"
```

---

## Contributing

1. Fork the repository
2. Add a new ingestion source in `src/ingest/`
3. Add a corresponding OpenClaw skill in `src/skills/`
4. Update `requirements.txt` if new dependencies are added
5. Submit a pull request with a clear description

---

## Tags

`RAG` `ChromaDB` `Vector Database` `OpenClaw` `Claude API` `Telegram Bot` `Ollama` `LangChain-free` `Knowledge Management` `Lightning AI` `Python` `Personal AI` `Second Brain` `Document Intelligence` `Multi-format Ingestion`