# File Send Skill
## Find and send files from the knowledge base over Telegram

When the user asks to receive, share, or download a file from their knowledge base,
search for it and send it directly to their Telegram chat.

### Trigger phrases
- "Send me the Project Proposal"
- "Share the Reid report"
- "Get me the Excel sheet from Finance project"
- "Forward me the requirements file"
- "Send the architecture diagram"
- "Download my notes on X"
- "Attach the config file"

### List triggers
- "List files in Reid project"
- "What files do I have in Finance project"
- "Show documents in my knowledge base"
- "What documents are in Person-Reid-Model"

### How to use

**Send a file (requires chat ID):**
```bash
cd ~/personal-knowledge-rag-os
python src/rag/rag_query.py "Send me the Project Proposal" --chat-id CHAT_ID
```

**List files in a project:**
```bash
python src/rag/rag_query.py "List files in Reid project" --chat-id CHAT_ID
```

**Search and send with project filter:**
```bash
python src/rag/rag_query.py "Send the requirements file" --project "Person-Reid-Model" --chat-id CHAT_ID
```

**Direct file sender:**
```bash
python src/telegram/file_sender.py "Project Proposal" --chat-id CHAT_ID
python src/telegram/file_sender.py "requirements" --project "Person-Reid-Model" --chat-id CHAT_ID
python src/telegram/file_sender.py "Reid project" --list --chat-id CHAT_ID
```

### Important notes
- File must have been ingested first via universal_ingest.py
- Files larger than 25 MB cannot be sent via Telegram (bot API limit)
- The chat ID is the user's Telegram chat ID — OpenClaw passes this automatically
- If multiple files match, the best match is sent and others are listed

### Response format
- Confirm which file was sent with its project and match score
- If file not found, explain why and suggest running ingestion
- If file too large, show the file path on the server instead