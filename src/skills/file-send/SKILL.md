# File Send Skill
## Find and send files from the knowledge base over Telegram

### Trigger phrases
- "Send me the Project Proposal"
- "Share the Reid report"
- "Get me the Excel sheet"
- "Forward me the requirements file"
- "List files in Reid project"
- "What files do I have"

### Commands
```bash
# Send a file
python src/rag/rag_query.py "Send me the Project Proposal" --chat-id CHAT_ID

# List files in project
python src/rag/rag_query.py "List files in Reid project" --chat-id CHAT_ID
```
