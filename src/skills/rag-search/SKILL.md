# RAG Search Skill
## Search the personal knowledge base using RAG

When the user asks a question about their notes, documents, PDFs, Notion pages,
or anything from their knowledge base, use this skill to retrieve relevant information.

### Trigger phrases
- "What did I write about..."
- "Find my notes on..."
- "Search my knowledge base for..."
- "What do I know about..."
- "Summarise my notes on..."
- "Find ideas about..."

### How to use
Run the RAG query script with the user's question:

```bash
cd ~/personal-knowledge-rag-os
python src/rag/rag_query.py "USER_QUESTION_HERE"
```

### Source filtering
To search only specific sources:

```bash
# Only PDFs
python src/rag/rag_query.py "question" --source pdf

# Only Notion
python src/rag/rag_query.py "question" --source notion

# Only Obsidian
python src/rag/rag_query.py "question" --source obsidian
```

### Response format
Return the synthesised answer directly to the user. Include the source attribution
that the script appends automatically.

### Knowledge base stats
To check what's in the knowledge base:
```bash
python src/rag/rag_query.py --stats
```
