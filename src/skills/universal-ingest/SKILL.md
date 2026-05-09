# Universal Ingest Skill
## Ingest all file types from my_knowledge/ into ChromaDB

### Trigger phrases
- "Ingest my files"
- "Index my knowledge"
- "Ingest the Reid project"
- "Re-index my documents"
- "Add new files to knowledge base"
- "Update my knowledge base"

### Commands
```bash
# Ingest everything
python src/ingest/universal_ingest.py

# Ingest specific project
python src/ingest/universal_ingest.py --project "Person-Reid-Model"

# Ingest single file
python src/ingest/universal_ingest.py --path my_knowledge/report.pdf

# Re-ingest clean
python src/ingest/universal_ingest.py --clean --project "Person-Reid-Model"
```
