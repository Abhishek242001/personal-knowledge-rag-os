# My Knowledge — Personal Files

This folder is where YOU put your personal knowledge files.
It is **gitignored** — your private documents will never be committed to git.

## What goes where

### 📄 pdfs/
Drop any PDF files you want to ingest:
- Research papers
- Books
- Articles saved as PDF
- Course notes
- Any document in PDF format

```bash
# After adding PDFs, run:
python src/ingest/pdf_ingest.py
```

### 📝 obsidian/
Copy or symlink your Obsidian vault here:

```bash
# Option A: Copy your vault
cp -r /path/to/your/obsidian/vault/* my_knowledge/obsidian/

# Option B: Symlink (recommended — stays in sync)
ln -s /path/to/your/obsidian/vault my_knowledge/obsidian
```

```bash
# After adding notes, run:
python src/ingest/obsidian_ingest.py
```

### 🌐 bookmarks.html (optional)
Export your browser bookmarks as HTML and place here.
Chrome: Settings → Bookmarks → Export bookmarks

---

## Re-ingesting after adding new files

```bash
# Re-ingest everything
python src/ingest/pdf_ingest.py
python src/ingest/notion_ingest.py
python src/ingest/obsidian_ingest.py

# Or ask your bot on Telegram:
# "Re-ingest my knowledge base"
```

## Checking what's in your knowledge base

```bash
python src/rag/rag_query.py --stats
```
