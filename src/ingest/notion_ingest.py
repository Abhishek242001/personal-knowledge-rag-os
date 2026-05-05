"""
Notion Ingestion Pipeline
=========================
Fetches all pages from your Notion workspace,
chunks them, and stores in ChromaDB.

Usage:
    python src/ingest/notion_ingest.py
"""

import os
import sys
import hashlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    from notion_client import Client
except ImportError:
    print("❌ Install notion-client: pip install notion-client")
    sys.exit(1)

sys.path.append(str(Path(__file__).parent.parent))
from embed.embedder import get_embedder
from rag.chromadb_store import get_collection

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def extract_text_from_block(block: dict) -> str:
    """Extract plain text from a Notion block."""
    block_type = block.get("type", "")
    content = block.get(block_type, {})

    if "rich_text" in content:
        return " ".join([rt.get("plain_text", "") for rt in content["rich_text"]])
    elif block_type == "child_page":
        return content.get("title", "")
    return ""


def get_page_content(notion: Client, page_id: str) -> str:
    """Get all text content from a Notion page."""
    lines = []
    try:
        blocks = notion.blocks.children.list(block_id=page_id)
        for block in blocks.get("results", []):
            text = extract_text_from_block(block)
            if text.strip():
                lines.append(text.strip())
    except Exception as e:
        print(f"   ⚠️  Could not fetch blocks: {e}")
    return "\n".join(lines)


def get_page_title(page: dict) -> str:
    """Extract title from a Notion page object."""
    props = page.get("properties", {})
    for prop in props.values():
        if prop.get("type") == "title":
            title_parts = prop.get("title", [])
            return " ".join([t.get("plain_text", "") for t in title_parts])
    return "Untitled"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def ingest_notion_pages():
    """Fetch and ingest all accessible Notion pages."""
    api_key = os.getenv("NOTION_API_KEY")
    if not api_key or "your-notion-key" in api_key:
        print("❌ NOTION_API_KEY not set in .env")
        return

    print(f"🧠 Personal Knowledge OS — Notion Ingestion")
    print(f"=============================================")

    notion = Client(auth=api_key)
    embedder = get_embedder()
    collection = get_collection()

    # Search for all pages
    print("🔍 Fetching pages from Notion...")
    results = notion.search(filter={"property": "object", "value": "page"})
    pages = results.get("results", [])
    print(f"📄 Found {len(pages)} page(s)")

    total_chunks = 0
    for page in pages:
        page_id = page["id"]
        title = get_page_title(page)
        url = page.get("url", "")

        print(f"\n📝 Processing: {title}")
        content = get_page_content(notion, page_id)

        if not content.strip():
            print(f"   ⚠️  Empty page, skipping")
            continue

        full_text = f"{title}\n\n{content}"
        chunks = chunk_text(full_text)
        page_chunks = 0

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 30:
                continue

            chunk_id = hashlib.md5(
                f"notion:{page_id}:c{i}:{chunk[:50]}".encode()
            ).hexdigest()

            embedding = embedder.embed(chunk)

            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "source": "notion",
                    "page_id": page_id,
                    "title": title,
                    "url": url,
                    "chunk_index": i,
                }]
            )
            page_chunks += 1
            total_chunks += 1

        print(f"   ✅ {page_chunks} chunks stored")

    print(f"\n=============================================")
    print(f"✅ Done! {total_chunks} chunks stored from Notion")


if __name__ == "__main__":
    ingest_notion_pages()
