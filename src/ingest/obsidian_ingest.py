"""
Obsidian Vault Ingestion Pipeline
===================================
Walks all .md files in my_knowledge/obsidian/,
chunks them, and stores in ChromaDB.

Usage:
    python src/ingest/obsidian_ingest.py
    python src/ingest/obsidian_ingest.py --vault /path/to/your/vault
"""

import os
import re
import sys
import hashlib
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))
from embed.embedder import get_embedder
from rag.chromadb_store import get_collection

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
OBSIDIAN_DIR = Path("my_knowledge/obsidian")


def clean_markdown(text: str) -> str:
    """Remove Obsidian-specific markdown syntax."""
    # Remove wiki links [[link]] → link
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    # Remove tags #tag
    text = re.sub(r'#\w+', '', text)
    # Remove markdown links [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Remove inline code
    text = re.sub(r'`[^`]+`', '', text)
    # Remove markdown headers #
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Remove bold/italic
    text = re.sub(r'\*{1,3}([^\*]+)\*{1,3}', r'\1', text)
    # Clean up extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter from a markdown file."""
    metadata = {}
    if text.startswith('---'):
        end = text.find('---', 3)
        if end != -1:
            frontmatter = text[3:end].strip()
            content = text[end + 3:].strip()
            for line in frontmatter.split('\n'):
                if ':' in line:
                    key, _, value = line.partition(':')
                    metadata[key.strip()] = value.strip()
            return metadata, content
    return metadata, text


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


def ingest_obsidian_vault(vault_dir: Path = OBSIDIAN_DIR):
    """Ingest all .md files from an Obsidian vault."""
    if not vault_dir.exists():
        print(f"❌ Vault directory not found: {vault_dir}")
        print(f"   Copy your Obsidian vault to: {vault_dir}/")
        return

    md_files = list(vault_dir.glob("**/*.md"))
    if not md_files:
        print(f"⚠️  No .md files found in {vault_dir}")
        return

    print(f"🧠 Personal Knowledge OS — Obsidian Ingestion")
    print(f"===============================================")
    print(f"📁 Found {len(md_files)} note(s) in {vault_dir}")

    embedder = get_embedder()
    collection = get_collection()

    total_chunks = 0
    for md_path in md_files:
        try:
            raw_text = md_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"   ⚠️  Could not read {md_path.name}: {e}")
            continue

        frontmatter, content = extract_frontmatter(raw_text)
        cleaned = clean_markdown(content)

        if len(cleaned.strip()) < 50:
            continue

        title = md_path.stem
        full_text = f"{title}\n\n{cleaned}"
        chunks = chunk_text(full_text)
        file_chunks = 0

        print(f"\n📝 Processing: {md_path.name}")

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 30:
                continue

            chunk_id = hashlib.md5(
                f"obsidian:{md_path.name}:c{i}:{chunk[:50]}".encode()
            ).hexdigest()

            embedding = embedder.embed(chunk)

            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "source": "obsidian",
                    "filename": md_path.name,
                    "title": title,
                    "path": str(md_path),
                    "chunk_index": i,
                    **{k: v for k, v in frontmatter.items() if isinstance(v, str)},
                }]
            )
            file_chunks += 1
            total_chunks += 1

        print(f"   ✅ {file_chunks} chunks stored")

    print(f"\n===============================================")
    print(f"✅ Done! {total_chunks} chunks stored from Obsidian vault")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Obsidian vault into ChromaDB")
    parser.add_argument("--vault", type=str, default=str(OBSIDIAN_DIR), help="Path to Obsidian vault")
    args = parser.parse_args()
    ingest_obsidian_vault(Path(args.vault))
