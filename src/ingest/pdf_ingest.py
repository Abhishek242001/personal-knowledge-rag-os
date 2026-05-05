"""
PDF Ingestion Pipeline
======================
Reads PDFs from my_knowledge/pdfs/, chunks them,
generates embeddings, and stores in ChromaDB.

Usage:
    python src/ingest/pdf_ingest.py
    python src/ingest/pdf_ingest.py --path my_knowledge/pdfs/paper.pdf
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    import fitz  # pymupdf
except ImportError:
    try:
        import PyPDF2 as fitz
        fitz = None
    except ImportError:
        print("❌ Install pymupdf: pip install pymupdf")
        sys.exit(1)

sys.path.append(str(Path(__file__).parent.parent))
from embed.embedder import get_embedder
from rag.chromadb_store import get_collection

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
PDF_DIR = Path("my_knowledge/pdfs")


def extract_text_pymupdf(pdf_path: Path) -> list[dict]:
    """Extract text page by page using pymupdf."""
    pages = []
    doc = fitz.open(str(pdf_path))
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


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


def ingest_pdf(pdf_path: Path, collection, embedder) -> int:
    """Ingest a single PDF file. Returns number of chunks stored."""
    print(f"\n📄 Processing: {pdf_path.name}")

    pages = extract_text_pymupdf(pdf_path)
    if not pages:
        print(f"   ⚠️  No text found in {pdf_path.name}")
        return 0

    total_chunks = 0
    for page_data in pages:
        chunks = chunk_text(page_data["text"])

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue

            chunk_id = hashlib.md5(
                f"{pdf_path.name}:p{page_data['page']}:c{i}:{chunk[:50]}".encode()
            ).hexdigest()

            embedding = embedder.embed(chunk)

            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "source": "pdf",
                    "filename": pdf_path.name,
                    "page": page_data["page"],
                    "chunk_index": i,
                    "path": str(pdf_path),
                }]
            )
            total_chunks += 1

    print(f"   ✅ {len(pages)} pages → {total_chunks} chunks stored")
    return total_chunks


def ingest_all_pdfs(pdf_dir: Path = PDF_DIR):
    """Ingest all PDFs in the given directory."""
    if not pdf_dir.exists():
        print(f"❌ Directory not found: {pdf_dir}")
        print(f"   Create it and add your PDFs: mkdir -p {pdf_dir}")
        return

    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    if not pdf_files:
        print(f"⚠️  No PDFs found in {pdf_dir}")
        print(f"   Add your PDF files to {pdf_dir}/")
        return

    print(f"🧠 Personal Knowledge OS — PDF Ingestion")
    print(f"==========================================")
    print(f"📁 Found {len(pdf_files)} PDF(s) in {pdf_dir}")

    embedder = get_embedder()
    collection = get_collection()

    total = 0
    for pdf_path in pdf_files:
        total += ingest_pdf(pdf_path, collection, embedder)

    print(f"\n==========================================")
    print(f"✅ Done! {total} chunks stored in ChromaDB")
    print(f"   Collection: {os.getenv('CHROMADB_COLLECTION', 'knowledge_os')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
    parser.add_argument("--path", type=str, help="Path to a single PDF file")
    parser.add_argument("--dir", type=str, default=str(PDF_DIR), help="Directory of PDFs")
    args = parser.parse_args()

    if args.path:
        embedder = get_embedder()
        collection = get_collection()
        ingest_pdf(Path(args.path), collection, embedder)
    else:
        ingest_all_pdfs(Path(args.dir))
