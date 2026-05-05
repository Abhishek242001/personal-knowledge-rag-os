"""
RAG Query Pipeline
==================
Takes a user question, retrieves relevant chunks from ChromaDB,
and feeds them to Claude for answer synthesis.

Usage:
    python src/rag/rag_query.py "What did I write about RAG?"
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))
from embed.embedder import get_embedder
from rag.chromadb_store import search, get_stats

try:
    import anthropic
except ImportError:
    print("❌ Install anthropic: pip install anthropic")
    sys.exit(1)

TOP_K = 5
MODEL = os.getenv("OPENCLAW_MODEL", "anthropic/claude-haiku-4-5").replace("anthropic/", "")


def format_chunks_for_prompt(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        source = meta.get("source", "unknown").upper()
        title = meta.get("title") or meta.get("filename") or "Untitled"
        relevance = chunk["relevance"]

        lines.append(f"[{i}] SOURCE: {source} | {title} (relevance: {relevance})")
        lines.append(chunk["document"])
        lines.append("")
    return "\n".join(lines)


def build_system_prompt() -> str:
    return """You are KnowledgeOS, a personal knowledge assistant with access to the user's notes, documents, PDFs, and Notion pages.

Your job is to answer questions by synthesising information from the retrieved context chunks provided.

Rules:
- Base your answer primarily on the retrieved context
- If the context doesn't contain enough information, say so clearly
- Cite the source (PDF name, Notion page title, etc.) when referencing specific information
- Be concise and direct — this is a personal assistant, not a formal report
- If you see connections between different sources, highlight them
- Use bullet points for lists, but prose for explanations"""


def query(question: str, top_k: int = TOP_K, source_filter: str = None) -> str:
    """
    Run a RAG query against the knowledge base.

    Args:
        question: The user's question
        top_k: Number of chunks to retrieve
        source_filter: Optional filter ('pdf', 'notion', 'obsidian', 'gmail')

    Returns:
        Claude's synthesised answer as a string
    """
    stats = get_stats()
    if stats["total_chunks"] == 0:
        return (
            "⚠️ Your knowledge base is empty!\n\n"
            "Run the ingestion pipelines first:\n"
            "```\n"
            "python src/ingest/pdf_ingest.py\n"
            "python src/ingest/notion_ingest.py\n"
            "python src/ingest/obsidian_ingest.py\n"
            "```"
        )

    # Embed the question
    embedder = get_embedder()
    query_embedding = embedder.embed(question)

    # Retrieve relevant chunks
    chunks = search(query_embedding, n_results=top_k, source_filter=source_filter)

    if not chunks:
        return "❌ No relevant content found in your knowledge base for that question."

    # Build prompt
    context = format_chunks_for_prompt(chunks)
    user_prompt = f"""Question: {question}

Retrieved context from your knowledge base:
---
{context}
---

Please answer the question based on the context above."""

    # Call Claude
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=build_system_prompt(),
        messages=[{"role": "user", "content": user_prompt}]
    )

    answer = response.content[0].text

    # Add source attribution footer
    sources_used = list(set([
        c["metadata"].get("title") or c["metadata"].get("filename", "Unknown")
        for c in chunks
    ]))
    footer = f"\n\n📚 Sources: {', '.join(sources_used[:3])}"
    if len(sources_used) > 3:
        footer += f" (+{len(sources_used) - 3} more)"

    return answer + footer


def weekly_digest() -> str:
    """Generate a weekly digest of forgotten ideas."""
    question = (
        "Surface 5 interesting ideas, notes, or concepts from my knowledge base "
        "that I might have forgotten about or haven't revisited recently. "
        "For each one, explain why it might be worth revisiting."
    )
    return query(question, top_k=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query your knowledge base")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--source", choices=["pdf", "notion", "obsidian", "gmail"], help="Filter by source")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of chunks to retrieve")
    parser.add_argument("--digest", action="store_true", help="Generate weekly digest")
    parser.add_argument("--stats", action="store_true", help="Show ChromaDB stats")
    args = parser.parse_args()

    if args.stats:
        stats = get_stats()
        print(f"📊 Knowledge Base Stats")
        print(f"Total chunks: {stats['total_chunks']}")
        for src, count in stats["by_source"].items():
            print(f"  {src}: {count} chunks")
    elif args.digest:
        print(weekly_digest())
    elif args.question:
        print(query(args.question, top_k=args.top_k, source_filter=args.source))
    else:
        parser.print_help()
