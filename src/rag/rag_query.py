"""
RAG Query Pipeline
==================
Takes a user question, retrieves relevant chunks from ChromaDB,
and feeds them to Claude for answer synthesis.

Also detects file-send requests and routes them to the file sender.

Usage:
    python src/rag/rag_query.py "What did I write about RAG?"
    python src/rag/rag_query.py "Send me the Project Proposal" --chat-id 123456
    python src/rag/rag_query.py "List files in Reid project" --chat-id 123456
    python src/rag/rag_query.py --project "Person-Reid-Model" "what does this project do?"
    python src/rag/rag_query.py --digest
    python src/rag/rag_query.py --stats
"""

import os
import re
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

FILE_SEND_TRIGGERS = [
    "send me", "share", "send the", "get me", "fetch",
    "download", "give me the file", "send file",
    "forward me", "attach", "send document", "send it",
]

LIST_TRIGGERS = [
    "list files", "what files", "show files", "what documents",
    "list documents", "show documents", "files in",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_chunks_for_prompt(chunks: list[dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, 1):
        meta      = chunk["metadata"]
        source    = meta.get("source", "unknown").upper()
        project   = meta.get("project_name", "")
        title     = meta.get("title") or meta.get("filename") or "Untitled"
        file_type = meta.get("file_type", "")
        relevance = chunk["relevance"]

        label = source
        if project and project != "root":
            label += f" | project:{project}"
        if file_type:
            label += f" | {file_type}"
        label += f" | {title} (relevance: {relevance})"

        lines.append(f"[{i}] {label}")
        lines.append(chunk["document"])
        lines.append("")
    return "\n".join(lines)


def build_system_prompt() -> str:
    return (
        "You are KnowledgeOS, a personal knowledge assistant with access to the user's "
        "notes, documents, PDFs, Word files, Excel sheets, code files, images, and Notion "
        "pages — organised into project folders.\n\n"
        "Rules:\n"
        "- Base your answer primarily on the retrieved context\n"
        "- If the context doesn't contain enough information, say so clearly\n"
        "- Cite the source (filename, project name) when referencing specific information\n"
        "- Be concise and direct\n"
        "- Highlight connections between different sources or projects\n"
        "- When answering about code files, explain what the code does in plain language"
    )


def is_file_send_request(question: str) -> bool:
    q = question.lower()
    return any(t in q for t in FILE_SEND_TRIGGERS)


def is_list_request(question: str) -> bool:
    q = question.lower()
    return any(t in q for t in LIST_TRIGGERS)


def extract_project_from_question(question: str):
    match = re.search(r'\bin\s+([A-Za-z0-9_\-]+)\s+project', question, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r'([A-Za-z0-9_\-]+)\s+project', question, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _search_with_project(query_embedding, top_k, project_name):
    from rag.chromadb_store import get_collection
    collection = get_collection()
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"project_name": project_name},
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "document":  doc,
                "metadata":  meta,
                "distance":  dist,
                "relevance": round(1 - dist, 3),
            })
        return chunks
    except Exception:
        return search(query_embedding, n_results=top_k)


# ── Core query ────────────────────────────────────────────────────────────────

def query(question: str, top_k: int = TOP_K,
          source_filter: str = None,
          project_filter: str = None,
          chat_id: str = None) -> str:

    stats = get_stats()
    if stats["total_chunks"] == 0:
        return (
            "⚠️ Your knowledge base is empty!\n\n"
            "Run ingestion first:\n"
            "  python src/ingest/universal_ingest.py"
        )

    # File send intent
    if is_file_send_request(question) and chat_id:
        try:
            from telegram.file_sender import find_and_send_file
            proj = project_filter or extract_project_from_question(question)
            return find_and_send_file(question, chat_id, project_filter=proj)
        except ImportError as e:
            return f"❌ File sender module error: {e}"

    # File list intent
    if is_list_request(question) and chat_id:
        try:
            from telegram.file_sender import list_files_in_project
            proj = project_filter or extract_project_from_question(question)
            if proj:
                return list_files_in_project(proj, chat_id=chat_id)
        except ImportError:
            pass

    # Normal RAG
    embedder        = get_embedder()
    query_embedding = embedder.embed(question)

    if project_filter:
        chunks = _search_with_project(query_embedding, top_k, project_filter)
    else:
        chunks = search(query_embedding, n_results=top_k, source_filter=source_filter)

    if not chunks:
        return "❌ No relevant content found in your knowledge base for that question."

    context     = format_chunks_for_prompt(chunks)
    user_prompt = (
        f"Question: {question}\n\n"
        f"Retrieved context from your knowledge base:\n---\n{context}\n---\n\n"
        f"Please answer the question based on the context above."
    )

    client   = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=build_system_prompt(),
        messages=[{"role": "user", "content": user_prompt}],
    )

    answer = response.content[0].text

    sources_used = list(set([
        c["metadata"].get("title") or c["metadata"].get("filename", "Unknown")
        for c in chunks
    ]))
    projects_used = list(set([
        c["metadata"].get("project_name", "")
        for c in chunks
        if c["metadata"].get("project_name", "") not in ("", "root")
    ]))

    footer = f"\n\n📚 Sources: {', '.join(sources_used[:3])}"
    if len(sources_used) > 3:
        footer += f" (+{len(sources_used) - 3} more)"
    if projects_used:
        footer += f"\n📁 Projects: {', '.join(projects_used[:3])}"

    return answer + footer


def weekly_digest() -> str:
    question = (
        "Surface 5 interesting ideas, notes, or concepts from my knowledge base "
        "that I might have forgotten about or haven't revisited recently. "
        "For each one, explain why it might be worth revisiting."
    )
    return query(question, top_k=10)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query your personal knowledge base")
    parser.add_argument("question",  nargs="?",  help="Question to ask")
    parser.add_argument("--source",  choices=["pdf", "notion", "obsidian", "gmail", "universal"],
                        help="Filter by ingestion source")
    parser.add_argument("--project", type=str,   help="Filter by project folder name")
    parser.add_argument("--top-k",   type=int,   default=TOP_K)
    parser.add_argument("--chat-id", type=str,   help="Telegram chat ID (enables file send)")
    parser.add_argument("--digest",  action="store_true")
    parser.add_argument("--stats",   action="store_true")
    args = parser.parse_args()

    if args.stats:
        stats = get_stats()
        print("📊 Knowledge Base Stats")
        print(f"Total chunks: {stats['total_chunks']}")
        for src, count in stats["by_source"].items():
            print(f"  {src}: {count} chunks")
    elif args.digest:
        print(weekly_digest())
    elif args.question:
        print(query(
            args.question,
            top_k=args.top_k,
            source_filter=args.source,
            project_filter=args.project,
            chat_id=args.chat_id,
        ))
    else:
        parser.print_help()