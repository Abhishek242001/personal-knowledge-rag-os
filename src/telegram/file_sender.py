"""
Telegram File Sender
=====================
Searches the knowledge base for a file matching the user's request
and sends it directly over Telegram.

Flow:
    1. User asks: "Send me the Project Proposal from Reid project"
    2. RAG search finds matching chunk with file_path in metadata
    3. Bot calls Telegram sendDocument with that absolute path
    4. File delivered to user in Telegram

Usage (standalone CLI test):
    python src/telegram/file_sender.py "Project Proposal" --chat-id YOUR_CHAT_ID

Usage (from rag_query.py):
    from telegram.file_sender import find_and_send_file
    result = find_and_send_file(question, chat_id)
"""

import os
import sys
import requests
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))
from embed.embedder import get_embedder
from rag.chromadb_store import get_collection

MAX_FILE_SIZE    = 25 * 1024 * 1024   # 25 MB Telegram bot limit
SEARCH_TOP_K     = 10                  # how many chunks to scan for file candidates
RELEVANCE_CUTOFF = 0.25               # minimum similarity score (0-1, higher = stricter)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")


# ── Telegram API helpers ──────────────────────────────────────────────────────

def send_message(chat_id: str, text: str) -> bool:
    """Send a plain text message to a Telegram chat."""
    if not BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not set")
        return False
    url  = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    resp = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=15)
    return resp.ok


def send_document(chat_id: str, file_path: Path, caption: str = "") -> dict:
    """
    Upload and send a file to a Telegram chat.
    Returns {"ok": bool, "error": str or None}
    """
    if not BOT_TOKEN:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN not set"}

    if not file_path.exists():
        return {"ok": False, "error": f"File not found on server: {file_path}"}

    size = file_path.stat().st_size
    if size > MAX_FILE_SIZE:
        mb = size / (1024 * 1024)
        return {"ok": False, "error": f"File too large ({mb:.1f} MB). Telegram limit is 25 MB."}

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(
                url,
                data={"chat_id": chat_id, "caption": caption[:1024]},
                files={"document": (file_path.name, f)},
                timeout=60,
            )
        if resp.ok:
            return {"ok": True, "error": None}
        else:
            return {"ok": False, "error": resp.json().get("description", "Unknown Telegram error")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── File search logic ─────────────────────────────────────────────────────────

def is_file_request(question: str) -> bool:
    """
    Detect if the user is asking to receive/send a file,
    not just asking a knowledge question.
    """
    triggers = [
        "send me", "share", "send the", "get me", "fetch",
        "download", "give me the file", "send file",
        "forward me", "attach", "send document",
    ]
    q_lower = question.lower()
    return any(t in q_lower for t in triggers)


def search_files(query: str, project_filter: str = None,
                 file_type_filter: str = None) -> list[dict]:
    """
    Search ChromaDB for file chunks matching the query.
    Returns list of unique file candidates sorted by best relevance.
    """
    embedder   = get_embedder()
    collection = get_collection()

    query_embedding = embedder.embed(query)

    # Build where filter
    where = {"source": "universal"}
    if project_filter:
        where = {"$and": [{"source": "universal"}, {"project_name": project_filter}]}
    if file_type_filter:
        where = {"$and": [{"source": "universal"}, {"file_type": file_type_filter}]}

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=SEARCH_TOP_K,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        print(f"Search error: {e}")
        return []

    # De-duplicate by file_path, keeping best (lowest distance) per file
    seen      = {}
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        relevance = round(1 - dist, 3)
        if relevance < RELEVANCE_CUTOFF:
            continue
        fp = meta.get("file_path", "")
        if not fp:
            continue
        if fp not in seen or relevance > seen[fp]["relevance"]:
            seen[fp] = {
                "file_path":    fp,
                "filename":     meta.get("filename", Path(fp).name),
                "project_name": meta.get("project_name", "root"),
                "file_type":    meta.get("file_type", "unknown"),
                "rel_path":     meta.get("rel_path", ""),
                "relevance":    relevance,
                "snippet":      doc[:200],
            }

    # Sort by relevance descending
    return sorted(seen.values(), key=lambda x: x["relevance"], reverse=True)


def find_and_send_file(question: str, chat_id: str,
                       project_filter: str = None) -> str:
    """
    Main entry point. Finds the best matching file and sends it over Telegram.
    Returns a status message string (for the bot to relay back if needed).
    """
    print(f"\n📎 File send request: {question}")

    candidates = search_files(question, project_filter=project_filter)

    if not candidates:
        msg = (
            "❌ No matching file found in your knowledge base.\n\n"
            "Make sure:\n"
            "• The file has been ingested (run universal_ingest.py)\n"
            "• Your description matches the file name or content"
        )
        send_message(chat_id, msg)
        return msg

    # Take the best match
    best = candidates[0]
    file_path = Path(best["file_path"])

    print(f"   Best match: {best['filename']} (relevance: {best['relevance']})")

    # If multiple strong candidates, list them
    strong = [c for c in candidates if c["relevance"] >= 0.55]
    if len(strong) > 1:
        # More than one strong match — ask user to clarify
        listing = "\n".join([
            f"  {i+1}. {c['filename']} [{c['project_name']}] (match: {c['relevance']})"
            for i, c in enumerate(strong[:5])
        ])
        msg = (
            f"📁 Found {len(strong)} possible files. Sending the best match:\n\n"
            f"{listing}\n\n"
            f"➡️  Sending: {best['filename']}"
        )
        send_message(chat_id, msg)

    # Build caption
    caption = (
        f"📄 {best['filename']}\n"
        f"📁 Project: {best['project_name']}\n"
        f"🔍 Match score: {best['relevance']}"
    )

    # Send the file
    result = send_document(chat_id, file_path, caption=caption)

    if result["ok"]:
        status = f"✅ Sent: {best['filename']}"
        print(f"   {status}")
        return status
    else:
        err_msg = (
            f"❌ Could not send {best['filename']}\n"
            f"Reason: {result['error']}\n\n"
            f"File location on server:\n{file_path}"
        )
        send_message(chat_id, err_msg)
        print(f"   ❌ Send failed: {result['error']}")
        return err_msg


def list_files_in_project(project_name: str, chat_id: str = None) -> str:
    """
    List all ingested files in a project.
    Optionally sends the list to Telegram.
    """
    collection = get_collection()

    try:
        results = collection.get(
            where={"$and": [{"source": "universal"}, {"project_name": project_name}]},
            include=["metadatas"],
        )
    except Exception as e:
        return f"Error fetching project files: {e}"

    if not results["metadatas"]:
        return f"No files found for project: {project_name}"

    # Unique files only
    seen  = {}
    for meta in results["metadatas"]:
        fp = meta.get("file_path", "")
        if fp and fp not in seen:
            seen[fp] = {
                "filename":  meta.get("filename", "?"),
                "file_type": meta.get("file_type", "?"),
                "size_kb":   meta.get("file_size_kb", "?"),
            }

    lines = [f"📁 Files in project: {project_name}\n"]
    for info in sorted(seen.values(), key=lambda x: x["filename"]):
        lines.append(
            f"  • {info['filename']}  [{info['file_type']}]  ({info['size_kb']} KB)"
        )

    msg = "\n".join(lines)

    if chat_id:
        send_message(chat_id, msg)

    return msg


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search and send files over Telegram")
    parser.add_argument("query",      help="What file to look for")
    parser.add_argument("--chat-id",  required=True, help="Telegram chat ID to send file to")
    parser.add_argument("--project",  help="Filter search to a specific project")
    parser.add_argument("--list",     action="store_true",
                        help="List all files in the project instead of searching")
    args = parser.parse_args()

    if args.list and args.project:
        print(list_files_in_project(args.project, chat_id=args.chat_id))
    else:
        result = find_and_send_file(args.query, args.chat_id, project_filter=args.project)
        print(result)