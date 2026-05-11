"""
Telegram File Sender
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

MAX_FILE_SIZE = 25 * 1024 * 1024
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")


def send_message(chat_id: str, text: str) -> bool:
    if not BOT_TOKEN:
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    resp = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=15)
    return resp.ok


def send_document(chat_id: str, file_path: Path, caption: str = "") -> dict:
    if not BOT_TOKEN:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN not set"}
    if not file_path.exists():
        return {"ok": False, "error": f"File not found: {file_path}"}
    size = file_path.stat().st_size
    if size > MAX_FILE_SIZE:
        return {"ok": False, "error": f"File too large ({size/(1024*1024):.1f} MB). Limit is 25 MB."}
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(
                url,
                data={"chat_id": chat_id, "caption": caption[:1024]},
                files={"document": (file_path.name, f)},
                timeout=60,
            )
        return {"ok": resp.ok, "error": None if resp.ok else resp.json().get("description")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def clean_query(query: str) -> str:
    """Remove action words to get just the file description."""
    q = query.lower()
    for word in ["send me the", "send me", "share the", "share", "get me the",
                 "get me", "fetch the", "fetch", "forward me the", "forward me",
                 "attach the", "attach", "send the", "send", "give me the", "give me"]:
        q = q.replace(word, "").strip()
    return q.strip() or query


def find_file_by_name(query: str, project_filter: str = None) -> list[dict]:
    """Search by filename keyword match directly in metadata."""
    collection = get_collection()
    results = collection.get(include=["metadatas"])

    q = clean_query(query).lower()
    query_words = [w for w in q.split() if len(w) > 2]

    seen = {}
    for meta in results["metadatas"]:
        filename = meta.get("filename", "").lower()
        fp = meta.get("file_path", "")
        if not fp:
            continue
        if project_filter and project_filter.lower() not in meta.get("project_name", "").lower():
            continue
        score = sum(1 for w in query_words if w in filename)
        if score > 0 and fp not in seen:
            seen[fp] = {
                "file_path":    fp,
                "filename":     meta.get("filename", "?"),
                "project_name": meta.get("project_name", "root"),
                "file_type":    meta.get("file_type", "?"),
                "rel_path":     meta.get("rel_path", ""),
                "relevance":    score,
            }

    return sorted(seen.values(), key=lambda x: x["relevance"], reverse=True)


def find_file_by_semantic(query: str, project_filter: str = None) -> list[dict]:
    """Semantic search fallback."""
    embedder = get_embedder()
    collection = get_collection()
    q = clean_query(query)
    embedding = embedder.embed(q)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=10,
        include=["metadatas", "distances"],
    )

    seen = {}
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        relevance = round(1 - dist, 3)
        if relevance < 0.10:
            continue
        fp = meta.get("file_path", "")
        if not fp:
            continue
        if project_filter and project_filter.lower() not in meta.get("project_name", "").lower():
            continue
        if fp not in seen or relevance > seen[fp]["relevance"]:
            seen[fp] = {
                "file_path":    fp,
                "filename":     meta.get("filename", "?"),
                "project_name": meta.get("project_name", "root"),
                "file_type":    meta.get("file_type", "?"),
                "rel_path":     meta.get("rel_path", ""),
                "relevance":    relevance,
            }

    return sorted(seen.values(), key=lambda x: x["relevance"], reverse=True)


def find_and_send_file(question: str, chat_id: str, project_filter: str = None) -> str:
    print(f"\n📎 File send request: {question}")

    # Try filename match first
    candidates = find_file_by_name(question, project_filter)
    print(f"   Filename matches: {len(candidates)}")

    # Fall back to semantic search
    if not candidates:
        candidates = find_file_by_semantic(question, project_filter)
        print(f"   Semantic matches: {len(candidates)}")

    if not candidates:
        msg = "❌ No matching file found.\nMake sure the file has been ingested."
        send_message(chat_id, msg)
        return msg

    best = candidates[0]
    file_path = Path(best["file_path"])
    print(f"   Sending: {best['filename']} (score: {best['relevance']})")

    caption = f"📄 {best['filename']}\n📁 Project: {best['project_name']}"
    result = send_document(chat_id, file_path, caption=caption)

    if result["ok"]:
        return f"✅ Sent: {best['filename']}"
    else:
        err = f"❌ Could not send {best['filename']}\nReason: {result['error']}"
        send_message(chat_id, err)
        return err


def list_files_in_project(project_name: str, chat_id: str = None) -> str:
    collection = get_collection()
    results = collection.get(include=["metadatas"])

    seen = {}
    for meta in results["metadatas"]:
        if project_name.lower() not in meta.get("project_name", "").lower():
            continue
        fp = meta.get("file_path", "")
        if fp and fp not in seen:
            seen[fp] = {
                "filename":  meta.get("filename", "?"),
                "file_type": meta.get("file_type", "?"),
                "size_kb":   meta.get("file_size_kb", "?"),
            }

    if not seen:
        return f"No files found for project: {project_name}"

    lines = [f"📁 Files in project: {project_name}\n"]
    for info in sorted(seen.values(), key=lambda x: x["filename"]):
        lines.append(f"  • {info['filename']}  [{info['file_type']}]  ({info['size_kb']} KB)")

    msg = "\n".join(lines)
    if chat_id:
        send_message(chat_id, msg)
    return msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query",     help="What file to look for")
    parser.add_argument("--chat-id", required=True)
    parser.add_argument("--project", help="Filter to specific project")
    parser.add_argument("--list",    action="store_true")
    args = parser.parse_args()

    if args.list and args.project:
        print(list_files_in_project(args.project, chat_id=args.chat_id))
    else:
        print(find_and_send_file(args.query, args.chat_id, project_filter=args.project))
