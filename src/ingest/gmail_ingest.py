"""
Gmail Thread Ingestion Pipeline
================================
Fetches Gmail threads, extracts key decisions and action items,
chunks them, and stores in ChromaDB.

Setup:
    1. Go to https://console.cloud.google.com
    2. Enable Gmail API
    3. Create OAuth 2.0 credentials
    4. Download credentials.json to config/gmail_credentials.json
    5. Run: pip install google-auth google-auth-oauthlib google-api-python-client

Usage:
    python src/ingest/gmail_ingest.py
    python src/ingest/gmail_ingest.py --max 50
"""

import os
import sys
import hashlib
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CREDENTIALS_PATH = "config/gmail_credentials.json"
TOKEN_PATH = "config/gmail_token.json"
MAX_THREADS = 100
CHUNK_SIZE = 400


def check_dependencies():
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        return True
    except ImportError:
        print("Gmail dependencies not installed.")
        print("Run: pip install google-auth google-auth-oauthlib google-api-python-client")
        return False


def get_gmail_service():
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    import json

    SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
    creds = None

    if Path(TOKEN_PATH).exists():
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not Path(CREDENTIALS_PATH).exists():
                print(f"Credentials file not found: {CREDENTIALS_PATH}")
                print("Download from Google Cloud Console and place at that path.")
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def extract_thread_text(service, thread_id: str) -> str:
    import base64
    thread = service.users().threads().get(userId="me", id=thread_id, format="full").execute()
    messages = thread.get("messages", [])
    parts = []

    for message in messages[:5]:  # limit to first 5 messages per thread
        payload = message.get("payload", {})
        body = payload.get("body", {})
        data = body.get("data", "")
        if data:
            text = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
            parts.append(text[:2000])

    return "\n\n".join(parts)


def get_thread_subject(thread, service) -> str:
    messages = thread.get("messages", [])
    if not messages:
        return "No subject"
    headers = messages[0].get("payload", {}).get("headers", [])
    for h in headers:
        if h.get("name", "").lower() == "subject":
            return h.get("value", "No subject")
    return "No subject"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def ingest_gmail_threads(max_threads: int = MAX_THREADS):
    if not check_dependencies():
        return

    sys.path.append(str(Path(__file__).parent.parent))
    from embed.embedder import get_embedder
    from rag.chromadb_store import get_collection

    print("Gmail Thread Ingestion")
    print("======================")

    try:
        service = get_gmail_service()
    except Exception as e:
        print(f"Gmail auth failed: {e}")
        return

    embedder = get_embedder()
    collection = get_collection()

    threads_result = service.users().threads().list(
        userId="me",
        maxResults=max_threads,
        q="is:important OR label:starred"
    ).execute()

    threads = threads_result.get("threads", [])
    print(f"Found {len(threads)} threads")

    total_chunks = 0
    for thread_meta in threads:
        thread_id = thread_meta["id"]

        try:
            thread = service.users().threads().get(
                userId="me", id=thread_id, format="full"
            ).execute()

            subject = get_thread_subject(thread, service)
            text = extract_thread_text(service, thread_id)

            if len(text.strip()) < 50:
                continue

            print(f"Processing: {subject[:60]}")
            chunks = chunk_text(f"{subject}\n\n{text}")

            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(
                    f"gmail:{thread_id}:c{i}:{chunk[:50]}".encode()
                ).hexdigest()

                embedding = embedder.embed(chunk)
                collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "source": "gmail",
                        "thread_id": thread_id,
                        "subject": subject,
                        "chunk_index": i,
                    }]
                )
                total_chunks += 1

        except Exception as e:
            print(f"  Error processing thread {thread_id}: {e}")
            continue

    print(f"\nDone. {total_chunks} chunks stored from Gmail.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Gmail threads into ChromaDB")
    parser.add_argument("--max", type=int, default=MAX_THREADS, help="Max threads to fetch")
    args = parser.parse_args()
    ingest_gmail_threads(max_threads=args.max)