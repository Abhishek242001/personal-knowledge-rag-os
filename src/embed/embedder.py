"""
Embedding Pipeline
==================
Supports OpenAI embeddings (cloud) or Ollama (local/free).
Configured via .env — set OPENAI_API_KEY for OpenAI
or OLLAMA_BASE_URL for local Ollama.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class OpenAIEmbedder:
    """Cloud embeddings using OpenAI text-embedding-3-small."""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "text-embedding-3-small"
        self.dimension = 1536
        print(f"✅ Embedder: OpenAI ({self.model})")

    def embed(self, text: str) -> list[float]:
        text = text.replace("\n", " ").strip()
        response = self.client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        texts = [t.replace("\n", " ").strip() for t in texts]
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]


class OllamaEmbedder:
    """Local embeddings using Ollama (free, no API key needed)."""

    def __init__(self):
        import requests
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.dimension = 768
        self.requests = requests
        print(f"✅ Embedder: Ollama ({self.model} @ {self.base_url})")

    def embed(self, text: str) -> list[float]:
        response = self.requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def get_embedder():
    """
    Auto-select embedder based on .env config.
    Priority: Ollama (if OLLAMA_BASE_URL set) → OpenAI → Error
    """
    ollama_url = os.getenv("OLLAMA_BASE_URL")
    openai_key = os.getenv("OPENAI_API_KEY")

    if ollama_url and "localhost" in ollama_url:
        return OllamaEmbedder()
    elif openai_key and "your-openai" not in openai_key:
        return OpenAIEmbedder()
    else:
        print("⚠️  No embedder configured!")
        print("   Option A: Set OPENAI_API_KEY in .env")
        print("   Option B: Set OLLAMA_BASE_URL=http://localhost:11434 in .env")
        print("   Install Ollama: https://ollama.ai")
        raise ValueError("No embedder configured. See above options.")
