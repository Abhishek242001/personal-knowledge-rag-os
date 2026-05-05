"""
Embedding Pipeline
==================
Supports multiple embedding providers:
- Voyage AI (cloud, free tier 200M tokens) — recommended with Claude
- OpenAI (cloud, paid) — best quality
- Ollama (local, free) — nomic-embed-text

Priority order (auto-detected from .env):
1. VOYAGE_API_KEY → Voyage AI
2. OPENAI_API_KEY → OpenAI
3. OLLAMA_BASE_URL → Ollama
"""

import os
from dotenv import load_dotenv

load_dotenv()


class VoyageEmbedder:
    def __init__(self):
        try:
            import voyageai
        except ImportError:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "voyageai", "-q"])
            import voyageai
        self.client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        self.model = "voyage-3-lite"
        self.dimension = 512
        print(f"✅ Embedder: Voyage AI ({self.model})")

    def embed(self, text: str) -> list[float]:
        text = text.replace("\n", " ").strip()[:8000]
        result = self.client.embed([text], model=self.model)
        return result.embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        texts = [t.replace("\n", " ").strip()[:8000] for t in texts]
        result = self.client.embed(texts, model=self.model)
        return result.embeddings


class OpenAIEmbedder:
    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openai", "-q"])
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
    def __init__(self):
        import requests
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.dimension = 768
        self.requests = requests

        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
        except Exception:
            raise ConnectionError(
                f"❌ Cannot connect to Ollama at {self.base_url}\n"
                f"   Install: curl -fsSL https://ollama.ai/install.sh | sh\n"
                f"   Pull model: ollama pull {self.model}"
            )
        print(f"✅ Embedder: Ollama ({self.model} @ {self.base_url})")

    def embed(self, text: str) -> list[float]:
        # Truncate to safe length — nomic-embed-text has 8192 token limit
        text = text.strip().replace("\n", " ")

        # Try with full text first, then fall back to shorter versions
        for max_len in [4000, 2000, 500]:
            try:
                response = self.requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text[:max_len]},
                    timeout=60
                )
                response.raise_for_status()
                return response.json()["embedding"]
            except self.requests.exceptions.HTTPError as e:
                if response.status_code == 500 and max_len > 500:
                    print(f"   ⚠️  Retrying with shorter text ({max_len} → {max_len // 2} chars)...")
                    continue
                raise e

        raise RuntimeError(f"❌ Ollama failed to embed even short text. Check ollama logs.")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def get_embedder():
    """
    Auto-select embedder based on .env config.
    Priority: Voyage AI → OpenAI → Ollama
    """
    voyage_key = os.getenv("VOYAGE_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "")

    if voyage_key and "YOUR" not in voyage_key and voyage_key.strip():
        return VoyageEmbedder()
    elif openai_key and "YOUR" not in openai_key and openai_key.strip():
        return OpenAIEmbedder()
    elif ollama_url and ollama_url.strip():
        return OllamaEmbedder()
    else:
        print("\n❌ No embedder configured in .env!")
        print("=" * 50)
        print("Option A — Voyage AI (FREE 200M tokens):")
        print("  1. https://dash.voyageai.com")
        print("  2. VOYAGE_API_KEY=your-key")
        print()
        print("Option B — OpenAI:")
        print("  1. https://platform.openai.com/api-keys")
        print("  2. OPENAI_API_KEY=sk-your-key")
        print()
        print("Option C — Ollama (local, free):")
        print("  1. curl -fsSL https://ollama.ai/install.sh | sh")
        print("  2. ollama pull nomic-embed-text")
        print("  3. OLLAMA_BASE_URL=http://localhost:11434")
        print("=" * 50)
        raise ValueError("No embedder configured.")


if __name__ == "__main__":
    embedder = get_embedder()
    test = embedder.embed("This is a test sentence for embedding.")
    print(f"✅ Test successful! Dimension: {len(test)}")