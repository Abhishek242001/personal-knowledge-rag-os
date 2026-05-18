#!/bin/bash
echo "📦 Installing dependencies..."
pip install -q python-docx python-pptx pandas openpyxl xlrd pymupdf anthropic chromadb python-dotenv requests notion-client

echo "🚀 Starting Ollama..."
ollama serve &
sleep 5

echo "📥 Pulling embedding model..."
ollama pull nomic-embed-text

echo "✅ Ready! Run: python src/ingest/universal_ingest.py"
