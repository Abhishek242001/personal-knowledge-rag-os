#!/bin/bash
# ============================================================
# Personal Knowledge OS — Start Script
# ============================================================
# Starts ngrok tunnel + OpenClaw gateway together
# Usage: bash start.sh
# ============================================================

set -e

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✅ Environment variables loaded"
else
    echo "❌ .env file not found!"
    echo "   Run: cp .env.example .env"
    echo "   Then fill in your API keys"
    exit 1
fi

# Create logs directory
mkdir -p logs

echo ""
echo "🧠 Starting Personal Knowledge OS..."
echo "======================================"

# --- Start ngrok tunnel ---
echo ""
echo "🔗 Starting ngrok tunnel on port $OPENCLAW_PORT..."
pkill -f "ngrok http" 2>/dev/null || true
sleep 1

if [ -n "$NGROK_URL" ] && [[ "$NGROK_URL" != *"your-tunnel"* ]]; then
    DOMAIN=$(echo $NGROK_URL | sed 's|https://||')
    ngrok http $OPENCLAW_PORT --url=$DOMAIN > logs/ngrok.log 2>&1 &
    echo "✅ ngrok started → $NGROK_URL"
else
    ngrok http $OPENCLAW_PORT > logs/ngrok.log 2>&1 &
    echo "✅ ngrok started (ephemeral URL — check logs/ngrok.log)"
fi

sleep 3

# --- Start OpenClaw gateway ---
echo ""
echo "🦞 Starting OpenClaw gateway..."
echo "======================================"
echo "   Telegram bot: @research_myknowledgeos_bot"
echo "   Port: $OPENCLAW_PORT"
echo "   Model: $OPENCLAW_MODEL"
echo ""
echo "   Press Ctrl+C to stop"
echo "======================================"
echo ""

openclaw gateway
