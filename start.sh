#!/bin/bash
# Personal Knowledge OS — Start Script
# Usage: bash start.sh

set -e

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "ERROR: .env file not found."
    echo "Run: cp .env.example .env and fill in your credentials."
    exit 1
fi

mkdir -p logs

echo "Starting Personal Knowledge OS..."

# Start ngrok tunnel
pkill -f "ngrok http" 2>/dev/null || true
sleep 1

if [ -n "$NGROK_URL" ] && [[ "$NGROK_URL" != *"YOUR"* ]]; then
    DOMAIN=$(echo $NGROK_URL | sed 's|https://||')
    ngrok http ${OPENCLAW_PORT:-18789} --url=$DOMAIN > logs/ngrok.log 2>&1 &
else
    ngrok http ${OPENCLAW_PORT:-18789} > logs/ngrok.log 2>&1 &
fi

echo "ngrok tunnel started"
sleep 3

# Start OpenClaw gateway
echo "Starting OpenClaw gateway on port ${OPENCLAW_PORT:-18789}..."
openclaw gateway