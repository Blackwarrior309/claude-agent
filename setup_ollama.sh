#!/usr/bin/env bash
# Setup script for local Ollama models (optimized for 4 GB VRAM GPU)
# Run once on your laptop before starting the agent.

set -e

echo "=== Autonomous MicroBusiness Agent — Ollama Setup ==="
echo "GPU target: 4 GB VRAM (RTX 3060 oder ähnlich)"
echo ""

# 1. Install Ollama if not present
if ! command -v ollama &>/dev/null; then
    echo "[1/4] Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "[1/4] Ollama already installed: $(ollama --version)"
fi

# 2. Start Ollama service
echo ""
echo "[2/4] Starting Ollama service..."
if systemctl is-active --quiet ollama 2>/dev/null; then
    echo "       Ollama service already running"
elif pgrep -x ollama &>/dev/null; then
    echo "       Ollama process already running"
else
    ollama serve &
    sleep 3
    echo "       Ollama started (background)"
fi

# 3. Pull models that fit in 4 GB VRAM
echo ""
echo "[3/4] Pulling models for 4 GB VRAM..."
echo ""

echo "  → phi3.5:mini (~2.2 GB) — Klassifikation, Routing, Parsing"
ollama pull phi3.5:mini

echo ""
echo "  → llama3.2:3b (~2.0 GB) — Zusammenfassungen, einfache Antworten"
ollama pull llama3.2:3b

echo ""
echo "  → nomic-embed-text (~0.3 GB) — Embeddings"
ollama pull nomic-embed-text

echo ""
echo "[4/4] Verifying models..."
ollama list

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Teste local models:"
echo "  ollama run phi3.5:mini 'Antworte mit einem Wort: Hallo'"
echo ""
echo "Starte den Agenten:"
echo "  export ANTHROPIC_API_KEY=sk-ant-..."
echo "  python main.py --local-status"
echo "  python main.py --status"
echo "  python main.py --cycles 1"
