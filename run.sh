#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
# Sports Media Guardian — Quick Setup & Run Script
# Usage: bash run.sh
# ─────────────────────────────────────────────────────────

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$ROOT/backend"
VENV="$ROOT/.venv"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   🛡  SPORTS MEDIA GUARDIAN — SETUP          ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── 1. Create virtual environment ────────────────────
if [ ! -d "$VENV" ]; then
  echo "📦 Creating Python virtual environment..."
  python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"

# ── 2. Install dependencies ───────────────────────────
echo "📦 Installing backend dependencies..."
pip install -q --upgrade pip
pip install -q -r "$BACKEND/requirements.txt"

# ── 3. Generate sample dataset ────────────────────────
if [ -z "$(ls -A $ROOT/dataset/original 2>/dev/null)" ]; then
  echo ""
  echo "🎨 Generating sample dataset images..."
  cd "$ROOT"
  python generate_sample_dataset.py
  echo ""
fi

# ── 4. Start backend ──────────────────────────────────
echo ""
echo "🚀 Starting FastAPI backend on http://localhost:8000 ..."
echo "   Frontend: http://localhost:8000"
echo "   API docs: http://localhost:8000/docs"
echo ""
echo "   Press Ctrl+C to stop."
echo ""

cd "$BACKEND"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
