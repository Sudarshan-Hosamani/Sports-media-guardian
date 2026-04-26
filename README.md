# 🛡️ Sports Media Guardian

> **Hackathon prototype** for detecting unauthorized use of sports media images using AI-powered image similarity.

---

## 📁 Project Structure

```
sports-media-guardian/
├── backend/
│   ├── main.py              # FastAPI backend — all logic here
│   └── requirements.txt     # Python dependencies
├── frontend/
│   └── index.html           # Single-file UI (served by FastAPI)
├── dataset/
│   ├── original/            # ← Put your original protected images here
│   ├── modified/            # ← Put tampered/modified versions here
│   └── unrelated/           # ← Put unrelated "safe" images here
├── generate_sample_dataset.py   # Creates synthetic test images
└── run.sh                       # One-command setup & run
```

---

## ⚡ Quick Start (Recommended)

```bash
bash run.sh
```

Then open → **http://localhost:8000**

---

## 🔧 Manual Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r backend/requirements.txt
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\activate

```

### 3. (Optional) Enable CLIP for better accuracy

```bash
pip install torch
pip install git+https://github.com/openai/CLIP.git
```

### 4. Generate sample dataset

```bash
python generate_sample_dataset.py
```

### 5. Start the server

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
### 6. Open the app

Go to **http://localhost:8000** in your browser.

---

## 🤖 How It Works

### Embedding Engine

| Mode | Method | Best For |
|------|--------|----------|
| **CLIP** (if installed) | OpenAI ViT-B/32 visual encoder | Production accuracy |
| **Fallback** | Pixel thumbnail + HSV color histogram | Zero-dependency demo |

### Similarity & Classification

```
cosine_similarity(query_embedding, dataset_embedding) → score [0–1]

score > 0.85  → 🚨 VIOLATION  (direct copy)
score 0.70–0.85 → ⚠️ TAMPERED  (cropped/filtered/resized)
score < 0.70  → ✅ SAFE       (no match)
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload-and-analyze` | Upload image → get top 3 matches |
| GET | `/dataset-info` | Current dataset stats |
| GET | `/reload-dataset` | Hot-reload dataset without restart |
| GET | `/docs` | Swagger UI (auto-generated) |

---

## 📸 Adding Your Own Dataset

Drop images into the correct folders, then hit **http://localhost:8000/reload-dataset**:

```
dataset/
  original/    ← official sports photos (e.g., match_01.jpg)
  modified/    ← cropped/filtered versions of originals
  unrelated/   ← completely different images (control group)
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`

---

## 🏆 Hackathon Notes

- **No Docker, no cloud, no database** — everything runs locally in-memory
- Dataset is loaded once at startup; embeddings cached in RAM
- Frontend is a single HTML file served by FastAPI (no build step)
- CLIP install is optional — the fallback works for demos
