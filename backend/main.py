"""
Sports Media Guardian - Backend API
Detects unauthorized use of sports media images using CLIP embeddings + cosine similarity.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import io
import os
import base64
from pathlib import Path
from typing import List, Dict, Any
import json

app = FastAPI(title="Sports Media Guardian API")

# Allow all origins for local hackathon use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Embedding Engine
# Try CLIP first; fall back to a lightweight perceptual hash + color histogram hybrid
# ─────────────────────────────────────────────

USE_CLIP = False
clip_model = None
clip_preprocess = None
clip_device = None

try:
    import torch
    import clip as openai_clip
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = openai_clip.load("ViT-B/32", device=clip_device)
    USE_CLIP = True
    print("✅ CLIP loaded successfully")
except Exception as e:
    print(f"⚠️  CLIP not available ({e}). Using lightweight fallback embeddings.")


def get_embedding(img: Image.Image) -> np.ndarray:
    """
    Returns a normalized embedding vector for an image.
    Uses CLIP if available, otherwise a perceptual hash + color histogram hybrid.
    """
    if USE_CLIP:
        return _clip_embedding(img)
    return _lightweight_embedding(img)


def _clip_embedding(img: Image.Image) -> np.ndarray:
    """CLIP visual embedding — robust to crops, resizes, minor edits."""
    import torch
    tensor = clip_preprocess(img).unsqueeze(0).to(clip_device)
    with torch.no_grad():
        features = clip_model.encode_image(tensor)
    vec = features.cpu().numpy().flatten().astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


def _lightweight_embedding(img: Image.Image) -> np.ndarray:
    """
    Fallback: combines a resized pixel vector + HSV color histogram.
    Handles moderate crops/resizes/blurs reasonably well.
    """
    # Resize to fixed size to normalize spatial structure
    thumb = img.convert("RGB").resize((32, 32), Image.LANCZOS)
    pixel_vec = np.array(thumb, dtype=np.float32).flatten() / 255.0

    # HSV histogram captures color distribution (blur/crop robust)
    hsv = img.convert("RGB").resize((64, 64))
    hsv_arr = np.array(hsv, dtype=np.float32)
    h_hist, _ = np.histogram(hsv_arr[:, :, 0], bins=32, range=(0, 255))
    s_hist, _ = np.histogram(hsv_arr[:, :, 1], bins=16, range=(0, 255))
    v_hist, _ = np.histogram(hsv_arr[:, :, 2], bins=16, range=(0, 255))
    color_vec = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)

    combined = np.concatenate([pixel_vec * 0.6, color_vec * 0.4])
    return combined / (np.linalg.norm(combined) + 1e-8)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors → [0, 1]."""
    return float(np.dot(a, b))


def classify(score: float) -> str:
    """Map similarity score to violation status."""
    if score > 0.85:
        return "Violation"
    elif score >= 0.70:
        return "Tampered"
    return "Safe"


# ─────────────────────────────────────────────
# Dataset — loaded once at startup into memory
# ─────────────────────────────────────────────

DATASET_DIR = Path(__file__).parent.parent / "dataset"
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# In-memory store: list of { path, category, embedding, thumbnail_b64 }
dataset: List[Dict[str, Any]] = []


def load_dataset():
    """Scan dataset folders, compute embeddings, store in memory."""
    global dataset
    dataset = []

    for category in ["original", "modified", "unrelated"]:
        folder = DATASET_DIR / category
        if not folder.exists():
            print(f"  ⚠️  Folder not found: {folder}")
            continue

        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in SUPPORTED_EXT:
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                emb = get_embedding(img)
                thumb = _make_thumbnail_b64(img)
                dataset.append({
                    "path": str(img_path),
                    "filename": img_path.name,
                    "category": category,
                    "embedding": emb,
                    "thumbnail": thumb,
                })
                print(f"  ✓ Loaded {category}/{img_path.name}")
            except Exception as e:
                print(f"  ✗ Failed {img_path.name}: {e}")

    print(f"\n📦 Dataset loaded: {len(dataset)} images\n")


def _make_thumbnail_b64(img: Image.Image, size=(200, 200)) -> str:
    """Convert PIL image to base64-encoded JPEG thumbnail."""
    thumb = img.copy()
    thumb.thumbnail(size, Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.on_event("startup")
def startup_event():
    print("🚀 Loading dataset...")
    load_dataset()


# ─────────────────────────────────────────────
# Main Endpoint
# ─────────────────────────────────────────────

@app.post("/upload-and-analyze")
async def upload_and_analyze(file: UploadFile = File(...)):
    """
    Accepts an uploaded image, computes its embedding,
    compares with dataset, returns top 3 matches with similarity scores.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # Read and decode uploaded image
    raw = await file.read()
    try:
        query_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    if not dataset:
        raise HTTPException(status_code=503, detail="Dataset is empty. Add images to /dataset/ folders.")

    # Compute query embedding
    query_emb = get_embedding(query_img)

    # Compute cosine similarity against all dataset images
    scores = [
        {**item, "score": cosine_similarity(query_emb, item["embedding"])}
        for item in dataset
    ]

    # Sort descending and take top 3
    top3 = sorted(scores, key=lambda x: x["score"], reverse=True)[:3]

    # Build response (drop raw numpy arrays — not JSON serializable)
    results = []
    for match in top3:
        sim = round(match["score"] * 100, 1)
        results.append({
            "filename": match["filename"],
            "category": match["category"],
            "similarity": sim,
            "status": classify(match["score"]),
            "thumbnail": match["thumbnail"],
        })

    # Encode uploaded image as thumbnail for display
    query_thumb = _make_thumbnail_b64(query_img, size=(400, 400))

    # Overall verdict = highest-severity result among top 3
    verdict_priority = {"Violation": 3, "Tampered": 2, "Safe": 1}
    overall = max(results, key=lambda r: verdict_priority[r["status"]])["status"]

    return JSONResponse({
        "query_thumbnail": query_thumb,
        "embedding_method": "CLIP (ViT-B/32)" if USE_CLIP else "Perceptual Hash + Color Histogram",
        "overall_status": overall,
        "matches": results,
    })


@app.get("/dataset-info")
def dataset_info():
    """Returns a summary of the loaded dataset."""
    summary = {}
    for item in dataset:
        summary[item["category"]] = summary.get(item["category"], 0) + 1
    return {
        "total": len(dataset),
        "breakdown": summary,
        "embedding_method": "CLIP (ViT-B/32)" if USE_CLIP else "Perceptual Hash + Color Histogram",
    }


@app.get("/reload-dataset")
def reload_dataset():
    """Hot-reload the dataset without restarting the server."""
    load_dataset()
    return {"message": f"Dataset reloaded. {len(dataset)} images loaded."}


# Serve frontend static files
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
