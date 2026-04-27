"""
Sports Media Guardian - Backend API
Layer 1: Sports relevance detection (Guardian)
Layer 2: Piracy detection using CLIP embeddings + cosine similarity
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import numpy as np
from PIL import Image
import io
import os
import base64
from pathlib import Path
from typing import List, Dict, Any

# Import Guardian sports classifier
try:
    from backend.guardian import is_sports_image
except ImportError:
    from guardian import is_sports_image

app = FastAPI(title="Sports Media Guardian API")

# ============================================
# CORS
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# EMBEDDING ENGINE
# Try CLIP first, fallback to lightweight embedding
# ============================================

USE_CLIP = False
clip_model = None
clip_preprocess = None
clip_device = None

try:
    import torch
#     import clip as openai_clip

    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = openai_clip.load(
        "ViT-B/32",
        device=clip_device
    )

    USE_CLIP = True
    print("✅ CLIP loaded successfully")

except Exception as e:
    print(f"⚠️ CLIP not available ({e})")
    print("Using lightweight fallback embeddings...")


def get_embedding(img: Image.Image) -> np.ndarray:
    if USE_CLIP:
        return _clip_embedding(img)
    return _lightweight_embedding(img)


def _clip_embedding(img: Image.Image) -> np.ndarray:
    import torch

    tensor = clip_preprocess(img).unsqueeze(0).to(clip_device)

    with torch.no_grad():
        features = clip_model.encode_image(tensor)

    vec = features.cpu().numpy().flatten().astype(np.float32)

    return vec / (np.linalg.norm(vec) + 1e-8)


def _lightweight_embedding(img: Image.Image) -> np.ndarray:
    thumb = img.convert("RGB").resize((32, 32), Image.LANCZOS)
    pixel_vec = np.array(thumb, dtype=np.float32).flatten() / 255.0

    hsv = img.convert("RGB").resize((64, 64))
    hsv_arr = np.array(hsv, dtype=np.float32)

    h_hist, _ = np.histogram(
        hsv_arr[:, :, 0],
        bins=32,
        range=(0, 255)
    )

    s_hist, _ = np.histogram(
        hsv_arr[:, :, 1],
        bins=16,
        range=(0, 255)
    )

    v_hist, _ = np.histogram(
        hsv_arr[:, :, 2],
        bins=16,
        range=(0, 255)
    )

    color_vec = np.concatenate([
        h_hist,
        s_hist,
        v_hist
    ]).astype(np.float32)

    combined = np.concatenate([
        pixel_vec * 0.6,
        color_vec * 0.4
    ])

    return combined / (np.linalg.norm(combined) + 1e-8)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def classify(score: float) -> str:
    if score > 0.85:
        return "Violation"
    elif score >= 0.70:
        return "Tampered"
    return "Safe"


# ============================================
# DATASET LOADING
# ============================================

DATASET_DIR = Path(__file__).parent.parent / "dataset"
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

dataset: List[Dict[str, Any]] = []


def _make_thumbnail_b64(img: Image.Image, size=(200, 200)) -> str:
    thumb = img.copy()
    thumb.thumbnail(size, Image.LANCZOS)

    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=75)

    return base64.b64encode(
        buf.getvalue()
    ).decode("utf-8")


def load_dataset():
    global dataset
    dataset = []

    for category in ["original", "modified", "unrelated"]:
        folder = DATASET_DIR / category

        if not folder.exists():
            print(f"⚠️ Folder not found: {folder}")
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

                print(f"✓ Loaded {category}/{img_path.name}")

            except Exception as e:
                print(f"✗ Failed {img_path.name}: {e}")

    print(f"\n📦 Dataset loaded: {len(dataset)} images\n")


@app.on_event("startup")
def startup_event():
    print("🚀 Loading dataset...")
    load_dataset()


# ============================================
# MAIN ENDPOINT
# ============================================

@app.post("/upload-and-analyze")
async def upload_and_analyze(file: UploadFile = File(...)):
    """
    Step 1: Sports relevance detection
    Step 2: Piracy detection if sports content
    """

    # Validate file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image."
        )

    # Read uploaded file
    raw = await file.read()

    try:
        query_img = Image.open(
            io.BytesIO(raw)
        ).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image."
        )

    # ============================================
    # LAYER 1: SPORTS FILTER (Guardian)
    # ============================================

    temp_path = f"temp_{file.filename}"

    try:
        with open(temp_path, "wb") as f:
            f.write(raw)
            label = "Sports Content"
            confidence = 0.95
            is_sport, label, confidence, _ = is_sports_image(temp_path)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if not is_sport:
        return JSONResponse({
            "overall_status": "Rejected",
            "reason": "Non-sports content detected",
            "detected_label": label,
            "confidence": round(float(confidence) * 100, 1),
            "status": "blocked"
        })

    # ============================================
    # LAYER 2: PIRACY DETECTION
    # ============================================

    if not dataset:
        raise HTTPException(
            status_code=503,
            detail="Dataset is empty. Add images to dataset folders."
        )

    query_emb = get_embedding(query_img)

    scores = []

    for item in dataset:
        score = cosine_similarity(
            query_emb,
            item["embedding"]
        )

        scores.append({
            **item,
            "score": score
        })

    top3 = sorted(
        scores,
        key=lambda x: x["score"],
        reverse=True
    )[:3]

    results = []

    for match in top3:
        similarity = round(match["score"] * 100, 1)

        results.append({
            "filename": match["filename"],
            "category": match["category"],
            "similarity": similarity,
            "status": classify(match["score"]),
            "thumbnail": match["thumbnail"]
        })

    query_thumb = _make_thumbnail_b64(
        query_img,
        size=(400, 400)
    )

    verdict_priority = {
        "Violation": 3,
        "Tampered": 2,
        "Safe": 1
    }

    overall = max(
        results,
        key=lambda r: verdict_priority[r["status"]]
    )["status"]

    return JSONResponse({
        "query_thumbnail": query_thumb,
        "sports_check": {
            "status": "approved",
            "detected_label": label,
            "confidence": round(float(confidence) * 100, 1)
        },
        "embedding_method": (
            "CLIP (ViT-B/32)"
            if USE_CLIP
            else "Perceptual Hash + Color Histogram"
        ),
        "overall_status": overall,
        "matches": results
    })


# ============================================
# INFO ENDPOINTS
# ============================================

@app.get("/dataset-info")
def dataset_info():
    summary = {}

    for item in dataset:
        summary[item["category"]] = (
            summary.get(item["category"], 0) + 1
        )

    return {
        "total": len(dataset),
        "breakdown": summary,
        "embedding_method": (
            "CLIP (ViT-B/32)"
            if USE_CLIP
            else "Perceptual Hash + Color Histogram"
        )
    }


@app.get("/reload-dataset")
def reload_dataset():
    load_dataset()

    return {
        "message": f"Dataset reloaded. {len(dataset)} images loaded."
    }


# ============================================
# FRONTEND STATIC FILES
# ============================================

frontend_dir = Path(__file__).parent.parent / "frontend"

if frontend_dir.exists():
    app.mount(
        "/",
        StaticFiles(
            directory=str(frontend_dir),
            html=True
        ),
        name="frontend"
    )


@app.get("/health")
def health():
    return {"status": "healthy"}