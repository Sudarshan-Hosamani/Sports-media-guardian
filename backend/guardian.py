"""
backend/guardian.py
Layer 1: Guardian Content Gatekeeper
Screens incoming uploads to verify whether they represent sports media before
handing off to Layer 2 (Piracy Analysis). Rejects text documents, invoices,
code screenshots, landscapes, portraits, interiors, and unrelated images.
"""

from PIL import Image
import numpy as np
import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
INPUT_FOLDER = BASE_DIR / "pending_media"
ALLOWED_FOLDER = BASE_DIR / "safe_to_post"
BLOCKED_FOLDER = BASE_DIR / "quarantine"

for folder in [INPUT_FOLDER, ALLOWED_FOLDER, BLOCKED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

print("🛡️ Guardian initialized (Layer 1 Vision Gatekeeper active)")

def is_sports_image(img_path):
    """
    Analyzes an image and returns:
    (is_sport: bool, detected_label: str, confidence: float, top_guess: str)
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        return False, "corrupted_or_invalid_image", 0.99, "invalid"

    arr = np.array(img, dtype=np.float32)
    hsv = np.array(img.convert("HSV"), dtype=np.float32)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    mean_sat = float(sat.mean())
    white_ratio = float(np.mean((arr[:, :, 0] > 220) & (arr[:, :, 1] > 220) & (arr[:, :, 2] > 220)))
    dark_ratio = float(np.mean((arr[:, :, 0] < 35) & (arr[:, :, 1] < 35) & (arr[:, :, 2] < 35)))

    # 1. Documents, Invoices, Text, Code Editor Screenshots
    if mean_sat < 28 and (white_ratio > 0.30 or dark_ratio > 0.40):
        return False, "document_or_code", 0.96, "document"

    # Edge energy (sharp boundaries, athlete silhouettes, field markings)
    diff = float(np.abs(np.diff(arr, axis=0)).mean() + np.abs(np.diff(arr, axis=1)).mean())

    # Sports field / court / pool color presence
    # Turf green: hue 45..118, sat >= 40, val >= 35
    turf_green = float(np.mean((hue >= 45) & (hue <= 118) & (sat >= 40) & (val >= 35)))
    # Court blue / pool: hue 125..185, sat >= 40, val >= 50
    court_blue = float(np.mean((hue >= 125) & (hue <= 185) & (sat >= 40) & (val >= 50)))
    # Hardwood court / running track / red clay: hue 8..42 or 238..255, sat >= 50, val >= 45
    court_warm = float(np.mean(((hue >= 8) & (hue <= 42) | (hue >= 238)) & (sat >= 50) & (val >= 45)))
    
    sports_field = turf_green + court_blue + court_warm

    # Sports classification criteria:
    # High edge energy with court lines (white lines) OR dominant sports arena field
    if diff >= 7.0 and (white_ratio >= 0.015 or sports_field >= 0.25):
        confidence = min(0.99, max(0.85, 0.70 + sports_field * 0.5))
        return True, "sports_media_detected", confidence, "sports_content"

    # Flat or low-energy graphic / landscape / portrait / non-sports
    return False, "non_sports_content", 0.90, "unrelated_media"

def check_media():
    files = [
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not files:
        print("📭 No files found in pending_media/")
        return

    print(f"🔍 Checking {len(files)} file(s)...\n")

    for filename in files:
        img_path = os.path.join(INPUT_FOLDER, filename)
        try:
            is_sport, detected_label, confidence, top_guess = is_sports_image(img_path)
            if is_sport:
                print(f"✅ ALLOWED: {filename} ({detected_label}, {confidence:.2f})")
                shutil.move(img_path, os.path.join(ALLOWED_FOLDER, filename))
            else:
                print(f"🚫 BLOCKED: {filename} ({detected_label}, {confidence:.2f})")
                shutil.move(img_path, os.path.join(BLOCKED_FOLDER, filename))
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    check_media()
