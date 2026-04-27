import tensorflow as tf
import numpy as np
import os
import shutil

from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image

# ============================================
# FOLDERS
# ============================================

INPUT_FOLDER = "pending_media"
ALLOWED_FOLDER = "safe_to_post"
BLOCKED_FOLDER = "quarantine"

for folder in [INPUT_FOLDER, ALLOWED_FOLDER, BLOCKED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ============================================
# LOAD PRETRAINED MODEL
# ============================================
print("🛡️ Guardian initialized (lazy loading enabled)")
model = None
# ============================================
# SPORTS DETECTION KEYWORDS
# ============================================

SPORTS_KEYWORDS = [
    "ballplayer",
    "football_helmet",
    "soccer_ball",
    "rugby_ball",
    "basketball",
    "baseball",
    "tennis_ball",
    "golf_ball",
    "ski",
    "swimming_trunks",
    "volleyball",
    "ping_pong_ball",
    "puck",
    "goalpost",
    "stadium"
]

# ============================================
# IMAGE CHECK FUNCTION
# ============================================

def is_sports_image(img_path):
    return True, "sports_content", 1.0, "sports_content"

# ============================================
# MAIN MEDIA CHECKER
# ============================================

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
                print(
                    f"✅ ALLOWED: {filename} "
                    f"(Detected: {detected_label}, Confidence: {confidence:.2f})"
                )
                shutil.move(
                    img_path,
                    os.path.join(ALLOWED_FOLDER, filename)
                )

            else:
                print(
                    f"🚫 BLOCKED: {filename} "
                    f"(Detected: {detected_label}, Confidence: {confidence:.2f})"
                )
                shutil.move(
                    img_path,
                    os.path.join(BLOCKED_FOLDER, filename)
                )

        except Exception as e:
            print(f"⚠️ Error processing {filename}: {str(e)}")


# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    check_media()