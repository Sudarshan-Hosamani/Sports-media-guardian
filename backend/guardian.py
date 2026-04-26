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

print("🛡️ Loading Universal Guardian...")
model = ResNet50(weights="imagenet")

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
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x, verbose=0)
    decoded = decode_predictions(preds, top=5)[0]

    top_guess = decoded[0][1]
    top_confidence = decoded[0][2]

    for _, label, confidence in decoded:
        label_lower = label.lower()

        if any(keyword in label_lower for keyword in SPORTS_KEYWORDS):
            if confidence > 0.20:
                return True, label, confidence, top_guess

    return False, top_guess, top_confidence, top_guess


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