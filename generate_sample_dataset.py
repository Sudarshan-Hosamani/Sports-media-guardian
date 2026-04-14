"""
generate_sample_dataset.py
Creates synthetic sports-like images in the dataset folders so the prototype
works out-of-the-box without requiring real photos.

Run once: python generate_sample_dataset.py
"""

from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
import os
from pathlib import Path

BASE = Path(__file__).parent / "dataset"

def make_sports_image(seed: int, width=400, height=300) -> Image.Image:
    """Generate a colorful fake 'sports action' photo using geometric shapes."""
    rng = np.random.default_rng(seed)
    
    # Background gradient (simulates stadium / field)
    bg_colors = [
        ((34, 139, 34), (144, 238, 144)),    # green field
        ((0, 0, 139), (135, 206, 250)),       # blue court / sky
        ((139, 0, 0), (255, 160, 122)),       # red arena
        ((75, 0, 130), (238, 130, 238)),      # purple court
        ((184, 134, 11), (255, 215, 0)),      # golden arena
    ]
    color_set = bg_colors[seed % len(bg_colors)]
    
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    
    # Simple vertical gradient background
    for y in range(height):
        t = y / height
        r = int(color_set[0][0] * (1 - t) + color_set[1][0] * t)
        g = int(color_set[0][1] * (1 - t) + color_set[1][1] * t)
        b = int(color_set[0][2] * (1 - t) + color_set[1][2] * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    # Draw court/field lines
    line_color = (255, 255, 255, 180)
    draw.rectangle([20, 20, width-20, height-20], outline=(255, 255, 255), width=3)
    draw.line([(width//2, 20), (width//2, height-20)], fill=(255, 255, 255), width=2)
    draw.ellipse([width//2-40, height//2-40, width//2+40, height//2+40], outline=(255, 255, 255), width=2)
    
    # Draw "players" (colored circles/rectangles)
    num_players = rng.integers(4, 8)
    for i in range(num_players):
        x = int(rng.integers(40, width-40))
        y = int(rng.integers(40, height-40))
        size = int(rng.integers(15, 30))
        player_color = tuple(int(c) for c in rng.integers(50, 220, 3))
        draw.ellipse([x-size, y-size, x+size, y+size], fill=player_color, outline=(0, 0, 0), width=2)
        # Jersey number
        draw.text((x-5, y-8), str(rng.integers(1, 99)), fill=(255, 255, 255))
    
    # Draw "ball"
    bx, by = int(rng.integers(50, width-50)), int(rng.integers(50, height-50))
    draw.ellipse([bx-10, by-10, bx+10, by+10], fill=(255, 165, 0), outline=(0,0,0), width=2)
    
    # Watermark / branding text
    draw.text((10, height-25), f"SPORTS MEDIA © 2024 | IMG-{seed:04d}", fill=(255, 255, 255))
    
    return img


def save(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=90)
    print(f"  ✓ {path}")


print("🎨 Generating sample dataset images...\n")

# ── ORIGINALS (5 unique sports scenes) ──────────────────────────────
print("📁 original/")
originals = {}
for i in range(5):
    img = make_sports_image(seed=i * 100)
    originals[i] = img
    save(img, BASE / "original" / f"sports_{i+1:02d}_original.jpg")

# ── MODIFIED (tampered versions of originals) ────────────────────────
print("\n📁 modified/")
for i, orig in originals.items():
    # Crop (simulates partial screenshot)
    w, h = orig.size
    crop = orig.crop((w//6, h//6, w - w//6, h - h//6))
    save(crop, BASE / "modified" / f"sports_{i+1:02d}_cropped.jpg")
    
    # Resize + slight blur (simulates re-upload)
    small = orig.resize((200, 150)).resize((400, 300))
    blurred = small.filter(ImageFilter.GaussianBlur(radius=1.5))
    save(blurred, BASE / "modified" / f"sports_{i+1:02d}_blurred.jpg")
    
    # Color-shifted (simulates filter applied)
    arr = np.array(orig, dtype=np.float32)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.3, 0, 255)   # boost red
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.7, 0, 255)   # reduce blue
    color_shifted = Image.fromarray(arr.astype(np.uint8))
    save(color_shifted, BASE / "modified" / f"sports_{i+1:02d}_filtered.jpg")

# ── UNRELATED (completely different images — should score Safe) ───────
print("\n📁 unrelated/")
unrelated_themes = [
    ((200, 200, 200), "DOCUMENT"),     # gray document
    ((240, 230, 140), "LANDSCAPE"),    # khaki landscape
    ((176, 224, 230), "INTERIOR"),     # powder blue room
    ((255, 228, 196), "PORTRAIT"),     # bisque portrait
    ((152, 251, 152), "NATURE"),       # pale green nature
]

for idx, (bg, label) in enumerate(unrelated_themes):
    img = Image.new("RGB", (400, 300), bg)
    draw = ImageDraw.Draw(img)
    # Add some random shapes to differentiate
    rng = np.random.default_rng(999 + idx)
    for _ in range(8):
        x1, y1 = int(rng.integers(0, 350)), int(rng.integers(0, 250))
        x2, y2 = x1 + int(rng.integers(20, 80)), y1 + int(rng.integers(20, 80))
        color = tuple(int(c) for c in rng.integers(50, 200, 3))
        draw.rectangle([x1, y1, x2, y2], fill=color)
    draw.text((10, 10), f"UNRELATED: {label}", fill=(0, 0, 0))
    save(img, BASE / "unrelated" / f"unrelated_{idx+1:02d}_{label.lower()}.jpg")

print(f"\n✅ Dataset generation complete!")
print(f"   original/  : 5 images")
print(f"   modified/  : 15 images (3 variants per original)")
print(f"   unrelated/ : 5 images")
