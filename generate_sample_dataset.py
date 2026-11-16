"""
generate_sample_dataset.py
Expanded sports media generator producing a diverse, professional 70-image dataset:
- 12 unique sports (Soccer, Basketball, Tennis, Cricket, Swimming, Track, Hockey, Baseball, Volleyball, Rugby, Badminton, Table Tennis)
- 48 realistic tampering variants (Crop/Zoom, Color Filter, Watermarked/Blur, Monotone/Contrast)
- 10 non-sports control images (Documents, Office, City, Nature, Portraits, etc.)
"""

from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import os
from pathlib import Path

BASE = Path(__file__).parent / "dataset"

SPORTS_CONFIGS = [
    {
        "name": "soccer",
        "bg_top": (28, 120, 28),
        "bg_bot": (45, 175, 45),
        "line_color": (255, 255, 255),
        "type": "pitch",
        "ball_color": (245, 245, 245),
        "ball_outline": (20, 20, 20),
        "label": "SOCCER CHAMPIONSHIP",
    },
    {
        "name": "basketball",
        "bg_top": (195, 115, 45),
        "bg_bot": (220, 140, 60),
        "line_color": (255, 255, 255),
        "type": "court",
        "ball_color": (230, 95, 20),
        "ball_outline": (50, 20, 0),
        "label": "PRO BASKETBALL LEAGUE",
    },
    {
        "name": "tennis",
        "bg_top": (25, 95, 175),
        "bg_bot": (35, 125, 210),
        "line_color": (255, 255, 255),
        "type": "tennis_court",
        "ball_color": (205, 245, 40),
        "ball_outline": (100, 140, 0),
        "label": "GRAND SLAM TENNIS",
    },
    {
        "name": "cricket",
        "bg_top": (38, 145, 38),
        "bg_bot": (50, 180, 50),
        "line_color": (240, 240, 240),
        "type": "cricket_pitch",
        "ball_color": (190, 20, 20),
        "ball_outline": (60, 0, 0),
        "label": "WORLD CRICKET TROPHY",
    },
    {
        "name": "swimming",
        "bg_top": (10, 85, 150),
        "bg_bot": (25, 160, 220),
        "line_color": (255, 220, 0),
        "type": "pool_lanes",
        "ball_color": (255, 255, 255),
        "ball_outline": (0, 70, 140),
        "label": "AQUATICS SPEEDWAY",
    },
    {
        "name": "track",
        "bg_top": (160, 45, 35),
        "bg_bot": (200, 70, 55),
        "line_color": (255, 255, 255),
        "type": "track_curves",
        "ball_color": (255, 215, 0),
        "ball_outline": (100, 80, 0),
        "label": "OLYMPIC SPRINT TRACK",
    },
    {
        "name": "hockey",
        "bg_top": (215, 235, 245),
        "bg_bot": (235, 248, 255),
        "line_color": (210, 30, 30),
        "type": "ice_rink",
        "ball_color": (20, 20, 20),
        "ball_outline": (100, 100, 100),
        "label": "NATIONAL ICE HOCKEY",
    },
    {
        "name": "baseball",
        "bg_top": (40, 135, 40),
        "bg_bot": (180, 125, 65),
        "line_color": (255, 255, 255),
        "type": "baseball_diamond",
        "ball_color": (250, 250, 250),
        "ball_outline": (190, 20, 20),
        "label": "MAJOR LEAGUE BASEBALL",
    },
    {
        "name": "volleyball",
        "bg_top": (225, 185, 115),
        "bg_bot": (245, 210, 145),
        "line_color": (255, 255, 255),
        "type": "court",
        "ball_color": (255, 220, 40),
        "ball_outline": (30, 70, 180),
        "label": "PRO BEACH VOLLEYBALL",
    },
    {
        "name": "rugby",
        "bg_top": (30, 110, 40),
        "bg_bot": (40, 150, 55),
        "line_color": (255, 255, 255),
        "type": "rugby_field",
        "ball_color": (240, 230, 200),
        "ball_outline": (80, 50, 20),
        "label": "INTERNATIONAL RUGBY",
    },
    {
        "name": "badminton",
        "bg_top": (20, 120, 80),
        "bg_bot": (30, 160, 110),
        "line_color": (255, 255, 255),
        "type": "court",
        "ball_color": (255, 255, 255),
        "ball_outline": (200, 200, 200),
        "label": "OPEN BADMINTON MASTERS",
    },
    {
        "name": "table_tennis",
        "bg_top": (20, 60, 130),
        "bg_bot": (30, 85, 175),
        "line_color": (255, 255, 255),
        "type": "tt_table",
        "ball_color": (255, 145, 20),
        "ball_outline": (100, 50, 0),
        "label": "WORLD TABLE TENNIS",
    },
]

def make_sports_image(cfg: dict, seed: int, width=440, height=320) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    # Background gradient
    t_top, t_bot = cfg["bg_top"], cfg["bg_bot"]
    for y in range(height):
        t = y / height
        r = int(t_top[0] * (1 - t) + t_bot[0] * t)
        g = int(t_top[1] * (1 - t) + t_bot[1] * t)
        b = int(t_top[2] * (1 - t) + t_bot[2] * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    lc = cfg["line_color"]

    # Field specific lines
    if cfg["type"] == "pitch":
        draw.rectangle([25, 25, width-25, height-25], outline=lc, width=3)
        draw.line([(width//2, 25), (width//2, height-25)], fill=lc, width=2)
        draw.ellipse([width//2-45, height//2-45, width//2+45, height//2+45], outline=lc, width=2)
        draw.rectangle([25, height//2-40, 75, height//2+40], outline=lc, width=2)
        draw.rectangle([width-75, height//2-40, width-25, height//2+40], outline=lc, width=2)
    elif cfg["type"] == "court":
        draw.rectangle([25, 25, width-25, height-25], outline=lc, width=3)
        draw.line([(width//2, 25), (width//2, height-25)], fill=lc, width=2)
        draw.arc([width//2-70, height//2-70, width//2+70, height//2+70], 0, 360, fill=lc, width=2)
        draw.arc([10, 50, 140, height-50], 270, 90, fill=lc, width=2)
        draw.arc([width-140, 50, width-10, height-50], 90, 270, fill=lc, width=2)
    elif cfg["type"] == "tennis_court":
        draw.rectangle([30, 30, width-30, height-30], outline=lc, width=3)
        draw.rectangle([50, 50, width-50, height-50], outline=lc, width=2)
        draw.line([(width//2, 30), (width//2, height-30)], fill=(255, 255, 255), width=4)  # Net
        draw.line([(50, height//2), (width-50, height//2)], fill=lc, width=2)
    elif cfg["type"] == "cricket_pitch":
        # Pitch strip in center
        draw.rectangle([width//2-35, 30, width//2+35, height-30], fill=(210, 180, 120), outline=lc, width=2)
        draw.line([(width//2-35, 60), (width//2+35, 60)], fill=lc, width=2)
        draw.line([(width//2-35, height-60), (width//2+35, height-60)], fill=lc, width=2)
        draw.ellipse([width//2-80, height//2-80, width//2+80, height//2+80], outline=(255, 255, 255, 120), width=1)
    elif cfg["type"] == "pool_lanes":
        for ly in range(40, height, 40):
            draw.line([(0, ly), (width, ly)], fill=lc, width=2)
    elif cfg["type"] == "track_curves":
        for rad in range(30, 180, 25):
            draw.arc([-60, -60, rad*2, rad*2], 0, 90, fill=lc, width=2)
            draw.arc([width-rad*2+60, height-rad*2+60, width+60, height+60], 180, 270, fill=lc, width=2)
    elif cfg["type"] == "ice_rink":
        draw.rectangle([25, 25, width-25, height-25], outline=(50, 100, 200), width=3)
        draw.line([(width//2, 25), (width//2, height-25)], fill=(200, 30, 30), width=3)
        draw.line([(width//4, 25), (width//4, height-25)], fill=(30, 80, 190), width=2)
        draw.line([(3*width//4, 25), (3*width//4, height-25)], fill=(30, 80, 190), width=2)
        draw.ellipse([width//2-35, height//2-35, width//2+35, height//2+35], outline=(30, 80, 190), width=2)
    elif cfg["type"] == "baseball_diamond":
        cx, cy = width//2, height//2 + 20
        d = 70
        draw.polygon([(cx, cy-d), (cx+d, cy), (cx, cy+d), (cx-d, cy)], outline=lc, width=3)
        draw.line([(cx, cy+d), (cx+d*2, cy-d//2)], fill=lc, width=2)
        draw.line([(cx, cy+d), (cx-d*2, cy-d//2)], fill=lc, width=2)
    elif cfg["type"] == "rugby_field":
        draw.rectangle([25, 25, width-25, height-25], outline=lc, width=3)
        for rx in range(60, width-60, 45):
            draw.line([(rx, 25), (rx, height-25)], fill=(255, 255, 255), width=1)
        draw.line([(width//2, 25), (width//2, height-25)], fill=lc, width=3)
    elif cfg["type"] == "tt_table":
        draw.rectangle([40, 50, width-40, height-50], outline=lc, width=3)
        draw.line([(width//2, 50), (width//2, height-50)], fill=(240, 240, 240), width=4)
        draw.line([(40, height//2), (width-40, height//2)], fill=lc, width=2)

    # Draw athletes (dynamic circles with numbers & uniforms)
    num_players = int(rng.integers(5, 9))
    for _ in range(num_players):
        px = int(rng.integers(50, width-50))
        py = int(rng.integers(45, height-45))
        psize = int(rng.integers(14, 24))
        pcolor = tuple(int(c) for c in rng.integers(40, 240, 3))
        draw.ellipse([px-psize, py-psize, px+psize, py+psize], fill=pcolor, outline=(255, 255, 255), width=2)
        draw.text((px-5, py-7), str(rng.integers(1, 99)), fill=(255, 255, 255))

    # Draw sports ball / object
    bx = int(rng.integers(80, width-80))
    by = int(rng.integers(60, height-60))
    bsize = 11
    draw.ellipse([bx-bsize, by-bsize, bx+bsize, by+bsize], fill=cfg["ball_color"], outline=cfg["ball_outline"], width=2)

    # Broadcast overlay banner
    draw.rectangle([10, height-30, width-10, height-8], fill=(15, 20, 30))
    draw.text((20, height-26), f"LIVE | {cfg['label']} | OFFICIAL BROADCAST", fill=(255, 255, 255))

    return img

def save(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=92)
    print(f"  ✓ {path.name}")

print("🎨 Generating 70-image expanded sports dataset...\n")

# Clean existing dataset folders first for clean slate
for sub in ["original", "modified", "unrelated"]:
    p = BASE / sub
    if p.exists():
        for f in p.glob("*.*"):
            try: f.unlink()
            except: pass

originals = []

print("📁 Generating dataset/original/ (12 sports)...")
for idx, cfg in enumerate(SPORTS_CONFIGS):
    img = make_sports_image(cfg, seed=100 + idx * 7)
    fname = f"sports_{idx+1:02d}_{cfg['name']}.jpg"
    save(img, BASE / "original" / fname)
    originals.append((idx+1, cfg['name'], img))

print("\n📁 Generating dataset/modified/ (48 tampered variants)...")
for num, sname, orig in originals:
    w, h = orig.size

    # Variant 1: Crop & Zoom (Cropped pirate clip)
    crop = orig.crop((w//5, h//5, w - w//5, h - h//5)).resize((w, h))
    save(crop, BASE / "modified" / f"sports_{num:02d}_{sname}_cropped.jpg")

    # Variant 2: Color grading / Social media filter
    arr = np.array(orig, dtype=np.float32)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.35, 0, 255)  # boost warm tones
    arr[:, :, 1] = np.clip(arr[:, :, 1] * 0.90, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.70, 0, 255)  # reduce blue
    filtered = Image.fromarray(arr.astype(np.uint8))
    save(filtered, BASE / "modified" / f"sports_{num:02d}_{sname}_filtered.jpg")

    # Variant 3: Watermark + Slight Blur (Re-encoded / Pirated watermark)
    blurred = orig.resize((w//2, h//2)).resize((w, h)).filter(ImageFilter.GaussianBlur(radius=1.2))
    bdraw = ImageDraw.Draw(blurred)
    bdraw.rectangle([w//2-70, h//2-20, w//2+70, h//2+20], fill=(0, 0, 0, 180))
    bdraw.text((w//2-55, h//2-8), "@PIRATE_STREAM", fill=(255, 50, 50))
    save(blurred, BASE / "modified" / f"sports_{num:02d}_{sname}_watermarked.jpg")

    # Variant 4: Monochrome / High-contrast edit
    gray = orig.convert("L").convert("RGB")
    g_arr = np.array(gray, dtype=np.float32)
    g_arr = np.clip((g_arr - 128) * 1.3 + 128, 0, 255).astype(np.uint8)
    monochrome = Image.fromarray(g_arr)
    save(monochrome, BASE / "modified" / f"sports_{num:02d}_{sname}_monochrome.jpg")

print("\n📁 Generating dataset/unrelated/ (10 control non-sports images)...")
unrelated_scenes = [
    ("document", (245, 245, 248), "OFFICIAL INVOICE & REPORT - CONFIDENTIAL"),
    ("code_editor", (30, 30, 35), "const express = require('express'); app.listen(3000);"),
    ("landscape", (210, 195, 140), "DESERT MOUNTAIN HORIZON SUNSET"),
    ("office_interior", (170, 180, 195), "CONFERENCE ROOM - FLOOR 12 INTERIOR"),
    ("portrait", (250, 220, 190), "STUDIO LIGHTING PORTRAIT PROFILE"),
    ("nature_forest", (85, 120, 75), "PINE FOREST BOTANICAL SANCTUARY"),
    ("city_night", (20, 25, 45), "METROPOLITAN DOWNTOWN HIGHWAY TRAFFIC"),
    ("abstract_art", (140, 60, 130), "MINIMALIST GEOMETRIC VECTOR COMPOSITION"),
    ("food_plate", (230, 150, 80), "GOURMET RESTAURANT CULINARY DISH"),
    ("star_constellation", (8, 12, 25), "ASTRONOMY DEEP SPACE OBSERVATORY"),
]

for idx, (label, bg, desc) in enumerate(unrelated_scenes):
    img = Image.new("RGB", (440, 320), bg)
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(700 + idx)

    # Specific patterns for non-sports
    if label == "document":
        for ly in range(40, 280, 18):
            lw = int(rng.integers(120, 380))
            draw.line([(30, ly), (30 + lw, ly)], fill=(70, 70, 80), width=2)
    elif label == "code_editor":
        draw.rectangle([10, 10, 430, 310], fill=(24, 24, 28))
        draw.text((25, 25), "// System Log & Config", fill=(100, 180, 100))
        for ly in range(50, 270, 22):
            draw.text((25, ly), f"function processData_{ly}() {{ return null; }}", fill=(170, 170, 180))
    elif label == "star_constellation":
        for _ in range(60):
            sx, sy = int(rng.integers(10, 430)), int(rng.integers(10, 310))
            draw.point((sx, sy), fill=(255, 255, 255))
    else:
        for _ in range(6):
            x1, y1 = int(rng.integers(10, 350)), int(rng.integers(10, 250))
            x2, y2 = x1 + int(rng.integers(30, 90)), y1 + int(rng.integers(30, 90))
            color = tuple(int(c) for c in rng.integers(40, 220, 3))
            draw.rectangle([x1, y1, x2, y2], fill=color)

    draw.rectangle([15, 280, 425, 305], fill=(0, 0, 0))
    draw.text((25, 286), f"CONTROL: {desc}", fill=(255, 255, 255))
    save(img, BASE / "unrelated" / f"unrelated_{idx+1:02d}_{label}.jpg")

print(f"\n✅ Total Dataset generated: 70 images!")
print(f"   - 12 Originals (12 different sports)")
print(f"   - 48 Modified (4 realistic tampering techniques per sport)")
print(f"   - 10 Unrelated (Diverse non-sports controls)")
