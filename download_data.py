import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import os

# 1. Load the "Non-Sports" data from CIFAR-10
ds = tfds.load('cifar10', split='train', shuffle_files=True)
output_path = "classifier_data/non_sports"
os.makedirs(output_path, exist_ok=True)

# 2. Save 500 images as JPEGs
print("📥 Downloading non-sports images...")
for i, example in enumerate(ds.take(500)):
    image = example["image"].numpy()
    img = Image.fromarray(image)
    img.save(f"{output_path}/random_{i}.jpg")

print(f"✅ Done! 500 images saved to {output_path}")