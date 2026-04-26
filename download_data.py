# Use this script ONLY for downloading or organizing data
# The training is already DONE in Google Colab.
print("This file is for data management. Use predict.py to run the AI.")

import tensorflow as tf
# from tensorflow.keras import layers, models
import os

# 1. Setup Data Folders
DATA_DIR = 'classifier_data'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 2. Load the Dataset from your folders
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary' # 0 for first folder, 1 for second
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# 3. Build the Model (Transfer Learning)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), 
    include_top=False, 
    weights='imagenet'
)
base_model.trainable = False # Freeze the pre-trained weights

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid') # Output 0 to 1
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train the Model
print("🚀 Starting training...")
model.fit(train_ds, validation_data=val_ds, epochs=5)

# 5. Save the result
model.save("sports_filter.h5")
print("✅ Done! Model saved as sports_filter.h5")