import os
# This line tells TensorFlow to be quiet and not show 100 warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

print("🏁 Script started...")

MODEL_PATH = os.path.join('models', 'sports_filter.keras')

if not os.path.exists(MODEL_PATH):
    print(f"❌ File NOT found at: {os.path.abspath(MODEL_PATH)}")
else:
    print(f"📂 Found model at: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Success! Model loaded.")
        print(f"🧠 Model Input Shape: {model.input_shape}")
    except Exception as e:
        print(f"💥 Error: {e}")

print("🏁 Script finished.")