import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the optimized model
model = load_model("glaucoma_detection_model.keras")

# Print model summary
model.summary()
