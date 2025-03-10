from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS for Render deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Local development
        "http://localhost:3000",  # Alternative local port
        "https://occusafe-pharmacy.netlify.app",  # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model - Using absolute path for Render deployment
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glaucoma_detection_model.keras")

# Initialize model variable
model = None

@app.on_event("startup")
async def load_model():
    """Load the model on startup"""
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model = None

def plot_to_base64(figure):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    figure.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def process_image(image_file):
    """Load and preprocess image for model prediction"""
    try:
        image_data = image_file.file.read()
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize for model
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        preprocessed = preprocess_input(img_array)
        
        return img, img_resized, preprocessed, img_array[0]
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Error processing image")

def generate_lime_explanation(img_array, preprocessed_image):
    """Generate LIME explanation for the prediction"""
    try:
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_array,
            lambda x: model.predict(preprocess_input(x)),
            top_labels=1,
            hide_color=0,
            num_samples=1000
        )
        return explanation
    except Exception as e:
        logger.error(f"Error generating LIME explanation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating explanation")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Glaucoma Detection API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict glaucoma from uploaded image"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        logger.info("Processing uploaded image")
        
        # Process image
        original_image, resized_image, preprocessed_image, img_array = process_image(file)
        
        # Get prediction
        prediction = model.predict(preprocessed_image)
        logger.info(f"Prediction generated: {prediction}")
        
        # Generate LIME explanation
        explanation = generate_lime_explanation(img_array, preprocessed_image)
        
        # Generate visualizations
        visualizations = {}
        
        # 1. Original Image
        plt.figure(figsize=(8, 8))
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')
        visualizations["original_image"] = f"data:image/png;base64,{plot_to_base64(plt.gcf())}"
        plt.close()
        
        # 2. Superpixel Segmentation
        segments = slic(img_array, n_segments=100, compactness=10)
        plt.figure(figsize=(8, 8))
        plt.imshow(mark_boundaries(img_array/255.0, segments))
        plt.title("Superpixel Segmentation")
        plt.axis('off')
        visualizations["superpixels_image"] = f"data:image/png;base64,{plot_to_base64(plt.gcf())}"
        plt.close()
        
        # 3. LIME Explanation (All contributions)
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            hide_rest=False,
            num_features=10
        )
        plt.figure(figsize=(8, 8))
        plt.imshow(mark_boundaries(temp/255.0, mask))
        plt.title("LIME Explanation (All Contributions)")
        plt.axis('off')
        visualizations["lime_explanation"] = f"data:image/png;base64,{plot_to_base64(plt.gcf())}"
        plt.close()
        
        # 4. Positive Contributions
        temp_pos, mask_pos = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            hide_rest=False,
            num_features=5
        )
        plt.figure(figsize=(8, 8))
        plt.imshow(mark_boundaries(temp_pos/255.0, mask_pos))
        plt.title("Positive Contributions")
        plt.axis('off')
        visualizations["lime_positive"] = f"data:image/png;base64,{plot_to_base64(plt.gcf())}"
        plt.close()
        
        # 5. Feature Importance Distribution
        importance = explanation.local_exp[explanation.top_labels[0]]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance)), [x[1] for x in importance])
        plt.title("Feature Importance of Superpixels")
        plt.xlabel("Superpixel Index")
        plt.ylabel("Importance")
        visualizations["feature_importance"] = f"data:image/png;base64,{plot_to_base64(plt.gcf())}"
        plt.close()
        
        # 6. Top Contributing Regions
        top_features = sorted(importance, key=lambda x: abs(x[1]), reverse=True)[:5]
        mask_top = np.zeros(segments.shape, dtype=bool)
        for feature, _ in top_features:
            mask_top = mask_top | (segments == feature)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img_array/255.0)
        plt.imshow(mask_top, alpha=0.3, cmap='Reds')
        plt.title("Top Contributing Regions")
        plt.axis('off')
        visualizations["top_contributing"] = f"data:image/png;base64,{plot_to_base64(plt.gcf())}"
        plt.close()
        
        # 7. Perturbed Images
        perturbed_images = []
        for feature, _ in top_features[:4]:
            temp_img = img_array.copy()
            mask = segments == feature
            temp_img[mask] = 0
            plt.figure(figsize=(8, 8))
            plt.imshow(temp_img/255.0)
            plt.title(f"Perturbed Image (Superpixel {feature})")
            plt.axis('off')
            perturbed_images.append(f"data:image/png;base64,{plot_to_base64(plt.gcf())}")
            plt.close()
        
        # 8. LIME Mask Overlay
        plt.figure(figsize=(8, 8))
        plt.imshow(img_array/255.0)
        plt.imshow(mask, alpha=0.3, cmap='coolwarm')
        plt.title("LIME Mask Overlay")
        plt.axis('off')
        visualizations["mask_overlay"] = f"data:image/png;base64,{plot_to_base64(plt.gcf())}"
        plt.close()
        
        visualizations["perturbed_images"] = perturbed_images
        
        logger.info("Successfully generated all visualizations")
        
        return JSONResponse({
            "predictions": prediction.tolist(),
            "images": visualizations
        })
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
