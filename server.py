"""
Hot Dog or Not Hot Dog - Backend API Server
A tribute to Jian Yang's SeeFood app from HBO's Silicon Valley

FastAPI server that serves predictions from the trained model.
"""

import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import tensorflow as tf

# Configuration
IMG_SIZE = 224
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hotdog_model.keras")

# Initialize FastAPI app
app = FastAPI(
    title="Hot Dog or Not Hot Dog API",
    description="A tribute to Jian Yang's SeeFood app from HBO's Silicon Valley",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None


def load_model():
    """Load the trained model."""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Please run train_model.py first."
            )
        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model prediction."""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to expected size
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # Rescale to [-1, 1] (same as training)
    img_array = img_array / 127.5 - 1.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Model will be loaded when first prediction is requested.")


@app.get("/")
async def root():
    """Serve the main HTML page."""
    html_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "Hot Dog or Not Hot Dog API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict whether an image contains a hot dog.
    
    Returns:
        prediction: "Hot Dog" or "Not Hot Dog"
        confidence: float between 0 and 1
        raw_score: the raw model output
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    try:
        # Load the model if not already loaded
        load_model()
        
        # Read and preprocess the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)[0][0]
        
        # Interpret results (0 = hotdog, 1 = nothotdog based on folder order)
        # Lower values mean "hotdog", higher values mean "not hotdog"
        is_hotdog = bool(prediction < 0.5)
        confidence = float(1 - prediction if is_hotdog else prediction)
        
        result = {
            "prediction": "Hot Dog" if is_hotdog else "Not Hot Dog",
            "confidence": confidence,
            "raw_score": float(prediction),
            "is_hotdog": is_hotdog
        }
        
        print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})")
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


# Serve static files (CSS, JS)
@app.get("/styles.css")
async def get_styles():
    """Serve CSS file."""
    css_path = os.path.join(BASE_DIR, "styles.css")
    if os.path.exists(css_path):
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS file not found")


@app.get("/app.js")
async def get_js():
    """Serve JavaScript file."""
    js_path = os.path.join(BASE_DIR, "app.js")
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="JavaScript file not found")


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🌭 HOT DOG OR NOT HOT DOG - API Server")
    print("   A tribute to Jian Yang's SeeFood from Silicon Valley")
    print("=" * 60)
    print("\nStarting server at http://localhost:8000")
    print("API docs available at http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
