import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import numpy as np
import io
import uvicorn
import cv2
from pathlib import Path
import base64
import warnings
warnings.filterwarnings('ignore')

# ─── Constantes ──────────────────────────────────────────────
IMG_SIZE = 224
OPTIMAL_THRESH = 0.45

# ─── Chemin du modèle ──────────────────────────────────────
current_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
MODEL_FILENAME = "DenseNet121_Pneumonia_20260619_2156.keras"
MODEL_PATH = current_dir / 'models' / MODEL_FILENAME

# ─── Chargement ──────────────────────────────────────────────
model = None
try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print(f"✅ Model '{MODEL_PATH}' loaded successfully.")
    print(f"📊 Model architecture: {model.__class__.__name__}")
    print(f"📈 Number of layers: {len(model.layers)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# ─── GradCAM Class ───────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for visual explanations.
    Generates heatmaps showing which image regions most influence the model's predictions.
    """
    def __init__(self, model, layer_name=None):
        """
        Initialize GradCAM with the model and target layer.
        
        Args:
            model: Keras model
            layer_name: Name of the convolutional layer to target. 
                       If None, automatically finds the last conv layer.
        """
        self.model = model
        
        # Auto-detect the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
        
        self.layer_name = layer_name
        print(f"🎯 GradCAM targeting layer: {self.layer_name}")
        
        try:
            self.grad_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=[model.get_layer(layer_name).output, model.output]
            )
        except Exception as e:
            print(f"❌ Error creating GradCAM model: {e}")
            self.grad_model = None
    
    def compute_heatmap(self, img_array):
        """
        Compute the GradCAM heatmap for an input image.
        
        Args:
            img_array: Input image array (batch, height, width, channels)
            
        Returns:
            Heatmap array with values between 0 and 1
        """
        if self.grad_model is None:
            return None
        
        try:
            with tf.GradientTape() as tape:
                conv_outputs, predictions = self.grad_model(img_array)
                loss = predictions[:, 0]
            
            # Compute gradients
            grads = tape.gradient(loss, conv_outputs)
            
            # Global average pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the activation map by the gradients
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            # Normalize heatmap
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
            
            return heatmap.numpy()
        except Exception as e:
            print(f"❌ Heatmap computation error: {e}")
            return None
    
    def superposer(self, original_image, heatmap, alpha=0.5):
        """
        Overlay the heatmap on the original image.
        
        Args:
            original_image: PIL Image
            heatmap: Heatmap array
            alpha: Blending factor (0-1)
            
        Returns:
            Overlayed image as numpy array
        """
        try:
            original_np = np.array(original_image)
            h, w = original_np.shape[:2]
            
            # Resize heatmap to match image dimensions
            heatmap_resized = cv2.resize(heatmap, (w, h))
            
            # Convert heatmap to color
            heatmap_normalized = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
            
            # Convert original image to BGR for OpenCV operations
            original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
            
            # Ensure shapes match
            if original_bgr.shape != heatmap_colored.shape:
                heatmap_colored = cv2.resize(heatmap_colored, (original_bgr.shape[1], original_bgr.shape[0]))
            
            # Blend images
            superimposed = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_colored, alpha, 0)
            superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
            
            return superimposed_rgb
        except Exception as e:
            print(f"❌ Superposition error: {e}")
            return np.array(original_image)

# ─── Initialize GradCAM ──────────────────────────────────────
grad_cam = None
if model is not None:
    grad_cam = GradCAM(model)
    print(f"✅ GradCAM initialized successfully.")

# ─── FastAPI ──────────────────────────────────────────────────
app = FastAPI(title="Pneumonia Detection API", version="3.0.0")

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probability: float
    heatmap: Optional[str] = None

@app.get("/")
async def root():
    return {
        "message": "Pneumonia Detection API with GradCAM Heatmap",
        "status": "running",
        "model_loaded": model is not None,
        "gradcam_initialized": grad_cam is not None,
        "threshold": OPTIMAL_THRESH,
        "model_path": str(MODEL_PATH)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH) if MODEL_PATH.exists() else "not found"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_pneumonia(file: UploadFile = File(...), return_heatmap: bool = True):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if grad_cam is None:
        raise HTTPException(status_code=503, detail="GradCAM not initialized.")
    
    try:
        contents = await file.read()
        original_image = Image.open(io.BytesIO(contents)).convert("RGB")
        processed_image = original_image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(processed_image).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction_prob = float(model.predict(img_array, verbose=0)[0][0])
        prediction_label = 1 if prediction_prob >= OPTIMAL_THRESH else 0
        predicted_class = "PNEUMONIA" if prediction_label == 1 else "NORMAL"
        confidence = prediction_prob if prediction_label == 1 else 1 - prediction_prob

        response_data = {
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "probability": prediction_prob,
            "heatmap": None
        }

        # Generate GradCAM heatmap if pneumonia detected and heatmap requested
        if return_heatmap and predicted_class == "PNEUMONIA" and grad_cam is not None:
            heatmap = grad_cam.compute_heatmap(img_array)
            if heatmap is not None:
                overlayed_image = grad_cam.superposer(original_image, heatmap, alpha=0.5)
                if overlayed_image is not None:
                    overlayed_pil = Image.fromarray(overlayed_image.astype(np.uint8))
                    buffered = io.BytesIO()
                    overlayed_pil.save(buffered, format="PNG", optimize=True)
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    response_data["heatmap"] = f"data:image/png;base64,{img_base64}"
        
        print(f"✅ Prediction: {response_data['prediction']} (Confidence: {response_data['confidence']})")
        return response_data
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 Starting Pneumonia Detection API (new model)")
    print("="*50)
    print(f"📁 Model path: {MODEL_PATH}")
    print(f"✅ Model loaded: {model is not None}")
    print(f"🎯 Threshold: {OPTIMAL_THRESH}")
    print("\n📍 API available at: http://localhost:8000")
    print("📚 Documentation: http://localhost:8000/docs")
    print("="*50 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)



# uvicorn api:app --host 127.0.0.1 --port 8000