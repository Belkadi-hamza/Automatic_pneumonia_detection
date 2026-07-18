from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from typing import List
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import base64
import cv2
from pathlib import Path
import zipfile
from datetime import datetime
import time

# ─── Chemins ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
MODEL_NAME = "DenseNet121_Pneumonia_20260619_2156.keras"
MODEL_PATH = BASE_DIR / "API" / "models" / MODEL_NAME

# ─── FastAPI ──────────────────────────────────────────────
app = FastAPI(title="PneumoScan API")

# ─── Informations du modèle (pour l'affichage) ──────────
MODEL_INFO = {
    "name": "DenseNet121",
    "version": "1.0.0",
    "algorithm": "Deep Learning CNN (DenseNet121)",
    "dataset": "Chest X-Ray Pneumonia (Kaggle)",
    "accuracy": "92.5%"
}

# ─── Chargement du modèle ──────────────────────────────────
model = None
try:
    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(str(MODEL_PATH))
        print(f"✅ Model loaded from {MODEL_PATH}")
        print(f"📊 Architecture: {model.__class__.__name__}, {len(model.layers)} layers")
    else:
        print(f"❌ Model not found at {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# ─── Helpers ──────────────────────────────────────────────
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def occlusion_heatmap(img_array, model, step=24, window_size=48):
    try:
        base_pred = float(model.predict(img_array, verbose=0)[0][0])
        img = img_array[0]
        h, w, _ = img.shape
        
        n_rows = max(1, (h - window_size) // step + 1)
        n_cols = max(1, (w - window_size) // step + 1)
        heatmap = np.zeros((n_rows, n_cols))
        
        for i in range(n_rows):
            for j in range(n_cols):
                occluded = img.copy()
                y_start = i * step
                x_start = j * step
                y_end = min(y_start + window_size, h)
                x_end = min(x_start + window_size, w)
                occluded[y_start:y_end, x_start:x_end, :] = 0
                
                occ_tensor = np.expand_dims(occluded, axis=0)
                occ_pred = float(model.predict(occ_tensor, verbose=0)[0][0])
                
                importance = base_pred - occ_pred
                heatmap[i, j] = max(0, importance)
        
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        heatmap_resized = cv2.resize(heatmap, (w, h))
        return heatmap_resized
    except Exception as e:
        print(f"Heatmap error: {e}")
        return None

def overlay_heatmap(original_image, heatmap, alpha=0.5):
    try:
        original_np = np.array(original_image)
        h, w = original_np.shape[:2]
        
        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(heatmap, (w, h))
        
        heatmap_normalized = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
        
        if original_bgr.shape != heatmap_colored.shape:
            heatmap_colored = cv2.resize(heatmap_colored, (original_bgr.shape[1], original_bgr.shape[0]))
        
        superimposed = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
        
        return superimposed_rgb
    except Exception as e:
        print(f"Overlay error: {e}")
        return np.array(original_image)

def image_to_base64(image, format="JPEG"):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    buffered = io.BytesIO()
    if format == "JPEG":
        image.save(buffered, format="JPEG", quality=90, optimize=True)
    else:
        image.save(buffered, format="PNG", optimize=True)
    return base64.b64encode(buffered.getvalue()).decode()

# ─── Routes ──────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home():
    html_path = TEMPLATES_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse(content="<h1>Erreur : fichier index.html non trouvé</h1>", status_code=404)
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH) if MODEL_PATH.exists() else "not found",
        "model_info": MODEL_INFO
    }

@app.post("/api/predict")
async def predict_images(
    images: List[UploadFile] = File(...),
    threshold: float = Form(0.45),
    heatmap_threshold: float = Form(0.50),
    occlusion_step: int = Form(24),
    occlusion_window: int = Form(48)
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for img_file in images:
        start_time = time.time()
        try:
            contents = await img_file.read()
            original = Image.open(io.BytesIO(contents)).convert("RGB")
            img_array = preprocess_image(original)
            prob = float(model.predict(img_array, verbose=0)[0][0])
            label = "PNEUMONIE" if prob >= threshold else "NORMAL"
            confidence = prob if label == "PNEUMONIE" else 1 - prob
            
            # Calcul du temps de traitement
            processing_time = time.time() - start_time
            processing_ms = int(processing_time * 1000)
            
            result = {
                "filename": img_file.filename,
                "label": label,
                "confidence": round(confidence * 100, 1),
                "probability": round(prob * 100, 1),
                "original_b64": image_to_base64(original),
                "heatmap_b64": None,
                "processing_time": round(processing_time, 2),
                "processing_ms": processing_ms,
                "risk_level": "High Risk" if label == "PNEUMONIE" else "No Risk",
                "recommendations": get_recommendations(label)
            }
            
            if label == "PNEUMONIE":
                heatmap = occlusion_heatmap(img_array, model, step=occlusion_step, window_size=occlusion_window)
                if heatmap is not None:
                    heatmap_overlay = overlay_heatmap(original, heatmap, alpha=0.6)
                    result["heatmap_b64"] = image_to_base64(Image.fromarray(heatmap_overlay), format="PNG")
            
            results.append(result)
        except Exception as e:
            results.append({
                "filename": img_file.filename,
                "error": str(e),
                "processing_time": 0,
                "processing_ms": 0
            })
    
    return JSONResponse(content=results)

def get_recommendations(label):
    if label == "PNEUMONIE":
        return [
            "Pneumonia detected - immediate medical consultation strongly recommended",
            "Contact your healthcare provider or visit emergency services",
            "Monitor symptoms closely including fever, cough, and breathing difficulties",
            "Ensure adequate rest and hydration while awaiting medical evaluation",
            "Avoid strenuous physical activities until cleared by healthcare provider"
        ]
    else:
        return [
            "No signs of pneumonia detected in the current X-ray analysis",
            "Continue regular health monitoring and preventive care",
            "Maintain good respiratory hygiene and healthy lifestyle",
            "Schedule routine check-ups as recommended by your healthcare provider",
            "Seek medical attention if respiratory symptoms develop"
        ]

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 PneumoScan Web Interface (DenseNet121)")
    print("="*50)
    print(f"📍 Interface: http://127.0.0.1:8000")
    print(f"📚 API Docs: http://127.0.0.1:8000/docs")
    print("="*50 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)