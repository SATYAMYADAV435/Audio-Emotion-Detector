from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import joblib
import numpy as np
import os
import shutil
import librosa
from pydantic import BaseModel

# Import our backend modules
# (Ensure backend is in pythonpath or run from root)
from backend.model import RNNEmotionClassifier
from backend.pipeline import N_MELS, MAX_TIME_FRAMES

app = FastAPI(title="Emotion Detector API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global State
model = None
pipeline = None
label_encoder = None

@app.on_event("startup")
async def load_artifacts():
    global model, pipeline, label_encoder
    
    # Load Label Encoder
    if os.path.exists('backend/label_encoder.joblib'):
        label_encoder = joblib.load('backend/label_encoder.joblib')
        num_classes = len(label_encoder.classes_)
    else:
        # Fallback or error
        print("Warning: Label encoder not found. Using default 8 classes.")
        num_classes = 8
        label_encoder = None

    # Load Model
    model = RNNEmotionClassifier(input_size=N_MELS, num_classes=num_classes).to(DEVICE)
    if os.path.exists('backend/best_model.pt'):
        model.load_state_dict(torch.load('backend/best_model.pt', map_location=DEVICE))
        model.eval()
        print("Model loaded.")
    else:
        print("Warning: best_model.pt not found. Model is untrained.")

    # Load Pipeline
    if os.path.exists('backend/pipeline.joblib'):
        pipeline = joblib.load('backend/pipeline.joblib')
        print("Pipeline loaded.")
    else:
        print("Warning: pipeline.joblib not found.")

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not model or not pipeline:
        raise HTTPException(status_code=503, detail="Model or Pipeline not loaded.")
        
    try:
        # Save temp file
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Transform using Pipeline
        # Pipeline expects a dataframe-like or list of paths.
        # We pass a dataframe with one row.
        import pandas as pd
        X_df = pd.DataFrame([{'path': temp_path}])
        
        # Pipeline transform
        # shape: (1, N_MELS * MAX_TIME)
        features_flat = pipeline.transform(X_df[['path']])
        
        # Reshape for RNN: (Batch=1, Time, Feat)
        feat_2d = features_flat[0].reshape(N_MELS, MAX_TIME_FRAMES).T
        
        # To Tensor
        tensor_in = torch.tensor(feat_2d, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(tensor_in)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = int(probs.argmax())
        
        # Decode Label
        if label_encoder:
            emotion = label_encoder.inverse_transform([pred_idx])[0]
        else:
            emotion = str(pred_idx)
        
        # Cleanup
        os.remove(temp_path)
        
        return {
            "emotion": emotion,
            "confidence": float(probs[pred_idx]),
            "probabilities": probs.tolist()
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
