import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Import our modules
from backend.dataset_loader import load_combined_dataset
from backend.pipeline import create_pipeline, N_MELS, MAX_TIME_FRAMES
from backend.model import RNNEmotionClassifier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ProcessedDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Feature is a flattened array from StandardScaler: (N_MELS * MAX_TIME)
        # Reshape to (MAX_TIME, N_MELS) because LSTM expects (Batch, Time, Feat)
        flat_feat = self.features[idx]
        
        # Reshape
        # Note: In pipeline, we flattened (N_MELS, MAX_TIME).
        # So we reshape back to (N_MELS, MAX_TIME) then transpose to (MAX_TIME, N_MELS)
        feat_2d = flat_feat.reshape(N_MELS, MAX_TIME_FRAMES).T
        
        return torch.tensor(feat_2d, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def train_main():
    print("Loading datasets...")
    df = load_combined_dataset()
    if df.empty:
        print("No data found. Please check dataset paths.")
        return

    print(f"Loaded {len(df)} samples.")
    
    # Encoder Labels
    le = LabelEncoder()
    df['label_idx'] = le.fit_transform(df['label'])
    
    # Save Label Encoder
    joblib.dump(le, 'backend/label_encoder.joblib')
    
    # Split Data preventing Leakage
    # GroupShuffleSplit ensures all samples from a speaker go to either Train or Val
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['speaker_id']))
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    print(f"Train samples: {len(train_df)} (Speakers: {train_df['speaker_id'].nunique()})")
    print(f"Val samples: {len(val_df)} (Speakers: {val_df['speaker_id'].nunique()})")
    
    # Check for leakage
    train_speakers = set(train_df['speaker_id'])
    val_speakers = set(val_df['speaker_id'])
    intersect = train_speakers.intersection(val_speakers)
    if intersect:
        print(f"WARNING: Data Leakage detected! Speakers in both sets: {intersect}")
    else:
        print("Success: No speaker leakage detected.")

    # Pipeline
    print("Fitting Preprocessing Pipeline on TRAIN data...")
    pipeline = create_pipeline()
    
    # Input to pipeline is the DF or list of paths.
    # Our pipeline supports list of paths in column 0 if using ColumnTransformer directly on DF,
    # or just list of paths. Our `create_pipeline` expects a DF-like where col 0 is processed.
    # Actually `AudioFeatureExtractor` takes X as list of paths.
    
    # Let's pass the 'path' column as a Dataframe to match ColumnTransformer expectations
    X_train_raw = train_df[['path']]
    X_val_raw = val_df[['path']]
    
    # Fit Pipeline
    X_train_processed = pipeline.fit_transform(X_train_raw)
    
    # Transform Val
    X_val_processed = pipeline.transform(X_val_raw)
    
    # Save Pipeline
    joblib.dump(pipeline, 'backend/pipeline.joblib')
    
    # Create PyTorch Datasets
    train_ds = ProcessedDataset(X_train_processed, train_df['label_idx'].values)
    val_ds = ProcessedDataset(X_val_processed, val_df['label_idx'].values)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    # Model
    model = RNNEmotionClassifier(input_size=N_MELS, num_classes=len(le.classes_)).to(DEVICE)
    
    # Training Config
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    best_acc = 0.0
    
    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            
            # Asserting Backpropagation
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                outputs = model(X)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        acc = correct / total
        avg_loss = running_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'backend/best_model.pt')
            print("  -> Saved Best Model")

    print(f"Training Complete. Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train_main()
