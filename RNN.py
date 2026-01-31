"""
RNN-based Emotion Detector + Whisper transcription + Cyberpunk UI

Features:
- Hybrid architecture: PyTorch LSTM for emotion + OpenAI Whisper for text.
- Techy/Cyberpunk Dashboard using Streamlit.
- Supports training on RAVDESS, IEMOCAP, MAD.
- Real-time spectrogram and waveform visualization (Oscilloscope style).
"""

import os
import math
import json
import glob
import time
import librosa
import random
import soundfile as sf
import numpy as np
import pandas as pd  # Added for Dataframes
from collections import Counter
from statistics import mode
from typing import List, Tuple
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg') # Non-interactive backend for safety
import matplotlib.pyplot as plt

import streamlit as st

# Optional: whisper from openai/whisper
try:
    import whisper
except Exception:
    whisper = None

# ----------------------------- Configuration -----------------------------
# User dataset paths (EDIT these)
RAVDESS_PATH = "content/RAVDESS DATASET"          #
IEMOCAP_PATH = "content/IEMCAP DATABASE"  #
MAD_PATH = "content/M.A.D. (MILITARY AUDIO DATASET)" #

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_RATE = 16000
CHUNK_SEC = 5.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SEC)
N_MELS = 64

EMOTION_LABELS = [
    'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'
]
LABEL2IDX = {label: i for i, label in enumerate(EMOTION_LABELS)}
IDX2LABEL = {i: label for label, i in LABEL2IDX.items()}

# ----------------------------- Backend Utilities -----------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_audio(path, sr=SAMPLE_RATE):
    wav, r = librosa.load(path, sr=sr, mono=True)
    return wav

def chunk_audio(wav: np.ndarray, chunk_sec: float = CHUNK_SEC) -> List[np.ndarray]:
    chunk_samples = int(SAMPLE_RATE * chunk_sec)
    chunks = []
    total = len(wav)
    for start in range(0, total, chunk_samples):
        end = min(start + chunk_samples, total)
        chunk = wav[start:end]
        if len(chunk) < chunk_samples:
            pad = np.zeros(chunk_samples - len(chunk), dtype=chunk.dtype)
            chunk = np.concatenate([chunk, pad])
        chunks.append(chunk)
    return chunks

def compute_mel(wav: np.ndarray, sr=SAMPLE_RATE, n_mels=N_MELS) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db  # shape (n_mels, T)

# ----------------------------- Dataset & Model -----------------------------

class EmotionDataset(Dataset):
    def __init__(self, manifest: List[Tuple[str, int]], augment=False):
        self.samples = manifest
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        wav = load_audio(path)
        mel = compute_mel(wav)
        max_time = 128
        if mel.shape[1] < max_time:
            pad = np.zeros((mel.shape[0], max_time - mel.shape[1]))
            mel = np.concatenate([mel, pad], axis=1)
        else:
            mel = mel[:, :max_time]
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)
        mel_tensor = torch.tensor(mel, dtype=torch.float32)
        return mel_tensor, label

class LSTMEmotionClassifier(nn.Module):
    def __init__(self, n_mels=N_MELS, hidden=128, num_layers=2, num_classes=len(EMOTION_LABELS)):
        super().__init__()
        self.n_mels = n_mels
        self.rnn = nn.LSTM(input_size=n_mels, hidden_size=hidden, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# ----------------------------- Training Logic -----------------------------

def make_ravdess_manifest(root_folder: str) -> List[Tuple[str, int]]:
    manifest = []
    emotion_map = {'01': 0, '02': 1, '03': 2, '04': 3, '05': 4, '06': 5, '07': 6, '08': 7}
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.wav'):
                parts = file.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    if emotion_code in emotion_map:
                        manifest.append((os.path.join(root, file), emotion_map[emotion_code]))
    return manifest

def make_iemocap_manifest(root_folder: str) -> List[Tuple[str, int]]:
    manifest = []
    emotion_map = {'neu': 0, 'hap': 2, 'sad': 3, 'ang': 4, 'fea': 5, 'sur': 7, 'dis': 6, 'fru': 4}
    for session in range(1, 6):
        emo_eval_path = os.path.join(root_folder, f'Session{session}', 'dialog', 'EmoEvaluation')
        if not os.path.isdir(emo_eval_path): continue
        for txt_file in glob.glob(os.path.join(emo_eval_path, '*.txt')):
            with open(txt_file, 'r') as f:
                for line in f:
                    if line.startswith('['):
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            wav_file = parts[0].strip('[]')
                            emotion = parts[3]
                            if emotion in emotion_map:
                                folder = wav_file.split('_')[0] + '_' + wav_file.split('_')[1]
                                wav_path = os.path.join(root_folder, f'Session{session}', 'sentences', 'wav', folder, f'{wav_file}.wav')
                                if os.path.exists(wav_path):
                                    manifest.append((wav_path, emotion_map[emotion]))
    return manifest

def make_mad_manifest(root_folder: str) -> List[Tuple[str, int]]:
    manifest = []
    emotion_map = {'01': 0, '02': 1, '03': 2, '04': 3, '05': 4, '06': 5, '07': 6, '08': 7}
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.wav'):
                parts = file.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    if emotion_code in emotion_map:
                        manifest.append((os.path.join(root, file), emotion_map[emotion_code]))
    return manifest

def train_model(train_manifest, val_manifest, save_path='best_emotion_model.pt', epochs=20, batch_size=16):
    train_ds = EmotionDataset(train_manifest)
    val_ds = EmotionDataset(val_manifest)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = LSTMEmotionClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val = 0.0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} Loss={running/len(train_loader):.4f} Val_Acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), save_path)
            print("Saved best model")
    return model

def train_from_datasets():
    train_manifest = []
    train_manifest += make_ravdess_manifest(RAVDESS_PATH)
    train_manifest += make_iemocap_manifest(IEMOCAP_PATH)
    train_manifest += make_mad_manifest(MAD_PATH)
    random.shuffle(train_manifest)
    split = int(0.9 * len(train_manifest))
    val_manifest = train_manifest[split:]
    train_manifest = train_manifest[:split]
    if not train_manifest:
        print('No files found. Check paths.')
    else:
        train_model(train_manifest, val_manifest)

# ----------------------------- Inference Logic -----------------------------

_whisper_model = None
def get_whisper_model(name='small'):
    global _whisper_model
    if _whisper_model is None:
        if whisper is None: raise RuntimeError('Whisper library missing')
        _whisper_model = whisper.load_model(name)
    return _whisper_model

def transcribe_chunk(chunk_wav: np.ndarray, whisper_name='small') -> dict:
    model = get_whisper_model(whisper_name)
    audio = chunk_wav.astype(np.float32)
    result = model.transcribe(audio)
    return result

def predict_emotion_on_chunk(model: nn.Module, chunk_wav: np.ndarray) -> Tuple[str, List[float]]:
    mel = compute_mel(chunk_wav)
    max_time = 128
    if mel.shape[1] < max_time:
        pad = np.zeros((mel.shape[0], max_time - mel.shape[1]))
        mel = np.concatenate([mel, pad], axis=1)
    else:
        mel = mel[:, :max_time]
    mel = (mel - mel.mean()) / (mel.std() + 1e-9)
    x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        label = IDX2LABEL.get(idx, 'unknown')
    return label, probs.tolist()

# ----------------------------- NEW TECHY UI -----------------------------

def get_waveform(wav, sr=SAMPLE_RATE, theme="Dark (Cyberpunk)"):
    fig, ax = plt.subplots(figsize=(12, 2))
    if theme == "Dark (Cyberpunk)":
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        color = '#00FFFF'
        tick_color = '#8B949E'
    else:
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#FFFFFF')
        color = '#000000'
        tick_color = '#6C757D'
    times = np.arange(len(wav)) / float(sr)
    ax.plot(times, wav, color=color, linewidth=0.6, alpha=0.9)
    ax.axis('off')
    plt.tight_layout()
    return fig

def get_spectrogram(wav, sr=SAMPLE_RATE, theme="Dark (Cyberpunk)"):
    fig, ax = plt.subplots(figsize=(12, 3))
    if theme == "Dark (Cyberpunk)":
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        tick_color = '#8B949E'
    else:
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#FFFFFF')
        tick_color = '#6C757D'
    D = librosa.stft(wav)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='inferno')
    ax.tick_params(axis='x', colors=tick_color)
    ax.tick_params(axis='y', colors=tick_color)
    ax.set_ylabel('FREQ (Hz)', color=tick_color, fontsize=8)
    ax.set_xlabel('TIME (s)', color=tick_color, fontsize=8)
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color=tick_color)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=tick_color, fontsize=8)
    plt.tight_layout()
    return fig

def run_ui():
    st.set_page_config(page_title="NEURAL AUDIO LAB", page_icon="🎛️", layout="wide", initial_sidebar_state="expanded")

    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ SYSTEM CONTROLS")
        st.markdown("`STATUS: ONLINE`")
        st.divider()
        uploaded_file = st.file_uploader("INPUT SIGNAL SOURCE", type=["wav", "mp3", "flac"])
        st.markdown("### ⚙️ PARAMETERS")
        theme = st.selectbox("THEME", options=["Dark (Cyberpunk)", "Light"], index=0)
        whisper_model = st.select_slider("TRANSCRIPTION ENGINE", options=["tiny", "base", "small"], value="small")
        chunk_sec = st.slider("CHUNK DURATION (sec)", min_value=1.0, max_value=10.0, value=CHUNK_SEC, step=0.5)
        n_mels_slider = st.slider("NUMBER OF MELS", min_value=32, max_value=128, value=N_MELS, step=16)
        show_spec = st.checkbox("ENABLE SPECTRAL ANALYSIS", value=True)
        st.divider()
        st.caption("v2.0 | RNN-LSTM | WHISPER")

    # Theme-based CSS
    if theme == "Dark (Cyberpunk)":
        css = """
        <style>
            .stApp { background-color: #0E1117; font-family: 'Courier New', Courier, monospace; }
            section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
            div[data-testid="stMetric"] { background-color: #21262D; border: 1px solid #30363D; border-radius: 4px; padding: 10px; }
            div[data-testid="stMetricLabel"] { color: #8B949E; font-size: 0.8rem; text-transform: uppercase; }
            div[data-testid="stMetricValue"] { color: #58A6FF; font-family: 'Roboto Mono', monospace; }
            h1, h2, h3 { color: #E6EDF3; font-family: 'Orbitron', sans-serif; text-transform: uppercase; letter-spacing: 2px; }
            div[data-testid="stDataFrame"] { border: 1px solid #30363D; }
        </style>
        """
    else:
        css = """
        <style>
            .stApp { background-color: #FFFFFF; font-family: 'Arial', sans-serif; }
            section[data-testid="stSidebar"] { background-color: #F8F9FA; border-right: 1px solid #DEE2E6; }
            div[data-testid="stMetric"] { background-color: #FFFFFF; border: 1px solid #DEE2E6; border-radius: 4px; padding: 10px; }
            div[data-testid="stMetricLabel"] { color: #6C757D; font-size: 0.8rem; text-transform: uppercase; }
            div[data-testid="stMetricValue"] { color: #007BFF; font-family: 'Arial', sans-serif; }
            h1, h2, h3 { color: #212529; font-family: 'Arial', sans-serif; text-transform: uppercase; letter-spacing: 2px; }
            div[data-testid="stDataFrame"] { border: 1px solid #DEE2E6; }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

    # Main Display
    st.title("NEURAL AUDIO LAB")
    st.markdown("`> INITIALIZING INFERENCE PIPELINE...`")

    if uploaded_file is None:
        st.info("AWAITING INPUT SIGNAL via SIDEBAR MODULE")
        return

    with st.spinner("PROCESSING AUDIO STREAM..."):
        # Save and Load
        tmp_path = os.path.join('static', uploaded_file.name)
        ensure_dir('static')
        with open(tmp_path, 'wb') as f: f.write(uploaded_file.getbuffer())
        
        wav = load_audio(tmp_path)
        duration = len(wav) / SAMPLE_RATE
        
        # Model Load
        model = LSTMEmotionClassifier().to(DEVICE)
        if os.path.exists('best_emotion_model.pt'):
            model.load_state_dict(torch.load('best_emotion_model.pt', map_location=DEVICE))
        else:
            st.warning("⚠️ PRE-TRAINED MODEL NOT FOUND. USING RANDOM WEIGHTS.")

        # Inference Loop
        chunks = chunk_audio(wav, chunk_sec=chunk_sec)
        results_data = []

        for i, c in enumerate(chunks):
            try:
                tr = transcribe_chunk(c, whisper_name=whisper_model)
                text = tr.get('text', '').strip()
            except: text = "[SIG_ERR]"

            label, probs = predict_emotion_on_chunk(model, c)
            results_data.append({
                "ID": i,
                "START (s)": i * chunk_sec,
                "DETECTED EMOTION": label.upper(),
                "CONFIDENCE": max(probs),
                "TRANSCRIPT": text
            })

    # 1. Metrics HUD
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("SAMPLE RATE", f"{SAMPLE_RATE} Hz")
    with col2: st.metric("DURATION", f"{duration:.2f} sec")
    with col3:
        emotions = [r['DETECTED EMOTION'] for r in results_data]
        dominant = max(set(emotions), key=emotions.count) if emotions else "N/A"
        st.metric("DOMINANT CLASS", dominant)
    with col4: st.metric("CHUNKS PROCESSED", len(chunks))
    st.markdown("---")

    # Tabs for organized display
    tab1, tab2, tab3 = st.tabs(["Overview", "Analysis", "Results"])

    with tab1:
        # Metrics HUD
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("SAMPLE RATE", f"{SAMPLE_RATE} Hz")
        with col2: st.metric("DURATION", f"{duration:.2f} sec")
        with col3:
            emotions = [r['DETECTED EMOTION'] for r in results_data]
            dominant = max(set(emotions), key=emotions.count) if emotions else "N/A"
            st.metric("DOMINANT CLASS", dominant)
        with col4: st.metric("CHUNKS PROCESSED", len(chunks))
        st.markdown("---")
        # Audio Player and Stats
        st.markdown("#### 🎧 MONITOR")
        st.audio(uploaded_file)
        st.markdown("#### ℹ️ SIGNAL STATS")
        st.markdown(f"- **RMS Level:** {np.sqrt(np.mean(wav**2)):.4f}\n- **Bit Depth:** 32-bit Float")

    with tab2:
        # Visuals
        st.markdown("#### 📈 AMPLITUDE ENVELOPE")
        st.pyplot(get_waveform(wav, theme=theme))
        if show_spec:
            st.markdown("#### 📉 SPECTRAL DENSITY")
            st.pyplot(get_spectrogram(wav, theme=theme))

    with tab3:
        # Data Log
        st.markdown("#### 🖥️ INFERENCE LOG")
        df = pd.DataFrame(results_data)
        st.dataframe(
            df,
            column_config={
                "CONFIDENCE": st.column_config.ProgressColumn("CONFIDENCE", format="%.2f", min_value=0, max_value=1),
                "DETECTED EMOTION": st.column_config.TextColumn("CLASS", width="small"),
                "TRANSCRIPT": st.column_config.TextColumn("DECODED SEQUENCE", width="large")
            },
            width='stretch',
            hide_index=True
        )

    # Re-Analyze Button
    if st.button("🔄 RE-ANALYZE"):
        st.rerun()

# ----------------------------- Entry Point -----------------------------
if __name__ == '__main__':
    # If running via Streamlit, sys.argv[0] usually contains 'streamlit'
    # If running via python cli, we check args.
    import argparse
    
    if '--train' in sys.argv:
        # CLI Training Mode
        parser = argparse.ArgumentParser()
        parser.add_argument('--train', action='store_true', help='Train model from datasets')
        args = parser.parse_args()
        if args.train:
            train_from_datasets()
    else:
        # UI Mode (Streamlit)
        try:
            # This checks if we are inside a Streamlit runtime
            from streamlit.web import cli as stcli
            run_ui()
        except ImportError:
            # Fallback if checks fail
            run_ui()